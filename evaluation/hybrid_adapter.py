import time
from typing import Any

import torch

from .classes import TranslationOutput, TranslationModelAdapter


class WaitKHybridMamba2Adapter(TranslationModelAdapter):
    def __init__(
        self,
        *,
        model,
        tokenizer,
        name: str = "HybridMamba2CrossAttn",
        device: str | torch.device = "cuda",
        target_lang: str = "rus_Cyrl",
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        synchronize_timing: bool = True,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.name = name
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.synchronize_timing = synchronize_timing
        self.target_lang_token_id = tokenizer.convert_tokens_to_ids(target_lang)
        if self.target_lang_token_id is None or self.target_lang_token_id < 0:
            raise ValueError(f"Unknown target language token: {target_lang}")

    @torch.inference_mode()
    def translate_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        wait_k: int,
        max_new_tokens: int,
        speed: int = 1,
        stop_after_eos_when_full_source_read: bool = True,
        **generation_kwargs: Any,
    ) -> list[TranslationOutput]:
        self.model.eval()

        source_ids = batch["source_ids"].to(self.device, non_blocking=True).long()
        source_mask = batch["source_mask"].to(self.device, non_blocking=True).bool()
        max_src_len = min(source_ids.size(1), self.model.cfg.max_source_len)
        source_ids = source_ids[:, :max_src_len]
        source_mask = source_mask[:, :max_src_len]
        source_lens = source_mask.long().sum(dim=1).clamp_min(1).clamp_max(max_src_len)

        cache_dtype = (
            self.amp_dtype
            if self.use_amp and self.device.type == "cuda"
            else next(self.model.parameters()).dtype
        )

        if self.synchronize_timing and self.device.type == "cuda":
            torch.cuda.synchronize()
        batch_start = time.perf_counter()

        with torch.autocast(
            device_type="cuda",
            enabled=self.use_amp and self.device.type == "cuda",
            dtype=self.amp_dtype,
        ):
            generated = self.model.generate_waitk(
                source_ids=source_ids,
                source_mask=source_mask,
                target_lang_token_id=self.target_lang_token_id,
                max_new_tokens=max_new_tokens,
                k=wait_k,
                speed=speed,
                stop_after_eos_when_full_source_read=stop_after_eos_when_full_source_read,
                cache_dtype=cache_dtype,
            )

        if self.synchronize_timing and self.device.type == "cuda":
            torch.cuda.synchronize()
        total_batch_time = time.perf_counter() - batch_start

        generated_cpu = generated.detach().cpu()
        generated_only = generated_cpu[:, 1:]
        hypothesis_texts = self.tokenizer.batch_decode(
            generated_only.tolist(),
            skip_special_tokens=True,
        )
        source_lens_cpu = source_lens.detach().cpu().tolist()

        outputs: list[TranslationOutput] = []
        batch_size = generated.size(0)
        for i in range(batch_size):
            hyp_ids = generated_only[i].tolist()
            if self.model.cfg.eos_token_id in hyp_ids:
                eos_pos = hyp_ids.index(self.model.cfg.eos_token_id)
                hyp_ids = hyp_ids[: eos_pos + 1]
            while hyp_ids and hyp_ids[-1] == self.model.cfg.pad_token_id:
                hyp_ids.pop()

            target_len = max(1, len(hyp_ids))
            delays = [
                min(source_lens_cpu[i], wait_k + j * speed)
                for j in range(target_len)
            ]
            outputs.append(
                TranslationOutput(
                    hypothesis_ids=hyp_ids,
                    hypothesis_text=hypothesis_texts[i],
                    delays=delays,
                    target_len=target_len,
                    first_token_latency_sec=None,
                    generation_time_sec=total_batch_time / max(1, batch_size),
                    extra={
                        "hybrid": True,
                        "causal_source_encode_once": True,
                        "target_lang_token_id": self.target_lang_token_id,
                    },
                )
            )
        return outputs
