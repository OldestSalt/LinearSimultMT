import time
from typing import Any

import torch

from .classes import TranslationOutput, TranslationModelAdapter


class WaitKTransformerAdapter(TranslationModelAdapter):
    """
    Computationally honest streaming adapter for WaitKTransformerMT.

    Difference from the fast adapter:
        - does NOT precompute full-source memory;
        - at each target step, encoder receives only visible source prefix;
        - this is slower, but fairer for wall-clock streaming efficiency.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        name: str = "HonestStreamingWaitKTransformer",
        device: str | torch.device = "cuda",
        target_lang: str = "rus_Cyrl",
        causal_encoder: bool = True,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        synchronize_timing: bool = True,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.name = name
        self.device = torch.device(device)

        self.causal_encoder = causal_encoder
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
        break_when_all_finished: bool = False,
        **generation_kwargs: Any,
    ) -> list[TranslationOutput]:
        self.model.eval()

        source_ids = batch["source_ids"].to(
            self.device,
            non_blocking=True,
        ).long()

        source_mask = batch["source_mask"].to(
            self.device,
            non_blocking=True,
        ).bool()

        max_src_len = min(
            source_ids.size(1),
            self.model.cfg.max_seq_len,
        )

        source_ids = source_ids[:, :max_src_len]
        source_mask = source_mask[:, :max_src_len]

        batch_size, src_len = source_ids.shape

        source_lens = source_mask.long().sum(dim=1).clamp_min(1).clamp_max(src_len)

        generated = torch.full(
            size=(batch_size, 1),
            fill_value=self.target_lang_token_id,
            dtype=torch.long,
            device=self.device,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device,
        )

        source_positions = torch.arange(
            src_len,
            device=self.device,
        ).unsqueeze(0)

        if self.synchronize_timing and self.device.type == "cuda":
            torch.cuda.synchronize()

        batch_start = time.perf_counter()
        generation_start = batch_start

        max_steps = min(
            max_new_tokens,
            max(1, self.model.cfg.max_seq_len - 1),
        )

        for step in range(max_steps):
            if generated.size(1) >= self.model.cfg.max_seq_len:
                break

            visible_lens = torch.minimum(
                source_lens,
                torch.full_like(source_lens, wait_k + step * speed),
            ).clamp_min(1)

            max_visible_len = min(
                src_len,
                wait_k + step * speed,
            )

            prefix_ids = source_ids[:, :max_visible_len]

            prefix_mask = (
                source_mask[:, :max_visible_len]
                & (source_positions[:, :max_visible_len] < visible_lens.unsqueeze(1))
            )

            target_mask = generated.ne(self.model.cfg.pad_token_id).long()

            with torch.autocast(
                device_type="cuda",
                enabled=self.use_amp and self.device.type == "cuda",
                dtype=self.amp_dtype,
            ):
                memory = self.model.encode(
                    source_ids=prefix_ids,
                    source_mask=prefix_mask.long(),
                    causal=self.causal_encoder,
                )

                hidden = self.model.decode(
                    target_input_ids=generated,
                    memory=memory,
                    target_input_mask=target_mask,
                    source_mask=prefix_mask.long(),
                    memory_mask=None,
                )

                next_logits = self.model.lm_head(hidden[:, -1, :])

            next_token = next_logits.argmax(dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.model.cfg.pad_token_id),
                next_token,
            )

            generated = torch.cat(
                [generated, next_token[:, None]],
                dim=1,
            )

            if stop_after_eos_when_full_source_read:
                finished |= (
                    next_token.eq(self.model.cfg.eos_token_id)
                    & (visible_lens >= source_lens)
                )
            else:
                finished |= next_token.eq(self.model.cfg.eos_token_id)

            if break_when_all_finished and bool(finished.all().item()):
                break

        if self.synchronize_timing and self.device.type == "cuda":
            torch.cuda.synchronize()

        first_token_latency_sec = time.perf_counter() - generation_start
        total_batch_time = time.perf_counter() - batch_start

        generated_cpu = generated.detach().cpu()
        generated_only = generated_cpu[:, 1:]

        hypothesis_texts = self.tokenizer.batch_decode(
            generated_only.tolist(),
            skip_special_tokens=True,
        )

        source_lens_cpu = source_lens.detach().cpu().tolist()

        outputs: list[TranslationOutput] = []

        for i in range(batch_size):
            hyp_ids = generated_only[i].tolist()

            if self.model.cfg.eos_token_id in hyp_ids:
                eos_pos = hyp_ids.index(self.model.cfg.eos_token_id)
                hyp_ids = hyp_ids[:eos_pos + 1]

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
                    first_token_latency_sec=first_token_latency_sec,
                    generation_time_sec=total_batch_time / max(1, batch_size),
                    extra={
                        "honest_streaming": True,
                    },
                )
            )

        return outputs