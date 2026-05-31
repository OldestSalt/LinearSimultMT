import time
from typing import Any

import torch

from .evaluator import *


class NLLBSimulMTAdapter:
    """
    SimulMT adapter for HuggingFace AutoModelForSeq2SeqLM, e.g. NLLB.

    Important:
        This adapter does NOT use model.generate().

    For NLLB decoder input we use:
        [decoder_start_token_id, target_lang_token_id, generated_token_1, ...]

    Source prefix format at every step:
        visible_source_tokens + eos_token + padding

    This is important for NLLB, because the encoder expects a properly
    terminated source sequence.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        name: str = "NLLB-Seq2Seq-SimulMT",
        device: str | torch.device = "cuda",
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        max_source_len: int | None = None,
        duplicate_eos_on_full_source: bool = False,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.name = name
        self.device = torch.device(device)

        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.max_source_len = max_source_len
        self.duplicate_eos_on_full_source = duplicate_eos_on_full_source

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self.decoder_start_token_id = getattr(
            model.config,
            "decoder_start_token_id",
            None,
        )

        if self.decoder_start_token_id is None:
            raise ValueError("model.config.decoder_start_token_id is None")

        self.target_lang_token_id = tokenizer.convert_tokens_to_ids("rus_Cyrl")

    def _make_source_prefix_batch(
        self,
        *,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        visible_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build source prefix batch:

            source[:visible_len] + eos + pad...

        Args:
            source_ids:
                [batch, src_len]

            source_mask:
                [batch, src_len]

            visible_lens:
                [batch], number of visible source tokens.

        Returns:
            prefix_source_ids:
                [batch, prefix_len]

            prefix_source_mask:
                [batch, prefix_len]
        """
        device = source_ids.device
        batch_size, src_len = source_ids.shape

        visible_lens = visible_lens.clamp_min(1).clamp_max(src_len)

        batch_idx = torch.arange(
            batch_size,
            device=device,
        )

        last_visible_idx = visible_lens - 1
        last_visible_tokens = source_ids[batch_idx, last_visible_idx]

        already_ends_with_eos = last_visible_tokens.eq(self.eos_token_id)

        if self.duplicate_eos_on_full_source:
            add_eos = torch.ones_like(already_ends_with_eos, dtype=torch.bool)
        else:
            add_eos = ~already_ends_with_eos

        output_lens = visible_lens + add_eos.long()
        prefix_len = int(output_lens.max().item())

        positions = torch.arange(
            prefix_len,
            device=device,
        ).unsqueeze(0)

        copy_mask = positions < visible_lens.unsqueeze(1)
        eos_mask = positions.eq(visible_lens.unsqueeze(1)) & add_eos.unsqueeze(1)
        valid_mask = copy_mask | eos_mask

        prefix_source_ids = torch.full(
            size=(batch_size, prefix_len),
            fill_value=self.pad_token_id,
            dtype=source_ids.dtype,
            device=device,
        )

        prefix_source_mask = valid_mask.long()

        src_positions = positions.clamp_max(src_len - 1).expand(batch_size, prefix_len)

        copied_tokens = source_ids.gather(
            dim=1,
            index=src_positions,
        )

        prefix_source_ids = torch.where(
            copy_mask,
            copied_tokens,
            prefix_source_ids,
        )

        prefix_source_ids = torch.where(
            eos_mask,
            torch.full_like(prefix_source_ids, self.eos_token_id),
            prefix_source_ids,
        )

        return prefix_source_ids, prefix_source_mask

    @torch.inference_mode()
    def translate_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        wait_k: int,
        max_new_tokens: int,
        speed: int = 1,
        mode: str = "simulmt",
        stop_after_eos_when_full_source_read: bool = True,
        **generation_kwargs: Any,
    ) -> list[TranslationOutput]:
        if mode not in {"simulmt", "offline"}:
            raise ValueError("mode must be either 'simulmt' or 'offline'")
    
        self.model.eval()
    
        batch_start = time.perf_counter()
    
        source_ids = batch["source_ids"].to(
            self.device,
            non_blocking=True,
        ).long()
    
        source_mask = batch["source_mask"].to(
            self.device,
            non_blocking=True,
        ).long()
    
        if self.max_source_len is not None:
            source_ids = source_ids[:, :self.max_source_len]
            source_mask = source_mask[:, :self.max_source_len]
    
        batch_size, src_len = source_ids.shape
        source_lens = source_mask.sum(dim=1).long().clamp_min(1)
    
        decoder_input_ids = torch.full(
            size=(batch_size, 2),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
    
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        decoder_input_ids[:, 1] = self.target_lang_token_id
    
        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device,
        )
    
        delays: list[list[int]] = [[] for _ in range(batch_size)]
    
        first_token_latency_sec = None
        generation_start = time.perf_counter()
    
        if mode == "offline":
            full_source_ids, full_source_mask = self._make_source_prefix_batch(
                source_ids=source_ids,
                source_mask=source_mask,
                visible_lens=source_lens,
            )
    
        for step in range(max_new_tokens):
            if mode == "simulmt":
                visible_lens = torch.minimum(
                    source_lens,
                    torch.full_like(source_lens, wait_k + step * speed),
                ).clamp_min(1)
    
                prefix_source_ids, prefix_source_mask = self._make_source_prefix_batch(
                    source_ids=source_ids,
                    source_mask=source_mask,
                    visible_lens=visible_lens,
                )
    
            else:
                visible_lens = source_lens
                prefix_source_ids = full_source_ids
                prefix_source_mask = full_source_mask
    
            decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).long()
    
            with torch.autocast(
                device_type="cuda",
                enabled=self.use_amp and self.device.type == "cuda",
                dtype=self.amp_dtype,
            ):
                outputs = self.model(
                    input_ids=prefix_source_ids,
                    attention_mask=prefix_source_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
    
                next_logits = outputs.logits[:, -1, :]
    
            next_token = next_logits.argmax(dim=-1)
    
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.pad_token_id),
                next_token,
            )
    
            if first_token_latency_sec is None:
                first_token_latency_sec = time.perf_counter() - generation_start
    
            active_before_step = ~finished
    
            for i in range(batch_size):
                if bool(active_before_step[i].item()):
                    delays[i].append(int(visible_lens[i].item()))
    
            decoder_input_ids = torch.cat(
                [decoder_input_ids, next_token[:, None]],
                dim=1,
            )
    
            if mode == "offline":
                finished |= next_token.eq(self.eos_token_id)
            elif stop_after_eos_when_full_source_read:
                finished |= (
                    next_token.eq(self.eos_token_id)
                    & (visible_lens >= source_lens)
                )
            else:
                finished |= next_token.eq(self.eos_token_id)
    
            if bool(finished.all().item()):
                break
    
        total_batch_time = time.perf_counter() - batch_start
    
        decoder_input_ids_cpu = decoder_input_ids.detach().cpu()
    
        generated_only = decoder_input_ids_cpu[:, 2:]
    
        hypothesis_texts = self.tokenizer.batch_decode(
            generated_only.tolist(),
            skip_special_tokens=True,
        )
    
        outputs: list[TranslationOutput] = []
    
        for i in range(batch_size):
            hyp_ids = generated_only[i].tolist()
    
            if self.eos_token_id in hyp_ids:
                eos_pos = hyp_ids.index(self.eos_token_id)
                hyp_ids = hyp_ids[:eos_pos + 1]
    
            while hyp_ids and hyp_ids[-1] == self.pad_token_id:
                hyp_ids.pop()
    
            outputs.append(
                TranslationOutput(
                    hypothesis_ids=hyp_ids,
                    hypothesis_text=hypothesis_texts[i],
                    delays=delays[i],
                    target_len=max(1, len(delays[i])),
                    first_token_latency_sec=first_token_latency_sec,
                    generation_time_sec=total_batch_time / max(1, batch_size),
                    extra={
                        "decoder_start_token_id": self.decoder_start_token_id,
                        "target_lang_token_id": self.target_lang_token_id,
                        "mode": mode,
                        "offline": mode == "offline",
                    },
                )
            )
    
        return outputs