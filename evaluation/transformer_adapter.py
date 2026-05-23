import torch
from typing import Any
import time
from .classes import TranslationOutput, TranslationModelAdapter


class WaitKTransformerAdapter(TranslationModelAdapter):
    def __init__(
        self,
        *,
        model,
        tokenizer,
        name: str = "WaitKTransformer",
        device: str | torch.device = "cuda",
        causal_encoder: bool = True,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.name = name
        self.device = torch.device(device)
        self.causal_encoder = causal_encoder
        self.use_amp = use_amp

        self.target_lang_token_id = tokenizer.convert_tokens_to_ids("rus_Cyrl")

    @torch.inference_mode()
    def translate_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        wait_k: int,
        max_new_tokens: int,
        speed: int = 1,
        **generation_kwargs: Any,
    ) -> list[TranslationOutput]:
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

        # Limit source length to model positional limit.
        max_src_len = min(
            source_ids.size(1),
            self.model.cfg.max_seq_len,
        )

        source_ids = source_ids[:, :max_src_len]
        source_mask = source_mask[:, :max_src_len]

        batch_size, src_len = source_ids.shape

        source_lens = source_mask.long().sum(dim=1).clamp_min(1)
        source_lens = source_lens.clamp_max(src_len)

        # One encoder forward for the whole batch.
        with torch.autocast(
            device_type="cuda",
            enabled=self.use_amp and self.device.type == "cuda",
        ):
            memory = self.model.encode(
                source_ids=source_ids,
                source_mask=source_mask,
                causal=self.causal_encoder,
            )

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

        delays: list[list[int]] = [[] for _ in range(batch_size)]
        first_token_latency_sec = None
        generation_start = time.perf_counter()

        # Used to build per-example visible source masks.
        source_positions = torch.arange(
            src_len,
            device=self.device,
        ).unsqueeze(0)

        for step in range(max_new_tokens):
            if generated.size(1) >= self.model.cfg.max_seq_len:
                break

            visible_lens = torch.minimum(
                source_lens,
                torch.full_like(source_lens, wait_k + step * speed),
            )

            visible_lens = visible_lens.clamp_min(1)

            # [B, S], masks source positions not yet visible.
            current_source_mask = (
                source_mask.bool()
                & (source_positions < visible_lens.unsqueeze(1))
            ).long()

            target_mask = generated.ne(self.model.cfg.pad_token_id).long()

            with torch.autocast(
                device_type="cuda",
                enabled=self.use_amp and self.device.type == "cuda",
            ):
                hidden = self.model.decode(
                    target_input_ids=generated,
                    memory=memory,
                    target_input_mask=target_mask,
                    source_mask=current_source_mask,
                    memory_mask=None,
                )

                next_logits = self.model.lm_head(hidden[:, -1, :])

            next_token = next_logits.argmax(dim=-1)

            # Do not continue generating real tokens for finished sequences.
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.model.cfg.pad_token_id),
                next_token,
            )

            if first_token_latency_sec is None:
                first_token_latency_sec = time.perf_counter() - generation_start

            active_before_step = ~finished

            for i in range(batch_size):
                if bool(active_before_step[i].item()):
                    delays[i].append(int(visible_lens[i].item()))

            generated = torch.cat(
                [generated, next_token[:, None]],
                dim=1,
            )

            finished |= (
                next_token.eq(self.model.cfg.eos_token_id)
                & (visible_lens >= source_lens)
            )

            if finished.all():
                break

        generated_cpu = generated.detach().cpu()

        hypotheses = self.tokenizer.batch_decode(
            generated_cpu.tolist(),
            skip_special_tokens=True,
        )

        total_batch_time = time.perf_counter() - batch_start

        outputs: list[TranslationOutput] = []

        for i in range(batch_size):
            hyp_ids = generated_cpu[i].tolist()

            # Trim after EOS for cleaner storage.
            if self.model.cfg.eos_token_id in hyp_ids:
                eos_pos = hyp_ids.index(self.model.cfg.eos_token_id)
                hyp_ids = hyp_ids[:eos_pos + 1]

            outputs.append(
                TranslationOutput(
                    hypothesis_ids=hyp_ids,
                    hypothesis_text=hypotheses[i],
                    delays=delays[i],
                    target_len=max(1, len(delays[i])),
                    first_token_latency_sec=first_token_latency_sec,
                    generation_time_sec=total_batch_time / max(1, batch_size),
                    extra={},
                )
            )

        return outputs