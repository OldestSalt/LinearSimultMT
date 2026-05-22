import time
from typing import Any
from .evaluator import *

import torch


class NLLBSimulMTAdapter:
    """
    Fast-ish SimulMT adapter for HuggingFace AutoModelForSeq2SeqLM, e.g. NLLB.

    Important:
        This adapter does NOT use model.generate().

    For NLLB decoder input we use:
        [decoder_start_token_id, target_lang_token_id, generated_token_1, ...]

    SimulMT policy:
        At target step j, the model sees source prefix of length:
            min(source_len, wait_k + j * speed)

    Because NLLB encoder is bidirectional, we must not encode full source once.
    We physically restrict source prefix at every step.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        name: str = "NLLB-Seq2Seq-SimulMT",
        device: str | torch.device = "cuda",
        target_lang: str = "rus_Cyrl",
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        max_source_len: int | None = None,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.name = name
        self.device = torch.device(device)

        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.max_source_len = max_source_len

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
        """
        Args:
            batch:
                TranslationDataset batch with source_ids and source_mask.

            wait_k:
                Wait-k latency parameter.

            max_new_tokens:
                Maximum number of generated target tokens, excluding
                decoder_start_token_id and target_lang_token_id.

            speed:
                How many source tokens become visible after each target step.

        Returns:
            list[TranslationOutput]
        """
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

        for step in range(max_new_tokens):
            visible_lens = torch.minimum(
                source_lens,
                torch.full_like(source_lens, wait_k + step * speed),
            ).clamp_min(1)

            max_visible_len = int(visible_lens.max().item())

            prefix_source_ids = source_ids[:, :max_visible_len]
            prefix_source_mask = source_mask[:, :max_visible_len].clone()

            # Per-sample prefix mask:
            # samples with shorter visible_len must not attend to later source tokens.
            positions = torch.arange(
                max_visible_len,
                device=self.device,
            ).unsqueeze(0)

            prefix_source_mask = (
                prefix_source_mask.bool()
                & (positions < visible_lens.unsqueeze(1))
            ).long()

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

            # Keep finished samples padded.
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

            if stop_after_eos_when_full_source_read:
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

        outputs: list[TranslationOutput] = []

        # Remove [decoder_start_token_id, target_lang_token_id] from decoded text.
        generated_only = decoder_input_ids_cpu[:, 2:]

        hypothesis_texts = self.tokenizer.batch_decode(
            generated_only.tolist(),
            skip_special_tokens=True,
        )

        for i in range(batch_size):
            hyp_ids = generated_only[i].tolist()

            if self.eos_token_id in hyp_ids:
                eos_pos = hyp_ids.index(self.eos_token_id)
                hyp_ids = hyp_ids[:eos_pos + 1]

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
                    },
                )
            )

        return outputs