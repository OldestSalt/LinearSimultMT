from dataclasses import dataclass
from typing import Any, Protocol
import time
import gc
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER


# =========================
# Outputs / protocol
# =========================

@dataclass
class TranslationOutput:
    hypothesis_ids: list[int]
    hypothesis_text: str
    delays: list[int]
    target_len: int
    first_token_latency_sec: float | None = None
    generation_time_sec: float | None = None
    extra: dict[str, Any] | None = None


class TokenizedTranslationModelAdapter(Protocol):
    name: str

    def translate_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        wait_k: int,
        max_new_tokens: int,
        **generation_kwargs: Any,
    ) -> list[TranslationOutput]:
        ...


# =========================
# Helpers
# =========================

def batch_decode_valid(
    tokenizer,
    ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    skip_special_tokens: bool = True,
) -> list[str]:
    """
    Decode a padded batch correctly.

    Args:
        ids:
            [batch, seq_len]

        mask:
            [batch, seq_len], 1/True for valid tokens.
    """
    ids_cpu = ids.detach().cpu()

    if mask is None:
        sequences = ids_cpu.tolist()
    else:
        mask_cpu = mask.detach().cpu().bool()
        sequences = [
            ids_cpu[i, mask_cpu[i]].tolist()
            for i in range(ids_cpu.size(0))
        ]

    return tokenizer.batch_decode(
        sequences,
        skip_special_tokens=skip_special_tokens,
    )


def lengths_from_mask(mask: torch.Tensor) -> list[int]:
    return mask.long().sum(dim=1).detach().cpu().tolist()
    
def valid_tokens(
    ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """
    Return only valid tokens from one sequence.
    """
    ids = ids.detach().cpu()

    if mask is not None:
        mask = mask.detach().cpu().bool()
        return ids[mask]

    if pad_token_id is not None:
        return ids[ids.ne(pad_token_id)]

    return ids


def decode_valid(
    tokenizer,
    ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    pad_token_id: int | None = None,
    skip_special_tokens: bool = True,
) -> str:
    ids = valid_tokens(
        ids,
        mask,
        pad_token_id=pad_token_id,
    )

    return tokenizer.batch_decode(
        ids,
        skip_special_tokens=skip_special_tokens,
    )


def token_lens(
    ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    pad_token_id: int | None = None,
) -> list[int]:
    return [int(
        valid_tokens(
            ids[i],
            mask[i],
            pad_token_id=pad_token_id,
        ).numel()
    ) for i in range(ids.size(0))]


def default_waitk_delays(
    *,
    source_len: int,
    target_len: int,
    wait_k: int,
) -> list[int]:
    return [
        min(source_len, wait_k + i)
        for i in range(target_len)
    ]


# =========================
# Quality metrics
# =========================

class MTQualityScorer:
    def __init__(self):
        self.bleu = BLEU(tokenize="13a")
        self.chrf = CHRF(word_order=2)
        self.ter = TER()

    def score(
        self,
        *,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> dict[str, float]:
        return {
            "BLEU": float(self.bleu.corpus_score(hypotheses, [references]).score),
            "chrF++": float(self.chrf.corpus_score(hypotheses, [references]).score),
            "TER": float(self.ter.corpus_score(hypotheses, [references]).score),
        }


# =========================
# Latency metrics
# =========================

@dataclass
class SentenceLatency:
    ap: float
    al: float
    dal: float
    laal: float
    atd_text: float


class WaitKLatencyScorer:
    def __init__(self, use_reference_len_for_gamma: bool = True):
        self.use_reference_len_for_gamma = use_reference_len_for_gamma

    def score_sentence(
        self,
        *,
        delays,
        source_len: int,
        target_len: int,
        reference_len: int | None = None,
    ) -> SentenceLatency:
        source_len = max(1, source_len)
        target_len = max(1, target_len)

        d = np.asarray(delays, dtype=np.float64)
        if len(d) == 0:
            d = np.asarray([source_len], dtype=np.float64)

        d = np.clip(d, 0, source_len)

        if len(d) < target_len:
            d = np.concatenate([d, np.full(target_len - len(d), d[-1])])
        elif len(d) > target_len:
            d = d[:target_len]

        effective_y_len = (
            reference_len
            if self.use_reference_len_for_gamma and reference_len is not None and reference_len > 0
            else target_len
        )

        gamma = max(effective_y_len / source_len, 1e-8)

        ap = float(d.sum() / (source_len * target_len))

        full_source_positions = np.where(d >= source_len)[0]
        tau = int(full_source_positions[0] + 1) if len(full_source_positions) else target_len
        tau = max(1, min(tau, target_len))

        ideal_prefix = np.arange(tau, dtype=np.float64) / gamma
        al = float(np.mean(d[:tau] - ideal_prefix))

        min_step = 1.0 / gamma
        d_dal = np.zeros_like(d)
        d_dal[0] = d[0]

        for i in range(1, target_len):
            d_dal[i] = max(d[i], d_dal[i - 1] + min_step)

        ideal_full = np.arange(target_len, dtype=np.float64) / gamma
        dal = float(np.mean(d_dal - ideal_full))

        adaptive_y_len = max(target_len, reference_len or target_len)
        adaptive_gamma = max(adaptive_y_len / source_len, 1e-8)
        laal = float(np.mean(d - (np.arange(target_len, dtype=np.float64) / adaptive_gamma)))

        aligned_source_pos = ((np.arange(target_len, dtype=np.float64) + 1.0) / target_len) * source_len
        atd_text = float(np.mean(np.maximum(0.0, d - aligned_source_pos)))

        return SentenceLatency(
            ap=ap,
            al=al,
            dal=dal,
            laal=laal,
            atd_text=atd_text,
        )

    def score_corpus(
        self,
        *,
        all_delays,
        source_lens,
        target_lens,
        reference_lens,
    ) -> dict[str, float]:
        rows = []

        for delays, src_len, tgt_len, ref_len in zip(
            all_delays,
            source_lens,
            target_lens,
            reference_lens,
        ):
            rows.append(
                self.score_sentence(
                    delays=delays,
                    source_len=src_len,
                    target_len=tgt_len,
                    reference_len=ref_len,
                )
            )

        return {
            "AP": float(np.mean([x.ap for x in rows])),
            "AL": float(np.mean([x.al for x in rows])),
            "DAL": float(np.mean([x.dal for x in rows])),
            "LAAL": float(np.mean([x.laal for x in rows])),
            "ATD_text": float(np.mean([x.atd_text for x in rows])),
        }


# =========================
# Adapter for WaitKTransformerMT
# =========================

class WaitKTransformerDatasetAdapter:
    """
    Faster adapter for WaitKTransformerMT.

    Main speedup:
        - encodes full source once with causal encoder;
        - at every generation step masks invisible source positions
          using source_mask;
        - decodes the whole batch at once.

    This is still honest for your model because the encoder is causal:
        encoder state h_i cannot contain information from source positions > i.
    """

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

# =========================
# Evaluation result
# =========================

@dataclass
class EvaluationResult:
    model_name: str
    wait_k: int
    metrics: dict[str, float]
    translations: pd.DataFrame

    def to_flat_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "wait_k": self.wait_k,
            **self.metrics,
        }


# =========================
# Dataset evaluator
# =========================

class TokenizedSimulMTEvaluator:
    """
    Faster evaluator for TranslationDataset.

    Improvements over the previous version:
        - adapter is expected to do real batched generation;
        - no per-batch cuda synchronize unless accurate_timing=True;
        - batched decoding of source/reference;
        - vectorized length calculation from masks.
    """

    def __init__(
        self,
        *,
        tokenizer,
        quality_scorer: MTQualityScorer | None = None,
        latency_scorer: WaitKLatencyScorer | None = None,
        pad_token_id: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.quality_scorer = quality_scorer or MTQualityScorer()
        self.latency_scorer = latency_scorer or WaitKLatencyScorer()
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

    def evaluate(
        self,
        model: TokenizedTranslationModelAdapter,
        dataset,
        *,
        wait_k: int,
        batch_size: int = 16,
        max_new_tokens: int = 64,
        show_progress: bool = True,
        accurate_timing: bool = False,
        **generation_kwargs: Any,
    ) -> EvaluationResult:
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        torch.cuda.reset_peak_memory_stats()

        gc.collect()

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        all_outputs: list[TranslationOutput] = []

        sources: list[str] = []
        references: list[str] = []
        source_lens: list[int] = []
        reference_lens: list[int] = []

        if accurate_timing:
            torch.cuda.synchronize()

        total_start = time.perf_counter()

        iterator = tqdm(
            dataloader,
            desc=f"Validating {model.name}, wait_k={wait_k}",
            leave=True,
        ) if show_progress else dataloader

        for batch in iterator:
            if accurate_timing:
                torch.cuda.synchronize()

            outputs = model.translate_batch(
                batch,
                wait_k=wait_k,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )

            if accurate_timing:
                torch.cuda.synchronize()

            batch_size_actual = batch["source_ids"].size(0)

            if len(outputs) != batch_size_actual:
                raise RuntimeError(
                    f"Adapter returned {len(outputs)} outputs for "
                    f"batch of {batch_size_actual} examples"
                )

            all_outputs.extend(outputs)

            sources.extend(
                batch_decode_valid(
                    self.tokenizer,
                    batch["source_ids"],
                    batch["source_mask"],
                    skip_special_tokens=True,
                )
            )

            references.extend(
                batch_decode_valid(
                    self.tokenizer,
                    batch["target_ids"],
                    batch["target_mask"],
                    skip_special_tokens=True,
                )
            )

            source_lens.extend(lengths_from_mask(batch["source_mask"]))
            reference_lens.extend(lengths_from_mask(batch["target_mask"]))

        if accurate_timing:
            torch.cuda.synchronize()

        total_time = time.perf_counter() - total_start

        hypotheses = [x.hypothesis_text for x in all_outputs]
        target_lens = [x.target_len for x in all_outputs]

        all_delays = []

        for src_len, tgt_len, out in zip(source_lens, target_lens, all_outputs):
            if out.delays:
                all_delays.append(out.delays)
            else:
                all_delays.append(
                    default_waitk_delays(
                        source_len=src_len,
                        target_len=tgt_len,
                        wait_k=wait_k,
                    )
                )

        quality = self.quality_scorer.score(
            sources=sources,
            hypotheses=hypotheses,
            references=references,
        )

        latency = self.latency_scorer.score_corpus(
            all_delays=all_delays,
            source_lens=source_lens,
            target_lens=target_lens,
            reference_lens=reference_lens,
        )

        total_target_tokens = int(sum(target_lens))
        total_source_tokens = int(sum(source_lens))

        first_token_latencies = [
            out.first_token_latency_sec
            for out in all_outputs
            if out.first_token_latency_sec is not None
        ]

        peak_gpu_memory_mb = (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else None
        )

        efficiency = {
            "total_time_sec": float(total_time),
            "ms_per_sentence": float(1000.0 * total_time / len(dataset)),
            "target_tokens_per_sec": float(total_target_tokens / max(total_time, 1e-8)),
            "source_tokens_per_sec": float(total_source_tokens / max(total_time, 1e-8)),
            "first_token_latency_sec": (
                float(np.mean(first_token_latencies))
                if first_token_latencies
                else None
            ),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        }

        metrics = {
            **quality,
            **latency,
            **efficiency,
        }

        translations = pd.DataFrame(
            {
                "source": sources,
                "reference": references,
                "hypothesis": hypotheses,
                "source_len": source_lens,
                "reference_len": reference_lens,
                "target_len": target_lens,
                "delays": all_delays,
                "generation_time_sec": [x.generation_time_sec for x in all_outputs],
                "first_token_latency_sec": [x.first_token_latency_sec for x in all_outputs],
                "hypothesis_ids": [x.hypothesis_ids for x in all_outputs],
            }
        )

        return EvaluationResult(
            model_name=model.name,
            wait_k=wait_k,
            metrics=metrics,
            translations=translations,
        )


# =========================
# Multi-model comparison
# =========================

def compare_models_on_translation_dataset(
    *,
    models: list[TokenizedTranslationModelAdapter],
    dataset,
    wait_k: int,
    evaluator: TokenizedSimulMTEvaluator,
    batch_size: int = 16,
    max_new_tokens: int = 64,
    num_workers: int = 0,
    save_translations_dir: str | None = None,
    **generation_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, EvaluationResult]]:
    results: dict[str, EvaluationResult] = {}
    rows: list[dict[str, Any]] = []

    if save_translations_dir is not None:
        os.makedirs(save_translations_dir, exist_ok=True)

    for model in models:
        result = evaluator.evaluate(
            model=model,
            dataset=dataset,
            wait_k=wait_k,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            num_workers=num_workers,
            **generation_kwargs,
        )

        results[model.name] = result
        rows.append(result.to_flat_dict())

        if save_translations_dir is not None:
            safe_name = model.name.replace("/", "_").replace(" ", "_")
            result.translations.to_csv(
                os.path.join(save_translations_dir, f"{safe_name}_wait{wait_k}.csv"),
                index=False,
            )

    summary = pd.DataFrame(rows)

    preferred_cols = [
        "model",
        "wait_k",
        "BLEU",
        "chrF++",
        "TER",
        "AL",
        "DAL",
        "AP",
        "LAAL",
        "ATD_text",
        "total_time_sec",
        "ms_per_sentence",
        "target_tokens_per_sec",
        "source_tokens_per_sec",
        "first_token_latency_sec",
        "peak_gpu_memory_mb",
    ]

    cols = [c for c in preferred_cols if c in summary.columns]
    other_cols = [c for c in summary.columns if c not in cols]

    return summary[cols + other_cols], results