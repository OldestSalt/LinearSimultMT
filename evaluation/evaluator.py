import time
import gc
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER
from .classes import *
from .helpers import *
from .metrics import *

class SimulMTEvaluator:
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
        model: TranslationModelAdapter,
        dataset,
        *,
        wait_k: int,
        batch_size: int = 16,
        max_new_tokens: int = 64,
        show_progress: bool = True,
        accurate_timing: bool = False,
        dataset_fraction: float = 1.0,
        **generation_kwargs: Any,
    ) -> EvaluationResult:
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        
        eval_dataset = make_fraction_subset(
            dataset,
            dataset_fraction=dataset_fraction,
        )
        
        if len(eval_dataset) == 0:
            raise ValueError("eval_dataset is empty")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        gc.collect()
        
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
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

        generation_total_time = float(
            sum(
                x.generation_time_sec or 0.0
                for x in all_outputs
            )
        )
        
        generation_efficiency = {
            "generation_total_time_sec": generation_total_time,
            "generation_ms_per_sentence": float(
                1000.0 * generation_total_time / len(eval_dataset)
            ),
            "generation_target_tokens_per_sec": float(
                sum(target_lens) / max(generation_total_time, 1e-8)
            ),
        }

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
            "ms_per_sentence": float(1000.0 * total_time / len(eval_dataset)),
            "target_tokens_per_sec": float(total_target_tokens / max(total_time, 1e-8)),
            "source_tokens_per_sec": float(total_source_tokens / max(total_time, 1e-8)),
            "first_token_latency_sec": (
                float(np.mean(first_token_latencies))
                if first_token_latencies
                else None
            ),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "dataset_fraction": float(dataset_fraction),
            "eval_dataset_size": int(len(eval_dataset)),
            "full_dataset_size": int(len(dataset)),
        }

        metrics = {
            **quality,
            **latency,
            **efficiency,
            **generation_efficiency,
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