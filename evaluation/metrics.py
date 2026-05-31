import numpy as np
from .classes import SentenceLatency
from sacrebleu.metrics import BLEU, CHRF, TER
import torch


class MTQualityScorer:
    def __init__(
        self,
        *,
        use_comet: bool = False,
        comet_model_name: str = "Unbabel/wmt22-comet-da",
        comet_batch_size: int = 8,
        comet_gpus: int | None = None,
    ):
        self.bleu = BLEU(tokenize="13a")
        self.chrf = CHRF(word_order=2)
        self.ter = TER()

        self.use_comet = use_comet
        self.comet_model_name = comet_model_name
        self.comet_batch_size = comet_batch_size
        self.comet_gpus = comet_gpus
        self.comet_model = None

        if use_comet:
            self._load_comet()

    def _load_comet(self):
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as e:
            raise ImportError(
                "COMET is not installed. Install it with: pip install unbabel-comet"
            ) from e

        model_path = download_model(self.comet_model_name)
        self.comet_model = load_from_checkpoint(model_path)

    def _score_comet(
        self,
        *,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> dict[str, float]:
        if self.comet_model is None:
            self._load_comet()

        data = [
            {
                "src": src,
                "mt": hyp,
                "ref": ref,
            }
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]

        gpus = self.comet_gpus
        if gpus is None:
            gpus = 1 if torch.cuda.is_available() else 0

        output = self.comet_model.predict(
            data,
            batch_size=self.comet_batch_size,
            gpus=gpus,
            progress_bar=True,
        )

        # Different COMET versions expose the system score slightly differently.
        if isinstance(output, tuple):
            _, system_score = output
        elif hasattr(output, "system_score"):
            system_score = output.system_score
        elif isinstance(output, dict) and "system_score" in output:
            system_score = output["system_score"]
        else:
            raise RuntimeError(f"Unsupported COMET output type: {type(output)}")

        return {
            "COMET": float(system_score),
        }

    def score(
        self,
        *,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> dict[str, float]:
        scores = {
            "BLEU": float(self.bleu.corpus_score(hypotheses, [references]).score),
            "chrF++": float(self.chrf.corpus_score(hypotheses, [references]).score),
            "TER": float(self.ter.corpus_score(hypotheses, [references]).score),
        }

        if self.use_comet:
            scores.update(
                self._score_comet(
                    sources=sources,
                    hypotheses=hypotheses,
                    references=references,
                )
            )

        return scores


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
