import numpy as np
from .classes import SentenceLatency
from sacrebleu.metrics import BLEU, CHRF, TER


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
