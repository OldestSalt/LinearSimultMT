from dataclasses import dataclass
from typing import Protocol, Any
import torch
import pandas as pd


@dataclass
class TranslationOutput:
    hypothesis_ids: list[int]
    hypothesis_text: str
    delays: list[int]
    target_len: int
    first_token_latency_sec: float | None = None
    generation_time_sec: float | None = None
    extra: dict[str, Any] | None = None


class TranslationModelAdapter(Protocol):
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


@dataclass
class SentenceLatency:
    ap: float
    al: float
    dal: float
    laal: float
    atd_text: float


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