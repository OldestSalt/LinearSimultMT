import torch
from .classes import TranslationModelAdapter


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


def make_fraction_subset(
    dataset,
    dataset_fraction: float,
):
    """
    Return deterministic prefix subset of dataset.

    dataset_fraction:
        1.0 -> full dataset
        0.1 -> first 10%
    """
    if not (0.0 < dataset_fraction <= 1.0):
        raise ValueError("dataset_fraction must be in the interval (0, 1].")

    if dataset_fraction == 1.0:
        return dataset

    subset_size = max(1, int(len(dataset) * dataset_fraction))
    indices = torch.arange(subset_size).tolist()

    return torch.utils.data.Subset(dataset, indices)