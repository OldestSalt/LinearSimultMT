import torch


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )

    return (loss.view_as(labels) * mask).sum() / mask.float().sum().clamp_min(1.0)


def masked_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:

    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

    kl = torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)

    return (kl * mask).sum() / mask.float().sum().clamp_min(1.0)


def masked_topk_kl_divergence(
    *,
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    student_log_probs = torch.nn.functional.log_softmax(
        student_logits,
        dim=-1,
    )

    student_topk_log_probs = student_log_probs.gather(
        dim=-1,
        index=teacher_topk_ids,
    )

    teacher_topk_probs = teacher_topk_probs.float()

    kl = torch.nn.functional.kl_div(
        student_topk_log_probs,
        teacher_topk_probs,
        reduction="none",
    ).sum(dim=-1)

    return (kl * mask).sum() / mask.float().sum().clamp_min(1.0)


def simulmt_distillation_loss(
    *,
    student_logits: torch.Tensor,
    dataset_labels: torch.Tensor,
    label_mask: torch.Tensor,

    use_kl_loss: bool,
    use_dataset_ce_loss: bool,

    kl_weight: float,
    dataset_ce_weight: float,

    teacher_topk_ids: torch.Tensor | None = None,
    teacher_topk_probs: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    device = student_logits.device

    total_loss = torch.zeros([], device=device)

    zero = torch.zeros([], device=device)

    kl_loss = zero
    teacher_ce_loss = zero
    dataset_ce_loss = zero

    if use_kl_loss and kl_weight > 0:
        if teacher_topk_ids is None or teacher_topk_probs is None:
            raise ValueError("Top-k KL is enabled, but teacher_topk_ids/probs are missing.")

        kl_loss = masked_topk_kl_divergence(
            student_logits=student_logits,
            teacher_topk_ids=teacher_topk_ids,
            teacher_topk_probs=teacher_topk_probs,
            mask=label_mask
        )

        total_loss = total_loss + kl_weight * kl_loss

    if use_dataset_ce_loss and dataset_ce_weight > 0:
        dataset_ce_loss = masked_cross_entropy(
            logits=student_logits,
            labels=dataset_labels,
            mask=label_mask,
        )

        total_loss = total_loss + dataset_ce_weight * dataset_ce_loss

    return {
        "loss": total_loss,
        "kl_loss": kl_loss.detach(),
        "teacher_ce_loss": teacher_ce_loss.detach(),
        "dataset_ce_loss": dataset_ce_loss.detach(),
    }