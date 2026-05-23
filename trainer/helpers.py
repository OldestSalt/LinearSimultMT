import torch
import mlflow
import mlflow.pytorch
from dataclasses import asdict
from pathlib import Path
from .configs import *


def log_configs_to_mlflow(model_cfg, train_cfg):
    mlflow.log_params({
        f"model.{k}": v
        for k, v in asdict(model_cfg).items()
    })

    mlflow.log_params({
        f"train.{k}": v
        for k, v in asdict(train_cfg).items()
    })


def log_gpu_memory_to_mlflow(step: int):
    mlflow.log_metrics(
        {
            "gpu.allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "gpu.reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "gpu.max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        },
        step=step,
    )


def save_and_log_checkpoint(
    *,
    path,
    student,
    optimizer,
    scaler,
    model_cfg,
    train_cfg,
    epoch,
    global_step,
    train_time
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "epoch": epoch,
            "global_step": global_step,
            "train_time": train_time
        },
        path,
    )

    # mlflow.log_artifact(str(path), artifact_path="checkpoints")


def prepend_decoder_start_token(
    target_input_ids: torch.Tensor,
    target_input_mask: torch.Tensor,
    *,
    decoder_start_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add decoder start token before the existing target prefix.

    Student input:
        [rus_Cyrl, y1, y2, ...]

    Teacher input:
        [</s>, rus_Cyrl, y1, y2, ...]

    The extra teacher output at position 0 must be removed later.
    """
    batch_size = target_input_ids.size(0)
    device = target_input_ids.device

    start_ids = torch.full(
        size=(batch_size, 1),
        fill_value=decoder_start_token_id,
        dtype=target_input_ids.dtype,
        device=device,
    )

    start_mask = torch.ones(
        size=(batch_size, 1),
        dtype=target_input_mask.dtype,
        device=device,
    )

    teacher_target_input_ids = torch.cat(
        [start_ids, target_input_ids],
        dim=1,
    )

    teacher_target_input_mask = torch.cat(
        [start_mask, target_input_mask],
        dim=1,
    )

    return teacher_target_input_ids, teacher_target_input_mask


@torch.no_grad()
def translate_with_latency(
    model,
    tokenizer,
    source: str,
    max_len: int = 100,
    k: int = 5,
    speed: int = 1,
    source_lang: str = "eng_Latn",
    target_lang: str = "rus_Cyrl",
) -> str:
    model.eval()
    device = next(model.parameters()).device

    tokenizer.src_lang = source_lang

    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=model.cfg.max_seq_len,
    ).to(device)

    source_len = int(inputs["attention_mask"].sum().item())
    visible_prefix_len = min(k, source_len, model.cfg.max_seq_len)

    decoder_start_token_id = model.cfg.eos_token_id
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)

    target_tokens = torch.tensor(
        [[target_lang_id]],
        dtype=torch.long,
        device=device,
    )

    i = 1

    while target_tokens.size(1) < min(max_len, model.cfg.max_seq_len):
        latency_inputs = inputs["input_ids"][:, :visible_prefix_len]
        latency_attention_mask = inputs["attention_mask"][:, :visible_prefix_len]

        memory = model.encode(
            source_ids=latency_inputs,
            source_mask=latency_attention_mask,
            causal=True,
        )

        target_mask = target_tokens.ne(model.cfg.pad_token_id).long()

        hidden = model.decode(
            memory=memory,
            target_input_ids=target_tokens,
            target_input_mask=target_mask,
            source_mask=latency_attention_mask,
            memory_mask=None,
        )

        logits = model.lm_head(hidden)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() != tokenizer.eos_token_id:
            target_tokens = torch.cat([target_tokens, next_token], dim=-1)

        print(f"Iteration {i}")
        print(f"\tInput: {tokenizer.batch_decode(latency_inputs, skip_special_tokens=False)[0]}")
        print(f"\tTarget: {tokenizer.batch_decode(target_tokens, skip_special_tokens=False)[0]}")
        print(f"\tGenerated token: {tokenizer.batch_decode(next_token, skip_special_tokens=False)[0]}")
        i += 1

        visible_prefix_len = min(
            visible_prefix_len + speed,
            source_len,
            model.cfg.max_seq_len,
        )

        if next_token.item() == tokenizer.eos_token_id and visible_prefix_len >= source_len:
            break

    return tokenizer.batch_decode(
        target_tokens,
        skip_special_tokens=True,
    )[0]


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def load_training_checkpoint(
    checkpoint_path: str,
    model_config_class,
    model_class,
    *,
    device: str = "cuda"
):
    """
    Load model, configs, optimizer, scaler and training progress
    from a full training checkpoint.

    Returns:
        student
        optimizer
        scaler
        model_cfg
        train_cfg
        state
        train_time
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_cfg = model_config_class(**checkpoint["model_cfg"])
    train_cfg = TrainConfig(**checkpoint["train_cfg"])
    train_time = checkpoint["train_time"]

    student = torch.compile(model_class(model_cfg))
    student.load_state_dict(checkpoint["model_state_dict"])
    student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.98),
    )

    if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler = torch.amp.GradScaler(
        enabled=train_cfg.use_amp and device == "cuda"
    )

    if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    state = {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", checkpoint.get("step", 0)),
        "optimizer_step": checkpoint.get("optimizer_step", checkpoint.get("step", 0)),
    }

    return student, optimizer, scaler, model_cfg, train_cfg, state, train_time