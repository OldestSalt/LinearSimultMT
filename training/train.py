import datetime
from pathlib import Path
import time

import torch
import mlflow
import numpy as np
import tqdm.notebook as tqdm

from .configs import *
from .dataset import TranslationDataset
from .losses import simulmt_distillation_loss
from .helpers import *
from .helpers import _seconds_from_train_time


def train_waitk_student(
    *,
    student: torch.nn.Module,
    train_dataset: TranslationDataset,
    model_cfg,
    train_cfg: TrainConfig,
    device: str | torch.device = "cuda",
    mlflow_run_name: str | None = None,

    resume_from_checkpoint: str | Path | None = None,
    resume_mlflow_run: bool = True,

    compile_model: bool = True,
    compile_mode: str = "max-autotune",

    checkpoint_dir: str | Path = "checkpoints",
    checkpoint_name_prefix: str = "student",

    save_latest: bool = True,
    log_checkpoints_to_mlflow: bool = False,

    strict_model_load: bool = True,
):
    """
    Train wait-k student with full checkpoint resume support.

    Resume restores:
        - model weights
        - optimizer state
        - AMP scaler state
        - epoch/global_step/optimizer_step
        - cumulative train_time
        - MLflow run id, if resume_mlflow_run=True
    """
    mlflow_run_name = mlflow_run_name or "waitk-student"

    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None
    previous_train_time_seconds = 0.0
    start_epoch = 0
    global_step = 0
    optimizer_step = 0
    resumed_mlflow_run_id = None

    if resume_from_checkpoint is not None:
        resume_from_checkpoint = Path(resume_from_checkpoint)

        resume_checkpoint = torch.load(
            resume_from_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        previous_train_time_seconds = float(
            resume_checkpoint.get(
                "train_time_seconds",
                _seconds_from_train_time(resume_checkpoint.get("train_time", 0.0)),
            )
        )

        # Saved epoch means "last epoch index that was being/has been processed".
        # For normal epoch-level checkpointing, continue from epoch + 1.
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1

        global_step = int(resume_checkpoint.get("global_step", 0))
        optimizer_step = int(resume_checkpoint.get("optimizer_step", 0))

        resumed_mlflow_run_id = resume_checkpoint.get("mlflow_run_id", None)

    student.to(device)

    if resume_checkpoint is not None:
        load_model_state_robust(
            student,
            resume_checkpoint["model_state_dict"],
            strict=strict_model_load,
        )

    if compile_model:
        student = torch.compile(
            student,
            mode=compile_mode,
        )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.98),
    )

    scaler = torch.amp.GradScaler(
        enabled=train_cfg.use_amp and device.type == "cuda"
    )

    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint and resume_checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(
                resume_checkpoint["optimizer_state_dict"]
            )

            # Move optimizer state tensors to the current device.
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

        if "scaler_state_dict" in resume_checkpoint and resume_checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(
                resume_checkpoint["scaler_state_dict"]
            )

    # Continue old MLflow run if possible.
    mlflow_run_id = (
        resumed_mlflow_run_id
        if resume_mlflow_run and resumed_mlflow_run_id is not None
        else None
    )

    run_context = (
        mlflow.start_run(run_id=mlflow_run_id)
        if mlflow_run_id is not None
        else mlflow.start_run(run_name=mlflow_run_name)
    )

    wall_start_time = time.perf_counter()

    with run_context as active_run:
        active_mlflow_run_id = active_run.info.run_id

        # Log configs only at fresh run start.
        # If resuming the same run, repeated log_params with changed values
        # may fail in MLflow.
        if resume_checkpoint is None or mlflow_run_id is None:
            log_configs_to_mlflow(model_cfg, train_cfg)

            mlflow.set_tags({
                "task": "SimulMT",
                "policy": "wait-k",
                "teacher": "NLLB",
                "architecture": student.__class__.__name__,
            })

        else:
            mlflow.set_tags({
                "resumed": "true",
                "resume_from_checkpoint": str(resume_from_checkpoint),
            })

        rng = np.random.default_rng()

        current_dataset = train_dataset

        if start_epoch >= train_cfg.epochs:
            print(
                f"Checkpoint epoch={start_epoch - 1}, "
                f"but train_cfg.epochs={train_cfg.epochs}. Nothing to train."
            )
            return student

        for epoch in range(start_epoch, train_cfg.epochs):
            student.train()

            sampler = None

            if train_cfg.short_epochs and train_cfg.batches_per_epoch > 0:
                sampler = torch.utils.data.RandomSampler(
                    train_dataset,
                    replacement=False,
                    num_samples=train_cfg.batches_per_epoch * train_cfg.batch_size,
                )

            dataloader = torch.utils.data.DataLoader(
                current_dataset,
                batch_size=train_cfg.batch_size,
                shuffle=not train_cfg.short_epochs,
                sampler=sampler,
                num_workers=getattr(train_cfg, "num_workers", 0),
                pin_memory=device.type == "cuda",
            )

            progress = tqdm.tqdm(
                dataloader,
                desc=f"epoch {epoch + 1}/{train_cfg.epochs}",
                leave=True,
            )

            optimizer.zero_grad(set_to_none=True)

            for micro_step, batch in enumerate(progress):
                source_ids = batch["source_ids"].to(device, non_blocking=True).long()
                target_ids = batch["target_ids"].to(device, non_blocking=True).long()
                source_mask = batch["source_mask"].to(device, non_blocking=True)
                target_mask = batch["target_mask"].to(device, non_blocking=True)

                target_input_ids = target_ids[:, :-1]
                dataset_labels = target_ids[:, 1:]

                target_input_mask = target_mask[:, :-1]
                label_mask = target_mask[:, 1:]

                teacher_topk_ids = None
                teacher_topk_probs = None

                with torch.no_grad():
                    if train_cfg.use_kl_loss:
                        teacher_topk_ids = batch["teacher_top32_ids"].to(
                            device,
                            non_blocking=True,
                        ).long()

                        teacher_topk_probs = torch.nn.functional.softmax(
                            batch["teacher_top32_logits"].to(
                                device,
                                non_blocking=True,
                            ).float(),
                            dim=-1,
                        )

                if isinstance(train_cfg.wait_k, list):
                    wait_k = int(rng.choice(train_cfg.wait_k))
                else:
                    wait_k = int(train_cfg.wait_k)

                with torch.amp.autocast(
                    enabled=train_cfg.use_amp and device.type == "cuda",
                    device_type="cuda",
                ):
                    student_logits = student.forward_waitk(
                        source_ids=source_ids,
                        target_input_ids=target_input_ids,
                        k=wait_k,
                        source_mask=source_mask,
                        target_input_mask=target_input_mask,
                    )

                    loss_dict = simulmt_distillation_loss(
                        student_logits=student_logits,
                        dataset_labels=dataset_labels,
                        label_mask=label_mask,

                        use_kl_loss=train_cfg.use_kl_loss,
                        use_dataset_ce_loss=train_cfg.use_dataset_ce_loss,

                        kl_weight=train_cfg.kl_weight,
                        dataset_ce_weight=train_cfg.dataset_ce_weight,

                        teacher_topk_ids=teacher_topk_ids,
                        teacher_topk_probs=teacher_topk_probs,
                    )

                    loss = loss_dict["loss"]
                    loss_for_backward = loss / train_cfg.gradient_accumulation_steps

                scaler.scale(loss_for_backward).backward()

                global_step += 1

                should_step = (
                    global_step % train_cfg.gradient_accumulation_steps == 0
                )

                if should_step:
                    if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            student.parameters(),
                            train_cfg.grad_clip,
                        )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)

                    optimizer_step += 1

                    elapsed_seconds = previous_train_time_seconds + (
                        time.perf_counter() - wall_start_time
                    )

                    metrics = {
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "train_time_sec": elapsed_seconds,

                        "loss.total": float(loss.detach().cpu()),
                        "loss.kl": float(loss_dict["kl_loss"].detach().cpu()),
                        "loss.dataset_ce": float(loss_dict["dataset_ce_loss"].detach().cpu()),

                        # Keep this for compatibility if the loss dict has it.
                        "loss.teacher_ce": float(
                            loss_dict.get(
                                "teacher_ce_loss",
                                torch.zeros([], device=device),
                            ).detach().cpu()
                        ),

                        "lr": optimizer.param_groups[0]["lr"],
                        "wait_k": wait_k,
                    }

                    mlflow.log_metrics(
                        metrics,
                        step=global_step,
                    )

                    if global_step % 100 == 0 and device.type == "cuda":
                        log_gpu_memory_to_mlflow(global_step)

                    progress.set_postfix(
                        loss=f"{metrics['loss.total']:.4f}",
                        kl=f"{metrics['loss.kl']:.4f}",
                        d_ce=f"{metrics['loss.dataset_ce']:.4f}",
                        opt_step=optimizer_step,
                    )

                    if (
                        train_cfg.save_every_steps is not None
                        and train_cfg.save_every_steps > 0
                        and optimizer_step % train_cfg.save_every_steps == 0
                    ):
                        train_time_delta = datetime.timedelta(
                            seconds=elapsed_seconds,
                        )

                        step_path = (
                            checkpoint_dir
                            / f"{checkpoint_name_prefix}_step_{optimizer_step}.pt"
                        )

                        save_and_log_checkpoint(
                            path=step_path,
                            student=student,
                            optimizer=optimizer,
                            scaler=scaler,
                            model_cfg=model_cfg,
                            train_cfg=train_cfg,
                            epoch=epoch,
                            global_step=global_step,
                            optimizer_step=optimizer_step,
                            train_time=train_time_delta,
                            mlflow_run_id=active_mlflow_run_id,
                            log_to_mlflow=log_checkpoints_to_mlflow,
                        )

                        if save_latest:
                            latest_path = checkpoint_dir / f"{checkpoint_name_prefix}_latest.pt"

                            save_and_log_checkpoint(
                                path=latest_path,
                                student=student,
                                optimizer=optimizer,
                                scaler=scaler,
                                model_cfg=model_cfg,
                                train_cfg=train_cfg,
                                epoch=epoch,
                                global_step=global_step,
                                optimizer_step=optimizer_step,
                                train_time=train_time_delta,
                                mlflow_run_id=active_mlflow_run_id,
                                log_to_mlflow=False,
                            )

            # Epoch checkpoint.
            elapsed_seconds = previous_train_time_seconds + (
                time.perf_counter() - wall_start_time
            )

            train_time_delta = datetime.timedelta(seconds=elapsed_seconds)

            epoch_path = checkpoint_dir / f"{checkpoint_name_prefix}_epoch_{epoch + 1}.pt"

            save_and_log_checkpoint(
                path=epoch_path,
                student=student,
                optimizer=optimizer,
                scaler=scaler,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                epoch=epoch,
                global_step=global_step,
                optimizer_step=optimizer_step,
                train_time=train_time_delta,
                mlflow_run_id=active_mlflow_run_id,
                log_to_mlflow=log_checkpoints_to_mlflow,
            )

            if save_latest:
                latest_path = checkpoint_dir / f"{checkpoint_name_prefix}_latest.pt"

                save_and_log_checkpoint(
                    path=latest_path,
                    student=student,
                    optimizer=optimizer,
                    scaler=scaler,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    train_time=train_time_delta,
                    mlflow_run_id=active_mlflow_run_id,
                    log_to_mlflow=False,
                )

    return student