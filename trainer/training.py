import torch
import mlflow
from .configs import *
from .dataset import TranslationDataset
from .losses import simulmt_distillation_loss
from .helpers import *
import tqdm.notebook as tqdm
import datetime
import numpy as np


def train_waitk_student(
    *,
    student: torch.nn.Module,
    train_dataset: TranslationDataset,
    model_cfg,
    train_cfg: TrainConfig,
    device: str | torch.device = "cuda",
    mlflow_run_name: str | None = None
):
    mlflow_run_name = mlflow_run_name or "waitk-student"
    start = datetime.datetime.now()
    with mlflow.start_run(run_name=mlflow_run_name):
        log_configs_to_mlflow(model_cfg, train_cfg)

        mlflow.set_tags({
            "task": "SimulMT",
            "policy": "wait-k",
            "teacher": "NLLB",
            "architecture": "VanillaTransformer"
        })

        device = torch.device(device)
        rng = np.random.default_rng()

        student.to(device)
        student = torch.compile(student, mode="max-autotune")

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.98),
        )

        scaler = torch.amp.GradScaler(
            enabled=train_cfg.use_amp and device.type == "cuda"
        )

        global_step = 0
        optimizer_step = 0
        current_dataset = train_dataset
        sampler = None

        for epoch in range(train_cfg.epochs):
            student.train()

            if train_cfg.short_epochs and train_cfg.batches_per_epoch > 0:
                sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False, num_samples=train_cfg.batches_per_epoch * train_cfg.batch_size)

            dataloader = torch.utils.data.DataLoader(
                current_dataset,
                batch_size=train_cfg.batch_size,
                shuffle=not train_cfg.short_epochs,
                sampler=sampler,
            )

            progress = tqdm.tqdm(
                dataloader,
                desc=f"epoch {epoch + 1}/{train_cfg.epochs}",
                leave=True,
            )

            optimizer.zero_grad(set_to_none=True)

            for micro_step, batch in enumerate(progress):
                source_ids = batch["source_ids"].to(device, non_blocking=True)
                target_ids = batch["target_ids"].to(device, non_blocking=True)
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
                        teacher_topk_ids = batch["teacher_top32_ids"].to(device, non_blocking=True)
                        teacher_topk_probs = torch.nn.functional.softmax(batch["teacher_top32_logits"].to(device, non_blocking=True), dim=-1)

                if isinstance(train_cfg.wait_k, list):
                    wait_k = rng.choice(train_cfg.wait_k)
                else:
                    wait_k = train_cfg.wait_k
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
                        teacher_topk_probs=teacher_topk_probs
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

                metrics = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "loss.total": float(loss.detach().cpu()),
                    "loss.kl": float(loss_dict["kl_loss"].cpu()),
                    "loss.teacher_ce": float(loss_dict["teacher_ce_loss"].cpu()),
                    "loss.dataset_ce": float(loss_dict["dataset_ce_loss"].cpu()),
                    "lr": optimizer.param_groups[0]["lr"]
                }

                mlflow.log_metrics(metrics, step=global_step)

                if global_step % 100 == 0:
                    log_gpu_memory_to_mlflow(global_step)

                progress.set_postfix(
                    loss=f"{metrics['loss.total']:.4f}",
                    kl=f"{metrics['loss.kl']:.4f}",
                    t_ce=f"{metrics['loss.teacher_ce']:.4f}",
                    d_ce=f"{metrics['loss.dataset_ce']:.4f}",
                )
            
            save_and_log_checkpoint(
                path=f"checkpoints/epoch_{epoch + 1}.pt",
                student=student,
                optimizer=optimizer,
                scaler=scaler,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                epoch=epoch,
                global_step=global_step,
                train_time=datetime.datetime.now() - start
            )