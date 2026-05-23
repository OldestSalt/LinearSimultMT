from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 5
    short_epochs: bool = False
    batches_per_epoch: int = 10000
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_workers: int = 0

    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    wait_k: int | list[int] = 5

    use_kl_loss: bool = False
    use_dataset_ce_loss: bool = True

    kl_weight: float = 0.0
    dataset_ce_weight: float = 0.3

    log_dir: str = "./runs/simulmt_waitk"
    save_every_steps: int = 1000

    use_amp: bool = True