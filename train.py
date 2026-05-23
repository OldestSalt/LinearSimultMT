from trainer import *
from model_classes import WaitKTransformer

model_cfg = SimulTransformerConfig(
    vocab_size=len(tokenizer),
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    max_seq_len=64,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

train_cfg = TrainConfig(
    epochs=5,
    short_epochs=False,
    batches_per_epoch=2000,
    batch_size=192,
    gradient_accumulation_steps=8,

    wait_k=10,

    use_kl_loss=True,
    use_dataset_ce_loss=True,

    kl_weight=1.0,
    dataset_ce_weight=1.0,

    lr=1e-4,
    use_amp=True,
)

student = WaitKTransformerMT(model_cfg)

dataset = TranslationDataset("./data/train_dataset.hdf5")

train_waitk_student(
    student=student,
    train_dataset=dataset,
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    device="cuda",
)