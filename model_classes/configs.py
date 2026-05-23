from dataclasses import dataclass


@dataclass
class SimulTransformerConfig:
    vocab_size: int

    d_model: int = 512
    nhead: int = 8

    num_encoder_layers: int = 6
    num_decoder_layers: int = 6

    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 32

    pad_token_id: int = 1
    eos_token_id: int = 2

    tie_embeddings: bool = True


@dataclass
class SimulMamba2Config:
    vocab_size: int

    d_model: int = 512
    num_layers: int = 12

    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1

    dropout: float = 0.1

    # Source and target are concatenated into one scheduled sequence.
    max_source_len: int = 64
    max_target_len: int = 64

    pad_token_id: int = 1
    eos_token_id: int = 2

    tie_embeddings: bool = True

    # Keep separate source/target embeddings to be closer to your Transformer.
    separate_source_target_embeddings: bool = True