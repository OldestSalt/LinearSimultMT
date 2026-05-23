import torch
from .configs import SimulTransformerConfig


class PositionalEncoding(torch.nn.Module):
    """
    Learnable positional embeddings.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)

        self.position_embedding = torch.nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
        )

        self.max_len = max_len

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize positional embeddings with a small normal distribution.
        """
        torch.nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [batch, seq_len, d_model]

        Returns:
            Tensor with added positional embeddings.
        """

        x = x + self.position_embedding(torch.arange(
            x.shape[1],
            device=x.device,
        ).unsqueeze(0))

        return self.dropout(x)


class WaitKTransformerMT(torch.nn.Module):
    """
    Encoder-decoder Transformer for honest wait-k SimulMT training.

    Training mode:
        - The encoder receives the full source sequence.
        - The encoder uses a causal source-side mask.
        - The decoder uses a causal target-side mask.
        - Cross-attention uses a wait-k memory mask.

    Inference mode:
        - The model receives only the currently visible source prefix.
        - Therefore no wait-k cross-attention mask is needed.
        - The encoder can still use a causal mask to reduce train/inference mismatch.
    """

    def __init__(self, cfg: SimulTransformerConfig):
        super().__init__()

        self.cfg = cfg

        self.src_embedding = torch.nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_token_id,
        )

        self.tgt_embedding = torch.nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_token_id,
        )

        self.src_pos = PositionalEncoding(
            d_model=cfg.d_model,
            max_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )

        self.tgt_pos = PositionalEncoding(
            d_model=cfg.d_model,
            max_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.num_encoder_layers,
            norm=torch.nn.LayerNorm(cfg.d_model),
        )

        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=cfg.num_decoder_layers,
            norm=torch.nn.LayerNorm(cfg.d_model),
        )

        self.lm_head = torch.nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tgt_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
                if module.padding_idx is not None:
                    torch.nn.init.zeros_(module.weight[module.padding_idx])

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Boolean causal mask for PyTorch Transformer modules.

        True means "forbidden".
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    @staticmethod
    def make_waitk_memory_mask(
        tgt_len: int,
        src_len: int,
        k: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Cross-attention wait-k mask.

        For decoder position j:
            j = 0 predicts the first target token after the decoder prefix.

        Allowed source length:
            visible_src_len = k + j

        True means "forbidden".
        """

        return ~(torch.arange(src_len, device=device)[None, :] < (k + torch.arange(tgt_len, device=device)[:, None]))

    def encode(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            source_ids:
                [batch, src_len]

            source_mask:
                [batch, src_len], 1 for valid tokens, 0 for padding.

            causal:
                If True, source token i cannot attend to future source tokens.
        """
        if source_mask is None:
            source_key_padding_mask = source_ids.eq(self.cfg.pad_token_id)
        else:
            source_key_padding_mask = ~(source_mask.bool())

        src_attn_mask = None
        if causal:
            src_attn_mask = self.make_causal_mask(source_ids.shape[1], source_ids.device)

        x = self.src_embedding(source_ids) * math.sqrt(self.cfg.d_model)
        x = self.src_pos(x)

        memory = self.encoder(
            src=x,
            mask=src_attn_mask,
            src_key_padding_mask=source_key_padding_mask
        )

        return memory

    def decode(
        self,
        target_input_ids: torch.Tensor,
        memory: torch.Tensor,
        target_input_mask: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            target_input_ids:
                [batch, tgt_len]

            memory:
                [batch, src_len, d_model]

            target_input_mask:
                [batch, tgt_len], 1 for valid tokens, 0 for padding.

            source_mask:
                [batch, src_len], 1 for valid tokens, 0 for padding.

            memory_mask:
                [tgt_len, src_len], True means forbidden.
        """
        if target_input_mask is None:
            target_key_padding_mask = target_input_ids.eq(self.cfg.pad_token_id)
        else:
            target_key_padding_mask = ~(target_input_mask.bool())

        if source_mask is None:
            memory_key_padding_mask = None
        else:
            memory_key_padding_mask = ~(source_mask.bool())

        tgt_mask = self.make_causal_mask(target_input_ids.shape[1], target_input_ids.device)

        y = self.tgt_embedding(target_input_ids) * math.sqrt(self.cfg.d_model)
        y = self.tgt_pos(y)

        hidden = self.decoder(
            tgt=y,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return hidden

    def forward_waitk(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        k: int,
        source_mask: torch.Tensor | None = None,
        target_input_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Honest batched wait-k training forward pass.

        The source is full, but the encoder is causal and cross-attention is
        restricted by the wait-k policy.
        """
        batch_size, src_len = source_ids.shape
        _, tgt_len = target_input_ids.shape

        if source_mask is None:
            source_mask = source_ids.ne(self.cfg.pad_token_id).long()

        if target_input_mask is None:
            target_input_mask = target_input_ids.ne(self.cfg.pad_token_id).long()

        memory = self.encode(
            source_ids=source_ids,
            source_mask=source_mask,
            causal=True,
        )

        memory_mask = self.make_waitk_memory_mask(
            tgt_len=tgt_len,
            src_len=src_len,
            k=k,
            device=source_ids.device,
        )

        hidden = self.decode(
            target_input_ids=target_input_ids,
            memory=memory,
            target_input_mask=target_input_mask,
            source_mask=source_mask,
            memory_mask=memory_mask,
        )

        logits = self.lm_head(hidden)

        return logits