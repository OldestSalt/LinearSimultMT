import math
from dataclasses import dataclass
from .configs import SimulHybridMamba2Config

import torch
from mamba_ssm import Mamba2


class HybridPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.position_embedding = torch.nn.Embedding(max_len, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.position_embedding(position_ids)
        return self.dropout(x)


class HybridMamba2Block(torch.nn.Module):
    """Pre-norm residual Mamba-2 block with an incremental step method."""

    def __init__(self, cfg: SimulHybridMamba2Config):
        super().__init__()
        self.norm = torch.nn.LayerNorm(cfg.d_model)
        self.mixer = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
        )
        self.dropout = torch.nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return self.mixer.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            dtype=dtype,
        )

    def step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        conv_state, ssm_state = cache
        residual = x
        y, conv_state, ssm_state = self.mixer.step(self.norm(x), conv_state, ssm_state)
        return residual + self.dropout(y), (conv_state, ssm_state)


class HybridMambaCrossDecoderLayer(torch.nn.Module):
    """
    Mamba-2 target-side recurrent mixer + Transformer cross-attention.

    The Mamba block gives O(1)-state incremental target processing.
    The cross-attention gives explicit source-target alignment, which pure Mamba lacks.
    """

    def __init__(self, cfg: SimulHybridMamba2Config):
        super().__init__()
        self.self_mixer = HybridMamba2Block(cfg)
        self.cross_norm = torch.nn.LayerNorm(cfg.d_model)
        self.memory_norm = torch.nn.LayerNorm(cfg.d_model)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_dropout = torch.nn.Dropout(cfg.dropout)

        self.ffn_norm = torch.nn.LayerNorm(cfg.d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(cfg.d_model, cfg.dim_feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(cfg.dropout),
            torch.nn.Linear(cfg.dim_feedforward, cfg.d_model),
        )
        self.ffn_dropout = torch.nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.self_mixer(x)
        q = self.cross_norm(x)
        kv = self.memory_norm(memory)
        cross, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.cross_dropout(cross)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return self.self_mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def step(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        cache: tuple[torch.Tensor, torch.Tensor],
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, new_cache = self.self_mixer.step(x, cache)
        q = self.cross_norm(x)
        kv = self.memory_norm(memory)
        cross, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.cross_dropout(cross)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x, new_cache


class WaitKHybridMamba2MT(torch.nn.Module):
    """
    Hybrid SimulMT model:
      causal Mamba-2 encoder over source
      Mamba-2 decoder over target prefix
      cross-attention from decoder states to currently visible source states

    Compatible with the existing trainer because it exposes forward_waitk(...).
    Efficient evaluation uses encode-once causal source states + incremental decoder cache.
    """

    def __init__(self, cfg: SimulHybridMamba2Config):
        super().__init__()
        if cfg.d_model % cfg.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.cfg = cfg

        self.src_embedding = torch.nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_token_id,
        )
        if cfg.separate_source_target_embeddings:
            self.tgt_embedding = torch.nn.Embedding(
                cfg.vocab_size,
                cfg.d_model,
                padding_idx=cfg.pad_token_id,
            )
        else:
            self.tgt_embedding = self.src_embedding

        self.src_pos = HybridPositionalEncoding(cfg.d_model, cfg.max_source_len, cfg.dropout)
        self.tgt_pos = HybridPositionalEncoding(cfg.d_model, cfg.max_target_len, cfg.dropout)

        self.encoder_layers = torch.nn.ModuleList(
            [HybridMamba2Block(cfg) for _ in range(cfg.num_encoder_layers)]
        )
        self.encoder_norm = torch.nn.LayerNorm(cfg.d_model)

        self.decoder_layers = torch.nn.ModuleList(
            [HybridMambaCrossDecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)]
        )
        self.decoder_norm = torch.nn.LayerNorm(cfg.d_model)

        self.lm_head = torch.nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tgt_embedding.weight

        self._init_non_mamba_weights()

    def _init_non_mamba_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                if module not in (self.src_pos.position_embedding, self.tgt_pos.position_embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
                if module.padding_idx is not None:
                    torch.nn.init.zeros_(module.weight[module.padding_idx])

    @staticmethod
    def make_waitk_memory_mask(
        tgt_len: int,
        src_len: int,
        k: int,
        device: torch.device,
        speed: int = 1,
    ) -> torch.Tensor:
        """Boolean cross-attention mask. True means forbidden."""
        tgt_pos = torch.arange(tgt_len, device=device)[:, None]
        src_pos = torch.arange(src_len, device=device)[None, :]
        visible = k + tgt_pos * speed
        return ~(src_pos < visible)

    def encode(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Source encoder. Mamba-2 is causal by construction, so causal=True is kept
        for API compatibility with WaitKTransformerMT.
        """
        del causal
        if source_ids.size(1) > self.cfg.max_source_len:
            source_ids = source_ids[:, : self.cfg.max_source_len]

        x = self.src_embedding(source_ids) * math.sqrt(self.cfg.d_model)
        x = self.src_pos(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.encoder_norm(x)

    def decode(
        self,
        target_input_ids: torch.Tensor,
        memory: torch.Tensor,
        target_input_mask: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_input_ids.size(1) > self.cfg.max_target_len:
            target_input_ids = target_input_ids[:, : self.cfg.max_target_len]

        memory_key_padding_mask = None
        if source_mask is not None:
            memory_key_padding_mask = ~source_mask.bool()

        x = self.tgt_embedding(target_input_ids) * math.sqrt(self.cfg.d_model)
        x = self.tgt_pos(x)
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.decoder_norm(x)

    def forward_waitk(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        k: int,
        source_mask: torch.Tensor | None = None,
        target_input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, src_len = source_ids.shape
        _, tgt_len = target_input_ids.shape
        if src_len > self.cfg.max_source_len:
            source_ids = source_ids[:, : self.cfg.max_source_len]
            if source_mask is not None:
                source_mask = source_mask[:, : self.cfg.max_source_len]
            src_len = source_ids.size(1)
        if tgt_len > self.cfg.max_target_len:
            target_input_ids = target_input_ids[:, : self.cfg.max_target_len]
            if target_input_mask is not None:
                target_input_mask = target_input_mask[:, : self.cfg.max_target_len]
            tgt_len = target_input_ids.size(1)

        if source_mask is None:
            source_mask = source_ids.ne(self.cfg.pad_token_id).long()
        if target_input_mask is None:
            target_input_mask = target_input_ids.ne(self.cfg.pad_token_id).long()

        memory = self.encode(source_ids=source_ids, source_mask=source_mask, causal=True)
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
        return self.lm_head(hidden)

    def allocate_decoder_cache(
        self,
        batch_size: int,
        max_seqlen: int | None = None,
        dtype=None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if max_seqlen is None:
            max_seqlen = self.cfg.max_target_len
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return [
            layer.allocate_inference_cache(batch_size, max_seqlen=max_seqlen, dtype=dtype)
            for layer in self.decoder_layers
        ]

    def _embed_target_step(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embedding(token_ids) * math.sqrt(self.cfg.d_model)
        x = x + self.tgt_pos.position_embedding(position_ids)
        return self.tgt_pos.dropout(x).unsqueeze(1)

    def decode_step(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self._embed_target_step(token_ids, position_ids)
        new_caches = []
        for layer, cache in zip(self.decoder_layers, caches):
            x, new_cache = layer.step(
                x,
                memory,
                cache=cache,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            new_caches.append(new_cache)
        x = self.decoder_norm(x)
        return x[:, 0, :], new_caches

    @torch.inference_mode()
    def generate_waitk(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        target_lang_token_id: int,
        max_new_tokens: int = 64,
        k: int = 5,
        speed: int = 1,
        stop_after_eos_when_full_source_read: bool = True,
        cache_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Efficient streaming inference.

        Key idea: encode the full source once with a causal Mamba encoder.
        This does not leak future tokens because memory[:, i] depends only on source[:i+1].
        Wait-k is enforced by cross-attention key_padding_mask at every decoder step.
        """
        self.eval()
        device = source_ids.device
        batch_size, src_len = source_ids.shape
        src_len = min(src_len, self.cfg.max_source_len)
        source_ids = source_ids[:, :src_len]
        if source_mask is None:
            source_mask = source_ids.ne(self.cfg.pad_token_id)
        else:
            source_mask = source_mask[:, :src_len].bool()

        source_lens = source_mask.long().sum(dim=1).clamp_min(1).clamp_max(src_len)
        source_positions = torch.arange(src_len, device=device).unsqueeze(0)

        memory = self.encode(source_ids=source_ids, source_mask=source_mask, causal=True)

        if cache_dtype is None:
            cache_dtype = next(self.parameters()).dtype
        caches = self.allocate_decoder_cache(
            batch_size=batch_size,
            max_seqlen=self.cfg.max_target_len,
            dtype=cache_dtype,
        )

        max_steps = min(max_new_tokens, max(1, self.cfg.max_target_len - 1))
        generated = torch.full(
            (batch_size, 1 + max_steps),
            fill_value=self.cfg.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        generated[:, 0] = target_lang_token_id

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        current_token = torch.full(
            (batch_size,),
            fill_value=target_lang_token_id,
            dtype=torch.long,
            device=device,
        )

        generated_len = 1
        for step in range(max_steps):
            visible_lens = torch.minimum(
                source_lens,
                torch.full_like(source_lens, k + step * speed),
            ).clamp_min(1)
            visible_source_mask = source_mask & (source_positions < visible_lens.unsqueeze(1))
            memory_key_padding_mask = ~visible_source_mask

            position_ids = torch.full(
                (batch_size,),
                fill_value=step,
                dtype=torch.long,
                device=device,
            )
            hidden, caches = self.decode_step(
                token_ids=current_token,
                position_ids=position_ids,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                caches=caches,
            )
            next_token = self.lm_head(hidden).argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.cfg.pad_token_id),
                next_token,
            )
            generated[:, generated_len] = next_token
            generated_len += 1

            if stop_after_eos_when_full_source_read:
                finished |= next_token.eq(self.cfg.eos_token_id) & (visible_lens >= source_lens)
            else:
                finished |= next_token.eq(self.cfg.eos_token_id)

            if bool(finished.all().item()):
                break
            current_token = next_token

        return generated[:, :generated_len]
