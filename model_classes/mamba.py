import torch
from mamba_ssm import Mamba2


class Mamba2Block(torch.nn.Module):
    """
    Pre-norm residual Mamba-2 block.

    State dict is compatible with the old version:
        norm.*
        mixer.*
    """

    def __init__(self, cfg):
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

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype=None,
    ):
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
        """
        Process one token.

        Args:
            x:
                [batch, 1, d_model]

            cache:
                (conv_state, ssm_state)

        Returns:
            y:
                [batch, 1, d_model]

            new_cache:
                updated (conv_state, ssm_state)
        """
        conv_state, ssm_state = cache

        residual = x

        y, conv_state, ssm_state = self.mixer.step(
            self.norm(x),
            conv_state,
            ssm_state,
        )

        y = residual + self.dropout(y)

        return y, (conv_state, ssm_state)


class WaitKMamba2MT(torch.nn.Module):
    """
    Pure Mamba-2 prefix-to-prefix model for wait-k SimulMT.

    Compatible with the previous state_dict.
    New methods:
        - allocate_inference_cache
        - step_token
        - step_token_indices
        - logits_from_hidden
        - generate_incremental_waitk
    """

    def __init__(self, cfg):
        super().__init__()

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

        self.position_embedding = torch.nn.Embedding(
            cfg.max_source_len + cfg.max_target_len + 4,
            cfg.d_model,
        )

        # 0 = source, 1 = target, 2 = pad
        self.segment_embedding = torch.nn.Embedding(
            3,
            cfg.d_model,
        )

        self.dropout = torch.nn.Dropout(cfg.dropout)

        self.layers = torch.nn.ModuleList(
            [Mamba2Block(cfg) for _ in range(cfg.num_layers)]
        )

        self.final_norm = torch.nn.LayerNorm(cfg.d_model)

        self.lm_head = torch.nn.Linear(
            cfg.d_model,
            cfg.vocab_size,
            bias=False,
        )

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tgt_embedding.weight

        self._init_non_mamba_weights()

    def _init_non_mamba_weights(self) -> None:
        for module in [
            self.src_embedding,
            self.position_embedding,
            self.segment_embedding,
        ]:
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.cfg.d_model ** -0.5,
            )

            if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                torch.nn.init.zeros_(module.weight[module.padding_idx])

        if self.tgt_embedding is not self.src_embedding:
            torch.nn.init.normal_(
                self.tgt_embedding.weight,
                mean=0.0,
                std=self.cfg.d_model ** -0.5,
            )

            if self.tgt_embedding.padding_idx is not None:
                torch.nn.init.zeros_(
                    self.tgt_embedding.weight[self.tgt_embedding.padding_idx]
                )

        if not self.cfg.tie_embeddings:
            torch.nn.init.xavier_uniform_(self.lm_head.weight)

    # ============================================================
    # Original training / teacher-forcing path
    # ============================================================

    def _build_waitk_sequence(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        k: int,
        source_mask: torch.Tensor,
        target_input_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized wait-k sequence builder.

        Schedule:
            src_0 ... src_{k-1}, tgt_0, src_k, tgt_1, src_{k+1}, ...
        """
        source_mask = source_mask.bool()
        target_input_mask = target_input_mask.bool()

        device = source_ids.device

        batch_size, src_len = source_ids.shape
        tgt_len = target_input_ids.shape[-1]

        source_lengths = source_mask.sum(dim=1)
        target_lengths = target_input_mask.sum(dim=1)

        total_len = src_len + tgt_len

        max_allowed_len = self.cfg.max_source_len + self.cfg.max_target_len + 4
        if total_len > max_allowed_len:
            raise ValueError(
                f"Scheduled sequence length {total_len} exceeds "
                f"configured maximum {max_allowed_len}."
            )

        scheduled_ids = torch.full(
            size=(batch_size, total_len),
            fill_value=self.cfg.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        scheduled_segments = torch.full(
            size=(batch_size, total_len),
            fill_value=2,
            dtype=torch.long,
            device=device,
        )

        scheduled_mask = torch.zeros(
            size=(batch_size, total_len),
            dtype=torch.long,
            device=device,
        )

        src_idx = torch.arange(
            src_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, src_len)

        targets_before_source = (src_idx - k + 1).clamp(
            min=0,
            max=tgt_len,
        )

        source_positions = src_idx + targets_before_source

        source_valid = source_mask

        batch_idx_src = torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        ).unsqueeze(1).expand(batch_size, src_len)

        scheduled_ids[
            batch_idx_src[source_valid],
            source_positions[source_valid],
        ] = source_ids[source_valid]

        scheduled_segments[
            batch_idx_src[source_valid],
            source_positions[source_valid],
        ] = 0

        scheduled_mask[
            batch_idx_src[source_valid],
            source_positions[source_valid],
        ] = 1

        tgt_idx = torch.arange(
            tgt_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, tgt_len)

        visible_source_before_target = torch.minimum(
            source_lengths.unsqueeze(1),
            torch.full_like(tgt_idx, k) + tgt_idx,
        )

        target_positions = visible_source_before_target + tgt_idx

        target_segments = torch.where(
            tgt_idx < target_lengths.unsqueeze(1),
            torch.ones_like(tgt_idx),
            torch.full_like(tgt_idx, 2),
        )

        scheduled_ids.scatter_(
            dim=1,
            index=target_positions,
            src=target_input_ids,
        )

        scheduled_segments.scatter_(
            dim=1,
            index=target_positions,
            src=target_segments,
        )

        scheduled_mask.scatter_(
            dim=1,
            index=target_positions,
            src=torch.ones_like(target_positions),
        )

        return scheduled_ids, scheduled_segments, scheduled_mask, target_positions

    def forward_waitk(
        self,
        source_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        *,
        k: int,
        source_mask: torch.Tensor | None = None,
        target_input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing / training path.

        Returns:
            logits:
                [batch, target_len, vocab_size]
        """
        if source_mask is None:
            source_mask = source_ids.ne(self.cfg.pad_token_id)

        if target_input_mask is None:
            target_input_mask = target_input_ids.ne(self.cfg.pad_token_id)

        scheduled_ids, scheduled_segments, _, target_positions = (
            self._build_waitk_sequence(
                source_ids=source_ids,
                target_input_ids=target_input_ids,
                k=k,
                source_mask=source_mask,
                target_input_mask=target_input_mask,
            )
        )

        batch_size, total_len = scheduled_ids.shape

        src_emb = self.src_embedding(scheduled_ids)
        tgt_emb = self.tgt_embedding(scheduled_ids)

        is_target = scheduled_segments.eq(1) | scheduled_segments.eq(2)

        x = torch.where(
            is_target.unsqueeze(-1),
            tgt_emb,
            src_emb,
        )

        positions = torch.arange(
            total_len,
            device=scheduled_ids.device,
        ).unsqueeze(0)

        x = (
            x
            + self.position_embedding(positions)
            + self.segment_embedding(scheduled_segments)
        )

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        gather_index = target_positions.unsqueeze(-1).expand(
            -1,
            -1,
            self.cfg.d_model,
        )

        target_hidden = x.gather(
            dim=1,
            index=gather_index,
        )

        return self.lm_head(target_hidden)

    # ============================================================
    # Incremental inference path
    # ============================================================

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int | None = None,
        dtype=None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if max_seqlen is None:
            max_seqlen = self.cfg.max_source_len + self.cfg.max_target_len + 4

        if dtype is None:
            dtype = next(self.parameters()).dtype

        return [
            layer.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=max_seqlen,
                dtype=dtype,
            )
            for layer in self.layers
        ]

    def _embed_scheduled_token(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_ids:
                [batch]

            segment_ids:
                [batch]
                0 = source, 1 = target, 2 = pad

            position_ids:
                [batch]

        Returns:
            x:
                [batch, d_model]
        """
        src_emb = self.src_embedding(token_ids)
        tgt_emb = self.tgt_embedding(token_ids)

        is_target = segment_ids.eq(1) | segment_ids.eq(2)

        x = torch.where(
            is_target.unsqueeze(-1),
            tgt_emb,
            src_emb,
        )

        x = (
            x
            + self.position_embedding(position_ids)
            + self.segment_embedding(segment_ids)
        )

        return self.dropout(x)

    def step_token(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process one token for every batch item.

        Args:
            token_ids:
                [batch]

            segment_ids:
                [batch]

            position_ids:
                [batch]

            caches:
                full-batch per-layer Mamba caches

        Returns:
            hidden:
                [batch, d_model]

            caches:
                updated caches
        """
        x = self._embed_scheduled_token(
            token_ids=token_ids,
            segment_ids=segment_ids,
            position_ids=position_ids,
        ).unsqueeze(1)

        new_caches = []

        for layer, cache in zip(self.layers, caches):
            x, new_cache = layer.step(x, cache)
            new_caches.append(new_cache)

        x = self.final_norm(x)

        return x[:, 0, :], new_caches

    def step_token_indices(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Process one token only for selected batch rows.
    
        Optimized version:
            - uses full-batch step when all rows are active;
            - does not clone full cache tensors;
            - updates cache in-place with index_copy_.
        """
        if indices.numel() == 0:
            empty = torch.empty(
                0,
                self.cfg.d_model,
                device=position_ids.device,
                dtype=self.src_embedding.weight.dtype,
            )
            return empty, caches
    
        batch_size = caches[0][0].size(0)
    
        # Fast path: all rows are active.
        # Avoid index_select/index_copy entirely.
        if indices.numel() == batch_size:
            return self.step_token(
                token_ids=token_ids,
                segment_ids=segment_ids,
                position_ids=position_ids,
                caches=caches,
            )
    
        x = self._embed_scheduled_token(
            token_ids=token_ids,
            segment_ids=segment_ids,
            position_ids=position_ids,
        ).unsqueeze(1)
    
        new_full_caches = []
    
        for layer, cache in zip(self.layers, caches):
            conv_state, ssm_state = cache
    
            sub_conv_state = conv_state.index_select(0, indices).contiguous()
            sub_ssm_state = ssm_state.index_select(0, indices).contiguous()
    
            x, new_sub_cache = layer.step(
                x,
                (sub_conv_state, sub_ssm_state),
            )
    
            new_conv_state, new_ssm_state = new_sub_cache
    
            # Inference-only: safe to update in-place.
            conv_state.index_copy_(0, indices, new_conv_state)
            ssm_state.index_copy_(0, indices, new_ssm_state)
    
            new_full_caches.append((conv_state, ssm_state))
    
        x = self.final_norm(x)
    
        return x[:, 0, :], new_full_caches
    
    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    @torch.inference_mode()
    def generate_incremental_waitk(
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
        Fast stateful wait-k generation.

        Returns:
            generated:
                [batch, 1 + generated_len]
                Includes initial target language token.

            delays:
                list of delays per example.
        """
        self.eval()

        device = source_ids.device
        batch_size, src_len = source_ids.shape

        if source_mask is None:
            source_mask = source_ids.ne(self.cfg.pad_token_id)
        else:
            source_mask = source_mask.bool()

        src_len = min(src_len, self.cfg.max_source_len)
        source_ids = source_ids[:, :src_len]
        source_mask = source_mask[:, :src_len]

        source_lens = source_mask.long().sum(dim=1).clamp_min(1).clamp_max(src_len)

        max_seqlen = min(
            self.cfg.max_source_len + self.cfg.max_target_len + 4,
            src_len + self.cfg.max_target_len + 4,
        )

        if cache_dtype is None:
            cache_dtype = next(self.parameters()).dtype
        
        caches = self.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            dtype=cache_dtype,
        )

        max_steps = min(
            max_new_tokens,
            max(1, self.cfg.max_target_len - 1),
        )
        
        generated = torch.full(
            size=(batch_size, 1 + max_steps),
            fill_value=self.cfg.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        generated[:, 0] = target_lang_token_id
        generated_len = 1

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        consumed_source = torch.zeros(batch_size, dtype=torch.long, device=device)
        position_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        delays: list[list[int]] = [[] for _ in range(batch_size)]

        hidden = torch.empty(
            batch_size,
            self.cfg.d_model,
            device=device,
            dtype=self.src_embedding.weight.dtype,
        )

        def feed_source_until(target_visible_lens: torch.Tensor):
            nonlocal caches, consumed_source, position_ids

            target_visible_lens = torch.minimum(
                target_visible_lens,
                source_lens,
            )

            while True:
                active = (~finished) & (consumed_source < target_visible_lens)
                indices = torch.nonzero(active, as_tuple=False).flatten()

                if indices.numel() == 0:
                    break

                src_pos = consumed_source.index_select(0, indices)

                token_ids = source_ids[indices, src_pos]

                segment_ids = torch.zeros_like(token_ids)
                pos_ids = position_ids.index_select(0, indices)

                _, caches = self.step_token_indices(
                    token_ids=token_ids,
                    segment_ids=segment_ids,
                    position_ids=pos_ids,
                    caches=caches,
                    indices=indices,
                )

                consumed_source.index_add_(
                    0,
                    indices,
                    torch.ones_like(indices, dtype=consumed_source.dtype),
                )

                position_ids.index_add_(
                    0,
                    indices,
                    torch.ones_like(indices, dtype=position_ids.dtype),
                )

        # 1. Initial source prefix: x[:k]
        initial_visible = torch.minimum(
            source_lens,
            torch.full_like(source_lens, k),
        ).clamp_min(1)

        feed_source_until(initial_visible)

        # 2. Initial target language token.

        target_lang_tokens = torch.full(
            size=(batch_size,),
            fill_value=target_lang_token_id,
            dtype=torch.long,
            device=device,
        )

        target_segments = torch.ones_like(target_lang_tokens)
        target_positions = position_ids.clone()

        hidden, caches = self.step_token(
            token_ids=target_lang_tokens,
            segment_ids=target_segments,
            position_ids=target_positions,
            caches=caches,
        )

        position_ids += 1

        max_steps = min(
            max_new_tokens,
            max(1, self.cfg.max_target_len - 1),
        )
        
        for step in range(max_steps):
            next_logits = self.logits_from_hidden(hidden)
            next_token = next_logits.argmax(dim=-1)
            
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.cfg.pad_token_id),
                next_token,
            )
                
            generated[:, generated_len] = next_token
            generated_len += 1
    
            if stop_after_eos_when_full_source_read:
                finished |= (
                    next_token.eq(self.cfg.eos_token_id)
                    & (consumed_source >= source_lens)
                )
            else:
                finished |= next_token.eq(self.cfg.eos_token_id)
    
            if bool(finished.all().item()):
                break
    
            next_visible = torch.minimum(
                source_lens,
                torch.full_like(source_lens, k + (step + 1) * speed),
            ).clamp_min(1)
    
            feed_source_until(next_visible)
    
            pad_token = torch.full_like(next_token, self.cfg.pad_token_id)
            pad_segment = torch.full_like(next_token, 2)
            target_segment = torch.ones_like(next_token)
            
            step_tokens = torch.where(finished, pad_token, next_token)
            step_segments = torch.where(finished, pad_segment, target_segment)
            
            hidden, caches = self.step_token(
                token_ids=step_tokens,
                segment_ids=step_segments,
                position_ids=position_ids,
                caches=caches,
            )
            
            position_ids += 1
    
            # No clone needed in inference.
            #hidden.index_copy_(0, active_indices, new_hidden)
    
            #position_ids.index_add_(
                #0,
                #active_indices,
                #torch.ones_like(active_indices, dtype=position_ids.dtype),
            #)
    
            if generated.size(1) >= self.cfg.max_target_len:
                break
    
        return generated[:, :generated_len]