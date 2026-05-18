    def _forward_aiter(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: NSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if self.need_pad_heads:
            q_kernel = q.view(
                -1, layer.tp_q_head_num, layer.head_dim
            ).repeat_interleave(self.head_repeat_factor, dim=1)
            o_kernel = q.new_empty(
                (
                    q.shape[0],
                    layer.tp_q_head_num * self.head_repeat_factor,
                    layer.v_head_dim,
                )
            )
        else:
            q_kernel = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            o_kernel = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = self.kv_indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)

        # ===== Runs D/E/F patch (additive): forward FP8 scales + env-gated UA branches =====
        # Bug fix (DSA report §6.1): _forward_aiter previously did not forward
        # FP8 q_scale/kv_scale to mla_decode_fwd, asserting inside
        # mla_decode_stage1_asm_fwd when --kv-cache-dtype=fp8_e4m3.
        import os as _os_de
        _q_scale_t = getattr(layer, "k_scale", None)        # tensor form (ASM)
        _kv_scale_t = getattr(layer, "k_scale", None)
        _q_scale_f = getattr(layer, "k_scale_float", None)  # scalar form (UA)
        _kv_scale_f = getattr(layer, "k_scale_float", None)

        if _os_de.environ.get("SGLANG_NSA_USE_UA_SPARSE_MLA") == "1":
            # Run F: NSA-routed sparse decode through Triton
            # unified_attention_sparse_mla (CSR variant) with FP8 KV scales.
            # Use the dedicated _fp8 variant to avoid colliding with the
            # autotune-agent's wrapper signature (which lacks q_scale/k_scale/v_scale).
            from aiter.ops.triton.attention.unified_attention_sparse_mla_fp8 import (
                unified_attention_sparse_mla as _ua_sparse_mla,
            )
            head_size_v = layer.v_head_dim
            seqused_k = torch.full(
                (bs,), self.nsa_index_topk,
                dtype=torch.int32, device=q_kernel.device,
            )
            _kvview = kv_cache.view(-1, 1, 1, layer.head_dim)
            _ua_sparse_mla(
                q=q_kernel,
                kv=_kvview,
                out=o_kernel,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_q=metadata.max_seq_len_q,
                seqused_k=seqused_k,
                max_seqlen_k=self.nsa_index_topk,
                softmax_scale=layer.scaling,
                topk_indices=None,
                block_table=None,
                kv_lora_rank=layer.v_head_dim,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                max_sparse_len=int(self.nsa_index_topk),  # avoid host sync per decode
                q_scale=_q_scale_t,
                k_scale=_q_scale_t,
                v_scale=_q_scale_t,
            )
        elif _os_de.environ.get("SGLANG_NSA_USE_UNIFIED_ATTN") == "1":
            # Run E: NSA-routed sparse decode through Triton unified_attention.
            from aiter.ops.triton.attention.unified_attention import (
                unified_attention as _ua_unified_attention,
            )
            head_size_v = layer.v_head_dim
            seqused_k = torch.full(
                (bs,), self.nsa_index_topk,
                dtype=torch.int32, device=q_kernel.device,
            )
            # block_table is required (assertion in unified_attention.py) but
            # unread on the SPARSE_KV branch — kv_indices is used instead.
            dummy_bt = torch.zeros(
                (bs, 1), dtype=torch.int32, device=q_kernel.device,
            )
            _kvview = kv_cache.view(-1, 1, 1, layer.head_dim)
            _ua_unified_attention(
                q=q_kernel,
                k=_kvview,
                v=_kvview[..., :head_size_v],
                out=o_kernel,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_q=metadata.max_seq_len_q,
                seqused_k=seqused_k,
                max_seqlen_k=self.nsa_index_topk,
                softmax_scale=layer.scaling,
                causal=False,
                window_size=(-1, -1),
                softcap=0.0,
                q_descale=None,
                k_descale=_q_scale_t,
                v_descale=_q_scale_t,
                block_table=dummy_bt,
                sinks=None,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
            )
        else:
            # Run D (and default): ASM mla_decode_fwd, now with FP8 scales forwarded.
            mla_decode_fwd(
                q_kernel,
                kv_cache.view(-1, 1, 1, layer.head_dim),
                o_kernel,
                metadata.cu_seqlens_q,
                kv_indptr,
                kv_indices,
                metadata.cu_seqlens_q,
                metadata.max_seq_len_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                q_scale=_q_scale_t,
                kv_scale=_kv_scale_t,
            )

        if self.need_pad_heads:
            o = o_kernel[:, :: self.head_repeat_factor, :]

        return o
