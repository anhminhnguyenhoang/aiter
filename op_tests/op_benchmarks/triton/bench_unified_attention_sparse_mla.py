import argparse

import torch
import triton

import aiter.mla
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)
from op_tests.triton_tests.attention.test_unified_attention_sparse_mla import (
    Param,
    chunk_input,
    generate_test_data,
)


def dense_physical_indices_to_csr(indices_in_kvcache):
    num_rows = indices_in_kvcache.shape[0]
    chunks = []
    indptr = [0]
    max_sparse_len = 0
    for row in range(num_rows):
        valid = indices_in_kvcache[row] != -1
        row_indices = indices_in_kvcache[row, valid]
        chunks.append(row_indices)
        row_len = int(row_indices.numel())
        indptr.append(indptr[-1] + row_len)
        max_sparse_len = max(max_sparse_len, row_len)

    if chunks:
        kv_indices = torch.cat(chunks).to(torch.int32)
    else:
        kv_indices = torch.empty(
            (0,), dtype=torch.int32, device=indices_in_kvcache.device
        )
    kv_indptr = torch.tensor(
        indptr, dtype=torch.int32, device=indices_in_kvcache.device
    )
    return kv_indptr, kv_indices, max_sparse_len


def make_inputs(args):
    total_dim = args.lora_dim + args.rope_dim
    test_p = Param(
        args.batch,
        args.sq,
        args.sk,
        d=total_dim,
        dv=args.lora_dim,
        h_q=args.heads,
        block_size=args.block_size,
        is_varlen=args.varlen,
        is_causal=False,
        is_fp8=False,
        topk=args.top_k,
        test_performance=True,
    )
    cache_seqlens, q, block_table, blocked_k, abs_indices, indices_in_kvcache = (
        generate_test_data(test_p)
    )
    (
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        q,
        block_table,
        blocked_k,
        _,
        indices_in_kvcache,
    ) = chunk_input(
        cache_seqlens,
        q,
        block_table,
        blocked_k,
        abs_indices,
        indices_in_kvcache,
    )
    kv_indptr, kv_indices, max_sparse_len = dense_physical_indices_to_csr(
        indices_in_kvcache
    )
    return (
        q,
        blocked_k,
        block_table,
        indices_in_kvcache,
        kv_indptr,
        kv_indices,
        max_sparse_len,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
    )


def run(args):
    (
        q,
        blocked_k,
        block_table,
        topk_indices,
        kv_indptr,
        kv_indices,
        max_sparse_len,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
    ) = make_inputs(args)

    softmax_scale = args.lora_dim**-0.5
    out_topk = torch.empty((*q.shape[:-1], args.lora_dim), device=q.device, dtype=q.dtype)
    out_csr = torch.empty_like(out_topk)
    out_mla = torch.empty_like(out_topk)
    total_q = q.shape[0]
    qk_head_dim = args.lora_dim + args.rope_dim
    kv_buffer_mla = blocked_k.reshape(-1, 1, 1, qk_head_dim)
    qo_indptr_mla = torch.arange(
        total_q + 1, dtype=torch.int32, device=q.device
    )
    kv_last_page_lens = torch.ones(total_q, dtype=torch.int32, device=q.device)

    def run_topk():
        unified_attention_sparse_mla(
            q,
            blocked_k,
            out_topk,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            topk_indices,
            block_table,
            args.lora_dim,
        )

    def run_csr():
        unified_attention_sparse_mla(
            q,
            blocked_k,
            out_csr,
            cu_seqlens_q,
            max_seqlen_q,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            None,
            block_table,
            args.lora_dim,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            max_sparse_len=max_sparse_len,
        )

    def run_mla_decode():
        aiter.mla.mla_decode_fwd(
            q,
            kv_buffer_mla,
            out_mla,
            qo_indptr_mla,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            1,
            1,
            1,
            softmax_scale,
        )

    if args.validate:
        run_topk()
        run_csr()
        torch.testing.assert_close(out_csr, out_topk, atol=1.5e-2, rtol=1e-2)
        run_mla_decode()
        torch.testing.assert_close(out_mla, out_csr, atol=1.5e-2, rtol=1e-2)

    topk_ms = triton.testing.do_bench(run_topk, warmup=args.warmup, rep=args.rep)
    csr_ms = triton.testing.do_bench(run_csr, warmup=args.warmup, rep=args.rep)
    mla_ms = triton.testing.do_bench(
        run_mla_decode, warmup=args.warmup, rep=args.rep
    )
    nnz = int(kv_indices.numel())
    print(
        "unified_attention_sparse_mla "
        f"batch={args.batch} sq={args.sq} sk={args.sk} heads={args.heads} "
        f"lora={args.lora_dim} rope={args.rope_dim} block={args.block_size} "
        f"top_k={args.top_k} nnz={nnz} max_sparse_len={max_sparse_len}"
    )
    print(f"topk_matrix_ms={topk_ms:.4f}")
    print(f"csr_ms={csr_ms:.4f}")
    print(f"mla_decode_fwd_ms={mla_ms:.4f}")
    print(f"csr_speedup_vs_topk={topk_ms / csr_ms:.3f}x")
    print(f"mla_decode_fwd_speedup_vs_csr={csr_ms / mla_ms:.3f}x")
    print(f"mla_decode_fwd_speedup_vs_topk={topk_ms / mla_ms:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse MLA top-k vs CSR")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sq", type=int, default=1)
    parser.add_argument("--sk", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--lora-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--validate", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
