"""Smoke test for unified_attention_sparse_mla invocation shape that the
Run-F patch in _forward_aiter will use.

Imports the in-container aiter from /sgl-workspace/aiter so it matches what
SGLang will see during serving.
"""
import sys
sys.path.insert(0, "/sgl-workspace/aiter")

import torch
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)

torch.manual_seed(0)
device = "cuda"

# Mimic NSA decode shapes at TP=4: 16 Q-heads, top-k=2048.
batch = 4
sq = 1
sk = 4096  # full KV
heads = 16
lora = 512
rope = 64
head_dim = lora + rope
top_k = 2048
block_size = 64

n_blocks = (sk + block_size - 1) // block_size
total_blocks = batch * n_blocks

# Allocate a flat KV cache view-able as [num_pages, 1, 1, head_dim] per SGLang call.
kv_full = (torch.randn(total_blocks * block_size, 1, 1, head_dim, device=device) / 10)
fp8_max = 448.0
amax = kv_full.abs().max().clamp_min(1e-8)
scale_val = (amax / fp8_max).item()
kv_fp8 = (kv_full.float() / scale_val).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
scale_t = torch.tensor(scale_val, dtype=torch.float32, device=device)

q = torch.randn(batch * sq, heads, head_dim, device=device).clamp_(-1, 1).to(torch.bfloat16)
out = torch.empty(batch * sq, heads, lora, device=device, dtype=torch.bfloat16)
cu_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * sq

# Build CSR indices: top_k absolute KV slot positions per query, all valid.
kv_indices_l = []
kv_indptr = [0]
for i in range(batch * sq):
    cur = top_k  # exactly top_k entries per query
    perm = torch.randperm(int(sk), device=device)[:cur].to(torch.int32) + i * sk
    kv_indices_l.append(perm)
    kv_indptr.append(kv_indptr[-1] + cur)
kv_indices = torch.cat(kv_indices_l).to(torch.int32)
kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device=device)

seqused_k = torch.full((batch,), top_k, dtype=torch.int32, device=device)

# Call exactly like _forward_aiter (Run F branch).
unified_attention_sparse_mla(
    q=q,
    kv=kv_fp8,
    out=out,
    cu_seqlens_q=cu_q,
    max_seqlen_q=1,
    seqused_k=seqused_k,
    max_seqlen_k=top_k,
    softmax_scale=lora ** -0.5,
    topk_indices=None,
    block_table=None,
    kv_lora_rank=lora,
    kv_indptr=kv_indptr,
    kv_indices=kv_indices,
    max_sparse_len=top_k,
    q_scale=scale_t,
    k_scale=scale_t,
    v_scale=scale_t,
)
torch.cuda.synchronize()
print("smoke OK; out.abs().max() =", out.abs().max().item())
print("aiter wrapper path:", end=" ")
import aiter.ops.triton.attention.unified_attention_sparse_mla as m
print(m.__file__)
