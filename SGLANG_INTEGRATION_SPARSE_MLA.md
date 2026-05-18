# Plugging `unified_attention_sparse_mla` into SGLang

This guide covers (a) the kernel's call shape so you can wire it into other
serving runtimes, and (b) the concrete patch + serve harness shipped in
`sglang_patches/run_f/` for SGLang's NSA backend on GLM-5-FP8.

For microbench reproduction and tuning, see `REPRODUCING_SPARSE_MLA.md`.

## TL;DR

```bash
# Inside the anguyenh-sglang-benchmark container, after copying the
# worktree's aiter/ops/triton/ files into /sgl-workspace/aiter/...
cd /sgl-workspace/aiter
python sglang_patches/run_f/apply_patch.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/nsa_backend.py \
    sglang_patches/run_f/new_forward_aiter.py
python sglang_patches/run_f/smoke_test.py            # standalone kernel check
bash   sglang_patches/run_f/serve_and_bench_nsa_F.sh F_tp4   # serve + bench
```

The patch is **idempotent w.r.t. the body region** but not against itself —
re-running on an already-patched file will look for `def _forward_aiter(` and
swap it again with the same body. Safe to re-run after `git checkout` of
`nsa_backend.py`.

## Kernel contract

The wrapper is `aiter.ops.triton.attention.unified_attention_sparse_mla.unified_attention_sparse_mla`.
There are **two input forms** and both run on the same launcher:

| Form | Sparsity input | When to use |
|---|---|---|
| Dense top-k | `topk_indices: [seq_len, TOP_K] int32` | Original `unified_attention` API |
| **CSR** | `kv_indptr: [seq_len+1] int32`, `kv_indices: [nnz] int32` | NSA / DSA; variable per-query length; pre-pruned indices |

CSR is the path used by the SGLang patch. It dispatches further:
- **3D split-K** when `total_num_q_blocks < UNIFIED_ATTENTION_SPARSE_MLA_NUM_CU` and
  `max_sparse_len >= UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD` (default 1024).
  This is the perf lever for low-batch decode — it parallelizes across KV
  segments so a single CTA does not own a whole query.
- **2D CSR** otherwise (high `batch * heads`, where the grid already fills the GPU).

### Required arguments

```python
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla,
)

unified_attention_sparse_mla(
    q,                     # [total_q_tokens, num_heads, lora + rope]   bf16
    kv,                    # [num_pages, page_size, 1, lora + rope]     bf16 or fp8_e4m3
    out,                   # [total_q_tokens, num_heads, lora]          bf16 (written in place)
    cu_seqlens_q,          # [batch + 1]  int32
    max_seqlen_q,          # int
    seqused_k,             # [batch]      int32  — only used to size the host loop
    max_seqlen_k,          # int
    softmax_scale,         # float (typically lora ** -0.5)
    topk_indices=None,     # mutually exclusive with kv_indptr/kv_indices
    block_table=None,      # required only by the dense top-k path
    kv_lora_rank=lora,     # int
    kv_indptr=kv_indptr,   # [seq_len + 1] int32
    kv_indices=kv_indices, # [nnz]         int32  (absolute KV-slot positions, NOT page ids)
    max_sparse_len=None,   # int; pass to avoid a host sync per call
    q_scale=None,          # FP8: per-tensor scalar tensor or python float
    k_scale=None,          # FP8: per-tensor scalar tensor or python float
    v_scale=None,          # FP8: per-tensor scalar tensor or python float
)
```

Important shape notes:
- `kv` is a 4D paged buffer `[num_pages, page_size, 1, head_dim]`. SGLang's
  flat KV cache is `.view(-1, 1, 1, head_dim)` (page_size=1).
- `kv_indices` entries are **absolute KV-slot positions** (token-level), not
  page IDs. The kernel translates `idx -> (page, slot) = (idx // page_size,
  idx % page_size)` internally.
- `out` is the lora-only output (`[..., lora]`), since MLA collapses rope
  after softmax.
- Pass `max_sparse_len` explicitly. Computing it from `kv_indptr` requires a
  device→host `.item()` sync per decode step.

### FP8 KV cache

K-cache and V-cache are the same buffer in MLA, so `k_scale == v_scale` in
practice. Per-tensor only — no per-channel or per-token. Q is normally bf16
so `q_scale=None`. The kernel:
- Folds `K_SCALE` into `qk_scale` (same compile-time effect as bf16 once
  applied).
- Applies `V_SCALE` post-softmax in the epilogue (`acc *= v_scale`).
- The 3D reduce kernel stays scale-agnostic (its inputs are already fp32
  partial outputs).

## How the SGLang patch works

SGLang's NSA backend has `class NativeSparseAttnBackend._forward_aiter()` in
`python/sglang/srt/layers/attention/nsa_backend.py` — that method takes the
NSA-decimated KV indices and routes them into a decode kernel. The patch
swaps the body for one that branches on env vars:

| Env var | Branch | Kernel |
|---|---|---|
| (none) | Run D | ASM `mla_decode_fwd` with FP8 scales forwarded |
| `SGLANG_NSA_USE_UNIFIED_ATTN=1` | Run E | Triton `unified_attention` (dense, CSR-via-flag) |
| `SGLANG_NSA_USE_UA_SPARSE_MLA=1` | Run F | Triton `unified_attention_sparse_mla` (CSR, FP8) |

The patcher is a 30-line scan that locates the method by signature and
swaps the body — no AST/regex magic. See `sglang_patches/run_f/apply_patch.py`.

### Current `_fp8` import path

`sglang_patches/run_f/new_forward_aiter.py` imports from
`aiter.ops.triton.attention.unified_attention_sparse_mla_fp8` — a frozen
older copy of the wrapper that pre-dates the 3D split-K work. It exists
purely to keep the Run-F SGLang patch from colliding with the autotune-agent
during the experiment.

**Recommended migration:** swap the import in `new_forward_aiter.py` to the
main wrapper. Both the main wrapper and the `_fp8` shim now accept identical
FP8 args (`q_scale, k_scale, v_scale`), so the call site is unchanged. The
main wrapper additionally gives you the 3D split-K path, which is where
~2-3× decode speedups at low batch come from.

```python
# new_forward_aiter.py (recommended)
from aiter.ops.triton.attention.unified_attention_sparse_mla import (
    unified_attention_sparse_mla as _ua_sparse_mla,
)
```

This migration was not validated end-to-end in this branch (the multi-GPU
SGLang serve run was deferred to avoid blocking colleagues). The standalone
smoke test in `smoke_test.py` covers the call surface and uses the `_fp8`
shim; rerun it against the main wrapper before flipping the production import.

## Test workflow

### 1) Standalone kernel smoke

Fastest sanity check — no SGLang involvement. Allocates a synthetic NSA-like
top-k CSR and runs one decode.

```bash
docker exec -it anguyenh-sglang-benchmark bash
cd /sgl-workspace/aiter
python sglang_patches/run_f/smoke_test.py
# Expected: prints `smoke OK; out.abs().max() = <small bf16>`
```

Edit the constants at the top of `smoke_test.py` to mimic your target shape
(batch, heads, top_k, sk).

### 2) Microbench vs ASM (offline)

See `REPRODUCING_SPARSE_MLA.md` § Microbench. The bench script does NOT
require the SGLang patch — it goes directly to the Triton wrapper.

```bash
python op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py \
    --batch 1 --sq 1 --sk 2048 \
    --heads 16 --lora-dim 512 --rope-dim 64 --block-size 64 \
    --top-k 2048 --dtype fp8 --validate
```

### 3) Apply the patch

```bash
cd /sgl-workspace/aiter
python sglang_patches/run_f/apply_patch.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/nsa_backend.py \
    sglang_patches/run_f/new_forward_aiter.py
# Reports: "Replacing lines N..M of <target>"
```

To revert: `cd /sgl-workspace/sglang && git checkout python/sglang/srt/layers/attention/nsa_backend.py`.

### 4) Serve + bench end-to-end

```bash
bash sglang_patches/run_f/serve_and_bench_nsa_F.sh F_tp4
```

`serve_and_bench_nsa_F.sh` launches the SGLang server, waits for `/health`,
runs a curl smoke test, runs a warmup, then a real `sglang.bench_serving`
sweep. Logs:
- `serve_<label>_tp4_<ts>.log` — server stdout/stderr
- `warmup_logs/<label>...` — warmup
- `benchmark_logs/<label>...` — final perf

Defaults: TP=4 on GPUs 4–7, in=8192, out=1024, concurrency=64,
`--kv-cache-dtype fp8_e4m3`, `--disable-cuda-graph`. Override via env:
- `TP_SIZE=8` for a wider TP shard
- `CUDA_VISIBLE_DEVICES=0,1,2,3` for different GPUs
- `NSA_DISABLE_CUDA_GRAPH=` (empty) to enable cuda-graph capture

**Multi-GPU note:** the default uses GPUs 4–7 for ~30 min. Coordinate with
collaborators before running.

### 5) A/B against ASM (Run D)

To compare against the baseline ASM kernel without re-patching:

```bash
# Same patch, but un-set the Run F env var
SGLANG_NSA_USE_UA_SPARSE_MLA= bash sglang_patches/run_f/serve_and_bench_nsa_F.sh D_tp4
```

(The patched `_forward_aiter` falls through to ASM `mla_decode_fwd` when
neither `SGLANG_NSA_USE_UA_SPARSE_MLA` nor `SGLANG_NSA_USE_UNIFIED_ATTN`
is set.)

## Tunable knobs (env)

Set these in the server's environment (e.g. inside
`serve_and_bench_nsa_F.sh`) before `sglang.launch_server`:

| Env var | Default | Effect |
|---|---|---|
| `SGLANG_NSA_USE_UA_SPARSE_MLA` | unset | `=1` routes NSA decode through the Triton sparse MLA kernel |
| `UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD` | 1024 | Minimum `max_sparse_len` to enable 3D split-K. Lower for short-K decode shapes |
| `UNIFIED_ATTENTION_SPARSE_MLA_DISABLE_SPLIT_K` | 0 | `=1` forces 2D path (A/B for split-K isolation) |
| `UNIFIED_ATTENTION_SPARSE_MLA_NUM_CU` | 256 | CU-count hint for picking `NUM_SEGMENTS_PER_SEQ`. MI355X has 304; default leaves headroom |
| `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE` | unset | `=1` swaps in `triton.autotune` for both 2D paths and the 3D path. Slow first call, useful for re-tuning on a new shape |

## Migrating to other serving runtimes

If you're integrating into vLLM / TGI / something custom, the contract is the
same as in the patched `_forward_aiter`:

1. Build `kv_indptr` (cumsum of valid per-query top-k counts) and
   `kv_indices` (absolute KV-slot positions) from your sparse selector.
2. Reshape your paged KV cache to `[num_pages, page_size, 1, head_dim]` (use
   `kv_cache.view(-1, 1, 1, head_dim)` when `page_size=1`).
3. Allocate `out` as `[total_q, num_heads, lora_dim]` in bf16.
4. Pass `max_sparse_len` explicitly (the kernel will derive it from
   `kv_indptr` with a host sync otherwise — bad for per-step decode).
5. For FP8 KV, pass `k_scale=v_scale=<scalar>` and leave `q_scale=None`
   (assuming bf16 Q).

The kernel handles head-padding internally only if you do it before calling
(`repeat_interleave` for GQA). It does not pad heads itself.

## File map

- `sglang_patches/run_f/apply_patch.py` — body-swap utility (idempotent w.r.t. method region).
- `sglang_patches/run_f/new_forward_aiter.py` — replacement `_forward_aiter` body with env-gated branches.
- `sglang_patches/run_f/smoke_test.py` — standalone kernel smoke.
- `sglang_patches/run_f/serve_and_bench_nsa_F.sh` — full serve+bench orchestration.
- `aiter/ops/triton/attention/unified_attention_sparse_mla.py` — main wrapper (3D split-K + FP8).
- `aiter/ops/triton/attention/unified_attention_sparse_mla_fp8.py` — frozen 2D-only variant currently imported by the SGLang patch.
