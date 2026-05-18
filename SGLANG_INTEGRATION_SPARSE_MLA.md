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

SGLang's `class NativeSparseAttnBackend._forward_aiter()` in
`python/sglang/srt/layers/attention/nsa_backend.py` is the entry point.
The class name is a misnomer — it actually serves DSA models (GLM-5,
DeepSeek-V3.2-Exp); see "Index distribution" below. That method takes
the sparse-selector's KV indices and routes them into a decode kernel.
The patch
swaps the body for one that branches on env vars:

| Env var | Branch | Kernel |
|---|---|---|
| (none) | Run D | ASM `mla_decode_fwd` with FP8 scales forwarded |
| `SGLANG_NSA_USE_UNIFIED_ATTN=1` | Run E | Triton `unified_attention` (dense, CSR-via-flag) |
| `SGLANG_NSA_USE_UA_SPARSE_MLA=1` | Run F | Triton `unified_attention_sparse_mla` (CSR, FP8) |

The patcher is a 30-line scan that locates the method by signature and
swaps the body — no AST/regex magic. See `sglang_patches/run_f/apply_patch.py`.

### What changes in `nsa_backend.py`

`_forward_aiter` lives around line 1870 of
`python/sglang/srt/layers/attention/nsa_backend.py`. Pristine SGLang has
it at 58 lines ending in a single `mla_decode_fwd(...)` call with no FP8
scales. The patch swaps the body for 158 lines with env-gated kernel
branches and the DSA shuffle hook.

**Invariants the patch preserves:**
- Method signature: `def _forward_aiter(self, q_all, kv_cache, page_table_1, layer, metadata, bs) -> torch.Tensor` is unchanged.
- All other methods in the file (including `_forward_aiter_extend` — the
  prefill path — at line 2012) are untouched.
- Module-level imports at the top of `nsa_backend.py` are unchanged. All
  new imports (`aiter.ops.triton...`, `os`, kernel wrappers) happen
  *inside* the swapped body. This is what keeps the patch a pure
  body-swap, no AST manipulation.
- The first 39 body lines (Q/O setup through `get_valid_kv_indices`) and
  the trailing 2 lines (`need_pad_heads` reshape + `return o`) are
  byte-identical to pristine.

**The diff** (only the middle 17 lines balloon to 117):

```diff
         kv_indices = self.kv_indices
         get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)

-        mla_decode_fwd(
-            q_kernel,
-            kv_cache.view(-1, 1, 1, layer.head_dim),
-            o_kernel,
-            metadata.cu_seqlens_q,
-            kv_indptr,
-            kv_indices,
-            metadata.cu_seqlens_q,
-            metadata.max_seq_len_q,
-            sm_scale=layer.scaling,
-            logit_cap=layer.logit_cap,
-        )
+        # ===== Runs D/E/F patch (additive): forward FP8 scales + env-gated UA branches =====
+        import os as _os_de
+        _q_scale_t  = getattr(layer, "k_scale", None)        # tensor form (ASM/UA)
+        _kv_scale_t = getattr(layer, "k_scale", None)
+        _q_scale_f  = getattr(layer, "k_scale_float", None)  # scalar form
+        _kv_scale_f = getattr(layer, "k_scale_float", None)
+
+        # Scatter-stress shuffle (benchmarking only).
+        if _os_de.environ.get("SGLANG_NSA_DSA_SHUFFLE") == "1":
+            for _i in range(bs):
+                _s = int(kv_indptr[_i].item())
+                _e = int(kv_indptr[_i + 1].item())
+                if _e > _s:
+                    kv_indices[_s:_e] = kv_indices[_s:_e][
+                        torch.randperm(_e - _s, device=kv_indices.device)
+                    ]
+
+        if _os_de.environ.get("SGLANG_NSA_USE_UA_SPARSE_MLA") == "1":
+            # Run F: sparse decode through Triton sparse-MLA kernel.
+            from aiter.ops.triton.attention.unified_attention_sparse_mla import (
+                unified_attention_sparse_mla as _ua_sparse_mla,
+            )
+            seqused_k = torch.full(
+                (bs,), self.nsa_index_topk,
+                dtype=torch.int32, device=q_kernel.device,
+            )
+            _kvview = kv_cache.view(-1, 1, 1, layer.head_dim)
+            _ua_sparse_mla(
+                q=q_kernel, kv=_kvview, out=o_kernel,
+                cu_seqlens_q=metadata.cu_seqlens_q,
+                max_seqlen_q=metadata.max_seq_len_q,
+                seqused_k=seqused_k,
+                max_seqlen_k=self.nsa_index_topk,
+                softmax_scale=layer.scaling,
+                topk_indices=None, block_table=None,
+                kv_lora_rank=layer.v_head_dim,
+                kv_indptr=kv_indptr, kv_indices=kv_indices,
+                max_sparse_len=int(self.nsa_index_topk),  # avoid host sync per decode
+                q_scale=_q_scale_t, k_scale=_q_scale_t, v_scale=_q_scale_t,
+            )
+        elif _os_de.environ.get("SGLANG_NSA_USE_UNIFIED_ATTN") == "1":
+            # Run E: sparse decode through Triton unified_attention.
+            from aiter.ops.triton.attention.unified_attention import (
+                unified_attention as _ua_unified_attention,
+            )
+            seqused_k = torch.full(
+                (bs,), self.nsa_index_topk,
+                dtype=torch.int32, device=q_kernel.device,
+            )
+            dummy_bt = torch.zeros((bs, 1), dtype=torch.int32, device=q_kernel.device)
+            _kvview = kv_cache.view(-1, 1, 1, layer.head_dim)
+            _ua_unified_attention(
+                q=q_kernel, k=_kvview, v=_kvview[..., :layer.v_head_dim],
+                out=o_kernel,
+                cu_seqlens_q=metadata.cu_seqlens_q,
+                max_seqlen_q=metadata.max_seq_len_q,
+                seqused_k=seqused_k,
+                max_seqlen_k=self.nsa_index_topk,
+                softmax_scale=layer.scaling,
+                causal=False, window_size=(-1, -1), softcap=0.0,
+                q_descale=None, k_descale=_q_scale_t, v_descale=_q_scale_t,
+                block_table=dummy_bt, sinks=None,
+                kv_indptr=kv_indptr, kv_indices=kv_indices,
+            )
+        else:
+            # Run D (default): ASM mla_decode_fwd, now with FP8 scales forwarded.
+            mla_decode_fwd(
+                q_kernel,
+                kv_cache.view(-1, 1, 1, layer.head_dim),
+                o_kernel,
+                metadata.cu_seqlens_q,
+                kv_indptr, kv_indices,
+                metadata.cu_seqlens_q,
+                metadata.max_seq_len_q,
+                sm_scale=layer.scaling, logit_cap=layer.logit_cap,
+                q_scale=_q_scale_t, kv_scale=_kv_scale_t,
+            )

         if self.need_pad_heads:
             o = o_kernel[:, :: self.head_repeat_factor, :]

         return o
```

**Five logical pieces:**

1. **FP8 scale lookup** (4 lines). Bug fix: pristine code never forwarded
   `layer.k_scale` to the decode kernel, so `--kv-cache-dtype fp8_e4m3`
   runs asserted inside `mla_decode_stage1_asm_fwd`. We grab both the
   tensor and float forms once and pass whichever each kernel expects.
2. **DSA shuffle** (env-gated, ~7 lines). Runs *before* the kernel
   fan-out so it affects whichever branch is selected. `bs` host syncs
   per decode — benchmarking only; never enable in production.
3. **Run F branch** (`SGLANG_NSA_USE_UA_SPARSE_MLA=1`). Calls
   `unified_attention_sparse_mla` with the CSR inputs. `topk_indices=None`
   and `block_table=None` force the CSR dispatch path inside the wrapper.
4. **Run E branch** (`SGLANG_NSA_USE_UNIFIED_ATTN=1`). Calls the dense
   Triton `unified_attention`; needs `dummy_bt` because there's a
   kernel-side assertion that's unread on the sparse-KV branch.
5. **Run D default** (no env var set). Original `mla_decode_fwd` call
   with `q_scale`/`kv_scale` added.

**Backup files** the container preserves for `diff` sanity checks:
- `nsa_backend.py.preDE` — pristine, before any Run D/E/F work.
- `nsa_backend.py.preF`  — after D/E but before Run F was added.

## Index distribution and the scatter-stress A/B

**Naming caveat first.** SGLang's `nsa_backend.py` is a misnomer. The
file/class predate the DeepSeek-V3.2 naming and the code path actually
serves **DSA** (DeepSeek Sparse Attention) models — GLM-5, Qwen3-Next,
and DeepSeek-V3.2-Exp all use DSA-style indexers, not the original
block-wise NSA. There is no "NSA model" in the GLM-5/DeepSeek family
that this backend serves. Env vars and file paths keep the `NSA`/`nsa`
prefix purely for consistency with the upstream SGLang namespace.

For the kernel this distinction is moot: the selector hands it a CSR
`(kv_indptr, kv_indices)` and the kernel never inspects the *source*.
What matters at the kernel layer is the **physical distribution** of
indices within each row:

- **Real-indexer distribution** (what the model actually emits): DSA
  picks token-level top-k, but in practice the lightning indexer's
  signal often correlates with page locality (long contiguous KV
  regions score similarly), so emitted indices may exhibit partial
  clustering. The exact distribution depends on the indexer weights,
  context length, and training corpus.
- **Uniformly scattered distribution** (worst case): consecutive
  entries land in different physical KV pages. Memory access is
  gather-dominated with no spatial reuse.

The `SGLANG_NSA_DSA_SHUFFLE=1` env hook converts the first into the
second by randomly permuting each row's `kv_indices` immediately after
the selector builds them and before any kernel branch (Runs D/E/F)
consumes them. The shuffle preserves the per-row index *set*, so
softmax output is identical up to numerical noise — only the kernel's
gather pattern changes. Costs `bs` host syncs per decode step (a few
μs); fine for benchmarking, never enable in production.

Use it to bound the kernel's performance: if real-indexer and shuffled
runs land within noise, the kernel's index-load path is not
coalescing-bound and there is no headroom from making the indexer more
locality-aware. If shuffled is much slower, the kernel is sensitive to
page-locality and worth optimizing for gather throughput.

| Combination | Recipe |
|---|---|
| **Run F, real-indexer distribution** | `bash serve_and_bench_nsa_F.sh F_real` |
| **Run F, scatter-stress (shuffled)** | `SGLANG_NSA_DSA_SHUFFLE=1 bash serve_and_bench_nsa_F.sh F_scatter` |
| **Run D (ASM), real-indexer** | `SGLANG_NSA_USE_UA_SPARSE_MLA= bash serve_and_bench_nsa_F.sh D_real` |
| **Run D (ASM), scatter-stress** | `SGLANG_NSA_USE_UA_SPARSE_MLA= SGLANG_NSA_DSA_SHUFFLE=1 bash serve_and_bench_nsa_F.sh D_scatter` |

Microbench observation (from `REPRODUCING_SPARSE_MLA.md` heads=128
table): randomly-scattered indices are within noise of block-clustered
synthetic indices at every batch — the kernel's int32 index-load path
isn't coalesced enough for clustering to translate into a measurable
kernel win. The end-to-end serve numbers should match; a large delta
would suggest SGLang dispatch overhead dominates differently (unlikely
but worth confirming).

To run against a different DSA model end-to-end (e.g. swap GLM-5 for
DeepSeek-V3.2-Exp), edit `--model` in `serve_and_bench_nsa_F.sh` — the
patch and kernel are model-agnostic, but be mindful of shape coverage
(heads=128, lora=512, rope=64 is the documented sweet spot; at heads=128
batch ≥ 32 the 2D CSR path loses ~3–4× to ASM `mla_decode_fwd` per
`REPRODUCING_SPARSE_MLA.md`'s known limitations).

## Test workflow

### 1) Standalone kernel smoke

Fastest sanity check — no SGLang involvement. Allocates a synthetic
DSA-style scattered top-k CSR (uniformly random per-row indices via
`torch.randperm`) and runs one decode.

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

### 6) Run F under scatter-stress (uniformly-random index shuffle)

A/B kernel behavior between the real DSA-indexer output and uniformly
scattered indices on the same model (no model swap needed — the shuffle
hook reorders within each row):

```bash
# Real DSA-indexer distribution (baseline)
bash sglang_patches/run_f/serve_and_bench_nsa_F.sh F_real_tp4

# Uniformly scattered (worst-case gather)
SGLANG_NSA_DSA_SHUFFLE=1 bash sglang_patches/run_f/serve_and_bench_nsa_F.sh F_scatter_tp4
```

The two labels (`F_real_tp4`, `F_scatter_tp4`) keep the log filenames
distinct. Per-decode shuffle cost is `bs` host syncs (a few μs) — far
below decode latency, so the comparison reflects kernel behavior rather
than the shuffle itself.

Same approach works for Run D and Run E:

```bash
# Run D under scatter-stress (kernel: ASM mla_decode_fwd)
SGLANG_NSA_USE_UA_SPARSE_MLA= SGLANG_NSA_DSA_SHUFFLE=1 \
    bash sglang_patches/run_f/serve_and_bench_nsa_F.sh D_scatter_tp4
```

Expected outcome: F_real vs F_scatter within noise (per microbench
heads=128 table in `REPRODUCING_SPARSE_MLA.md`). A large delta would
mean the kernel's gather throughput is the bottleneck and there's
headroom in making the index-load path more coalesced — worth
investigating if seen.

To benchmark a different DSA model (e.g. DeepSeek-V3.2-Exp), edit
`--model` in `serve_and_bench_nsa_F.sh` and rerun — the patch and
kernel are model-agnostic.

## Tunable knobs (env)

Set these in the server's environment (e.g. inside
`serve_and_bench_nsa_F.sh`) before `sglang.launch_server`:

| Env var | Default | Effect |
|---|---|---|
| `SGLANG_NSA_USE_UA_SPARSE_MLA` | unset | `=1` routes NSA decode through the Triton sparse MLA kernel |
| `SGLANG_NSA_USE_UNIFIED_ATTN` | unset | `=1` routes NSA decode through Triton `unified_attention` (Run E). Mutually exclusive with `SGLANG_NSA_USE_UA_SPARSE_MLA` |
| `SGLANG_NSA_DSA_SHUFFLE` | unset | `=1` randomly permutes each row's `kv_indices` before the kernel call, forcing uniformly-scattered access as a worst-case gather A/B against the real DSA-indexer output. Name keeps the `NSA` prefix to match SGLang's namespace; the backend is misnamed (serves DSA models). Orthogonal to the kernel-selection vars — combine with any of D/E/F |
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
- `aiter/ops/triton/attention/unified_attention_sparse_mla.py` — main wrapper (3D split-K + FP8); imported by the SGLang patch.
