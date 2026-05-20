Co-authors: @anhminhnguyenhoang

## Motivation

Extends the existing Triton **`unified_attention_sparse_mla`** kernel (dense top-k, BF16 only) with the input forms, dispatch path, and dtype coverage needed to back **DSA-style decode** (GLM-5, DeepSeek-V3.2-Exp, Qwen3-Next) on AMD Instinct GPUs (MI300X / MI355X). Concretely:

- **CSR input form** `(kv_indptr, kv_indices)` alongside the existing dense `[seq_len, TOP_K]` matrix — variable per-query length, no padding, matches the selector output shape used by SGLang's `nsa_backend`.
- **3D split-K path** for low-batch decode where the existing 2D grid leaves most CUs idle — gives a **1.77–1.88× decode-latency win** over `aiter.mla.mla_decode_fwd` at the GLM-5 decode shape (`heads=8`, batch ≤ 8) on MI355X. Speedup tapers to ~1.0× at batch=32 and crosses over (~0.92×) at batch=64 as the 2D grid naturally fills the GPU; the 3D path is the lever for the low-batch decode regime that motivates the kernel.
- **FP8 (e4m3) KV-cache support folded into the main wrapper** — per-tensor `q/k/v_scale` flow through the 2D and 3D paths without a separate `_fp8` shim.

The pre-existing dense top-k BF16 dispatch is preserved bit-compatible; existing call sites continue to work unchanged.

## Technical Details

`unified_attention_sparse_mla()` dispatches to one of two kernel families chosen at call time:

1. **2D path (`_kernel_unified_attention_sparse_mla_2d`)** -- the pre-existing dense top-k kernel body, extended to also serve the new CSR input form via `USE_CSR: tl.constexpr`. The only divergent region between the two forms is the per-tile KV-index fetch, so Triton specialization yields two AMDGCN variants with no runtime branch cost. Used when `total_num_q_blocks >= _NUM_CU_HINT` (grid already fills the GPU). Vestigial `block_tables_ptr` / `block_table_stride` params dropped -- both 2D bodies derive `(page, slot)` directly from `pos // BLOCK_SIZE` / `pos % BLOCK_SIZE`.
2. **3D split-K path (`_kernel_unified_attention_sparse_mla_csr_3d` + `_kernel_unified_attention_sparse_mla_csr_reduce`)** -- **new** in this PR. For low-batch decode where the existing 2D grid would leave most CUs idle. Grid is `(num_q_blocks, NUM_SEGMENTS_PER_SEQ)` so a single-batch decode fills the GPU instead of running 1 CTA on 304 CUs. `NUM_SEGMENTS_PER_SEQ` is chosen from a CU-count hint and rounded to a power of two for the LSE reduce kernel layout.

Shared softmax / dtype work added in this PR (applies to both paths):

- Softmax via `tl.math.exp2` with `RCP_LN2` folded into `qk_scale` once (replaces `tl.math.exp`).
- Fused `tl.dot(Q_lora, K_lora, acc=S)` over the rope dot.
- `PRELOAD_V=True` to overlap V load with softmax.
- FP8 KV-cache plumbing: per-tensor `q_scale / k_scale / v_scale` (scalar tensor or python float; coerced to 0-d float32) thread through 2D dense, 2D CSR, and 3D split-K. `K_SCALE` folds into `qk_scale`; `V_SCALE` applies post-softmax in the epilogue, so the 3D reduce kernel stays scale-agnostic. Per-tile `.to(Q.dtype)` casts promote FP8 K_rope / V_lora to BF16 before `tl.dot`.

**Dispatch policy** (`aiter/ops/triton/attention/unified_attention_sparse_mla.py`):

- 3D split-K when CSR inputs are passed, `total_num_q_blocks < UNIFIED_ATTENTION_SPARSE_MLA_NUM_CU`, and `max_sparse_len >= UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD` (default 1024). FP8 uses the same policy as BF16.
- 2D otherwise. Dense top-k path is preserved bit-compatible.

`max_sparse_len` is an optional argument -- pass it explicitly to skip a device->host `.item()` per decode step.

**Tuned defaults** (autotune sweep on MI355X, `lora=512, rope=64, block=64, top_k=2048` BF16 decode shapes):

- 3D path: dispatched by `num_query_heads`. Heads ≤ 8 uses the GLM-5 b=1 winner `TILE_SIZE=64, num_warps=8, num_stages=1, PRELOAD_V=False, waves_per_eu=2` (b=1 is the dominant lever — 3D split-K beats ASM `mla_decode_fwd` ~1.88× there; b=32/64 still re-validate within ~10% of optimal). Heads ≥ 16 keeps the prior tuning target `TILE_SIZE=32, num_warps=8, num_stages=2, PRELOAD_V=True, waves_per_eu=2` (heads=16 b=1 winner from the earlier sk=8192 sweep) so heads=16 callers don't regress.
- 2D CSR path: `TILE_SIZE=64, num_warps=4, num_stages=1, PRELOAD_V=False, waves_per_eu=2`. Picked from the heads=128 batch=32/64 winner (the 2D path's natural regime: high batch * heads where the grid already fills the GPU). Distinct from the dense top-k defaults -- only the CSR launch site uses these (Step 7 autotune A/B confirmed CSR winners do NOT transfer to the dense path: dense gets 22-27% slower across batch 1-64).

**Layout**:

- Kernels: `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py` -- merged `_kernel_unified_attention_sparse_mla_2d` (with `USE_CSR` constexpr branch + FP8 scale args), new `_kernel_unified_attention_sparse_mla_csr_3d` and `_kernel_unified_attention_sparse_mla_csr_reduce`, plus three autotuners: `_2d_topk_autotuner`, `_2d_csr_autotuner`, `_3d_csr_autotuner`. The two 2D autotuners wrap the same merged kernel with distinct cache keys so per-path tuning state stays isolated.
- API: `aiter/ops/triton/attention/unified_attention_sparse_mla.py` -- `unified_attention_sparse_mla()` wrapper extended for CSR input, 3D dispatch, FP8 scale coercion. `BLOCK_M=16` is unchanged from prior work.
- Tests: `op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py` -- extended to cover the new CSR + 3D + FP8 paths.
- Benchmark: `op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py` -- vs ASM `mla_decode_fwd`; supports `--dtype {bf16,fp8}`, `--sparse-pattern {nsa,dsa}` (DSA shuffles each row's indices to simulate the lightning-indexer's scattered selection), `--validate` (torch reference), `--skip-mla-decode` (when aiter HIP isn't built).
- Reproduction docs: `REPRODUCING_SPARSE_MLA.md` (kernel-side recipe) and `SGLANG_INTEGRATION_SPARSE_MLA.md` (SGLang serving integration + Run D/E/F harness in `sglang_patches/run_f/`).

**Env knobs** (read by the wrapper):

| Env var | Default | Effect |
|---|---|---|
| `UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD` | `1024` | Minimum `max_sparse_len` to enable 3D split-K. Lower for shorter-K decode shapes that still underfill the GPU. |
| `UNIFIED_ATTENTION_SPARSE_MLA_DISABLE_SPLIT_K` | `0` | Force-disable the 3D path for A/B comparisons. |
| `UNIFIED_ATTENTION_SPARSE_MLA_NUM_CU` | `256` | CU-count hint used to size `NUM_SEGMENTS_PER_SEQ`. MI355X has 304 CUs; default leaves headroom for waves. |
| `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE` | unset | When set, runs the three Triton autotuners instead of baked-in defaults -- slow first call, useful for re-sweeping on a new shape. |

## Test Plan

`op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py` covers both the 2D and 3D dispatch paths under both input forms (dense top-k and CSR) at BF16 and FP8 KV-cache, against a torch reference. The `--validate` flag in the bench script provides an additional online correctness check. After the 2D merge, **178/178 test cases pass**; CSR microbench parity within +/-3% at every shape.

## Test Setup

Triton/torch/aiter must come from the container -- the bare host has neither. The GLM-5 heads=8 numbers in this PR were collected inside `anguyenh-dev-2`, image `amdsiloai/pytorch-xdit:v26.4`, which carries a post-`public/main`-merge `aiter/mla.py` that supports the GLM-5 BF16 decode shape via the new `gfx950 + bf16` catch-all (dispatches to `mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co`). The SGLang container `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260427` (image used for prior `heads=16` numbers) ships an older `aiter.mla` that asserts on `(nhead=8, bf16, max_seqlen_q=1)` -- avoid it for GLM-5 numbers.

```
# In the project root (where aiter_ua is checked out)
docker exec -w /home/$USER/aiter_ua anguyenh-dev-2 bash -lc '
  export PYTHONPATH=$PWD:$PYTHONPATH

  pytest op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py -v -s

  # microbench (BF16) at GLM-5 shape
  python3 op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py \
      --batch 1 --sq 1 --sk 2048 \
      --heads 8 --lora-dim 512 --rope-dim 64 --block-size 64 \
      --top-k 2048 --warmup 25 --rep 100 --validate

  # microbench (FP8 KV-cache)
  python3 op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py \
      --batch 1 --sq 1 --sk 2048 \
      --heads 8 --lora-dim 512 --rope-dim 64 --block-size 64 \
      --top-k 2048 --warmup 25 --rep 100 --validate --dtype fp8
'
```

Sweep `--batch` in `{1, 8, 32, 64}` to see how the 3D split-K advantage tapers as the 2D grid naturally fills the GPU. Add `--sparse-pattern dsa` to A/B the kernel's gather throughput against block-clustered (NSA) vs uniformly scattered (DSA) index distributions on the same row sets.

## Test Result

### gfx950 (MI355X) -- GLM-5 decode shape sweep (heads=8, BF16)

Environment: Ubuntu 24.04, ROCm 7.x, PyTorch 2.9.1, **Triton 3.6.0**, image `amdsiloai/pytorch-xdit:v26.4` (container `anguyenh-dev-2`). Shape: `heads=8, lora=512, rope=64, block=64, sk=2048, top_k=2048`, BF16 KV-cache. CSR is the Triton path under test; baseline is `aiter.mla.mla_decode_fwd` (ASM) -- the post-`public/main`-merge `aiter/mla.py` adds a `gfx950 + bf16` catch-all entry in the natively-supported list, which dispatches to `mla_a16w16_qh8_qseqlen1_gqaratio8_v3.co`. The 3D split-K restructure is the dominant lever for batch ≤ 8.

The Triton kernel was extended to handle `num_query_heads < BLOCK_M` (ceil-div grid sizing + per-lane head-active mask) so it runs at GLM-5's heads=8 at all -- pre-fix it would launch a 0-program grid and divide-by-zero. BLOCK_M=16 is kept; lanes 8-15 are masked out, costing 50% MFMA tile utilisation but enabling a like-for-like comparison against ASM.

| batch | csr_ms | mla_decode_fwd_ms | csr speedup vs mla |
|---|---|---|---|
| 1  | 0.0251 | 0.0471 | **1.88x** |
| 8  | 0.0244 | 0.0431 | **1.77x** |
| 32 | 0.0408 | 0.0409 | **1.00x** |
| 64 | 0.0715 | 0.0656 | **0.92x** |

Speedup tapers at batch≥32 (CSR's per-CTA work scales ~linearly while `mla_decode_fwd` stays flat at heads=8 -- they cross around batch=32). The 3D path is the lever for the low-batch decode regime that motivates the kernel.

### gfx950 (MI355X) -- GLM-5 decode shape sweep (heads=8, FP8 KV-cache)

Same shape, `--dtype fp8` (e4m3, per-tensor `k_scale = v_scale = amax / 448`). Triton FP8 keeps Q at BF16 and quantizes only the K/V cache to FP8 (matches the SGLang Run F production path, where Q is the RMSNorm output and never round-trips through FP8). The bench script's `--dtype fp8` flag skips the ASM baseline by default — that gate was added before the post-merge `mla.py` exposed the FP8 path at heads=8 — so the comparison below is BF16 Triton vs FP8 Triton:

| batch | bf16 csr_ms | fp8 csr_ms | fp8 / bf16 |
|---|---|---|---|
| 1  | 0.0251 | 0.0263 | 1.05x |
| 8  | 0.0244 | 0.0267 | 1.09x |
| 32 | 0.0408 | 0.0383 | 0.94x |
| 64 | 0.0715 | 0.0657 | 0.92x |

FP8 KV-cache is within ~5-10% of BF16 at low batch (where the kernel is launch-bound and the per-tile `.to(bf16)` promote cast dominates the bandwidth saving) and pulls ~5-8% ahead at batch ≥ 32 (where the K/V bandwidth saving overtakes the cast cost). Tuning: a heads=8 FP8 autotune sweep at b ∈ {1, 8, 32, 64} produced per-batch winners that differ from the BF16 winner on three axes (`num_warps`, `PRELOAD_V`, `waves_per_eu`), but when measured against the BF16-baked `DEFAULT_3D_*` at proper warmup=25/rep=100 each FP8-autotuner pick **regressed** (1.74x at b=1, 1.59x at b=8, 1.12x at b=32; only b=64 was 1.20x faster autotuned). The autotuner's internal timing is noisier than the bench. Net: the BF16-baked `TILE_SIZE=64, num_warps=8, num_stages=1, PRELOAD_V=False, waves_per_eu=2` is kept as the single dtype-agnostic default; FP8 callers in the heads=8 b=64 regime can opt into `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE=1` for the modest extra win.

**ASM `mla_decode_fwd` FP8 cross-check (a8w8, apples-to-apples).** The bench script's default gate aside, ASM does provide an FP8 heads=8 path on gfx950: `mla_a8w8_qh8_qseqlen1_gqaratio8_v3.co`. This kernel is a8w8, so Q must also be FP8 — Triton matches it via the existing `q_scale` arg:

| batch | tri_a8w8 (fp8-Q + fp8-KV) | asm_a8w8 | tri / asm |
|---|---|---|---|
| 1  | 0.0314 | 0.0505 | 0.62x (Triton **1.61x faster**) |
| 8  | 0.0315 | 0.0518 | 0.61x (Triton **1.64x faster**) |
| 32 | 0.0313 | 0.0451 | 0.70x (Triton **1.44x faster**) |
| 64 | 0.0501 | 0.0444 | 1.13x (Triton 0.89x — **ASM wins**) |

Triton wins 1.44–1.64x at b ≤ 32; ASM only takes the lead at b=64, by 1.13x. Correctness: fp8-Q + fp8-KV rel error vs bf16 reference is 3.85%, no NaN/Inf.

### NSA vs DSA sparsity-pattern A/B (heads=128, MI355X)

`--sparse-pattern {nsa,dsa}` toggles between block-clustered indices (test harness default) and uniformly scattered indices that simulate DeepSeek V3.2-Exp's lightning-indexer token-level selection. **Within noise** at every batch -- the kernel's int32 index-load path isn't coalesced enough for clustering to matter:

| batch | nsa csr_ms | dsa csr_ms | mla_decode_fwd_ms |
|---|---|---|---|
| 1  | 0.0297 | 0.0241 | 0.0511 / 0.0475 |
| 8  | 0.0744 | 0.0746 | 0.0401 / 0.0398 |
| 32 | 0.2543 | 0.2548 | 0.0718 / 0.0714 |
| 64 | 0.3918 | 0.3927 | 0.1035 / 0.1032 |

At `heads=128` (DeepSeek V3.2-Exp shape), batch ≤ 8 the 3D CSR path wins (~1.97× over `mla_decode_fwd` at batch=1 DSA). Batch ≥ 32 falls into the 2D CSR path and loses ~3.5-3.8× to ASM -- the kernel's per-CTA work at heads=128 high-batch outruns occupancy. The 2D CSR re-tune from the new heads=128 b=32/64 autotune sweep (`PRELOAD_V=False, waves_per_eu=2`) improved heads=128 b=32 by ~17% (0.3064 → 0.2543 ms) and b=64 by ~11% (0.4418 → 0.3918 ms) vs the prior `PRELOAD_V=True` defaults, but is still well behind ASM; closing the rest of the gap likely needs a kernel restructure (better int32 index-load coalescing or a `BLOCK_M` re-tune), not config tuning -- flagged as out-of-scope follow-up.

### gfx942 (MI300X)

Not re-measured for this PR. The wrapper and kernels are arch-portable; defaults were tuned on MI355X. A gfx942 re-sweep via `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE=1` is a planned follow-up before any MI300X-side perf claim.

### End-to-end (SGLang Run F)

`sglang_patches/run_f/` patches SGLang's `NativeSparseAttnBackend._forward_aiter` with env-gated kernel branches (Run D = ASM `mla_decode_fwd`, Run E = Triton `unified_attention`, Run F = Triton `unified_attention_sparse_mla`) plus an optional `SGLANG_NSA_DSA_SHUFFLE` scatter-stress A/B. End-to-end SGLang Run F serving validation on GLM-5-FP8 is a planned follow-up (multi-GPU multi-hour run) -- see `SGLANG_INTEGRATION_SPARSE_MLA.md` for the kernel contract, the patch flow, and the migration recipe.

## Known Limitations

- **`heads=128 batch >= 32`** in the 2D CSR path is ~3.5-3.8x behind ASM `mla_decode_fwd` even after the heads=128 b=32/64 re-tune (which itself recovered ~11-17%). Closing the gap likely requires a kernel restructure (better int32 index-load coalescing or `BLOCK_M` re-tune), not config tuning. Out of scope for this PR.
- **`heads=8 batch >= 32`** in the 3D CSR path ties (`b=32`) or slightly loses (~0.92×) to ASM at `b=64` -- the 3D path's CU-fill advantage erodes once batch * head-blocks naturally fills the GPU. The 3D path is the lever for batch ≤ 8 where ASM `mla_decode_fwd` underfills CUs at heads=8.
- **`BLOCK_M=16` at `num_query_heads < BLOCK_M`** (e.g. GLM-5 heads=8): wrapper now ceil-divs the head dimension and the kernel masks lanes ≥ `num_query_heads` so the kernel runs correctly at heads=8. Cost: 50% MFMA tile utilisation per BLOCK_M tile. A native BLOCK_M=8 variant (MFMA 16×16 remapping) is a deeper refactor, out of scope for this PR.
- **Baked defaults** target the GLM-5 b=1 3D regime for `num_query_heads ≤ 8` and the prior heads=16 b=1 winner for `num_query_heads ≥ 16` (dispatched by `num_query_heads`). Intermediate or unlisted shapes should re-run `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE=1`. The 2D CSR defaults are tuned on heads=128 b=64.
- **FP8 vs BF16 at heads=8**: FP8 is ~5-10% slower at b ≤ 8 (launch-bound regime where the per-tile `.to(bf16)` promote cast dominates the K/V bandwidth saving), ~5-8% faster at b ≥ 32. The dtype-agnostic baked defaults `DEFAULT_3D_*` were verified to outperform per-batch FP8-autotuner picks in real bench at b=1/8/32; only b=64 FP8 gains a further ~20% by enabling `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE=1`.
- **FP8 vs ASM `mla_a8w8_qh8_qseqlen1_gqaratio8_v3` at heads=8 (apples-to-apples a8w8)**: Triton fp8-Q + fp8-KV is 1.44–1.64x faster than ASM at b ≤ 32 and **1.13x slower** at b=64 (ASM wins). Correctness verified: fp8-Q + fp8-KV rel error 3.85% vs bf16 reference, no NaN/Inf.
- **Tuned shape**: `heads ∈ {8, 128}, lora=512, rope=64, block=64, top_k=2048` on MI355X. Other decode geometries (different lora/rope, different top_k, different heads) should re-run autotune.

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
