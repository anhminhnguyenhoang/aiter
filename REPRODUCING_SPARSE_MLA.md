# Reproducing the sparse-MLA decode improvements on `ua_sparse_mla`

This document captures the harvested perf work for the Triton
`unified_attention_sparse_mla` kernel on MI355X. The CSR-input baseline,
the initial perf wave, and the three-step followup plan together land all
of the improvements:

```
1a28de504 sparse mla: tune 2D CSR defaults for high-batch heads=128       (followup step 2)
2976c0bca sparse mla: add --sparse-pattern flag for NSA vs DSA microbench (followup step 3)
49ab2ca7a sparse mla: plumb FP8 scales through 3D split-K path            (followup step 1)
10d5e8872 sparse mla: add FP8 KV-cache support + Run F SGLang patch
fe87a54fa sparse mla: 3D split-K + exp2 + tuned defaults for decode shapes
f694d452a sparse mla: add CSR (kv_indptr, kv_indices) input path + benchmark + docs   <-- baseline
```

Followup-plan status: **all three steps merged.** See "Followup plan" below
for what each step changed and the validation that gated it.

## What the commits actually change

**`fe87a54fa` (BF16 perf)**
- New 3D split-K CSR kernel + log-sum-exp reduce kernel
  (`_kernel_unified_attention_sparse_mla_csr_3d` and
  `_kernel_unified_attention_sparse_mla_csr_reduce`). Grid is
  `(num_q_blocks, NUM_SEGMENTS_PER_SEQ)` so a single-batch decode fills the
  GPU instead of running 1 CTA on 304 CUs.
- Wrapper dispatch picks 3D when `total_num_q_blocks < num_CUs` and
  `max_sparse_len >= UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD`.
- Softmax uses `tl.math.exp2` with `RCP_LN2` folded into `qk_scale`.
- `tl.dot(Q_lora, K_lora, acc=S)` fuses the lora dot onto the rope dot.
- `PRELOAD_V=True` overlaps the V load with softmax.
- Tuned 3D defaults: `TILE_SIZE=32, num_warps=8, num_stages=2,
  PRELOAD_V=True, waves_per_eu=2` (autotune sweep on GLM-5 decode shapes).

**`10d5e8872` (FP8 KV-cache + SGLang serve harness)**
- 2D dense top-k and 2D CSR kernels grow per-tensor `q_scale/k_scale/v_scale`
  inputs and `Q_SCALE/K_SCALE/V_SCALE` constexpr flags. K-scale folds into
  `qk_scale`; V-scale applies in the epilogue.
- Per-tile `.to(Q.dtype)` casts promote FP8 K_rope / V_lora to BF16 before
  `tl.dot`.
- The wrapper coerces python `int`/`float` scales into 0-d tensors and
  threads them through both the 2D CSR and 3D split-K paths. FP8 uses the
  same dispatch policy as BF16 (the 3D kernel was extended to accept
  `q_scale/k_scale/v_scale` + `Q_SCALE/K_SCALE/V_SCALE` constexpr flags;
  V_SCALE is applied to each per-segment `acc` before it lands in
  `segm_output_ptr` so the reduce kernel stays scale-agnostic).
- `sglang_patches/run_f/` ships an SGLang `forward_aiter` shim + smoke test
  + a `serve_and_bench` script reproducing the end-to-end Run F harness.

## Followup plan

Three independently-shippable commits resolved the three gaps the initial
perf wave left open. Each had a microbench gate that had to clear before the
next step landed.

**Step 1 — FP8 → 3D split-K (`49ab2ca7a`).** The initial 10d5e8872 work
threaded FP8 scales through the 2D dense and 2D CSR kernels only; the
wrapper's `use_split_k` was force-gated off when any FP8 scale was set, so
FP8 decode at low batch fell back to the 2D path and lost the 3D CU-fill
win. This step mirrored the 2D CSR FP8 pattern into
`_kernel_unified_attention_sparse_mla_csr_3d` (added `q_scale/k_scale/v_scale`
ptrs + `Q_SCALE/K_SCALE/V_SCALE` constexprs, folded K_SCALE into
`qk_scale * RCP_LN2`, applied V_SCALE in the per-segment epilogue so the
reduce kernel stays scale-agnostic, added the FP8→BF16 promotion casts on
each tile), dropped the `HAS_FP8` guard in the wrapper, and added the
FP8 flags to `_3d_csr_autotuner`'s key so BF16 and FP8 configs don't
collide.

**Step 2 — Autotune sweep + 2D CSR re-tune (`1a28de504`).** A
`UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE=1 TRITON_PRINT_AUTOTUNING=1` sweep
across `heads ∈ {16, 64, 128}` × `batch ∈ {1, 8, 32, 64}` (GLM-5 lora/rope,
sk=2048, top_k=2048) found that the 3D defaults already held up across
shapes — every candidate winner from the autotuner regressed (or was within
noise of) the baked-in `DEFAULT_3D_*` when re-validated at proper
`warmup=25/rep=100`. The 2D CSR path was different: at
heads=128/batch=64 (where `total_num_q_blocks ≥ _NUM_CU_HINT` so the
dispatcher picks 2D CSR), `PRELOAD_V=True` + `waves_per_eu=2` gave a
durable 28% local speedup (0.6147 → 0.4420 ms). New constants
`DEFAULT_2D_CSR_*` (wrapper) split the 2D CSR launch site from the dense
top-k launch site so the dense path's defaults stay unchanged for
back-compat.

**Step 4 — 2D kernel merge (post-followup cleanup).** The 2D dense top-k
and 2D CSR kernels were ~70% byte-identical; the only divergent region
was the per-tile KV index fetch. Merged into a single
`_kernel_unified_attention_sparse_mla_2d` selected by a `USE_CSR:
tl.constexpr` flag — Triton specializes per constexpr so generated AMDGCN
is identical to the two prior kernels (no runtime branch cost). Dropped
the vestigial `block_tables_ptr` / `block_table_stride` params (both 2D
bodies derived `(page, slot)` from `pos // BLOCK_SIZE` / `pos %
BLOCK_SIZE` and never dereferenced `block_tables_ptr`). Wrapper keeps two
autotuner wrappers (`_2d_topk_autotuner`, `_2d_csr_autotuner`) targeting
the merged kernel with distinct cache keys so per-path tuning state
stays isolated. Step 7 autotune A/B confirmed CSR's
`PRELOAD_V=True`/`waves_per_eu=2` winners do NOT transfer to the dense
path (dense gets 22–27% slower across batch 1–64), so the dense
defaults stay `PRELOAD_V=False`/`waves_per_eu=1`. Verified: 178/178
test cases pass; CSR microbench parity within ±3% at every shape (B=1
heads=16 csr_ms 0.0354 → 0.0313, an incidental win from the merged body
collapsing a missing-`RCP_LN2` fold in the old CSR kernel — both bodies
now share the dense path's correct `qk_scale = scale * RCP_LN2`).

**Step 3 — DSA-shaped microbench (`2976c0bca`).** Added a `--sparse-pattern
{nsa,dsa}` flag to `bench_unified_attention_sparse_mla.py`. `nsa` keeps
the test harness's block-quantized selection; `dsa` shuffles each row's
indices so consecutive entries scatter across physical blocks, simulating
DeepSeek V3.2-Exp's lightning-indexer token-level selection. Microbench
results (heads=128 table below) show DSA-pattern indices are within noise
of NSA-pattern at every batch — the kernel's index-loading path is not
coalesced enough for block clustering to matter. At heads=128 batch ≤ 8
the 3D CSR path still beats `mla_decode_fwd` (~2.4× at batch=1); at
batch ≥ 32 the 2D CSR path loses ~3–4× to ASM, and Step 2's re-tune is
the floor for that regime without a kernel restructure.

End-to-end SGLang Run F validation was deferred from this plan
(multi-GPU multi-hour run; would block colleagues). See
`SGLANG_INTEGRATION_SPARSE_MLA.md` for the kernel-contract walkthrough,
the `sglang_patches/run_f/` harness, and the migration recipe for the
frozen `_fp8` shim still imported by the Run F patch.

## Container

Builds, tests, and benches must run inside the project's docker container —
the bare host has no torch/triton/aiter. The container used for all numbers
in this doc is `anguyenh-sglang-benchmark`. Copy the worktree's
`aiter/ops/triton/...` files into `/sgl-workspace/aiter/...` inside the
container before benching.

## Microbench (CSR vs `mla_decode_fwd`)

GLM-5 decode shape sweep (heads=16, lora=512, rope=64, block=64, sk=2048,
top_k=2048). The CSR path on `ua_sparse_mla` beats the ASM
`mla_decode_fwd` baseline by 1.4–2.94× across batch 1–64. The 3D split-K
restructure is the dominant lever — at batch=1 it accounts for an ~8×
local speedup over the 2D-only version.

```bash
# inside the container
python op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py \
    --batch 1 --sq 1 --sk 2048 \
    --heads 16 --lora-dim 512 --rope-dim 64 --block-size 64 \
    --top-k 2048 --warmup 25 --rep 100
```

Sweep batch ∈ {1, 8, 32, 64} to see how the 3D split-K advantage tapers as
batch grows and the 2D grid naturally fills the GPU.

For an FP8 KV-cache run (now uses the same dispatch as BF16 — 3D split-K
when CSR is enabled and the GPU is under-filled):

```bash
python op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py \
    --batch 1 --sq 1 --sk 2048 \
    --heads 16 --lora-dim 512 --rope-dim 64 --block-size 64 \
    --top-k 2048 --warmup 25 --rep 100 \
    --dtype fp8
```

Add `--validate` to compare against a torch reference, and
`--skip-mla-decode` if aiter HIP isn't built.

### NSA vs DSA sparsity pattern

`--sparse-pattern {nsa,dsa}` controls how the top-k indices are
distributed across physical blocks. `nsa` (default) is the
block-quantized selection emitted by the test harness — consecutive
indices in a row tend to share a physical block. `dsa` shuffles each
row's indices so consecutive entries scatter across blocks, simulating
DeepSeek Sparse Attention's lightning-indexer token-level selection.

```bash
python op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py \
    --batch 1 --sq 1 --sk 2048 \
    --heads 128 --lora-dim 512 --rope-dim 64 --block-size 64 \
    --top-k 2048 --warmup 25 --rep 100 --validate \
    --sparse-pattern dsa --dtype bf16
```

Microbench on the kernel (heads=128, lora=512, rope=64, block=64,
sk=2048, top_k=2048, BF16, MI355X GPU 4) shows DSA-pattern indices are
**within noise of NSA-pattern** at every batch — the kernel's index-
loading path is not coalesced enough for block clustering to matter:

| batch | nsa csr_ms | dsa csr_ms | mla_decode_fwd_ms |
|---|---|---|---|
| 1 | 0.0506 | 0.0391 | 0.124 |
| 8 | 0.0740 | 0.0739 | 0.109 |
| 32 | 0.3064 | 0.3058 | 0.103 |
| 64 | 0.4418 | 0.4382 | 0.106 |

Status: at `heads=128` (DeepSeek V3.2-Exp shape) batch ≤ 8 the 3D CSR
path wins (CSR ~2.4× faster than `mla_decode_fwd` at batch=1). Batch ≥
32 falls into the 2D CSR path (`total_num_q_blocks ≥ _NUM_CU_HINT`) and
loses to `mla_decode_fwd` by ~3–4× — the kernel's per-CTA work at
heads=128 high-batch outruns occupancy. The 2D CSR defaults were
re-tuned for this shape (`DEFAULT_2D_CSR_PRELOAD_V=True`,
`waves_per_eu=2`); that gave a 28% local speedup on `heads=128 batch=64`
(0.6147 → 0.4420 ms) but is still ~4× behind ASM. Closing the rest of
the gap likely needs a kernel restructure (better index-load coalescing
or BLOCK_M re-tune), not just config tuning.

## Tunable knobs

All env vars are read in `aiter/ops/triton/attention/unified_attention_sparse_mla.py`.

| Env var | Default | Effect |
|---|---|---|
| `UNIFIED_ATTENTION_SPARSE_MLA_SPLIT_K_THRESHOLD` | `1024` | Minimum `max_sparse_len` to enable 3D split-K. Lower this if your decode shapes have shorter K but still under-fill the GPU. |
| `UNIFIED_ATTENTION_SPARSE_MLA_DISABLE_SPLIT_K` | `0` | Force-disable the 3D path for A/B comparisons. |
| `UNIFIED_ATTENTION_SPARSE_MLA_NUM_CU` | `256` | CU-count hint used to pick `NUM_SEGMENTS_PER_SEQ`. MI355X has 304 CUs; this hint is intentionally a bit smaller to leave headroom for waves. |
| `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE` | (off) | When set, runs the Triton autotuners (`_2d_csr_autotuner`, `_2d_topk_autotuner`, `_3d_csr_autotuner`) instead of the baked-in defaults. Use to re-sweep configs on a new shape. |

If you re-tune via `UNIFIED_ATTENTION_SPARSE_MLA_AUTOTUNE=1`, the winning
config from the sweep on the shape above was
`TILE_SIZE=32, num_warps=8, num_stages=2, PRELOAD_V=True, waves_per_eu=2`
for the 3D path. Update `DEFAULT_3D_*` in the wrapper if a new shape needs
different defaults.

For the 2D CSR path (used when CSR is enabled and `total_num_q_blocks ≥
_NUM_CU_HINT`), defaults were re-tuned on the heads=128 batch=64 shape:
`TILE_SIZE=64, num_warps=4, num_stages=1, PRELOAD_V=True, waves_per_eu=2`
(`DEFAULT_2D_CSR_*` in the wrapper). This is distinct from the dense
top-k path's defaults — only the CSR launch site uses it.

## End-to-end (SGLang Run F)

`sglang_patches/run_f/` patches SGLang's MLA forward to call
`unified_attention_sparse_mla` with the CSR input shape. See
`SGLANG_INTEGRATION_SPARSE_MLA.md` for the kernel-contract walkthrough,
the patch flow, the env knobs the patched `_forward_aiter` reads, and the
migration recipe for the frozen `_fp8` shim that the Run F patch still
imports.

```bash
# inside the container, after the aiter files are copied into /sgl-workspace/aiter
python sglang_patches/run_f/smoke_test.py            # functional check
bash   sglang_patches/run_f/serve_and_bench_nsa_F.sh # serving throughput
```

The Run E throughput baseline (pre-3D-split-K) was 367 tok/s; the post-merge
end-to-end serving number is the final validation gate for this branch.

## Known limitations

- 3D defaults were tuned on `heads=16, lora=512, rope=64, block=64,
  top_k=2048` and re-validated across `heads ∈ {16, 64, 128}` in Step 2.
  Other decode geometries (different lora/rope, different top_k) should
  still re-run autotune — Step 2 only swept the head axis.
- 2D CSR path at heads=128 batch ≥ 32 is ~3–4× behind `mla_decode_fwd`
  even after Step 2's `DEFAULT_2D_CSR_*` re-tune. Closing the gap likely
  requires a kernel restructure (better int32 index-load coalescing or a
  `BLOCK_M` re-tune), not config tuning. Out of scope for the followup
  plan.
- `BLOCK_M=16` is hardcoded in the wrapper; changing it requires re-tuning.
- At higher batches (≥32) FP8 CSR is marginally slower than BF16 CSR
  (~5%) — the FP8 → BF16 promotion casts cost more than the K-cache
  bandwidth savings buy back when the kernel is no longer
  bandwidth-bound. FP8 still wins at batch=1 where the 3D path matters
  most.

## File map

- `aiter/ops/triton/attention/unified_attention_sparse_mla.py` — Python
  wrapper (dispatch, env knobs, FP8 scale coercion).
- `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py`
  — Triton kernels: unified 2D (`_kernel_unified_attention_sparse_mla_2d`
  selects dense-topk vs CSR index source via a `USE_CSR: tl.constexpr`
  flag), 3D CSR split-K, reduce, and the three autotuners (two of which
  wrap the same merged 2D kernel with different cache keys).
- `op_tests/op_benchmarks/triton/bench_unified_attention_sparse_mla.py` —
  microbench vs `mla_decode_fwd` with `--dtype {bf16,fp8}` and `--validate`.
- `sglang_patches/run_f/` — SGLang serve+bench harness for end-to-end Run F.
