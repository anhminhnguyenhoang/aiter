# jdbba Performance Snapshot — MI355X / gfx950

**Kernel:** `jagged_dense_bmm_broadcast_add` (Meta HSTU grouped-GEMM, broadcast bias)

**Date:** 2026-07-07 (post D256 XCD retune; production 256×256 tile fold; CUDA-graph validated)  
**Hardware:** AMD Instinct MI355X / gfx950 (CDNA4)  
**Checkout:** `/home/anguyenh/aiter_jdbba_flydsl` (HEAD)  
**FlyDSL:** 0.2.2 — timed via CUDA-graph warm replay (`--cudagraph --graph-l2 warm`, peak throughput)  
**Triton:** 3.7.0 (`triton_rocm-3.7.0+git9c288bc5`, Meta HSTU reference kernel; `do_bench` timing)  
**Correctness:** `cos=1.0000` on all shapes (graph replay verified via `test_jdbba_graph.py`)  
**Metric note:** Speedup = Triton time / FlyDSL time (&gt;1 means FlyDSL wins). Autoresearch log uses `ratio` = FlyDSL/Triton (&lt;1 beats).

---

## Benchmark shapes (Mi=7680)

Development / autotune grid. Not deployed in production.

### Uniform regime

| B | D | KOUT | Mi | FlyDSL (ms) | Triton (ms) | Speedup | FlyDSL (TFLOPS) | Triton (TFLOPS) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 256 | 256 | 7680 | **0.257** | 0.269 | **1.05×** | **470** | 449 |
| 120 | 512 | 512 | 7680 | **0.725** | 0.754 | 1.04× | **666** | 641 |
| 1024 | 256 | 256 | 7680 | **2.091** | 2.098 | 1.00× | 493 | 491 |
| 1024 | 512 | 512 | 7680 | **5.649** | 6.470 | **1.15×** | **730** | 638 |

### Skew regime

Skew distribution: `M_i = Mi × U(0,1)⁴`, ~20% empty groups.

| B | D | KOUT | Mi | FlyDSL (ms) | Triton (ms) | Speedup | FlyDSL (TFLOPS) | Triton (TFLOPS) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 256 | 256 | 7680 | **0.049** | 0.071 | **1.45×** | **402** | 278 |
| 120 | 512 | 512 | 7680 | **0.134** | 0.164 | **1.22×** | **599** | 483 |
| 1024 | 256 | 256 | 7680 | **0.357** | 0.404 | 1.13× | 441 | 390 |
| 1024 | 512 | 512 | 7680 | **1.003** | 1.134 | **1.13×** | **627** | 548 |

---

## Production shapes — MoE/expert

Deployed problem: `M_total=524288` (8192/expert), `N=1024`, `K=512`, `G=64`, bf16.

| Slack param | Value | Bench arg |
|-------------|-------|-----------|
| G (groups/experts) | 64 | `-b 64` |
| K (reduction) | 512 | `-d 512` |
| N (output) | 1024 | `-kout 1024` |
| rows/expert | 8192 | `-mi 8192` |
| M_total (uniform) | 524288 | 64 × 8192 |

**Dispatch (2026-07-07):** `B64D512K1024N8192` winner — `tile_m=tile_n=256`, `xcd_c=60`, `xcd_w=8`, `use_mfma_k32=true` (was fallback 128×128). Uniform only; skew routes compact `BLOCK_M=128` (unchanged).

**Vendor ceiling (GPU 5):** hipBLASLt dense `torch.mm` at M=524288, N=1024, K=512 → **0.718 ms / 766 TFLOPS**. Production uniform **751 TFLOPS (~98% of ceiling)**.

### Uniform regime

| B | D | KOUT | Mi | FlyDSL (ms) | Triton (ms) | Speedup | FlyDSL (TFLOPS) | Triton (TFLOPS) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 512 | 1024 | 8192 | **0.732** | 0.931 | **1.27×** | **751** | 586 |

### Skew regime

| B | D | KOUT | Mi | FlyDSL (ms) | Triton (ms) | Speedup | FlyDSL (TFLOPS) | Triton (TFLOPS) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 512 | 1024 | 8192 | **0.181** | 0.211 | **1.17×** | **626** | 486 |

---

## Summary

- **Benchmark shapes (peak vs Triton):** 730 TFLOPS / **1.15×** (B1024 D512 uniform); skew up to **1.45×** (B120 D256).
- **Production shapes (post 256×256 fold):** **1.27×** uniform / **1.17×** skew. **751 TFLOPS** uniform (~**98%** of hipBLASLt dense ceiling).
- **Tile fold:** 128×128 → 256×256 on production uniform **−8%** graph-warm time (0.794 → 0.732 ms), cos=1.0.
- **Dispatch:** D256 XCD retune Mi=7680 `xcd_c=48/xcd_w=24`; Mi=16384 `xcd_c=90/xcd_w=16`; production `B64D512K1024N8192` 256×256 `@ xcd_c=60/w=8`.
- **CUDA-graph:** `JaggedDenseBmmGraph` in `aiter/ops/flydsl/jagged_dense_bmm_graph.py`.
- **bf16 gfx950 ceiling:** benchmark D512 + production MoE at vendor-near efficiency; see [jdbba-autoresearch-results.md](jdbba-autoresearch-results.md).
- **Historical runs + devcontainer setup:** [jdbba_benchmark_results.md](jdbba_benchmark_results.md).

---

## Reproduce

Requires Triton 3.7.0 wheel on `jdbba-flydsl-dev` (ROCm 7.2.4, PyTorch 2.10):

```bash
WHEEL="https://download.pytorch.org/whl/nightly/triton_rocm-3.7.0%2Bgit9c288bc5-cp312-cp312-linux_x86_64.whl"

# Benchmark shapes — CUDA-graph warm + Triton
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host -e HIP_VISIBLE_DEVICES=5 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/workspaces/meta/aiter:/workspaces/generative-recommenders \
  -v /home/anguyenh/aiter_jdbba_flydsl:/workspaces/meta/aiter \
  -v /home/anguyenh/generative-recommenders:/workspaces/generative-recommenders \
  jdbba-flydsl-dev bash -lc "
    pip install -q flydsl==0.2.2
    pip install --force-reinstall -q \"\$WHEEL\"
    cd /workspaces/meta/aiter/op_tests/flydsl_tests
    python3 bench_jagged_dense_bmm_perf.py --regime both --metric time -test -mi 7680 \
      --flydsl-only --cudagraph --graph-l2 warm
    python3 bench_jagged_dense_bmm_perf.py --regime both --metric time -test -mi 7680 --triton-only
  "

# Production shapes — CUDA-graph warm + Triton
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host -e HIP_VISIBLE_DEVICES=5 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/workspaces/meta/aiter:/workspaces/generative-recommenders \
  -v /home/anguyenh/aiter_jdbba_flydsl:/workspaces/meta/aiter \
  -v /home/anguyenh/generative-recommenders:/workspaces/generative-recommenders \
  jdbba-flydsl-dev bash -lc "
    pip install -q flydsl==0.2.2
    pip install --force-reinstall -q \"\$WHEEL\"
    cd /workspaces/meta/aiter/op_tests/flydsl_tests
    python3 bench_jagged_dense_bmm_perf.py -b 64 -d 512 -kout 1024 -mi 8192 \
      --regime both --metric time -test --flydsl-only --cudagraph --graph-l2 warm
    python3 bench_jagged_dense_bmm_perf.py -b 64 -d 512 -kout 1024 -mi 8192 \
      --regime both --metric time -test --triton-only
  "

# Graph capture/replay correctness
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host -e HIP_VISIBLE_DEVICES=5 \
  -e FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  -e PYTHONPATH=/workspaces/meta/aiter:/workspaces/generative-recommenders \
  -v /home/anguyenh/aiter_jdbba_flydsl:/workspaces/meta/aiter \
  -v /home/anguyenh/generative-recommenders:/workspaces/generative-recommenders \
  jdbba-flydsl-dev bash -lc '
    cd /workspaces/meta/aiter/op_tests/flydsl_tests
    python3 test_jdbba_graph.py
  '
```
