# jdbba Kernel Benchmark — FlyDSL vs Triton

**Kernel:** `jagged_dense_bmm_broadcast_add` (Meta HSTU grouped-GEMM, broadcast bias)

---

## Current MI355X / gfx950 numbers

**See [jdbba_performance_snapshot_2026-07-07.md](jdbba_performance_snapshot_2026-07-07.md)** for the canonical headline + MoE tables, summary, reproduce commands, and raw log (post D256 XCD retune, commit `024e999a9`).

The 2026-07-06 run below is retained only as historical context (pre-retune; skew compact-tile-map landing). Do not use those tables for current standing.

### Historical note (2026-07-06, superseded)

That run flipped B1024_D256 uniform from 0.99× to 1.01× via XCD re-sweep (`xcd_c` 32→60, `xcd_w` 8→16) and closed skew gaps by enabling the compacted-tile-map skew grid on gfx950. Subsequent Batch 2 sweeps (2026-07-07) retuned D256 further (`xcd_c=48/xcd_w=24` at Mi=7680; `xcd_c=90/xcd_w=16` at Mi=16384) — see snapshot and `jdbba-autoresearch-results.md` § BATCH 2.

---

| Component        | Version |
|------------------|---------|
| Base image       | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (built as `jdbba-flydsl-dev`) |
| Container OS     | Ubuntu 24.04 |
| ROCm (container) | `7.2.53211` |
| Python           | 3.12.3 |
| PyTorch          | `2.10.0+rocm7.2.4.git3d3aa833` |
| Triton           | `3.6.0+rocm7.2.4.git4ed88892` (`AITER_USE_SYSTEM_TRITON=1`) |
| FlyDSL           | `0.2.2` |
| aiter            | editable install from bind-mounted checkout |
| GPU arch seen    | `gfx950` |

### Bind mounts (gfx950 run)

| Host path | Container path | RW |
|-----------|----------------|----|
| `/home/anguyenh/aiter_jdbba_flydsl` | `/workspaces/meta/aiter` | yes |
| `/home/anguyenh/aiter_jdbba_flydsl` | `/workspaces/aiter` | yes |
| `/home/anguyenh/generative-recommenders` | `/workspaces/generative-recommenders` | yes |
| `/home/anguyenh/KernelForge` | `/workspaces/KernelForge` | yes |

### Devcontainer rebuild & launch (gfx950 run)

```bash
# Build image from jdbba-docs/.devcontainer
docker build -t jdbba-flydsl-dev \
  -f jdbba-docs/.devcontainer/Dockerfile.rocm \
  jdbba-docs/.devcontainer/

# Start container
docker rm -f jdbba-flydsl 2>/dev/null
docker run -d --name jdbba-flydsl \
  --ipc=host --network=host --shm-size=128g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e AMD_VISIBLE_DEVICES=5 \
  -e HIP_VISIBLE_DEVICES=5 \
  -v /home/anguyenh/aiter_jdbba_flydsl:/workspaces/aiter \
  -v /home/anguyenh/aiter_jdbba_flydsl:/workspaces/meta/aiter \
  -v /home/anguyenh/generative-recommenders:/workspaces/generative-recommenders \
  -v /home/anguyenh/KernelForge:/workspaces/KernelForge \
  -w /workspaces/meta/aiter \
  jdbba-flydsl-dev \
  bash -lc 'sleep infinity'

# Install deps inside container
docker exec jdbba-flydsl bash -lc '
  cd /workspaces/meta/aiter
  pip install --break-system-packages -q flydsl==0.2.2
  AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 \
    pip install --break-system-packages -e . --no-build-isolation -q
  pip install --break-system-packages -e /workspaces/KernelForge -q
'
```

### Run the benchmark (gfx950 run)

Use the reproduce block in [jdbba_performance_snapshot_2026-07-07.md](jdbba_performance_snapshot_2026-07-07.md). Timing uses `triton.testing.do_bench` (CUDA-event, L2-flushed each rep). TFLOPS uses actual packed length `L`: `2·L·D·N / time`. GPU device binding varies by run; speedup ratios are portable.

---

## Historical run — MI300X / gfx942 (2026-06-10)

**Hardware:** MI300X / gfx942 (CDNA3)
**Date:** 2026-06-10
**Branch:** `anguyenh/flydsl-jdbba`

### Uniform regime (time [ms], lower is better)

| B    | D   | KOUT | Mi   | FlyDSL (ms) | Triton (ms) | Speedup |
|------|-----|------|------|-------------|-------------|---------|
| 120  | 256 | 256  | 7680 | 0.4740      | 0.6521      | 1.38×   |
| 120  | 512 | 512  | 7680 | 1.3520      | 1.8934      | 1.40×   |
| 1024 | 256 | 256  | 7680 | 4.0384      | 5.4542      | 1.35×   |
| 1024 | 512 | 512  | 7680 | 11.3004     | 16.1430     | 1.43×   |

### Skew regime (time [ms], lower is better)

| B    | D   | KOUT | Mi   | FlyDSL (ms) | Triton (ms) | Speedup |
|------|-----|------|------|-------------|-------------|---------|
| 120  | 256 | 256  | 7680 | 0.1209      | 0.1562      | 1.29×   |
| 120  | 512 | 512  | 7680 | 0.3274      | 0.4019      | 1.23×   |
| 1024 | 256 | 256  | 7680 | 0.8075      | 1.0668      | 1.32×   |
| 1024 | 512 | 512  | 7680 | 2.4925      | 3.1512      | 1.26×   |

### Summary (gfx942)

- **Uniform:** FlyDSL is **1.35–1.43×** faster than Triton across all shapes.
- **Skew:** FlyDSL is **1.23–1.32×** faster than Triton across all shapes.
- All outputs match reference (`cos=1.0000`).
- Goal is up to 2× over Triton; the C-shuffle epilogue HBM round-trip is the current floor.

### Container image & toolchain (gfx942 run)

| Component       | Version |
|-----------------|---------|
| Base image      | `rocm/primus:v26.2` |
| Container OS    | Ubuntu 24.04.4 LTS |
| ROCm (container)| `7.2.0` |
| Python          | 3.12.3 |
| PyTorch         | `2.10.0a0+git449b176` (HIP `7.2.26015`) |
| Triton          | `3.6.0` (`AITER_USE_SYSTEM_TRITON=1`) |
| FlyDSL          | `0.1.9` (`flydsl==0.1.9.dev599`) |
| aiter           | editable install from bind-mounted checkout |
| GPU arch seen   | `gfx942` |

### Bind mounts (gfx942 run)

| Host path | Container path | RW |
|-----------|----------------|----|
| `/home/anguyenh/aiter` | `/workspaces/meta/aiter` | yes |
| `/home/anguyenh/mvonstra-amd/recsys-kernels` | `/workspaces/meta/recsys-kernels` | yes |

### Devcontainer launch (gfx942 run)

```bash
docker run -d --name jdbba-flydsl \
  --ipc=host --network=host --shm-size=128g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --cap-add CAP_SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /home/anguyenh/aiter:/workspaces/meta/aiter \
  -v /home/anguyenh/mvonstra-amd/recsys-kernels:/workspaces/meta/recsys-kernels \
  -w /workspaces/meta/recsys-kernels/recsys_harness \
  rocm/primus:v26.2 \
  bash -lc '
    set -e
    cd /workspaces/meta/aiter
    pip install --quiet --pre "flydsl==0.1.9.dev599"
    AITER_USE_SYSTEM_TRITON=1 PREBUILD_KERNELS=0 pip install -e . --no-build-isolation --quiet
    exec sleep infinity
  '
```
