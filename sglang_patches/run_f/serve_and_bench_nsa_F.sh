#!/usr/bin/env bash
# NSA-routed sparse decode benchmark orchestration for Run F.
#
# Run F = Triton unified_attention_sparse_mla with FP8 per-tensor KV scales.
# Routed via SGLANG_NSA_USE_UA_SPARSE_MLA=1 inside _forward_aiter.
#
# Args:
#   $1 = label (e.g., F_tp4)
#
# Logs (inside container):
#   /workspace/serve_${LABEL}_tp${TP}_${TS}.log
#   /workspace/warmup_logs/${LABEL}_tp${TP}_in8192_out1024_conc64_warmup.${TS}.log
#   /workspace/benchmark_logs/${LABEL}_tp${TP}_in8192_out1024_conc64_benchmark.${TS}.log

set -Eeuo pipefail

LABEL="${1:?label required}"
TS="$(date +%Y%m%d_%H%M%S)"

# Parallelization: defaults to TP=4 on GPUs 4-7.
TP_SIZE="${TP_SIZE:-4}"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

# Cuda-graph toggle: NSA_DISABLE_CUDA_GRAPH=1 adds --disable-cuda-graph.
NSA_DISABLE_CUDA_GRAPH="${NSA_DISABLE_CUDA_GRAPH:-1}"

INPUT_LEN=8192
OUTPUT_LEN=1024
CONCURRENCY=64
NUM_PROMPTS=$((CONCURRENCY * 10))

SERVE_LOG="/workspace/serve_${LABEL}_tp${TP_SIZE}_${TS}.log"
WARMUP_LOG="/workspace/warmup_logs/${LABEL}_tp${TP_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}_conc${CONCURRENCY}_warmup.${TS}.log"
BENCH_LOG="/workspace/benchmark_logs/${LABEL}_tp${TP_SIZE}_in${INPUT_LEN}_out${OUTPUT_LEN}_conc${CONCURRENCY}_benchmark.${TS}.log"

mkdir -p /workspace/warmup_logs /workspace/benchmark_logs

echo "[$(date -Is)] === ${LABEL} | TP=${TP_SIZE} | GPUs=${CUDA_VISIBLE_DEVICES_VAL} | DISABLE_CG='${NSA_DISABLE_CUDA_GRAPH}' | TS=${TS} ==="

# Ensure no stale server.
pkill -f sglang.launch_server 2>/dev/null || true
sleep 5

# Launch server.
echo "[$(date -Is)] launching NSA server -> ${SERVE_LOG}"
(
  export SGLANG_USE_AITER=1
  export SGLANG_AITER_MLA_PERSIST=0
  export SGLANG_AITER_FP8_PREFILL_ATTN=0
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"
  export SAFETENSORS_FAST_GPU=1
  export UNIFIED_ATTN_AUTOTUNE=1
  export SGLANG_NSA_USE_UA_SPARSE_MLA=1
  # Run F is the UA sparse MLA branch; leave UNIFIED_ATTN unset so it doesn't
  # tee into Run E's branch.
  unset SGLANG_NSA_USE_UNIFIED_ATTN || true
  cd /sgl-workspace/sglang
  CG_FLAG=""
  if [ -n "${NSA_DISABLE_CUDA_GRAPH}" ]; then
    CG_FLAG="--disable-cuda-graph"
  fi
  python -m sglang.launch_server \
    --model zai-org/GLM-5-FP8 \
    --attention-backend nsa \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend aiter \
    --host 0.0.0.0 --port 5060 \
    --tp-size "${TP_SIZE}" \
    --served-model-name glm-5-fp8 \
    --trust-remote-code \
    --watchdog-timeout 12000 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --kv-cache-dtype fp8_e4m3 \
    ${CG_FLAG} \
    > "${SERVE_LOG}" 2>&1
) &
SERVER_PID=$!
echo "[$(date -Is)] server PID=${SERVER_PID}"

# Wait for /health (up to 25 min).
DEADLINE=$(( $(date +%s) + 1500 ))
READY=0
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  if curl -sf --max-time 3 http://localhost:5060/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[$(date -Is)] server PID died; tail of serve log:"
    tail -80 "${SERVE_LOG}"
    exit 1
  fi
  sleep 15
done

if [ "${READY}" -ne 1 ]; then
  echo "[$(date -Is)] server did not become ready in 25 min; killing"
  kill "${SERVER_PID}" 2>/dev/null || true
  pkill -f sglang.launch_server 2>/dev/null || true
  tail -80 "${SERVE_LOG}"
  exit 1
fi
echo "[$(date -Is)] server ready"

# Curl smoke test (FP8 KV must not assert; answer must be coherent).
echo "[$(date -Is)] curl smoke test"
curl -sS -X POST http://localhost:5060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"glm-5-fp8\",
       \"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],
       \"max_tokens\":32,\"temperature\":0.0}" \
  | tee "/workspace/smoke_${LABEL}_${TS}.json"
echo
echo "[$(date -Is)] smoke OK"

# Warmup.
echo "[$(date -Is)] warmup -> ${WARMUP_LOG}"
HF_HUB_OFFLINE=0 python3 -m sglang.bench_serving \
  --backend sglang --port 5060 \
  --model zai-org/GLM-5-FP8 \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" --max-concurrency "${CONCURRENCY}" \
  > "${WARMUP_LOG}" 2>&1 || { echo "warmup failed"; tail -30 "${WARMUP_LOG}"; pkill -f sglang.launch_server 2>/dev/null || true; exit 2; }

# Main bench.
echo "[$(date -Is)] main bench -> ${BENCH_LOG}"
HF_HUB_OFFLINE=0 python3 -m sglang.bench_serving \
  --backend sglang --port 5060 \
  --model zai-org/GLM-5-FP8 \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" --max-concurrency "${CONCURRENCY}" \
  > "${BENCH_LOG}" 2>&1 || { echo "bench failed"; tail -30 "${BENCH_LOG}"; pkill -f sglang.launch_server 2>/dev/null || true; exit 2; }

# Teardown.
echo "[$(date -Is)] killing server"
kill "${SERVER_PID}" 2>/dev/null || true
pkill -f sglang.launch_server 2>/dev/null || true
sleep 10
echo "[$(date -Is)] === ${LABEL} done ==="
echo "logs:"
echo "  serve:  ${SERVE_LOG}"
echo "  smoke:  /workspace/smoke_${LABEL}_${TS}.json"
echo "  warmup: ${WARMUP_LOG}"
echo "  bench:  ${BENCH_LOG}"
