#!/usr/bin/env bash
# qwen38-sglang.sh — serve Qwen3.8-27B (NVFP4) on a DGX Spark / GB10 with SGLang.
# DRAFT — review before running.
#
# Modeled on the SGLang cookbook's DGX Spark cell (validated on GB10), adapted:
#   * MODEL defaults to RadixArk/Qwen3.8-27B-NVFP4 — the SGLang-compatible NVFP4
#     checkpoint. (Unsloth NVFP4 quants are vLLM-only per their model card:
#     lm_head is FP8-quant and SGLang cannot load it.)
#   * PORT defaults to 8889 and SERVED_MODEL_NAME is overridable, so an existing
#     client pin (e.g. an agent-delegation config pointing at this host:port with a
#     fixed model name) can be preserved with a one-line env override.
#   * GB10 host-level clock cap (nvidia-smi -lgc via a systemd unit) is untouched
#     and stays in effect — container-level flags do not change SM clocks.
#     Recommended cap: sudo nvidia-smi -lgc 0,2200 (NOT 2190 — that is stale).
#   * NOTE: SGLang defaults to thinking ON for reasoning models. For short chat
#     replies, set chat_template_kwargs thinking:false per request, or the visible
#     content comes back empty (finish_reason "length"). This is expected, not a bug.
#   * CPU pinning defaults to GB10's 10 Cortex-X5 cores (5-9,15-19); set CPUSET=""
#     to disable.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-27b-sglang}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"   # model-specific GB10 build (torch 2.13+cu130, SGLang qwen38.27b)
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-27b-sglang}"
HOST="0.0.0.0"
PORT="${PORT:-8889}"
WORK_DIR="${WORK_DIR:-$HOME/qwen38-sglang}"    # holds ./cache (HF + Triton) — persists warm kernels across restarts
HF_HOME="${WORK_DIR}/cache/huggingface"
TRITON_CACHE_DIR="${WORK_DIR}/cache/triton"
LOG_FILE="${WORK_DIR}/sglang.log"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

# --- tuning (defaults = the measured-optimal GB10 cell) ---
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"     # native. 1M: set CONTEXT_LENGTH=1000000 (+YARN auto)
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-10}"
SPEC_STEPS="${SPEC_STEPS:-3}"                  # MTP head trained for 3 steps (on-device sweep: peak)
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"                  # topk=1 requires DRAFT = STEPS + 1
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"     # validated Spark cell (2048 = smoother mixed-load ITL)
CPUSET="${CPUSET:-5-9,15-19}"
MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK:-0}"   # 1 frees one GDN state slot (S 4->3)
# 1 = also apply the YaRN rope_parameters override (factor 4.0 at 262K — the "1M-ready"
#    blend; identical JSON to the common vLLM --hf-overrides recipe). Repo-validated path
#    is native 262K / YaRN off (0). If you raise CONTEXT_LENGTH past 262144, SGLang needs
#    the longer-context override:
YARN_OVERRIDE="${YARN_OVERRIDE:-0}"
if (( CONTEXT_LENGTH > 1000000 )); then echo "CONTEXT_LENGTH > 1000000 unsupported"; exit 1; fi
if (( CONTEXT_LENGTH < 262144 )); then echo "CONTEXT_LENGTH < 262144 unsupported"; exit 1; fi
if [[ "${SPEC_TOPK}" != "1" ]] || (( SPEC_DRAFT != SPEC_STEPS + 1 )); then
  echo "SPEC_TOPK must be 1 and SPEC_DRAFT = SPEC_STEPS+1 (MTP chain)"; exit 1
fi

NEED_YARN=0
if (( CONTEXT_LENGTH > 262144 )) && { [[ "${YARN_OVERRIDE}" == "1" ]] || (( CONTEXT_LENGTH == 1000000 )); }; then
  YARN_FACTOR="$(awk -v n="${CONTEXT_LENGTH}" 'BEGIN{printf "%.0f", n/262144}')"
  NEED_YARN=1
fi
# Same rope_parameters block the vLLM recipe passes under text_config:
ROPE_OVERRIDE='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
if (( NEED_YARN )); then
  CONTEXT_ARGS=(--json-model-override-args "${ROPE_OVERRIDE}" --context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER=(-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
else
  CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER=()
fi
# (YARN_OVERRIDE=1 at 262K: pass the same rope JSON without the context-length bump:
#  CONTEXT_ARGS=(--json-model-override-args "$ROPE_OVERRIDE" --context-length 262144))

# GDN state pool = concurrency x 4 slots (extra_buffer_lazy + overlap scheduler;
# speculative verify window is a SEPARATE engine buffer — do not fold draft tokens in).
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * (4 - MAMBA_SKIP_DECODE_LOCK) ))

command -v docker >/dev/null || { echo "docker not on PATH"; exit 1; }
command -v curl >/dev/null || { echo "curl not on PATH"; exit 1; }
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"

# idempotent container handling
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "already running on ${PORT}"; exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi
# cutover safety: refuse to start if something else is already serving on ${PORT}
if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "ERROR: ${PORT} already serving — stop the old engine container first (delegation stays up only after the new one is ready; if it isn't, the port check above is your tripwire)"; exit 1
fi

echo "launching ${MODEL} via SGLang on ${HOST}:${PORT}"
echo "  ctx=${CONTEXT_LENGTH} (yarn-override=${YARN_OVERRIDE}), mamba pool=${MAMBA_CACHE_SIZE}, MTP ${SPEC_STEPS}/${SPEC_TOPK}/${SPEC_DRAFT}"
echo "  image=${IMAGE}, cpuset=${CPUSET:-<none>}, log=${LOG_FILE}"

PIN_ARGS=(); [[ -n "${CPUSET}" ]] && PIN_ARGS=(--cpuset-cpus "${CPUSET}")

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  "${PIN_ARGS[@]}" \
  "${ALLOW_LONGER[@]}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.triton" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static 0.95 \
  --attention-backend flashinfer \
  --chunked-prefill-size "${CHUNKED_PREFILL}" \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  "${CONTEXT_ARGS[@]}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${SPEC_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --host "${HOST}" \
  --port "${PORT}"

echo "waiting for readiness at ${READY_URL} (first run downloads ~22 GB of weights; 15-30 min typical)..."
for i in $(seq 1 360); do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "CONTAINER EXITED — log tail:"; docker logs "${CONTAINER_NAME}" 2>&1 | tail -60; exit 1
  fi
  if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
    echo "READY after ~$((i*5))s"
    curl -s "${READY_URL}"; echo
    # post-boot verification (expect: context_len, max_running_requests == MAX_CONCURRENT_REQUESTS,
    # mamba pool == MAX_CONCURRENT_REQUESTS x 4):
    docker logs "${CONTAINER_NAME}" 2>&1 | grep -E "context_len|max_running_requests|max_mamba_cache_size" | head -5 || true
    exit 0
  fi
  sleep 5
done
echo "TIMED OUT — log tail:"; docker logs "${CONTAINER_NAME}" 2>&1 | tail -80; exit 1
# stop / rollback:  docker stop "${CONTAINER_NAME}"  (and restart the previous engine if needed)
