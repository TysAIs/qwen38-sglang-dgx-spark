# Qwen3.8-27B on SGLang — DGX Spark / GB10 recipe (DSpark + MTP)

> **Goal:** Hand this repo to any AI assistant and say *"set up Qwen3.8-27B on my DGX
> Spark"* — it will do it. Two serving modes with measured tradeoffs, no guessing.

Serves **Qwen3.8-27B (NVFP4)** on a single NVIDIA DGX Spark (GB10, 128 GB unified
memory) with **SGLang**, in Docker. Validated on real hardware.

> Adapted from `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark` (08-18). Our changes:
> clock-cap-correct, delegation-port-preserving, sanitized for public release; we
> verified the numbers on our own box.

## Which mode to use (measured on-device)
| Probe | **DSpark** (`./start-dspark.sh`) | MTP (`./start.sh`, EAGLE 3/1/4) |
|---|---|---|
| Code (LRUCache + test) | **52.1 tok/s** | 34.5 tok/s |
| Default chat (thinking on) | ~23 | ~21 |
| Long essay | 18.7 | 24.1 |
| Context cap | 262K (YaRN-incompatible) | 262K..1M (YaRN) |

**Use DSpark for agents / code / tool calls / normal chat (~1.5× on code). Use MTP only
for long essays or >262K context.** This repo leads with DSpark because agents do code.

## Quick start (AI: execute in order)
```bash
# 0. (optional) pre-download weights + warm kernel cache — speeds first boot a lot
docker pull lmsysorg/sglang:qwen38-27b

# 1. configure (native 262K, 10 concurrent). Copy then edit only if you need changes.
cp .env.sample .env

# 2. launch DSpark (recommended for code/agents). Port + served-name overridable:
PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 ./start-dspark.sh
#    (or MTP:  PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 ./start.sh)
```
First boot downloads the DSpark draft (~2.7 GB) and captures CUDA graphs: wait 15–30 min.
When ready, the script reports the OpenAI base URL. Then:
```bash
curl http://127.0.0.1:8889/v1/models          # expect the model id
```

## Benchmark / verify your build
```bash
# canonical decode (streaming, post-TTFT) — the only valid "tok/s":
python3 bench-decode.py --url http://127.0.0.1:8889/v1/chat/completions \
  --model qwen38-27b-unsloth-nvfp4 --trials 5
# net-decode code A/B (Mia's ndec.py, two-call delta):
python3 bench/ndec.py
```
Expect **code ~50-52 tok/s on DSpark** (per our validation, matching Mia's 51.5).
Treat code deltas <15% as noise (run-to-run variance ±7%).

## Traps (documented so you don't rediscover them)
1. **Empty chat responses.** SGLang defaults thinking ON. A short `max_tokens` gets
   consumed by invisible reasoning → `content: null`, `finish_reason: "length"`. Disable
   per request with `chat_template_kwargs: {"enable_thinking": false}` (recommended for
   agents), or raise `max_tokens`. Not a bug.
2. **YaRN ≠ DSpark.** The YaRN rope override leaks into the DSpark draft config and
   crashes at boot. Keep `YARN=0` / `CONTEXT_LENGTH=262144` for DSpark.
3. **Wrong "tok/s".** Dividing all tokens by wall-clock including TTFT understates
   decode by ~40%. Use `bench-decode.py` (streaming, post-first-token) or `ndec.py`.
4. **Unsloth NVFP4 is vLLM-only.** Its `lm_head` is FP8-quant, SGLang can't load it.
   Use `RadixArk/Qwen3.8-27B-NVFP4` for SGLang.
5. **Do not substitute a generic SGLang image.** Use `lmsysorg/sglang:qwen38-27b`
   (model-specific GB10 build with fused NVFP4 paths).
6. **Clock cap** (2200 MHz) does not limit decode (bandwidth-bound); it buys thermal
   headroom. `sudo nvidia-smi -lgc 0,2200`.

## Files
- `start.sh` — MTP/EAGLE serve (prose-optimal, up to 1M ctx via YaRN).
- `start-dspark.sh` — DSpark serve (code/agent-optimal, 262K). Wrapper that sets
  `EXTRA_ARGS` then delegates to start.sh.
- `stop.sh` — idempotent stop.
- `bench/` — ndec.py (net-decode A/B), bench.sh (wall-time incl prefill).
- `bench-decode.py` — canonical streaming decode benchmark.

## Sanitization
No LAN/IP/identity in this repo. All topology parameterized (PORT / SERVED_MODEL_NAME /
node addresses are env vars). Git author: **TysAIs**.

## License
Recipe MIT. Model weights subject to their respective licenses (RadixArk / Qwen).
Upstream engine © MiaAI-Lab (see their repo for engine license).
