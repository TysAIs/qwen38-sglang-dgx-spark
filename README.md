# Qwen3.8-27B on SGLang — DGX Spark / GB10 recipe (DFlash2 + DSpark + MTP)

> **Goal:** Hand this repo to any AI assistant and say *"set up Qwen3.8-27B on my DGX
> Spark"* — it will do it. Three serving modes with measured tradeoffs, no guessing.

Serves **Qwen3.8-27B (NVFP4)** on a single NVIDIA DGX Spark (GB10, 128 GB unified
memory) with **SGLang**, in Docker. Validated on real hardware.

> Adapted from `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark` (08-19). Our changes:
> DFlash2 A/B-validated, env-override regression fixed in the vendored `start.sh`
> (see Trap #1), delegation-port-preserving, sanitized for public release; numbers
> verified on our own box.

## Which mode to use (measured on-device, same box / same tool / same day)

| Probe | **DFlash2** (`./start-dflash.sh`) | DSpark (`./start-dspark.sh`) | MTP (`./start.sh`, EAGLE 3/1/4) |
|---|---|---|---|
| Code (LRUCache + test) | **65.1 tok/s** | 52.6 | 34.5 |
| Long essay | **27.2** | 18.7 | 24.1 |
| Everyday chat | 23.4 | ~23 | ~21 |
| Context cap | 262K (no YaRN) | 262K (no YaRN) | 262K..1M (YaRN) |

**Use DFlash2 — it wins on code AND long-form writing, ties chat.** DSpark is the
rollback (same repo, one command). MTP only for >262K context. Code deltas <15% are
noise (run-to-run ±7%).

## Quick start (AI: execute in order)

```bash
# 1. configure (native 262K, 16 concurrent). Copy, edit only if you need changes.
cp .env.sample .env

# 2. launch DFlash2 (recommended). Port + served-name overridable:
PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 ./start-dflash.sh
#    (or DSpark:  PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 ./start-dspark.sh
#     or MTP:     PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 ./start.sh)
```

**DFlash2 has no released image yet** (merged upstream 2026-08-19, after every
published SGLang tag) — `start-dflash.sh` builds `lmsysorg/sglang:qwen38-27b-dflash2`
automatically on first run (needs git + network once, ~6 min) via `patch/`, then
pulls the ~2.7 GB DFlash2 draft. First boot also captures CUDA graphs: expect
5–15 min to ready. When ready, the script prints the OpenAI base URL:

```bash
curl http://127.0.0.1:8889/v1/models          # expect the model id
```

## Benchmark / verify your build

```bash
# canonical decode (streaming, post-TTFT) — the only valid "tok/s":
python3 bench-decode.py --url http://127.0.0.1:8889/v1/chat/completions \
  --model qwen38-27b-unsloth-nvfp4 --trials 5
# net-decode code/essay A/B (ndec.py, two-call delta):
python3 bench/ndec.py
```

Expect **code ~65 tok/s on DFlash2** (per our validation; DSpark ~52). Treat code
deltas <15% as noise.

## Traps (documented so you don't rediscover them)

1. **Upstream env-override regression — FIXED in this repo.** The 08-19 upstream
   `start.sh` hardcodes `SERVED_MODEL_NAME`/`CONTAINER_NAME`/`PORT` (the old
   `${VAR:-default}` pattern was lost) and its `.env` loader whitelists only
   QUANT/YARN/CONTEXT_LENGTH/MAX_CONCURRENT_REQUESTS. Symptom: your `PORT=8889`
   override is silently ignored and it comes up as `qwen3.8-27b-sglang` on :8888.
   **The vendored `start.sh` here already carries the 3-line fix** — do not
   overwrite it with a fresh upstream copy without re-applying:
   `SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"` (same for
   `CONTAINER_NAME`, `PORT`).
2. **Never run two engines at once.** DSpark and DFlash2 each allocate ~90% of the
   GB10's 128 GB. Starting the second while the first is up = CUDA OOM + unified-memory
   thrash that wedges the node (SSH banner timeout) for minutes until it self-recovers.
   Swap order: stop the old engine, confirm the port is free, then start the new one.
3. **Empty chat responses.** SGLang defaults thinking ON. A short `max_tokens` gets
   consumed by invisible reasoning → `content: null`, `finish_reason: "length"`. Disable
   per request with `chat_template_kwargs: {"enable_thinking": false}` (recommended for
   agents), or raise `max_tokens`. Not a bug.
4. **YaRN ≠ DSpark/DFlash2.** The YaRN rope override leaks into the draft config and
   crashes at boot. Keep `YARN=0` / `CONTEXT_LENGTH=262144` for both spec-decode modes.
5. **Wrong "tok/s".** Dividing all tokens by wall-clock including TTFT understates
   decode by ~40%. Use `bench-decode.py` (streaming, post-first-token) or `ndec.py`.
6. **Unsloth NVFP4 is vLLM-only.** Its `lm_head` is FP8-quant, SGLang can't load it.
   Use `RadixArk/Qwen3.8-27B-NVFP4` for SGLang. DFlash2 additionally needs the
   quantized-head selector patch (in `patch/dflash2_nvfp4_head.patch`) — a naive
   dequant-once approach hard-reboots the box at graph capture.
7. **Do not substitute a generic SGLang image.** Use `lmsysorg/sglang:qwen38-27b`
   (model-specific GB10 build with fused NVFP4 paths) as the base.
8. **Clock cap** (2200 MHz) does not limit decode (bandwidth-bound); it buys thermal
   headroom. `sudo nvidia-smi -lgc 0,2200`.

## Files

- `start.sh` — MTP/EAGLE serve (prose-optimal, up to 1M ctx via YaRN). **Vendored
  with the env-override fix** (Trap #1).
- `start-dspark.sh` — DSpark serve (rollback mode, 262K).
- `start-dflash.sh` — **DFlash2 serve (default)**: auto-builds the image on first
  run, pulls the block-diffusion draft, mem 0.90.
- `patch/` — DFlash2 image build (build-dflash2-image.sh + dflash2_nvfp4_head.patch
  + sha256-verified overlay).
- `stop.sh` — idempotent stop.
- `bench/` — ndec.py (net-decode A/B), bench.sh (wall-time incl prefill).
- `bench-decode.py` — canonical streaming decode benchmark.

## Sanitization

No LAN/IP/identity in this repo. All topology parameterized (PORT / SERVED_MODEL_NAME /
node addresses are env vars). Git author: **TysAIs**.

## License

Recipe MIT. Model weights subject to their respective licenses (RadixArk / Qwen).
Upstream engine © MiaAI-Lab (see their repo for engine license).
