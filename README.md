# Qwen3.8-27B on SGLang — DGX Spark / GB10 turnkey recipe

> **Goal:** Hand this repo (or its contents) to any AI assistant and say *"set up
> Qwen3.8-27B on my DGX Spark"* — it will do the whole thing. No prior knowledge needed.

A working recipe for serving **Qwen3.8-27B (NVFP4)** on a single NVIDIA DGX Spark
(GB10, 128 GB unified memory) with **SGLang**. Validated on real hardware.

> Fork/adaptation of `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`. Our changes:
> clock-cap-correct (2200 MHz), delegation-port-preserving, sanitized for public release.

## Why SGLang (and not vLLM) for this model
- The popular `unsloth/Qwen3.8-27B-NVFP4` quant is **vLLM-only** — its `lm_head` is
  quantized to FP8 and SGLang cannot load it (stated on the model card).
- Use **`RadixArk/Qwen3.8-27B-NVFP4`** instead (W4A4 NVFP4 MLP + FP8 attention, SGLang-compatible).
- Measured decode: **~20 tok/s** at context depths 0–32K on a single GB10 (at 2200 MHz cap).

## Hardware assumptions (AI: verify these on the target box)
- 1× DGX Spark (GB10), 128 GB unified memory, CUDA 13 / torch 2.13.
- Image: `lmsysorg/sglang:qwen38-27b` (model-specific GB10 build; already includes the
  fused NVFP4 paths — do NOT substitute a generic SGLang image).
- **GPU clock cap recommended**: `sudo nvidia-smi -lgc 0,2200` (thermal headroom; see gb10-vllm-ops).

## Step-by-step (AI: execute in order)
```bash
# 1. pre-pull image + download weights (one-time, ~22 GB)
docker pull lmsysorg/sglang:qwen38-27b
docker run --rm -e HF_HUB_DISABLE_XET=1 lmsysorg/sglang:qwen38-27b \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RadixArk/Qwen3.8-27B-NVFP4')"

# 2. launch (port 8889, served-model-name overridable for client compatibility)
PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 bash qwen38-aragorn-sglang.sh
```
Wait ~15–30 min on first run (weights download into the container cache). Then:
```bash
curl http://127.0.0.1:8889/v1/models   # expect the model id
```

## Key serving flags (and why)
| Flag | Value | Note |
|---|---|---|
| `--mem-fraction-static` | 0.95 | GB10 unified memory; leave host headroom |
| `--attention-backend` | flashinfer | trtllm_mha is SM100-only |
| `--kv-cache-dtype` | fp8_e4m3 | explicit FP8 KV |
| `--speculative-algorithm` | EAGLE | MTP head, 3 steps / topk 1 / 4 draft |
| `--mamba-*-*` | 4 flags | GDN (hybrid SSM) pool sizing for this arch |
| `--cpuset-cpus` | 5-9,15-19 | GB10's 10 Cortex-X5 big cores |
| `--context-length` | 262144 | native; 1M via YaRN override (see script) |

## Benchmark (verify your build)
Use the canonical decode benchmark (streaming, post-TTFT; the only number that
should be called "tok/s"):
```bash
python3 bench-decode.py --url http://127.0.0.1:8889/v1/chat/completions \
  --model qwen38-27b-unsloth-nvfp4 --trials 5
```
Expect ~18–22 tok/s single-stream at 256-token prompt.

## Sanitization
No LAN/IP/identity in this repo. All topology parameterized via env vars. Git author: **TysAIs**.

## License
Recipe MIT. Model weights subject to their respective licenses (RadixArk / Qwen).
