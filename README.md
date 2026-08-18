# Qwen3.8-27B on SGLang — DGX Spark / GB10 recipe

A working recipe for serving **Qwen3.8-27B (NVFP4)** on a single NVIDIA DGX Spark
(GB10, 128 GB unified memory) with **SGLang**. Validated on real hardware.

> This is a fork/adaptation of `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`. Our changes:
> clock-cap-aware, delegation-port-preserving, sanitized for public release.

## Why SGLang (and not vLLM) for this model
- The popular `unsloth/Qwen3.8-27B-NVFP4` quant is **vLLM-only** — its `lm_head` is
  quantized to FP8 and SGLang cannot load it (stated on the model card).
- Use **`RadixArk/Qwen3.8-27B-NVFP4`** instead (W4A4 NVFP4 MLP + FP8 attention, SGLang-compatible).
- Measured decode: **~20 tok/s** at context depths 0–32K on a single GB10 (at 2190 MHz cap).

## Hardware assumptions
- 1× DGX Spark (GB10), 128 GB unified memory, CUDA 13 / torch 2.13.
- Image: `lmsysorg/sglang:qwen38-27b` (model-specific GB10 build; already includes the
  fused NVFP4 paths — do NOT substitute a generic SGLang image).
- **GPU clock cap recommended**: `sudo nvidia-smi -lgc 0,2190` (thermal headroom; see gb10-vllm-ops).

## Quick start
```bash
# pre-pull + pre-download weights (one-time, ~22 GB)
docker pull lmsysorg/sglang:qwen38-27b
docker run --rm -e HF_HUB_DISABLE_XET=1 lmsysorg/sglang:qwen38-27b \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RadixArk/Qwen3.8-27B-NVFP4')"

# launch (port 8889, served-model-name overridable for client compatibility)
PORT=8889 SERVED_MODEL_NAME=qwen38-27b-unsloth-nvfp4 bash qwen38-aragorn-sglang.sh
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

## Sanitization
No LAN/IP/identity in this repo. Topology is parameterized via env vars with `[VERIFY]`
markers. Git author must be **TysAIs**.

## License
Recipe MIT. Model weights are subject to their respective licenses (RadixArk / Qwen).
