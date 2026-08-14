# MiniMax H3 Sol-Attn for AMD ROCm

DBJ's conservative, opt-in long-sequence Sol-Attn patch for native ComfyUI MiniMax H3 on AMD ROCm.

[![ComfyUI Registry](https://img.shields.io/badge/ComfyUI_Registry-install-blue)](https://registry.comfy.org/nodes/minimax-h3-sol-attn-rocm)

```bash
comfy node install minimax-h3-sol-attn-rocm
```

## What this release does

`MiniMaxH3FastPatch` follows ComfyUI's `ModelAttentionBackend`, leaving Comfy Kitchen attention as the dense backend and fallback. It activates only on eligible MiniMax H3 self-attention calls of at least 12,288 tokens.

Validated policy:

- 90% estimated attention-mass route coverage
- 50% minimum exact-block floor
- exact conditioning, reference, and target-audio rows
- INT8 QK/PV attention arithmetic
- dense fallback on unsupported or malformed calls

The patch is approximate and disabled unless the user explicitly enables it.

Validated environment: Linux, ComfyUI 0.32.0, comfy-kitchen 0.2.31, and RX 7900 XTX/gfx1100. Other GPUs and ComfyUI versions are unvalidated.

## Placement and LoRA

```text
MiniMaxH3INT8FastLoader
  → LoraLoaderBypassModelOnly (optional)
  → ModelAttentionBackend (comfy kitchen attention)
  → MiniMaxH3FastPatch
  → MiniMaxH3BlockCacheT8 (optional)
  → MiniMaxH3SigmaShift
  → scheduler / guider / sampler
```

A fixed-strength bypass LoRA is compatible because Sol-Attn sees the resulting current-call Q/K/V tensors. Match the LoRA variant and its 4-step or 8-step schedule to the FL2VA or Ref2VA base.

## Companion packages

- [MiniMax H3 INT8 Fast for AMD ROCm](https://registry.comfy.org/nodes/minimax-h3-int8-fast-rocm)
- [MiniMax H3 Block Cache](https://registry.comfy.org/nodes/minimax-h3-block-cache)
- [MiniMax H3 Turbo LoRAs](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/main)

## Provenance and license boundary

This release is derived from [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton), pinned at `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b`. See [`H3_COMMUNITY_R01.md`](H3_COMMUNITY_R01.md) for frozen release details.

The upstream repository does not currently declare a software license. This fork does not add or imply one.
