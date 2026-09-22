<h1 align="center">ComfyUI-SolAttn</h1>


# MiniMax H3 Sol-Attn for AMD ROCm

AMD ROCm support for MiniMax H3 with the `MiniMaxH3ChunkFeedForward` node for H3 FFN chunking.

## MiniMax H3 FFN Chunking

This fork adds the H3-specific:

```text
MiniMaxH3ChunkFeedForward
```

node, which was missing from the AMD ROCm version of Sol-Attn.

[`ComfyUI-H3-Multishot`](https://github.com/jlucasmcrell/ComfyUI-H3-Multishot) uses this node for its `chunk_ffn` VRAM/SPEED option. FFN chunking splits large H3 feed-forward operations into smaller token chunks, reducing peak activation memory and making large H3 generations more practical on GPUs with limited VRAM.

Without this node, H3 Multishot reports:

```text
[H3Memory] auto_chunk_ffn: ComfyUI-sol-attn not installed - cannot chunk.
```

With this fork, H3 Multishot can activate the FFN chunking path:

```text
[MiniMax H3 FFN] patched 52 MLPs (chunks=2, min_tokens=8192)
[MiniMax H3 FFN] active (... tokens, 2 chunks)
```

The added `MiniMaxH3ChunkFeedForward` implementation is independent of the attention kernel; this change specifically provides the H3 FFN chunking functionality expected by H3 Multishot on AMD ROCm.



Credit to the original author: https://github.com/DrBearJew/ComfyUI-SolAttn_triton

<h4 align="center">
  Experimental Triton implementation of Sol-Attn for ComfyUI
</h4>


<p align="center">
  <a href="https://arxiv.org/abs/2607.24027"><img src="https://img.shields.io/badge/📄_Paper-arXiv-b31b1b?style=flat-square" alt="Paper"/></a>
  <a href="https://github.com/NVlabs/Sana/tree/sol-engine/techniques/sparse_backends/sol_attn"><img src="https://img.shields.io/badge/💻_Code-Sol--Attn-76b900?style=flat-square" alt="Code"/></a>
  <a href="https://nvlabs.github.io/Sana/Sol-Attn/"><img src="https://img.shields.io/badge/🌐_Project-Page-blue?style=flat-square" alt="Project Page"/></a>
</p>

---

## Overview

[Sol-Attn](https://arxiv.org/abs/2607.24027) is a training-free sparse attention
method for accelerating image and video generation. This community extension
integrates a Triton implementation of Sol-Attn into ComfyUI.

> [!NOTE]
> This project is a work in progress. It has been tested on RTX 4090/5090 and
> RX 7900 XTX (gfx1100) GPUs with MiniMax H3.

## Usage notes

Triton kernels are compiled on first use, so the first run will be slower.

Use `start_percent`, `end_percent`, and `tau` to balance generation quality and
speed.

### MiniMax H3 fast preset

`MiniMax H3 Long-Sequence Attention (Experimental)` is a conservative one-input
preset for the native H3 model. It keeps conditioning rows and blocks `0-2,-1`
dense and activates only from 12K tokens. Shorter sequences stay dense because
the measured 8,501-token H3 run gained only 1.6% in denoise time.

Long calls use an explicit accepted-block mask: ranked block summaries must cover
90% of estimated attention mass and at least 50% of all KV blocks stay exact.
The stable grouped kernel then runs exact QK/PV in INT8; the pooled approximation
remains BF16. This corrected the prior `tau=1.2` long-sequence collapse without
changing H3 diffusion quantization, the Qwen encoder, or either VAE. A variable-
length compact-gather prototype caused an illegal-address reset under H3 async
offload and was removed; the accepted mask uses the proven grouped exact loop.

On an RX 7900 XTX (gfx1100, ROCm 7.15, PyTorch 2.14 nightly), a matched cold
1024×576×90-frame, 20-step run retained 50.0–86.0% exact blocks (54.7% mean).
The candidate completed in 447.2 seconds versus 472.9 seconds dense; active
sampling steps fell from roughly 17.5–21.3 to 14.3–15.4 seconds. It produced
90 unique coherent frames, while the memory-bound dense control developed late
frame corruption. Ambient audio matched closely: RMS -34.17 versus -33.86 dB
and peak -14.96 versus -14.36 dB.

Composed with the following `MiniMaxH3BlockCacheT8` at its released H3-style
threshold `0.08`, the same cold gate completed in 327.6 seconds: 1.44× wall-clock
and 1.64× denoising speedup versus dense. It retained 90 coherent unique frames;
audio stayed within 0.77 dB RMS and 0.87 dB peak. Both nodes require their explicit
`enabled` opt-in and remain disabled in prepared workflows by default.

Isolated H3-shape attention measurements were:

| tokens / shape | dense BF16 | Sol BF16 | Sol INT8 |
|---|---:|---:|---:|
| 5,504 × 56 × 128 | 13.20 ms | 5.82 ms | 7.58 ms |
| 21,760 × 56 × 128 | 772.21 ms | 115.58 ms | 71.71 ms |

These microbenchmarks are not the end-to-end result above. Keep the node
experimental until additional prompts, frame counts, and hardware pass the same
visual/audio gate.

## Examples

### Test output

https://github.com/user-attachments/assets/8d9ed820-0417-4d68-9d1c-5199534bed3b

### SageAttention vs. Sol-Attn

<table>
<tr>
<td align="center"><b>SageAttention</b></td>
<td align="center"><b>Sol-Attn</b></td>
</tr>
<tr>
<td width="50%">
<video src="https://github.com/user-attachments/assets/27f201ea-6bfc-4f43-826c-51809eed9d15" controls muted loop></video>
</td>
<td width="50%">
<video src="https://github.com/user-attachments/assets/73f63d14-2166-4f62-b098-e817ec1d7704" controls muted loop></video>
</td>
</tr>
</table>

<img width="482" height="500" alt="Sol-Attn example result" src="https://github.com/user-attachments/assets/27ae9886-aa3e-4470-a507-3a7c52b24be5" />

## Citation

If you find Sol-Attn useful in your work, please cite the paper:

```bibtex
@article{li2026solattn,
  title={Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification},
  author={Li, Haopeng and Li, Yitong and Chen, Junsong and Ye, Tian and Liu, Haozhe and Yu, Jincheng and Wang, Duomin and Zhang, Ruihua and Xie, Zeke and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2607.24027},
  year={2026}
}
```
