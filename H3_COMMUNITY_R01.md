# MiniMax H3 Sol-Attn ROCm r01

This release branch adds `MiniMaxH3FastPatch`, a conservative H3-only long-sequence Sol-Attn route.

- Upstream: <https://github.com/kijai/ComfyUI-SolAttn_triton>
- Pinned base: `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b`
- Validated stack: Linux, ComfyUI 0.32.0, comfy-kitchen 0.2.31, RX 7900 XTX/gfx1100
- Policy: starts at 12,288 tokens, 90% estimated-mass route coverage, 50% exact floor, exact conditioning rows, INT8 QK/PV
- Release-kit SHA-256: `855d84f5532484fa2c209c25f593d428cc310700e5d00f88aa0b1d5e9831d78c`

The node is approximate and explicitly opt-in. Place it after ComfyUI's `ModelAttentionBackend` configured as `comfy kitchen attention`. Other GPUs and ComfyUI versions are unvalidated. The upstream repository does not currently declare a software license; this fork does not add or imply one.
