"""Block summaries and routing thresholds shared by both forward kernels."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from ._autotune_log import AUTOTUNE_EXTRAS as _AUTOTUNE_EXTRAS, wrap as _wrap_autotune


BLOCK_SIZE = 64
# kc/vc are zero-padded to this many blocks so group tiles load unmasked;
# covers every GROUP the forward kernels autotune over.
GROUP_PAD = 64
_LOGGED_ROUTE_PLANS = set()


def tau_vector(tau, heads, device):
    """Broadcast tau to the per-head vector the kernels index."""
    return torch.full((heads,), float(tau), device=device, dtype=torch.float32)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=2)
        for warps in (4, 8)
    ],
    key=["T"],
    **_AUTOTUNE_EXTRAS,
)
@triton.jit
def _reduce_kc_kernel(
    k_ptr,
    kc,
    T,
    s_b, s_t, s_h,  # k strides; last dim contiguous
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    values = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    summary = tl.sum(values, axis=0) / block_len
    tl.store(
        kc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=2)
        for warps in (4, 8)
    ],
    key=["T"],
    **_AUTOTUNE_EXTRAS,
)
@triton.jit
def _reduce_vc_kernel(
    v_ptr,
    vc,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    d_tile, block, batch_head = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    offsets = d_tile * TILE_D + tl.arange(0, TILE_D)
    values = tl.load(
        v_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + offsets[None, :],
        mask=(rows < T)[:, None] & (offsets < D)[None, :],
        other=0.0,
    )
    summary = tl.sum(values, axis=0)
    tl.store(
        vc + ((batch * N + block) * H + head) * D + offsets,
        summary,
        mask=offsets < D,
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["T"],
    **_AUTOTUNE_EXTRAS,
)
@triton.jit
def _diag_threshold_kernel(
    q_ptr,
    kc_mean,
    kc_var_diag,
    global_threshold,
    softmax_scale,
    T,
    s_b, s_t, s_h,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE_D: tl.constexpr,
    tau_ptr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    q_start = q_block * BLOCK
    q_len = tl.minimum(BLOCK, T - q_start).to(tl.float32)
    d_offsets = tl.arange(0, TILE_D)
    valid_d = d_offsets < D
    q_rows = q_start + tl.arange(0, BLOCK)
    q_values = tl.load(
        q_ptr + batch * s_b + q_rows[:, None].to(tl.int64) * s_t + head * s_h + d_offsets[None, :],
        mask=(q_rows < T)[:, None] & valid_d[None, :],
        other=0.0,
    )
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean + batch_head * D + d_offsets, mask=valid_d, other=0.0)
    var_kc = tl.load(kc_var_diag + batch_head * D + d_offsets, mask=valid_d, other=0.0)

    TAU = tl.load(tau_ptr + head)
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(
        q_centroid * q_centroid * var_kc, axis=0
    ) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)

    tl.store(
        global_threshold + (batch * N + q_block) * H + head,
        mean + TAU * std,
    )


_wrap_autotune(_reduce_kc_kernel, "kc reduction")
_wrap_autotune(_reduce_vc_kernel, "vc reduction")
_wrap_autotune(_diag_threshold_kernel, "routing threshold")


def _reduce_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, padded, heads, head_dim = k.shape
    tokens = padded if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))
    blocks_padded = ((blocks + GROUP_PAD - 1) // GROUP_PAD) * GROUP_PAD
    kc = torch.zeros(
        (batch, blocks_padded, heads, head_dim),
        device=k.device,
        dtype=torch.bfloat16,
    )
    vc = torch.zeros_like(kc)
    grid = (triton.cdiv(head_dim, tile_d), blocks, batch * heads)
    _reduce_kc_kernel[grid](
        k,
        kc,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    _reduce_vc_kernel[grid](
        v,
        vc,
        tokens,
        v.stride(0), v.stride(1), v.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded, heads, head_dim = q.shape
    tokens = padded if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    tile_d = min(128, triton.next_power_of_2(head_dim))

    # kc is small enough that the moments are cheaper in torch than a kernel.
    kv = kc[:, :blocks].float().permute(0, 2, 1, 3)            # [B,H,blocks,D]
    kc_mean = kv.mean(dim=2).contiguous()
    kc_var_diag = (kv - kc_mean.unsqueeze(2)).pow(2).mean(dim=2).contiguous()

    global_threshold = torch.empty(
        (batch, blocks, heads),
        device=q.device,
        dtype=torch.float32,
    )
    _diag_threshold_kernel[(blocks, batch * heads)](
        q,
        kc_mean,
        kc_var_diag,
        global_threshold,
        scale,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tile_d,
        tau_vector(tau, heads, q.device),
    )
    return global_threshold


def build_route_plan(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tokens: int,
    scale: float,
    coverage: float,
    min_exact_fraction: float,
    sink_blocks: tuple[int, int] = (0, 0),
    sink_q: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an explicit exact-block plan in ``[B, query, H, key]`` order.

    Blocks are accepted in descending estimated attention mass until ``coverage``
    is reached, with a fixed exact-block floor. Local and conditioning blocks are
    mandatory. The forward consumes the mask through its stable grouped exact loop.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if not 0.0 <= min_exact_fraction <= 1.0:
        raise ValueError("min_exact_fraction must be in [0, 1]")

    batch, _, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    full_blocks, tail = divmod(tokens, BLOCK_SIZE)
    centroids = []
    if full_blocks:
        full = q[:, :full_blocks * BLOCK_SIZE].unflatten(
            1, (full_blocks, BLOCK_SIZE)
        )
        centroids.append(full.mean(dim=2, dtype=torch.float32))
    if tail:
        centroids.append(q[:, full_blocks * BLOCK_SIZE:tokens].mean(
            dim=1, dtype=torch.float32
        ).unsqueeze(1))
    q_centroids = torch.cat(centroids, dim=1).to(kc.dtype)

    # [B,H,Q,K]. A key-mean shift, used by the INT8 path, adds the same value
    # to every K score and therefore leaves this ranking and normalized mass intact.
    scores = torch.matmul(
        q_centroids.permute(0, 2, 1, 3),
        kc[:, :blocks].permute(0, 2, 3, 1),
    ).float().mul_(float(scale))
    lengths = torch.full(
        (blocks,), BLOCK_SIZE, device=q.device, dtype=torch.float32
    )
    lengths[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    mass = torch.softmax(scores + lengths.log().view(1, 1, 1, blocks), dim=-1)
    order = scores.argsort(dim=-1, descending=True)
    sorted_mass = mass.gather(-1, order)
    required = (sorted_mass.cumsum(dim=-1) < coverage).sum(dim=-1) + 1
    floor = max(1, int(blocks * min_exact_fraction + 0.999999))
    required.clamp_(min=floor, max=blocks)

    positions = torch.arange(blocks, device=q.device).view(1, 1, 1, blocks)
    selected_sorted = positions < required.unsqueeze(-1)
    selected = torch.zeros_like(selected_sorted).scatter_(-1, order, selected_sorted)

    q_index = torch.arange(blocks, device=q.device).view(1, 1, blocks, 1)
    k_index = torch.arange(blocks, device=q.device).view(1, 1, 1, blocks)
    selected |= (q_index - k_index).abs() <= 1
    sink_start, sink_end = (max(0, int(x)) for x in sink_blocks)
    selected[..., sink_start:min(sink_end, blocks)] = True
    sink_q_start, sink_q_end = (max(0, int(x)) for x in sink_q)
    selected[:, :, sink_q_start:min(sink_q_end, blocks), :] = True

    counts = selected.sum(dim=-1, dtype=torch.int32)
    key = (tokens, heads, coverage, min_exact_fraction, sink_blocks, sink_q)
    if key not in _LOGGED_ROUTE_PLANS:
        _LOGGED_ROUTE_PLANS.add(key)
        active = counts[:, :, min(sink_q_end, blocks):]
        observed = active if active.numel() else counts
        logging.info(
            "[sol_attn] accepted-block plan: %d blocks, exact min/mean/max "
            "%.1f%%/%.1f%%/%.1f%%",
            blocks,
            100.0 * observed.min().item() / blocks,
            100.0 * observed.float().mean().item() / blocks,
            100.0 * observed.max().item() / blocks,
        )
    return (
        selected.permute(0, 2, 1, 3).contiguous().to(torch.uint8),
        counts.permute(0, 2, 1).contiguous(),
    )


def prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (kc, vc, threshold). ``tokens`` is the true sequence length."""
    kc, vc = _reduce_kv(k, v, tokens)
    threshold = _compute_diag_threshold(q, kc, tau=tau, scale=scale, tokens=tokens)
    return kc, vc, threshold


__all__ = ["build_route_plan", "prepare"]
