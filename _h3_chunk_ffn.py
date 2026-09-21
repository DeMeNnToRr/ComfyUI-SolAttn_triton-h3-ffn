"""MiniMax H3 chunked FFN compatibility patch.

This is the backend-independent H3 MLP chunking portion of Sol-Attn.
It does not use or import the NVIDIA Sol-Attn kernel.

It patches:
    diffusion_model.blocks[*].mlp.forward
    diffusion_model.token_refiner.blocks[*].mlp.forward

The H3 MLP operates on [tokens, hidden], so chunking is performed
along dimension 0.
"""

import logging

import torch

from comfy_api.latest import ComfyExtension, io


log = logging.getLogger(__name__)


class _ChunkLog:
    def __init__(self):
        self.active = False

    def hit(self, tokens, chunks):
        if not self.active:
            log.info(
                "[MiniMax H3 FFN] active (%d tokens, %d chunks)",
                tokens,
                chunks,
            )
            self.active = True


def _make_chunked_forward(original_forward, chunks, min_tokens, chunk_log):
    def forward(x):
        # H3 MLP input is [tokens, hidden].
        #
        # Keep the original implementation for:
        #   - unexpected tensor ranks
        #   - shorter sequences
        #   - training/autograd
        if x.ndim != 2 or x.shape[0] < min_tokens or x.requires_grad:
            return original_forward(x)

        chunk_log.hit(x.shape[0], chunks)

        output = torch.empty_like(x)
        offset = 0

        for part in x.chunk(chunks, dim=0):
            end = offset + part.shape[0]
            output[offset:end].copy_(original_forward(part))
            offset = end

        return output

    # Allows later patches to recover the true original forward.
    forward._minimax_h3_ffn_fallback = original_forward

    return forward


class MiniMaxH3ChunkFeedForward(io.ComfyNode):
    """Chunk MiniMax H3 MLP activations to reduce peak VRAM."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkFeedForward",
            display_name="MiniMax H3 Chunk FeedForward",
            is_experimental=True,
            category="model_patches/memory",
            description=(
                "Chunk MiniMax H3 token-local feed-forward activations "
                "to reduce peak VRAM. Independent of the selected "
                "attention backend."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="Enable H3 FFN chunking.",
                ),
                io.Int.Input(
                    "chunks",
                    default=2,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "Number of chunks. More chunks reduce peak MLP "
                        "activation memory but add overhead."
                    ),
                ),
                io.Int.Input(
                    "min_tokens",
                    default=8192,
                    min=256,
                    max=131072,
                    step=256,
                    tooltip=(
                        "Only chunk H3 MLPs when the packed sequence "
                        "has at least this many tokens."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model, enabled=True, chunks=2, min_tokens=8192):
        return io.NodeOutput(
            cls().patch(model, enabled, chunks, min_tokens)[0]
        )

    def patch(self, model, enabled=True, chunks=2, min_tokens=8192):
        chunks = int(chunks)
        min_tokens = int(min_tokens)

        if not enabled or chunks == 1:
            return (model,)

        diffusion_model = model.get_model_object("diffusion_model")

        blocks = getattr(diffusion_model, "blocks", None)
        token_refiner = getattr(diffusion_model, "token_refiner", None)
        refiner_blocks = getattr(token_refiner, "blocks", None)

        if (
            diffusion_model.__class__.__name__ != "MiniMaxH3Model"
            or blocks is None
            or refiner_blocks is None
        ):
            log.warning(
                "[MiniMax H3 FFN] expected a MiniMax H3 model; "
                "returning it unchanged"
            )
            return (model,)

        patched = model.clone()

        paths = [
            f"diffusion_model.blocks.{i}.mlp.forward"
            for i in range(len(blocks))
        ]

        paths.extend(
            f"diffusion_model.token_refiner.blocks.{i}.mlp.forward"
            for i in range(len(refiner_blocks))
        )

        chunk_log = _ChunkLog()

        installed = 0

        for path in paths:
            original_forward = patched.get_model_object(path)

            # If another instance of this patch is already present,
            # recover the true underlying forward rather than stacking
            # chunk wrappers.
            if hasattr(original_forward, "_minimax_h3_ffn_fallback"):
                original_forward = (
                    original_forward._minimax_h3_ffn_fallback
                )

            patched.add_object_patch(
                path,
                _make_chunked_forward(
                    original_forward,
                    chunks,
                    min_tokens,
                    chunk_log,
                ),
            )

            installed += 1

        log.info(
            "[MiniMax H3 FFN] patched %d MLPs "
            "(chunks=%d, min_tokens=%d)",
            installed,
            chunks,
            min_tokens,
        )

        return (patched,)


class H3ChunkFFNExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ChunkFeedForward]


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ChunkFeedForward": MiniMaxH3ChunkFeedForward,
}

__all__ = [
    "MiniMaxH3ChunkFeedForward",
    "H3ChunkFFNExtension",
    "NODE_CLASS_MAPPINGS",
]
