"""HostEmbedding: the token embedding table kept in pinned host memory, gathered on the GPU (design-core B3; WP4).

The input embedding is the largest dense tensor of most targets (Qwen3: 151936 x 2048 bf16 = 0.58 GiB; Gemma-4:
262144 x 2816 = 1.38 GiB) and each decode step reads one row of it. HostEmbedding keeps the bf16 table in exact-size
page-locked host memory (moe_tiering.hostmem) and gathers the requested rows into a GPU tensor with the same zero-copy
kernel that stages cold expert blobs (moe_tiering.kernels.copy_blobs: dst[i] = src[ids[i]], int32 words), on the current
stream, without a host synchronization, so it also runs inside a captured decode graph.

Output = table[ids] (a bit-exact copy of nn.Embedding's lookup), times `embed_scale.to(table dtype)` when the
replaced module had one (Gemma4TextScaledWordEmbedding.forward: `super().forward(ids) * embed_scale.to(weight.dtype)`,
modeling_gemma4 5.19 :1446). The loader builds it after transformers' initialize_weights() has set embed_scale, and
quantizes a tied lm_head from the table before the table is pinned (moe_tiering.loader).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import hostmem


class HostEmbedding(nn.Module):
    def __init__(self, table: torch.Tensor, embed_scale: torch.Tensor | None = None, device=None,
                 padding_idx: int | None = None):
        """table: [V, H] CPU tensor (bf16, as loaded); embed_scale: the replaced module's 0-dim buffer or None;
        device: the GPU the outputs go to."""
        super().__init__()
        if table.dim() != 2 or table.device.type != "cpu":
            raise ValueError(f"HostEmbedding needs a 2-D CPU table, got {tuple(table.shape)} on {table.device}")
        row = table.shape[1] * table.element_size()
        if row % 4:
            raise ValueError(f"rows of {row} bytes: the gather copies whole int32 words")
        pinned = table.contiguous()
        if not pinned.is_pinned():
            pinned = hostmem.pin_copy(pinned)
        object.__setattr__(self, "table", pinned)  # unregistered: model.to() must never move it to the GPU
        self.num_embeddings, self.embedding_dim = table.shape
        self.padding_idx = padding_idx
        self.device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        if embed_scale is not None:
            self.register_buffer("embed_scale", embed_scale.detach().clone().to(self.device), persistent=False)
        else:
            self.embed_scale = None
        self._want = torch.ones(1, dtype=torch.int32, device=self.device)

    @property
    def weight(self) -> torch.Tensor:
        """The pinned host table (for code that reads the embedding's dtype or shape)."""
        return self.table

    def _ones(self, n: int) -> torch.Tensor:
        if self._want.numel() < n:
            self._want = torch.ones(max(n, 2 * self._want.numel()), dtype=torch.int32, device=self.device)
        return self._want[:n]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        from .kernels import copy_blobs  # Triton; imported on first use

        ids = input_ids.reshape(-1)
        if ids.device != self.device:
            ids = ids.to(self.device)
        ids = ids.to(torch.int32)
        n = ids.numel()
        out = torch.empty(n, self.embedding_dim, dtype=self.table.dtype, device=self.device)
        if n:
            copy_blobs(self.table.view(torch.uint8).view(self.num_embeddings, -1),
                       out.view(torch.uint8).view(n, -1), ids, self._ones(n))
        out = out.view(*input_ids.shape, self.embedding_dim)
        if self.embed_scale is not None:
            out = out * self.embed_scale.to(out.dtype)
        return out

    def nbytes(self) -> int:
        return self.table.numel() * self.table.element_size()

    def extra_repr(self) -> str:
        return (f"{self.num_embeddings}, {self.embedding_dim}, pinned host {self.table.dtype}, "
                f"scaled={self.embed_scale is not None}")
