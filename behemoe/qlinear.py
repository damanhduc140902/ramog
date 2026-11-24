"""Weight-only quantized nn.Linear for the dense part of the model (attention projections, lm_head).

Decode (few tokens): one grouped dequant-GEMV launch over a single-slot pool. Prefill: the grouped dequant-GEMM
of the MoE prefill with a single "expert" (weights dequantized in registers, never materialized). Same packed
format as the experts (group-wise asymmetric, fp16 scale/min).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import moe as _moe
from .kernels import block_m_for, grouped_gemv, moe_gemm
from .quant import Packed, dequantize, quantize

DECODE_MAX_TOKENS = 8
_BLOCKS: dict = {}  # (device, tokens, block_m) -> (order, blocks): the one-expert block table of moe_gemm


def _one_expert_blocks(n: int, block_m: int, device):
    key = (str(device), n, block_m)
    if key not in _BLOCKS:
        b = torch.arange(-(-n // block_m), device=device)
        blocks = torch.stack([torch.zeros_like(b), b * block_m, (n - b * block_m).clamp(max=block_m)], 1)
        _BLOCKS[key] = (torch.arange(n, device=device, dtype=torch.int32), blocks.to(torch.int32))
    return _BLOCKS[key]


class QuantLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int = 4, group: int = 64,
                 device=None, chunk_rows: int = 16384):
        """weight [out, in] may live on the CPU (e.g. read from a file) with `device` the GPU: it is moved and
        quantized `chunk_rows` rows at a time, so the float32 scratch stays bounded (the 151936-row lm_head of
        Qwen3 needs > 4 GiB in one piece). Quantization is per group within a row: chunking changes nothing."""
        super().__init__()
        out_f, in_f = weight.shape
        dev = weight.device if device is None else torch.device(device)
        parts = [quantize(weight[r0:r0 + chunk_rows].to(dev).float(), bits, group)
                 for r0 in range(0, out_f, chunk_rows)]
        cat = (lambda ts: ts[0]) if len(parts) == 1 else torch.cat
        self.register_buffer("qweight", cat([p.qweight for p in parts]).unsqueeze(0).contiguous())  # [1, out, in*b/8]
        self.register_buffer("scale", cat([p.scale for p in parts]).unsqueeze(0).contiguous())
        self.register_buffer("minimum", cat([p.minimum for p in parts]).unsqueeze(0).contiguous())
        del parts
        self.register_buffer("bias", bias.to(dev).clone() if bias is not None else None)
        self.bits, self.group = bits, group
        self.in_features, self.out_features = in_f, out_f
        self.register_buffer("_slot", torch.zeros(DECODE_MAX_TOKENS, dtype=torch.int32, device=dev), persistent=False)
        self.register_buffer("_tok", torch.arange(DECODE_MAX_TOKENS, dtype=torch.int32, device=dev), persistent=False)

    @property
    def weight(self) -> torch.Tensor:  # dequantized view for code that reads .weight (e.g. tied heads)
        p = Packed(self.qweight[0], self.scale[0], self.minimum[0], self.bits, self.group,
                   (self.out_features, self.in_features))
        return dequantize(p, torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        t = x2.shape[0]
        if t <= DECODE_MAX_TOKENS:
            y = grouped_gemv(x2.contiguous(), self._tok[:t], self._slot[:t], self.qweight, self.scale, self.minimum,
                             self.bits, self.group)
            y = y.to(x.dtype)
        elif _moe.PREFILL_IMPL == "dequant":  # the previous prefill, kept as the reference
            y = torch.empty(t, self.out_features, device=x.device, dtype=x.dtype)
            step = 8192  # dequantize a block of rows at a time to bound the temporary memory
            for r0 in range(0, self.out_features, step):
                r1 = min(self.out_features, r0 + step)
                p = Packed(self.qweight[0, r0:r1], self.scale[0, r0:r1], self.minimum[0, r0:r1], self.bits,
                           self.group, (r1 - r0, self.in_features))
                y[:, r0:r1] = x2 @ dequantize(p, x.dtype).t()
        else:
            y = torch.empty(t, self.out_features, device=x.device, dtype=x.dtype)
            xc = x2.contiguous()
            step = max(64, (64 << 20) // self.out_features)  # tokens per pass: <= 256 MiB of float32 output
            for r0 in range(0, t, step):
                n = min(t, r0 + step) - r0
                bm = block_m_for(n, 1)
                order, blocks = _one_expert_blocks(n, bm, x.device)
                out = torch.empty(n, self.out_features, device=x.device, dtype=torch.float32)
                moe_gemm(xc[r0:r0 + n], order, blocks, None, self.qweight, self.scale, self.minimum, self.bits,
                         self.group, num_experts=1, out=out, block_m=bm, mode=0)
                y[r0:r0 + n] = out
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.scale, self.minimum))


def quantize_linears(module: nn.Module, names: tuple[str, ...], bits: int = 4, group: int = 64) -> int:
    """Replace child nn.Linear modules whose attribute name is in `names` (recursively). Returns bytes saved."""
    saved = 0
    for parent in list(module.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in names and child.in_features % group == 0:
                q = QuantLinear(child.weight.data, child.bias.data if child.bias is not None else None, bits, group)
                saved += child.weight.numel() * child.weight.element_size() - q.nbytes()
                setattr(parent, child_name, q)
    return saved
