"""Group-wise asymmetric low-bit weights: packing, dequantization (reference) and memory accounting.

Layout for a weight W[out, in] (nn.Linear convention), group size G along `in`:
    qweight  uint8 [out, in * bits / 8]   consecutive values of a row packed little-end first
    scale    fp16  [out, in / G]
    minimum  fp16  [out, in / G]          w ~= q * scale + minimum
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Packed:
    qweight: torch.Tensor
    scale: torch.Tensor
    minimum: torch.Tensor
    bits: int
    group: int
    shape: tuple[int, int]  # (out, in)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.scale, self.minimum))

    def to(self, device, non_blocking: bool = False) -> Packed:
        return Packed(self.qweight.to(device, non_blocking=non_blocking), self.scale.to(device, non_blocking=non_blocking),
                      self.minimum.to(device, non_blocking=non_blocking), self.bits, self.group, self.shape)

    def pin(self) -> Packed:
        return Packed(self.qweight.pin_memory(), self.scale.pin_memory(), self.minimum.pin_memory(),
                      self.bits, self.group, self.shape)


def bits_per_weight(bits: int, group: int) -> float:
    return bits + 32 / group


@torch.no_grad()
def quantize(w: torch.Tensor, bits: int, group: int = 64, clip_ratios=(1.0, 0.9, 0.8, 0.7, 0.6)) -> Packed:
    """RTN with a per-group clipping search that minimizes squared error. w: [out, in]."""
    n_out, n_in = w.shape
    assert n_in % group == 0 and 8 % bits == 0, (w.shape, bits, group)
    wg = w.float().reshape(n_out, n_in // group, group)
    wmin, wmax = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    qmax = 2**bits - 1
    best_q = torch.zeros_like(wg, dtype=torch.uint8)
    best_s = torch.zeros_like(wmin)
    best_m = torch.zeros_like(wmin)
    best_err = torch.full_like(wmin, float("inf"))
    for r in clip_ratios:
        lo, hi = wmin * r, wmax * r
        # fp16-round the parameters first so the search scores what the kernel will actually use
        scale = ((hi - lo) / qmax).clamp(min=1e-8).half().float()
        lo = lo.half().float()
        q = ((wg - lo) / scale).round().clamp(0, qmax)
        err = (q * scale + lo - wg).pow(2).sum(-1, keepdim=True)
        better = err < best_err
        best_q = torch.where(better, q.to(torch.uint8), best_q)
        best_s = torch.where(better, scale, best_s)
        best_m = torch.where(better, lo, best_m)
        best_err = torch.where(better, err, best_err)
    per_byte = 8 // bits
    q = best_q.reshape(n_out, n_in // per_byte, per_byte).to(torch.int32)
    packed = torch.zeros(n_out, n_in // per_byte, dtype=torch.int32, device=w.device)
    for i in range(per_byte):
        packed |= q[..., i] << (bits * i)
    return Packed(packed.to(torch.uint8), best_s.squeeze(-1).half(), best_m.squeeze(-1).half(), bits, group, (n_out, n_in))


@torch.no_grad()
def dequantize(p: Packed, dtype=torch.float32) -> torch.Tensor:
    n_out, n_in = p.shape
    per_byte = 8 // p.bits
    b = p.qweight.to(torch.int32)
    vals = torch.stack([(b >> (p.bits * i)) & (2**p.bits - 1) for i in range(per_byte)], dim=-1)
    q = vals.reshape(n_out, n_in // p.group, p.group).float()
    w = q * p.scale.float()[..., None] + p.minimum.float()[..., None]
    return w.reshape(n_out, n_in).to(dtype)
