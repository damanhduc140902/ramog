"""k4row weights: low-bit codes, one byte of 4-bit scale codes per group, fp16 super-scales per row.

Format of a weight W [n = out, k = in] with b bits per weight and group g along `in` (the cold tier uses b = 2,
g = 16: 2 + 8/16 + 32/k bits per weight, 2.524 bpw averaged over a Qwen3 expert; b = 4 works the same way):
    qweight  uint8 [n, k * b / 8]   codes q in 0..2^b-1, consecutive values packed little-end first (ramog.quant)
    sb       uint8 [n, k / g]       low nibble sc in 1..15, high nibble mc in 0..15
    d, dmin  fp16  [n]              w = d * sc * q - dmin * mc
This is the llama.cpp Q2_K sub-block scheme with one super-block per row.

The quantizer is the q3 research track's "q3-awx-k4row" (quant_methods/q3_methods.py: q3_quant(w, b, g, calib=X,
aw=True, x=True, sbits=4, sup=0); results/q3/REPORT.md). Credit: the q3 track. It is copied here so that the engine
does not import quant_methods, keeping the arithmetic operation for operation (the same float32 torch ops in the
same order, hence the same decisions). Changes: it returns the integer codes instead of dequantized weights, it
runs batched over many experts with a per-expert importance, and the importance comes from statistics of the
routed inputs (sum of squares and count) instead of the raw tokens. Per group of g weights it minimises
sum_j a_j (w_j - w^_j)^2:
  1. fit(): 121 clip pairs (lo, hi) = (r_lo * min, r_hi * max), r in {1.00, 0.95, ..., 0.50} independently, then
     2 rounds of weighted least-squares refit of (step, min) given the codes, each kept only if the error drops;
  2. code_scales(): per row d = max step / 15 and dmin = max(-min) / 15 (fp16); (sc, mc) are chosen jointly in a
     +-2 window around round(step / d), round(-min / dmin) on the same weighted error.
Groups the search cannot code (no candidate with a finite error: d or dmin underflow in fp16, e.g. an all-zero
row, where q3 keeps an uncoded step/min that the format cannot store) fall back to the rounded codes; quantize()
counts them (`fallback`, 0 on every real Qwen3 expert seen so far).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

RATIOS_X = tuple(round(1.0 - 0.05 * i, 2) for i in range(11))  # 1.0 .. 0.5 for lo and hi independently
SBITS = 4
LV = 2**SBITS - 1  # largest scale code
LS_ITERS = 2
RADIUS = 2


@dataclass
class KPacked:
    qweight: torch.Tensor  # uint8 [..., n, k * bits / 8]
    sb: torch.Tensor  # uint8 [..., n, k / group]
    d: torch.Tensor  # fp16 [..., n]
    dmin: torch.Tensor  # fp16 [..., n]
    bits: int
    group: int
    fallback: int = 0  # groups that needed the fallback codes

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.sb, self.d, self.dmin))


def bits_per_weight(bits: int, group: int, k: int) -> float:
    return bits + 2 * SBITS / group + 32 / k


# -- importance ---------------------------------------------------------------------------------------

def importance_from_stats(sumsq: torch.Tensor, count) -> torch.Tensor:
    """Per-input-channel importance of each expert from its routed inputs' statistics.

    sumsq: [E, k] (or [k]) sum over the expert's routed tokens of x^2; count: [E] (or a scalar) number of tokens.
    E = sumsq / n, a = E + (0.02 + 4/n) * mean(E), a = a / mean(a); a = ones when n == 0 (q3's importance() written
    on the statistics). torch's mean multiplies the sum by the float32 reciprocal 1/n, so sumsq * (1/n) reproduces
    x.pow(2).mean(0) bit for bit when sumsq is x.pow(2).sum(0) taken on the same device. Computed on sumsq's
    device (pass GPU tensors to match a GPU q3 run exactly); returns float32."""
    one = sumsq.dim() == 1
    s = sumsq.float().reshape(-1, sumsq.shape[-1])
    counts = torch.as_tensor(count).reshape(-1).tolist()
    if len(counts) != s.shape[0]:
        raise ValueError(f"{len(counts)} counts for {s.shape[0]} experts")
    out = torch.ones_like(s)
    for e, n in enumerate(counts):
        n = int(n)
        if n <= 0:
            continue
        x = s[e] * (1.0 / n)
        x = x + (0.02 + 4.0 / n) * x.mean()
        out[e] = x / x.mean().clamp(min=1e-30)
    return out[0] if one else out


# -- search (q3_methods.fit / code_scales with aw=True, x=True, sbits=4, sup=0, no lut / mask) -------------

def _deq(wg, s, m, qmax):
    return m + s * ((wg - m) / s).round().clamp(0, qmax)


def _fit(wg, qmax, am):
    """Per-group (step, min) minimising sum am * (w - q(w))^2. wg [..., G, g]; am broadcastable to wg."""
    wmin, wmax = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    best_e = torch.full_like(wmin, float("inf"))
    best_s = torch.zeros_like(wmin)
    best_m = torch.zeros_like(wmin)
    for rl in RATIOS_X:
        for rh in RATIOS_X:
            lo, hi = wmin * rl, wmax * rh
            s = ((hi - lo) / qmax).clamp(min=1e-10)
            e = (am * (_deq(wg, s, lo, qmax) - wg).pow(2)).sum(-1, keepdim=True)
            b = e < best_e
            best_e = torch.where(b, e, best_e)
            best_s = torch.where(b, s, best_s)
            best_m = torch.where(b, lo, best_m)
    for _ in range(LS_ITERS):  # weighted least-squares refit of (step, min) given the codes
        c = ((wg - best_m) / best_s).round().clamp(0, qmax)
        sa, sc, scc = am.sum(-1, keepdim=True), (am * c).sum(-1, keepdim=True), (am * c * c).sum(-1, keepdim=True)
        sw, scw = (am * wg).sum(-1, keepdim=True), (am * c * wg).sum(-1, keepdim=True)
        det = sa * scc - sc * sc
        ok = det > 1e-9 * (sa * scc).clamp(min=1e-30)
        det = torch.where(ok, det, torch.ones_like(det))
        s = (sa * scw - sc * sw) / det
        m = (scc * sw - sc * scw) / det
        ok = ok & (s > 1e-10)
        s = torch.where(ok, s, best_s)
        m = torch.where(ok, m, best_m)
        e = (am * (_deq(wg, s, m, qmax) - wg).pow(2)).sum(-1, keepdim=True)
        b = e < best_e
        best_e = torch.where(b, e, best_e)
        best_s = torch.where(b, s, best_s)
        best_m = torch.where(b, m, best_m)
    return best_s, best_m


def _code_scales(wg, am, s, m, qmax):
    """Joint (sc, mc) search, one super-block per row. wg [B, R, G, g]; s, m [B, R, G, 1].
    Returns sc, mc [B, R, G, 1] (float, integral), d, dm [B, R, 1, 1] (fp16-representable), fallback mask."""
    B, R, G, _ = wg.shape
    sb = s.view(B, R, 1, G)
    mb = (-m).clamp(min=0).view(B, R, 1, G)
    d = (sb.amax(-1, keepdim=True) / LV).clamp(min=1e-12).half().float()
    dm = (mb.amax(-1, keepdim=True) / LV).clamp(min=1e-12).half().float()
    sc0 = (sb / d).round().clamp(0, LV)
    mc0 = (mb / dm).round().clamp(0, LV)
    best_e = torch.full_like(s, float("inf"))
    best_sc = torch.zeros_like(s)
    best_mc = torch.zeros_like(s)
    for ds in range(-RADIUS, RADIUS + 1):
        sc = (sc0 + ds).clamp(1, LV)
        st = (d * sc).view(B, R, G, 1)
        for dmc in range(-RADIUS, RADIUS + 1):
            mc = (mc0 + dmc).clamp(0, LV)
            mn = (-dm * mc).view(B, R, G, 1)
            e = (am * (_deq(wg, st, mn, qmax) - wg).pow(2)).sum(-1, keepdim=True)
            b = e < best_e
            best_e = torch.where(b, e, best_e)
            best_sc = torch.where(b, sc.view(B, R, G, 1), best_sc)
            best_mc = torch.where(b, mc.view(B, R, G, 1), best_mc)
    # q3 keeps the uncoded (step, min) where no candidate has a finite error (d or dmin underflowed in fp16);
    # the format cannot store that, so such groups take the rounded codes.
    fallback = ~(best_e < float("inf"))
    if bool(fallback.any()):
        fb_sc = torch.nan_to_num(sc0, nan=1.0).clamp(1, LV).view(B, R, G, 1)
        fb_mc = torch.nan_to_num(mc0, nan=0.0).clamp(0, LV).view(B, R, G, 1)
        best_sc = torch.where(fallback, fb_sc, best_sc)
        best_mc = torch.where(fallback, fb_mc, best_mc)
    return best_sc, best_mc, d, dm, fallback


def _quantize_tile(wg, am, qmax):
    s, m = _fit(wg, qmax, am)
    sc, mc, d, dm, fallback = _code_scales(wg, am, s, m, qmax)
    B, R, G, _ = wg.shape
    st = d.view(B, R, 1, 1) * sc  # the coded step and min exactly as q3's code_scales() returns them
    mn = -dm.view(B, R, 1, 1) * mc
    q = torch.nan_to_num(((wg - mn) / st).round(), nan=0.0).clamp(0, qmax)
    sbyte = sc.to(torch.uint8) | (mc.to(torch.uint8) << 4)
    return q.to(torch.uint8), sbyte.view(B, R, G), d.view(B, R).half(), dm.view(B, R).half(), int(fallback.sum())


# -- packing ------------------------------------------------------------------------------------------

def pack_codes(q: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 codes [..., k] -> [..., k * bits / 8] bytes, consecutive values little-end first (ramog.quant)."""
    per_byte = 8 // bits
    v = q.reshape(*q.shape[:-1], q.shape[-1] // per_byte, per_byte).to(torch.int32)
    packed = torch.zeros(v.shape[:-1], dtype=torch.int32, device=q.device)
    for i in range(per_byte):
        packed |= v[..., i] << (bits * i)
    return packed.to(torch.uint8)


def unpack_codes(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., k * bits / 8] bytes -> int32 codes [..., k]."""
    per_byte = 8 // bits
    b = qweight.to(torch.int32)
    vals = torch.stack([(b >> (bits * i)) & (2**bits - 1) for i in range(per_byte)], dim=-1)
    return vals.reshape(*qweight.shape[:-1], qweight.shape[-1] * per_byte)


# -- public API ---------------------------------------------------------------------------------------

@torch.no_grad()
def quantize(w: torch.Tensor, bits: int, group: int, importance: torch.Tensor | None = None, *,
             max_elems: int = 1 << 26) -> KPacked:
    """w: [n, k] or [E, n, k] (nn.Linear layout, any float dtype; computed in float32 on w's device).
    importance: None (all ones, i.e. plain weight MSE), [k] (shared) or [E, k] (one per expert; see
    importance_from_stats). The search runs on tiles of at most `max_elems` weights (whole experts when they fit,
    else blocks of whole rows of one expert); every decision is per group or per row, so the tiling does not
    change the result. Peak scratch is about 6 float32 copies of a tile (1.5 GiB at the default)."""
    single = w.dim() == 2
    W = w.unsqueeze(0) if single else w
    E, n, k = W.shape
    if 8 % bits or k % group or group % (8 // bits):
        raise ValueError(f"k4row: bits={bits}, group={group} do not fit k={k}")
    G = k // group
    qmax = 2**bits - 1
    dev = W.device
    if importance is None:
        a = torch.ones(1, k, device=dev).expand(E, k)
    else:
        a = importance.to(dev, torch.float32).reshape(-1, k)
        if a.shape[0] == 1:
            a = a.expand(E, k)
        elif a.shape[0] != E:
            raise ValueError(f"importance has {a.shape[0]} rows for {E} experts")
    qweight = torch.empty(E, n, k * bits // 8, dtype=torch.uint8, device=dev)
    sb = torch.empty(E, n, G, dtype=torch.uint8, device=dev)
    d = torch.empty(E, n, dtype=torch.float16, device=dev)
    dmin = torch.empty(E, n, dtype=torch.float16, device=dev)
    if n * k <= max_elems:
        e_step, r_step = max(1, max_elems // (n * k)), n
    else:
        e_step, r_step = 1, max(1, max_elems // k)
    fallback = 0
    for e0 in range(0, E, e_step):
        e1 = min(E, e0 + e_step)
        am = a[e0:e1].reshape(e1 - e0, 1, G, group)
        for r0 in range(0, n, r_step):
            r1 = min(n, r0 + r_step)
            wg = W[e0:e1, r0:r1].float().reshape(e1 - e0, r1 - r0, G, group)
            qq, sbb, dd, dmm, fb = _quantize_tile(wg, am, qmax)
            qweight[e0:e1, r0:r1] = pack_codes(qq.view(e1 - e0, r1 - r0, k), bits)
            sb[e0:e1, r0:r1] = sbb
            d[e0:e1, r0:r1] = dd
            dmin[e0:e1, r0:r1] = dmm
            fallback += fb
            del wg, qq
    if single:
        qweight, sb, d, dmin = qweight[0], sb[0], d[0], dmin[0]
    return KPacked(qweight, sb, d, dmin, bits, group, fallback)


@torch.no_grad()
def dequantize(qweight: torch.Tensor, sb: torch.Tensor, d: torch.Tensor, dmin: torch.Tensor, bits: int, group: int,
               dtype=torch.float32) -> torch.Tensor:
    """[..., n, k*bits/8], [..., n, k/group], [..., n], [..., n] -> w [..., n, k] (the quantizer's float32 math:
    min + step * q with step = d * sc and min = -dmin * mc, as q3's dequantized output)."""
    q = unpack_codes(qweight, bits)
    lead, k = q.shape[:-1], q.shape[-1]
    q = q.reshape(*lead, k // group, group).float()
    s = sb.to(torch.int32)
    sc = (s & 15).float().unsqueeze(-1)
    mc = (s >> 4).float().unsqueeze(-1)
    st = d.float()[..., None, None] * sc
    mn = -dmin.float()[..., None, None] * mc
    return (mn + st * q).reshape(*lead, k).to(dtype)


def dequantize_packed(p: KPacked, dtype=torch.float32) -> torch.Tensor:
    return dequantize(p.qweight, p.sb, p.d, p.dmin, p.bits, p.group, dtype)
