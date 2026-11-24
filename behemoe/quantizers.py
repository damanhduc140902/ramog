"""Low-bit weight quantizers for the cold tier, as fake-quant (quantize -> dequantize) for quality studies.

All methods are group-wise asymmetric scalar quantizers, so the existing dequant-GEMV kernel family applies:
    rtn      round-to-nearest with a per-group clipping search (the current engine format)
    hqq      HQQ (Badri & Shaji, 2023): half-quadratic optimization of the zero point under an lp (p<1) error
    had-*    the same, after an incoherence rotation W -> W (D H): random signs D and a (block) Walsh-Hadamard
             matrix H along the input dimension. At inference the input is rotated instead (x -> H^T D x), once
             per layer for gate_up (shared input) and per pair for down. Rotation spreads outliers evenly over a
             group, which is what low-bit scalar quantization needs.
    *-dq8    group scale and min stored as 8-bit codes against an fp16 per-row-block pair ("double quant"),
             which buys smaller groups at almost the same bits per weight.
Weights are [out, in] (nn.Linear layout); groups run along `in`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


# -- rotation -------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _hadamard(n: int, device: str) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device)


def hadamard_block(n_in: int) -> int:
    """Largest power-of-two block (<= 4096) that tiles the input dimension."""
    b = 1
    while n_in % (b * 2) == 0 and b * 2 <= 4096:
        b *= 2
    return b


def rotation(n_in: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (n_in,), generator=g) * 2 - 1).float().to(device)
    b = hadamard_block(n_in)
    return signs, _hadamard(b, str(device)), b


def rotate_in(w: torch.Tensor, signs, h, b) -> torch.Tensor:
    """w [out, in] -> w (D H_blockdiag)."""
    out, n_in = w.shape
    return ((w * signs).reshape(out, n_in // b, b) @ h).reshape(out, n_in)


def unrotate_in(w: torch.Tensor, signs, h, b) -> torch.Tensor:
    out, n_in = w.shape
    return ((w.reshape(out, n_in // b, b) @ h.t()).reshape(out, n_in)) * signs


# -- scalar group quantizers (return dequantized weights) -----------------------------------------

def _groups(w: torch.Tensor, group: int) -> torch.Tensor:
    out, n_in = w.shape
    return w.reshape(out, n_in // group, group)


def rtn_clip(w: torch.Tensor, bits: int, group: int, ratios=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """Per-group min/max RTN, choosing the clip ratio with the lowest squared error. Returns (deq, step, min)."""
    wg = _groups(w.float(), group)
    wmin, wmax = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    qmax = 2**bits - 1
    best = torch.empty_like(wg)
    best_s = torch.empty_like(wmin)
    best_m = torch.empty_like(wmin)
    best_err = torch.full_like(wmin, float("inf"))
    for r in ratios:
        lo, hi = wmin * r, wmax * r
        s = ((hi - lo) / qmax).clamp(min=1e-10)
        deq = ((wg - lo) / s).round().clamp(0, qmax) * s + lo
        err = (deq - wg).pow(2).sum(-1, keepdim=True)
        m = err < best_err
        best = torch.where(m, deq, best)
        best_s = torch.where(m, s, best_s)
        best_m = torch.where(m, lo, best_m)
        best_err = torch.where(m, err, best_err)
    return best.reshape(w.shape), best_s, best_m


def hqq(w: torch.Tensor, bits: int, group: int, iters: int = 20, p: float = 0.7, beta: float = 10.0,
        kappa: float = 1.01):
    """HQQ zero-point optimization (scale from min/max). Returns (deq, step, min)."""
    wg = _groups(w.float(), group)
    qmax = 2**bits - 1
    wmin, wmax = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    inv = qmax / (wmax - wmin).clamp(min=1e-10)  # HQQ's "scale": q = w * inv + zero
    zero = -wmin * inv
    best_err = torch.full_like(wmin, float("inf"))
    best = wg.clone()
    best_zero = zero.clone()
    for _ in range(iters):
        q = (wg * inv + zero).round().clamp(0, qmax)
        r = (q - zero) / inv
        err = (wg - r).abs().mean(-1, keepdim=True)
        better = err < best_err
        best = torch.where(better, r, best)
        best_zero = torch.where(better, zero, best_zero)
        best_err = torch.where(better, err, best_err)
        e = wg - r
        e = torch.sign(e) * torch.relu(e.abs() - (1.0 / beta) * e.abs().clamp(min=1e-12).pow(p - 1))
        zero = (q - (wg - e) * inv).mean(-1, keepdim=True)
        beta *= kappa
    step = 1.0 / inv
    return best.reshape(w.shape), step, -best_zero * step


def _dq8(step: torch.Tensor, mn: torch.Tensor, rows_per_block: int = 1):
    """Quantize per-group (step, min) to 8 bits against per-row max magnitudes (fp16). Returns rounded copies."""
    out = []
    for t in (step, mn):
        a = t.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)  # per row
        out.append((t / a * 127).round().clamp(-127, 127) / 127 * a)
    return out


def requant_with(w: torch.Tensor, step: torch.Tensor, mn: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Re-round w against given (possibly reduced-precision) step/min."""
    wg = _groups(w.float(), group)
    q = ((wg - mn) / step.clamp(min=1e-10)).round().clamp(0, 2**bits - 1)
    return (q * step + mn).reshape(w.shape)


# -- plugin registry ------------------------------------------------------------------------------
# Extra methods live in quant_methods/<name>.py and register themselves:
#
#     from behemoe.quantizers import register
#
#     @register("my-method", bpw=lambda bits, group: bits + 16 / group)
#     def my_method(w, bits, group, seed=0, calib=None):  # w: [out, in] float32 on the GPU
#         ...                                            # calib: [n, in] inputs routed to this expert, or None
#         return w_hat                                    # dequantized, same shape
#
# bpw must count everything stored per weight (codes, scales, codebooks amortized, outliers, ...).

REGISTRY: dict[str, tuple] = {}


def register(name: str, bpw):
    def deco(fn):
        REGISTRY[name] = (fn, bpw)
        return fn
    return deco


def load_plugins() -> None:
    import importlib
    import pkgutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "quant_methods"
    if not root.is_dir():
        return
    import sys

    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    for mod in pkgutil.iter_modules([str(root)]):
        try:
            importlib.import_module(f"quant_methods.{mod.name}")
        except Exception as e:  # noqa: BLE001 - one broken plugin must not break everyone's runs
            print(f"[quantizers] skipping plugin {mod.name}: {type(e).__name__}: {e}")


# -- public entry point ---------------------------------------------------------------------------

METHODS = ("rtn", "hqq", "had-rtn", "had-hqq", "rtn-dq8", "hqq-dq8", "had-rtn-dq8", "had-hqq-dq8")


def bits_per_weight(method: str, bits: int, group: int) -> float:
    if method in REGISTRY:
        return float(REGISTRY[method][1](bits, group))
    if method.endswith("dq8"):
        return bits + 16 / group  # 8-bit step + 8-bit min per group (+ negligible per-row fp16)
    return bits + 32 / group  # fp16 step + fp16 min per group


@torch.no_grad()
def fake_quant(w: torch.Tensor, method: str, bits: int, group: int, seed: int = 0,
               calib: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize-dequantize w [out, in] with `method`; returns a tensor like w (same dtype/device).

    calib: optional [n, in] activations routed to this expert (for calibration-aware methods)."""
    if method in REGISTRY:
        out = REGISTRY[method][0](w.float(), bits, group, seed=seed, calib=calib)
        return out.to(w.dtype)
    x = w.float()
    rot = None
    if method.startswith("had-"):
        rot = rotation(x.shape[1], seed, x.device)
        x = rotate_in(x, *rot)
        base = method[4:]
    else:
        base = method
    dq = base.endswith("-dq8")
    base = base.removesuffix("-dq8")
    if base == "rtn":
        deq, step, mn = rtn_clip(x, bits, group)
    elif base == "hqq":
        deq, step, mn = hqq(x, bits, group)
    else:
        raise ValueError(method)
    if dq:
        step, mn = _dq8(step, mn)
        deq = requant_with(x, step, mn, bits, group)
    if rot is not None:
        deq = unrotate_in(deq, *rot)
    return deq.to(w.dtype)
