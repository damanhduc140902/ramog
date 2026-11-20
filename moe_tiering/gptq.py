"""GPTQ on the k4row grid (static groups, act-order, damping 0.01): the dense-c quantizer of the quality sprint.

Ported from quant_methods/t1_dense.py (track T1; results/quality/REPORT.md), keeping its arithmetic operation for
operation so the engine does not import quant_methods (as moe_tiering.kquant does for the q3 quantizer):
  1. grid: per (row, group) the k4row step and min, chosen on the ORIGINAL weights by the importance-weighted
     search of moe_tiering.kquant (121 clip pairs, 2 least-squares refits, joint +-2 search of the 4-bit scale codes);
     fixed before GPTQ runs ("static groups"), so act-order and the k4row scale coding stay exact;
  2. H = X^T X / n from tools/dense_stats.py (xtx float32 summed in float64, divided here in float64), dead
     columns set to 1 on the diagonal, act-order permutation (descending diag), damping 0.01 * mean(diag H);
  3. U = chol(H^-1)^T (upper), then GPTQ / OPTQ error feedback (Frantar et al. 2210.17323) in float32 on W's device,
     lazy blocks of 128 columns, every value rounded to its group's grid.

Two ways to factorize H (step 2), the only float64 part:
  path="cpu"  numpy float64 on the CPU, exactly t1_dense.hinv_prep: reproduces T1's codes bit for bit (the
              container's torch has no CPU LAPACK; numpy's OpenBLAS is deterministic here: tests/test_gptq.py checks
              the factors against T1's on-disk cache, 97 of 97 bitwise equal for Qwen3-30B);
  path="gpu"  torch float64 on the GPU (cholesky, cholesky_inverse, cholesky of the inverse): about 60x faster at
              k = 4096 on an MI250. Checked by residuals (max |H Hinv - I| and max |U^T U - Hinv| / max |Hinv|) and
              finiteness; any exception or residual above `tol` falls back to the CPU path (T1 saw one transient GPU
              Cholesky failure on a well-conditioned float64 matrix). The float32 factor it yields differs from the
              CPU one in the last bits of a few entries, so a few codes differ (qbench KL within noise, see report).
The error-feedback loop itself (step 3) runs where W lives, as in T1 (a GPU, float32): run it on the same kind of
GPU for bit-identical codes.

quantize() returns moe_tiering.kquant.KPacked (codes, sb = sc | mc << 4, fp16 d / dmin per row), the format engine
.kernels decodes with fmt=1; kquant.dequantize() of it equals T1's dequantized output bit for bit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from . import kquant

DAMP = 0.01
BLOCK = 128
GPU_TOL = 1e-6  # residual bound of the GPU factorization (measured ~1e-12 on Qwen3-30B's H at k = 2048 / 4096)


# ---------------------------------------------------------------------------------------------------- statistics

def hessian(entry: dict) -> torch.Tensor:
    """H = xtx / count in float64 on the CPU from a tools/dense_stats.py entry ({xtx [k, k] float32, count, sumsq}),
    as t1_dense.hinv_cached computes it (xtx.double() / float(count))."""
    if entry is None or entry.get("xtx") is None:
        raise ValueError("GPTQ needs xtx statistics (tools/dense_stats.py)")
    return entry["xtx"].detach().to("cpu").contiguous().double() / float(entry["count"])


def importance(entry: dict | None, k: int, device) -> torch.Tensor | None:
    """kquant.importance_from_stats on a dense entry ({sumsq [k], count}) on `device` -> [k] float32, mean 1
    (t1_dense.importance). None when the entry has no sumsq."""
    if entry is None or entry.get("sumsq") is None:
        return None
    return kquant.importance_from_stats(entry["sumsq"].to(device, torch.float32), entry["count"]).reshape(k)


# ---------------------------------------------------------------------------------------------------- factorization

@dataclass
class Prep:
    """The factorized inverse Hessian: U [k, k] float32 on the GPTQ device (upper), perm (act-order) or None, dead
    columns (bool [k]); path = how it was computed ("cpu", "gpu", or "gpu->cpu" after a fallback)."""

    U: torch.Tensor
    perm: torch.Tensor | None
    dead: torch.Tensor
    path: str
    info: dict = field(default_factory=dict)


def _pre(H: torch.Tensor, damp: float, act_order: bool):
    """Dead columns, act-order permutation and damping, exactly as t1_dense.hinv_prep (CPU, float64).
    Returns (damped H as a numpy array, perm or None, dead)."""
    import numpy as np

    Hc = H.detach().to("cpu", torch.float64).clone()
    k = Hc.shape[0]
    dead = torch.diag(Hc) == 0
    Hc[dead, dead] = 1
    perm = torch.argsort(torch.diag(Hc), descending=True) if act_order else None
    if perm is not None:
        Hc = Hc[perm][:, perm]
    Hn = Hc.numpy()
    Hn = Hn + damp * float(np.mean(np.diag(Hn))) * np.eye(k)
    return Hn, perm, dead


def prep_cpu(H: torch.Tensor, damp: float = DAMP, act_order: bool = True, device="cpu") -> Prep:
    """t1_dense.hinv_prep: numpy float64 Cholesky check, inverse, symmetrization, Cholesky of the inverse."""
    import numpy as np

    t0 = time.time()
    Hn, perm, dead = _pre(H, damp, act_order)
    np.linalg.cholesky(Hn)  # raises unless positive definite
    Hi = np.linalg.inv(Hn)
    Hi = (Hi + Hi.T) / 2
    U = torch.from_numpy(np.ascontiguousarray(np.linalg.cholesky(Hi).T))
    dev = torch.device(device)
    return Prep(U.float().to(dev), None if perm is None else perm.to(dev), dead.to(dev), "cpu",
                {"seconds": time.time() - t0})


def prep_gpu(H: torch.Tensor, damp: float = DAMP, act_order: bool = True, device="cuda", tol: float = GPU_TOL) -> Prep:
    """The same factor computed in float64 on the GPU, with a residual check and the CPU path as fallback.
    The preprocessing (dead columns, permutation, damping) is the CPU path's, so only the O(k^3) part moves."""
    import numpy as np

    t0 = time.time()
    Hn, perm, dead = _pre(H, damp, act_order)
    dev = torch.device(device)
    info: dict = {}
    try:
        Hg = torch.from_numpy(Hn).to(dev)
        L, ok = torch.linalg.cholesky_ex(Hg)
        if int(ok) != 0:
            raise RuntimeError(f"cholesky of H failed (info {int(ok)})")
        Hi = torch.cholesky_inverse(L)
        Hi = (Hi + Hi.T) / 2
        U, ok = torch.linalg.cholesky_ex(Hi, upper=True)
        if int(ok) != 0:
            raise RuntimeError(f"cholesky of H^-1 failed (info {int(ok)})")
        eye = torch.eye(Hg.shape[0], device=dev, dtype=torch.float64)
        r_inv = float((Hg @ Hi - eye).abs().max())
        r_chol = float((U.T @ U - Hi).abs().max() / Hi.abs().max())
        info.update(resid_inv=r_inv, resid_chol=r_chol)
        if not (np.isfinite(r_inv) and np.isfinite(r_chol) and r_inv <= tol and r_chol <= tol
                and bool(torch.isfinite(U).all())):
            raise RuntimeError(f"residuals {r_inv:.3e} / {r_chol:.3e} above {tol:g}")
        del Hg, L, Hi, eye
    except Exception as e:  # noqa: BLE001 - any GPU solver problem: the exact CPU path
        p = prep_cpu(H, damp, act_order, dev)
        p.path = "gpu->cpu"
        p.info.update(info, fallback=str(e)[:200], seconds=time.time() - t0)
        return p
    info["seconds"] = time.time() - t0
    return Prep(U.float().contiguous(), None if perm is None else perm.to(dev), dead.to(dev), "gpu", info)


def factorize(H: torch.Tensor, damp: float = DAMP, act_order: bool = True, device="cuda", path: str = "cpu") -> Prep:
    if path == "cpu":
        return prep_cpu(H, damp, act_order, device)
    if path == "gpu":
        return prep_gpu(H, damp, act_order, device)
    raise ValueError(f"unknown GPTQ factorization path {path!r} (cpu | gpu)")


# ---------------------------------------------------------------------------------------------------- grid

@torch.no_grad()
def grid(W: torch.Tensor, bits: int, group: int, a: torch.Tensor | None, max_elems: int = 1 << 26):
    """The static k4row grid of W [n, k] (float32): moe_tiering.kquant's search (= t1_dense.kq_grid with sbits 4,
    radius 2), tiled like kquant.quantize. Returns (sc, mc) uint8 [n, G], (d, dmin) fp16 [n], and the coded step
    st = d * sc and min mn = -dmin * mc as float32 [n, G] (exactly as the format dequantizes them)."""
    n, k = W.shape
    if 8 % bits or k % group or group % (8 // bits):
        raise ValueError(f"k4row: bits={bits}, group={group} do not fit k={k}")
    G = k // group
    qmax = 2**bits - 1
    dev = W.device
    am_all = (torch.ones(1, 1, G, group, device=dev) if a is None
              else a.to(dev, torch.float32).reshape(1, 1, G, group))
    sc = torch.empty(n, G, dtype=torch.uint8, device=dev)
    mc = torch.empty(n, G, dtype=torch.uint8, device=dev)
    d = torch.empty(n, dtype=torch.float16, device=dev)
    dm = torch.empty(n, dtype=torch.float16, device=dev)
    st = torch.empty(n, G, device=dev)
    mn = torch.empty(n, G, device=dev)
    r_step = n if n * k <= max_elems else max(1, max_elems // k)
    fallback = 0
    for r0 in range(0, n, r_step):
        r1 = min(n, r0 + r_step)
        wg = W[r0:r1].float().reshape(1, r1 - r0, G, group)
        s, m = kquant._fit(wg, qmax, am_all)
        bsc, bmc, dd, dmm, fb = kquant._code_scales(wg, am_all, s, m, qmax)
        st[r0:r1] = (dd.view(1, r1 - r0, 1, 1) * bsc).view(r1 - r0, G)  # kquant._quantize_tile / t1 _code_scales
        mn[r0:r1] = (-dmm.view(1, r1 - r0, 1, 1) * bmc).view(r1 - r0, G)
        sc[r0:r1] = bsc.view(r1 - r0, G).to(torch.uint8)
        mc[r0:r1] = bmc.view(r1 - r0, G).to(torch.uint8)
        d[r0:r1] = dd.view(r1 - r0).half()
        dm[r0:r1] = dmm.view(r1 - r0).half()
        fallback += int(fb.sum())
        del wg, s, m, bsc, bmc
    return sc, mc, d, dm, st, mn, fallback


# ---------------------------------------------------------------------------------------------------- GPTQ

@torch.no_grad()
def gptq_codes(W: torch.Tensor, prep: Prep, st: torch.Tensor, mn: torch.Tensor, qmax: int, group: int,
               block: int = BLOCK, return_q: bool = False):
    """t1_dense.gptq with the uniform grid, returning the integer codes. W [n, k] float32; st/mn [n, G].
    Returns codes uint8 [n, k], or (codes, dequantized weight float32 [n, k] = mn + st * codes) with return_q
    (T1's return value; not needed to pack, so not stored by default)."""
    U, perm, dead = prep.U, prep.perm, prep.dead
    n, k = W.shape
    if perm is not None and not torch.equal(perm.sort().values, torch.arange(k, device=perm.device)):
        raise RuntimeError("gptq: the act-order permutation is not a permutation of the columns")
    if U.shape != (k, k) or not bool(torch.isfinite(U).all()):
        raise RuntimeError("gptq: bad inverse-Hessian factor")
    W = W.clone()
    W[:, dead] = 0
    col_group = torch.arange(k) // group  # CPU: python ints index the grid (views; same values as T1's gather)
    if perm is not None:
        W = W[:, perm]
        col_group = col_group[perm.cpu()]
    cg = col_group.tolist()
    Q = torch.zeros_like(W) if return_q else None
    C = torch.zeros(W.shape, dtype=torch.uint8, device=W.device)
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            g = cg[i1 + i]
            s, m = st[:, g], mn[:, g]
            c = torch.nan_to_num(((w - m) / s).round(), nan=0.0).clamp(0, qmax)
            q = m + s * c
            if return_q:
                Q[:, i1 + i] = q
            C[:, i1 + i] = c.to(torch.uint8)
            e = (w - q) / U1[i, i]
            W1[:, i:] -= e.unsqueeze(1) @ U1[i, i:].unsqueeze(0)
            E1[:, i] = e
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    del W
    if perm is not None:
        inv = torch.argsort(perm)
        C = C[:, inv]
        Q = Q[:, inv] if return_q else None
    return (C, Q) if return_q else C


# ---------------------------------------------------------------------------------------------------- public API

@torch.no_grad()
def quantize(W: torch.Tensor, bits: int, group: int, importance: torch.Tensor | None = None,
             H: torch.Tensor | dict | None = None, *, damp: float = DAMP, act_order: bool = True, path: str = "cpu",
             block: int = BLOCK, prep: Prep | None = None, info: dict | None = None) -> kquant.KPacked:
    """W [n, k] (any float dtype, on the device the search and the GPTQ loop run on; float32 math) -> KPacked.

    importance: [k] input-channel weights of the grid search (importance(entry, k, dev)) or None (plain MSE).
    H: None -> plain k4row rounding (moe_tiering.kquant.quantize, = t1_dense apply_grid); else GPTQ with H [k, k]
       (a tensor, e.g. hessian(entry)) or a dense_stats entry dict. prep: an already factorized H (q/k/v share one
       input, so one factorization serves all three). info: filled with the path, timings and fallback count."""
    if W.dim() != 2:
        raise ValueError(f"quantize: W must be [out, in], got {tuple(W.shape)}")
    if H is None and prep is None:
        t0 = time.time()
        kp = kquant.quantize(W.float(), bits, group, importance)
        if info is not None:
            info.update(gptq=False, grid_s=time.time() - t0, fallback=kp.fallback)
        return kp
    Wf = W.float()
    dev = Wf.device
    t0 = time.time()
    sc, mc, d, dm, st, mn, fb = grid(Wf, bits, group, importance)
    t1 = time.time()
    if prep is None:
        Hm = hessian(H) if isinstance(H, dict) else H
        prep = factorize(Hm, damp, act_order, dev, path)
    t2 = time.time()
    codes = gptq_codes(Wf, prep, st, mn, 2**bits - 1, group, block)
    sb = sc | (mc << 4)
    kp = kquant.KPacked(kquant.pack_codes(codes, bits), sb, d, dm, bits, group, fb)
    if info is not None:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        info.update(gptq=True, path=prep.path, grid_s=t1 - t0, factor_s=t2 - t1, loop_s=time.time() - t2,
                    fallback=fb, **{f"prep_{k}": v for k, v in prep.info.items() if k != "seconds"})
    return kp
