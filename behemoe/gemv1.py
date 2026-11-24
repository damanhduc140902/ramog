"""One-token GEMV for k4row weights (the dense part at decode): y[N] = dequant(W) @ x, one matrix, no pair tables.

kernels.grouped_gemv serves the experts' (token, slot) pairs; used for a dense projection at batch 1 it reads x with a
stride of 32 / BITS, gives a narrow matrix only a few dozen programs, and always zero-fills a float32 output that a
separate kernel then casts (MI250, Qwen3-30B 8-bit attention: k_proj 16 us for 1 MB, q_proj 34 us for 8.7 MB).
Here every program owns BLOCK_N rows over a K range, reads x contiguously, unpacks [BLOCK_N, WORDS, PER] codes in
registers and reduces per scale group; with SPLIT_K 1 it writes the output dtype directly (no fill, no cast, and
deterministic). k4row (behemoe.kquant): W[n, j] = d[n] * sc[n, g] * q[n, j] - dmin[n] * mc[n, g], g = j // GROUP,
code byte (sc | mc << 4).

    y = gemv1(x, qweight, sb, d, dmin, bits, group, out_dtype=torch.bfloat16)      x: [K] or [1, K]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def bytes_fit(k: int, bits: int, group: int) -> bool:
    """Whether the byte-wise kernels have a launch shape for rows of k weights (a whole number of steps, each a whole
    number of scale groups); otherwise callers keep the word-wise kernels."""
    if bits not in (4, 8):
        return False
    per = 32 // bits
    return any(k % (w * per) == 0 and (w * per) % group == 0 for w in (8, 16, 32, 64))


def _configs():
    out = []
    for block_n in (1, 2, 4, 8, 16, 32):
        for words in (8, 16, 32, 64):
            for warps in (1, 2, 4, 8):
                out.append(triton.Config({"BLOCK_N": block_n, "WORDS": words}, num_warps=warps))
    return out


def _prune(configs, named_args, **kwargs):
    per = 32 // kwargs["BITS"]
    k, n = named_args["K"], named_args["N"]
    ok = [c for c in configs if k % (c.kwargs["WORDS"] * per) == 0 and (c.kwargs["WORDS"] * per) % kwargs["GROUP"] == 0
          and c.kwargs["BLOCK_N"] * c.kwargs["WORDS"] <= 2048 and not (n >= 65536 and c.kwargs["BLOCK_N"] < 8)]
    if not ok:
        raise ValueError(f"gemv1: no configuration fits K {k}, {kwargs['BITS']} bits, group {kwargs['GROUP']}")
    return ok


@triton.autotune(configs=_configs(), key=["N", "K", "BITS", "GROUP"],  # OUT_BF16 not in the key: one config, one sum order
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemv1(x_ptr, qw_ptr, sb_ptr, d_ptr, dm_ptr, y_ptr, N, K, stride_qw, stride_sb,
           BITS: tl.constexpr, GROUP: tl.constexpr, OUT_BF16: tl.constexpr,
           BLOCK_N: tl.constexpr, WORDS: tl.constexpr):
    PER: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    KB: tl.constexpr = WORDS * PER  # weights per step
    GPS: tl.constexpr = KB // GROUP  # scale groups per step
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = rows < N
    c = tl.arange(0, WORDS)
    shifts = tl.arange(0, PER) * BITS
    gs = tl.arange(0, GPS)
    qb = qw_ptr + rows[:, None].to(tl.int64) * stride_qw + c[None, :]
    sbb = sb_ptr + rows[:, None].to(tl.int64) * stride_sb + gs[None, :]
    acc = tl.zeros((BLOCK_N, GPS), dtype=tl.float32)
    accm = tl.zeros((BLOCK_N, GPS), dtype=tl.float32)
    for it in range(0, K // KB):
        w = tl.load(qb + it * WORDS, mask=ok[:, None], other=0)
        q = ((w[:, :, None] >> shifts[None, None, :]) & MASK).to(tl.float32)  # [BLOCK_N, WORDS, PER]
        xv = tl.load(x_ptr + it * KB + tl.arange(0, KB)).to(tl.float32)
        xw = tl.reshape(xv, (WORDS, PER))
        t = tl.sum(tl.reshape(q * xw[None, :, :], (BLOCK_N, GPS, GROUP)), axis=2)
        xs = tl.sum(tl.reshape(xv, (GPS, GROUP)), axis=1)
        code = tl.load(sbb + it * GPS, mask=ok[:, None], other=0).to(tl.int32)
        acc += t * (code & 15).to(tl.float32)
        accm += (code >> 4).to(tl.float32) * xs[None, :]
    d = tl.load(d_ptr + rows, mask=ok, other=0).to(tl.float32)
    dm = tl.load(dm_ptr + rows, mask=ok, other=0).to(tl.float32)
    out = d * tl.sum(acc, axis=1) - dm * tl.sum(accm, axis=1)
    if OUT_BF16:
        tl.store(y_ptr + rows, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=ok)
    else:
        tl.store(y_ptr + rows, out, mask=ok)


def gemv1(x: torch.Tensor, qweight: torch.Tensor, sb: torch.Tensor, d: torch.Tensor, dmin: torch.Tensor, bits: int,
          group: int, out_dtype=torch.bfloat16, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [K] / [1, K] (any float dtype), qweight [N, K*bits/8] uint8 (or int32 view), sb uint8 [N, K/group],
    d / dmin fp16 [N] -> y [N] in out_dtype (bfloat16 or float32)."""
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"gemv1: out_dtype {out_dtype}")
    xk = x.reshape(-1)
    k = xk.numel()
    qw32 = qweight if qweight.dtype == torch.int32 else qweight.view(torch.int32)
    n = qw32.shape[0]
    y = out if out is not None else torch.empty(n, device=x.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_N"]),)

    _gemv1[grid](xk, qw32, sb, d, dmin, y, n, k, qw32.stride(0), sb.stride(0), BITS=bits, GROUP=group,
                 OUT_BF16=out_dtype == torch.bfloat16)
    return y


@triton.autotune(configs=_configs(), key=["N", "K", "BITS", "GROUP"],  # OUT_BF16 not in the key: one config, one sum order
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _gemv1b(x_ptr, qb_ptr, sb_ptr, d_ptr, dm_ptr, y_ptr, N, K, stride_qb, stride_sb,
            BITS: tl.constexpr, GROUP: tl.constexpr, OUT_BF16: tl.constexpr,
            BLOCK_N: tl.constexpr, WORDS: tl.constexpr):
    """Byte-wise variant (BITS 8 or 4): codes read as uint8 [BLOCK_N, bytes] (8-bit: one code per byte; 4-bit: the
    low nibble is the even weight, as in the int32 words' little-endian order), 2-D tensors only."""
    PER: tl.constexpr = 32 // BITS
    KB: tl.constexpr = WORDS * PER  # weights per step
    NB: tl.constexpr = WORDS * 4  # bytes per step
    GPS: tl.constexpr = KB // GROUP
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = rows < N
    bcol = tl.arange(0, NB)
    gs = tl.arange(0, GPS)
    qb = qb_ptr + rows[:, None].to(tl.int64) * stride_qb + bcol[None, :]
    sbb = sb_ptr + rows[:, None].to(tl.int64) * stride_sb + gs[None, :]
    acc = tl.zeros((BLOCK_N, GPS), dtype=tl.float32)
    accm = tl.zeros((BLOCK_N, GPS), dtype=tl.float32)
    for it in range(0, K // KB):
        b = tl.load(qb + it * NB, mask=ok[:, None], other=0).to(tl.int32)
        if BITS == 8:
            q = b.to(tl.float32)
        else:
            q = tl.interleave((b & 15).to(tl.float32), (b >> 4).to(tl.float32))  # [BLOCK_N, KB]
        xv = tl.load(x_ptr + it * KB + tl.arange(0, KB)).to(tl.float32)
        t = tl.sum(tl.reshape(q * xv[None, :], (BLOCK_N, GPS, GROUP)), axis=2)
        xs = tl.sum(tl.reshape(xv, (GPS, GROUP)), axis=1)
        code = tl.load(sbb + it * GPS, mask=ok[:, None], other=0).to(tl.int32)
        acc += t * (code & 15).to(tl.float32)
        accm += (code >> 4).to(tl.float32) * xs[None, :]
    d = tl.load(d_ptr + rows, mask=ok, other=0).to(tl.float32)
    dm = tl.load(dm_ptr + rows, mask=ok, other=0).to(tl.float32)
    out = d * tl.sum(acc, axis=1) - dm * tl.sum(accm, axis=1)
    if OUT_BF16:
        tl.store(y_ptr + rows, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=ok)
    else:
        tl.store(y_ptr + rows, out, mask=ok)


def gemv1b(x, qweight, sb, d, dmin, bits, group, out_dtype=torch.bfloat16, out=None):
    """gemv1 with byte-wise code reads (bits 8 or 4)."""
    if bits not in (4, 8):
        return gemv1(x, qweight, sb, d, dmin, bits, group, out_dtype, out)
    xk = x.reshape(-1)
    k = xk.numel()
    qb = qweight if qweight.dtype == torch.uint8 else qweight.view(torch.uint8)
    n = qb.shape[0]
    y = out if out is not None else torch.empty(n, device=x.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_N"]),)

    _gemv1b[grid](xk, qb, sb, d, dmin, y, n, k, qb.stride(0), sb.stride(0), BITS=bits, GROUP=group,
                  OUT_BF16=out_dtype == torch.bfloat16)
    return y
