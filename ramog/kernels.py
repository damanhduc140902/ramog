"""Triton kernels (NVIDIA and AMD): grouped dequant-GEMV over a pool of packed expert weights.

One launch computes y[p] = W[slot[p]] @ x[token[p]] for every (token, expert) pair p of a MoE layer, reading
the packed low-bit weights directly (no dequantized copy in memory). Two weight formats (constexpr FMT):
  * FMT 0 (rtn, ramog.quant): fp16 step and min per group, folded in per word:
        sum (q*s + m) x = s * sum q x + m * sum x;
  * FMT 1 (k4row, ramog.kquant): one byte per group (sc | mc << 4) and fp16 (d, dmin) per row, applied once
    in the epilogue:  y = d * sum_g sc_g (q.x)_g - dmin * sum_g mc_g (sum x)_g.
Decode is memory bound, so:
  * weights are read as int32 words (8 x 4-bit or 16 x 2-bit values per lane), WORDS words per row per
    iteration, so each row streams >= 128 contiguous bytes;
  * K can be split across programs (atomic add) so a few experts still fill the GPU.
The loop runs whole WORDS steps only, so a launch is right only when WORDS divides the row's K * BITS / 32 words.
Rows no primary config divides (2-bit gpt-oss: 180 words, Gemma-4 down: 44, from host memory) take the tail configs
(WORDS 4 / 2 / 1, SPLIT_K 1), which the autotuners offer only then: every other shape keeps its candidate set.
The packed uint8 layout from ramog.quant is viewed as int32 without copying (little-endian).

Prefill (moe_gemm): the same weights and formats, but pairs are grouped by expert (moe_blocks) and each program
multiplies a block of up to BLOCK_M pairs with a tile of one expert, dequantized in registers once per K step
(tl.dot on bf16, or float32 where the GPU has no bf16 matrix units), so a prompt reads each expert's weights once
per block instead of once per pair.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for block_n in (32, 64, 128):
        for split_k in (1, 2, 4):
            for words in (16, 32):
                for warps in (2, 4):
                    out.append(triton.Config({"BLOCK_N": block_n, "SPLIT_K": split_k, "WORDS": words}, num_warps=warps))
    # Narrow single-warp row blocks: on Turing (GTX 1650) the single-pair 4-bit dense GEMVs (attention projections,
    # lm_head) run about 2x faster with them than with any config above (lm_head 4.8 -> 2.2 ms, q_proj 146 -> 68 us,
    # o_proj 231 -> 135 us), and short-K expert matrices like WORDS=4. The autotuner only picks them where faster.
    for block_n, split_k, words, warps in ((8, 1, 32, 1), (8, 2, 32, 1), (8, 1, 4, 1), (8, 8, 8, 1),
                                           (16, 1, 32, 1), (16, 4, 32, 1), (16, 1, 4, 2), (32, 1, 4, 4)):
        out.append(triton.Config({"BLOCK_N": block_n, "SPLIT_K": split_k, "WORDS": words}, num_warps=warps))
    return out


def _host_configs():
    # Weights read over the host link: throughput is set by how many small reads are in flight, and measured
    # best with narrow row blocks and no K split (see tests/bench_cold.py).
    out = []
    for block_n in (16, 32, 64):
        for warps in (1, 2, 4):
            for words in (8, 16):
                out.append(triton.Config({"BLOCK_N": block_n, "SPLIT_K": 1, "WORDS": words}, num_warps=warps))
    return out


TAIL_WORDS = (4, 2, 1)


def _tail_configs(host: bool):
    # Appended to both autotuned lists; _prune offers them only for rows that no primary config divides, so every
    # other shape keeps its candidate set. Row blocks and warps of the list they extend, SPLIT_K 1 (no atomics).
    blocks, warps = ((16, 32, 64), (1, 2, 4)) if host else ((32, 64, 128), (2, 4))
    out = []
    for words in TAIL_WORDS:
        for block_n in blocks:
            for w in warps:
                c = triton.Config({"BLOCK_N": block_n, "SPLIT_K": 1, "WORDS": words}, num_warps=w)
                c.tail = True  # read by _prune
                out.append(c)
    return out


def row_words(k: int, bits: int) -> int:
    """int32 words per weight row (K * BITS / 32); raises when K * BITS is not a whole number of words."""
    if k * bits % 32:
        raise ValueError(f"K {k} x {bits} bits is not a whole number of int32 words per row")
    return k * bits // 32


def _prune(configs, named_args, **kwargs):
    # never falls back to a config that does not divide the row: its launch would skip the last words of every row
    words = row_words(named_args["K"], kwargs["BITS"])
    fit = [c for c in configs if words % c.kwargs["WORDS"] == 0]
    primary = [c for c in fit if not getattr(c, "tail", False)]
    if primary or fit:
        return primary or fit
    raise ValueError(f"no grouped GEMV config divides {words} words per row (K {named_args['K']}, "
                     f"{kwargs['BITS']} bits)")


@triton.jit
def _gemv_body(
    x_ptr, tok_ptr, slot_ptr, exp_ptr, qw_ptr, sc_ptr, mn_ptr, d_ptr, dm_ptr, y_ptr,
    N, K, P_BUCKET,
    stride_x, stride_qw_slot, stride_qw_row, stride_sc_slot, stride_sc_row, stride_d_slot, stride_y,
    BITS: tl.constexpr, GROUP: tl.constexpr, FMT: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr,
    WORDS: tl.constexpr, MODE: tl.constexpr,
):
    """MODE 0: weights at slot[p]. MODE 1 (hot): skip pairs with slot[p] < 0. MODE 2 (cold): skip pairs with
    slot[p] >= 0 and read expert exp[p] instead (typically from host memory, zero-copy).
    FMT 0: sc_ptr/mn_ptr = fp16 step/min per group (d_ptr/dm_ptr unused). FMT 1: sc_ptr = uint8 (sc | mc << 4)
    per group, d_ptr/dm_ptr = fp16 per row with slot stride stride_d_slot (mn_ptr unused)."""
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    tok = tl.load(tok_ptr + pid_p).to(tl.int64)
    slot = tl.load(slot_ptr + pid_p).to(tl.int64)
    if MODE == 1:
        if slot < 0:
            return
    if MODE == 2:
        if slot >= 0:
            return
        slot = tl.load(exp_ptr + pid_p).to(tl.int64)
    PER: tl.constexpr = 32 // BITS
    WPG: tl.constexpr = GROUP // PER  # words per group
    MASK: tl.constexpr = (1 << BITS) - 1

    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = rows < N
    c = tl.arange(0, WORDS)
    qb = qw_ptr + slot * stride_qw_slot + rows[:, None].to(tl.int64) * stride_qw_row + c[None, :]
    sb = sc_ptr + slot * stride_sc_slot + rows[:, None].to(tl.int64) * stride_sc_row
    if FMT == 0:
        mb = mn_ptr + slot * stride_sc_slot + rows[:, None].to(tl.int64) * stride_sc_row
    xb = x_ptr + tok * stride_x

    n_it = (K // PER) // WORDS
    per = tl.cdiv(n_it, SPLIT_K)
    i0 = pid_k * per
    i1 = tl.minimum(i0 + per, n_it)
    acc = tl.zeros((BLOCK_N, WORDS), dtype=tl.float32)
    if FMT == 1:
        acc_m = tl.zeros((BLOCK_N, WORDS), dtype=tl.float32)
    for it in range(i0, i1):
        w = tl.load(qb + it * WORDS, mask=ok[:, None], other=0)
        gi = (it * WORDS + c) // WPG
        if FMT == 0:
            s = tl.load(sb + gi[None, :], mask=ok[:, None], other=0).to(tl.float32)
            m = tl.load(mb + gi[None, :], mask=ok[:, None], other=0).to(tl.float32)
        else:
            code = tl.load(sb + gi[None, :], mask=ok[:, None], other=0).to(tl.int32)
        t = tl.zeros((BLOCK_N, WORDS), dtype=tl.float32)
        xs = tl.zeros((WORDS,), dtype=tl.float32)
        for r in tl.static_range(PER):
            xr = tl.load(xb + (it * WORDS + c) * PER + r).to(tl.float32)
            t += ((w >> (r * BITS)) & MASK).to(tl.float32) * xr[None, :]
            xs += xr
        if FMT == 0:
            acc += t * s + m * xs[None, :]
        else:
            acc += t * (code & 15).to(tl.float32)
            acc_m += (code >> 4).to(tl.float32) * xs[None, :]
    if FMT == 0:
        out = tl.sum(acc, axis=1)
    else:
        d = tl.load(d_ptr + slot * stride_d_slot + rows, mask=ok, other=0).to(tl.float32)
        dm = tl.load(dm_ptr + slot * stride_d_slot + rows, mask=ok, other=0).to(tl.float32)
        out = d * tl.sum(acc, axis=1) - dm * tl.sum(acc_m, axis=1)
    tl.atomic_add(y_ptr + pid_p * stride_y + rows, out, mask=ok)


@triton.jit
def _copy_blobs(src_ptr, dst_ptr, ids_ptr, want_ptr, n_words, src_stride, dst_stride,
                CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """dst[i] = src[ids[i]] for every i with want[i] != 0, CHUNK words per program (src may be host memory)."""
    i = tl.program_id(0)
    c = tl.program_id(1)
    if tl.load(want_ptr + i) == 0:
        return
    e = tl.load(ids_ptr + i).to(tl.int64)
    base = c * CHUNK
    for off in range(0, CHUNK, BLOCK):
        w = base + off + tl.arange(0, BLOCK)
        ok = w < n_words
        v = tl.load(src_ptr + e * src_stride + w, mask=ok)
        tl.store(dst_ptr + i * dst_stride + w, v, mask=ok)


def copy_blobs(src: torch.Tensor, dst: torch.Tensor, ids: torch.Tensor, want: torch.Tensor,
               chunk: int = 8192, block: int = 1024) -> None:
    """Gather rows of a (possibly pinned-host) uint8 blob table into a GPU buffer, on the current stream.

    src: [E, B] uint8, dst: [P, B] uint8, ids: [P] int32 rows of src, want: [P] (bool/int) which rows to copy.
    """
    s32 = src.view(torch.int32)
    d32 = dst.view(torch.int32)
    n_words = s32.shape[1]
    grid = (ids.numel(), triton.cdiv(n_words, chunk))
    _copy_blobs[grid](s32, d32, ids, want, n_words, s32.stride(0), d32.stride(0), CHUNK=chunk, BLOCK=block,
                      num_warps=4)


_KEY = ["N", "K", "BITS", "GROUP", "FMT", "P_BUCKET", "MODE"]
# restore_value (not reset_to_zero): the tuner's trial launches add into y and are undone after each trial, so a
# launch that accumulates into an existing `out` (hot + cold sources of one layer) is also right on the call that
# triggers the tuning (reset_to_zero used to wipe what the earlier sources had added).
_grouped_gemv = triton.autotune(configs=_configs() + _tail_configs(host=False), key=_KEY, restore_value=["y_ptr"],
                                prune_configs_by={"early_config_prune": _prune})(_gemv_body)
_grouped_gemv_host = triton.autotune(configs=_host_configs() + _tail_configs(host=True), key=_KEY,
                                     restore_value=["y_ptr"],
                                     prune_configs_by={"early_config_prune": _prune})(_gemv_body)


# -- decode GEMV, byte-wise k4row (FMT 1, 4 or 8 bits) --------------------------------------------------------------
# _gemv_body reads x with a stride of 32 / BITS and unpacks int32 words in a 3-D register tensor; for k4row at 4 and
# 8 bits this kernel reads the codes as bytes (8-bit: one code per byte; 4-bit: low nibble first, the words'
# little-endian order) and x contiguously, 2-D tensors only (ramog.gemv1's dense-part kernel, here with the pair
# tables and modes). Same math, same contract; on an MI250 the dense projections ran 2.3-3.5x faster this way.

def _bconfigs(host: bool):
    out = []
    blocks = (2, 4, 8, 16) if host else (1, 2, 4, 8, 16, 32)
    for block_n in blocks:
        for words in (8, 16, 32, 64):
            for warps in (1, 2, 4, 8):
                out.append(triton.Config({"BLOCK_N": block_n, "WORDS": words}, num_warps=warps))
    return out


def _bprune(configs, named_args, **kwargs):
    per = 32 // kwargs["BITS"]
    k = named_args["K"]
    ok = [c for c in configs if k % (c.kwargs["WORDS"] * per) == 0 and (c.kwargs["WORDS"] * per) % kwargs["GROUP"] == 0
          and c.kwargs["BLOCK_N"] * c.kwargs["WORDS"] <= 2048]
    if not ok:
        raise ValueError(f"no byte-wise GEMV config fits K {k}, {kwargs['BITS']} bits, group {kwargs['GROUP']}")
    return ok


@triton.jit
def _gemv_bytes(x_ptr, tok_ptr, slot_ptr, exp_ptr, qb_ptr, sb_ptr, d_ptr, dm_ptr, y_ptr, N, K, P_BUCKET,
                stride_x, stride_qb_slot, stride_qb_row, stride_sb_slot, stride_sb_row, stride_d_slot, stride_y,
                BITS: tl.constexpr, GROUP: tl.constexpr, MODE: tl.constexpr, BLOCK_N: tl.constexpr,
                WORDS: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)
    tok = tl.load(tok_ptr + pid_p).to(tl.int64)
    slot = tl.load(slot_ptr + pid_p).to(tl.int64)
    if MODE == 1:
        if slot < 0:
            return
    if MODE == 2:
        if slot >= 0:
            return
        slot = tl.load(exp_ptr + pid_p).to(tl.int64)
    PER: tl.constexpr = 32 // BITS
    KB: tl.constexpr = WORDS * PER
    NB: tl.constexpr = WORDS * 4
    GPS: tl.constexpr = KB // GROUP
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = rows < N
    bcol = tl.arange(0, NB)
    gs = tl.arange(0, GPS)
    qb = qb_ptr + slot * stride_qb_slot + rows[:, None].to(tl.int64) * stride_qb_row + bcol[None, :]
    sbb = sb_ptr + slot * stride_sb_slot + rows[:, None].to(tl.int64) * stride_sb_row + gs[None, :]
    xb = x_ptr + tok * stride_x
    acc = tl.zeros((BLOCK_N, GPS), dtype=tl.float32)
    accm = tl.zeros((BLOCK_N, GPS), dtype=tl.float32)
    for it in range(0, K // KB):
        b = tl.load(qb + it * NB, mask=ok[:, None], other=0).to(tl.int32)
        if BITS == 8:
            q = b.to(tl.float32)
        else:
            q = tl.interleave((b & 15).to(tl.float32), (b >> 4).to(tl.float32))
        xv = tl.load(xb + it * KB + tl.arange(0, KB)).to(tl.float32)
        t = tl.sum(tl.reshape(q * xv[None, :], (BLOCK_N, GPS, GROUP)), axis=2)
        xs = tl.sum(tl.reshape(xv, (GPS, GROUP)), axis=1)
        code = tl.load(sbb + it * GPS, mask=ok[:, None], other=0).to(tl.int32)
        acc += t * (code & 15).to(tl.float32)
        accm += (code >> 4).to(tl.float32) * xs[None, :]
    d = tl.load(d_ptr + slot * stride_d_slot + rows, mask=ok, other=0).to(tl.float32)
    dm = tl.load(dm_ptr + slot * stride_d_slot + rows, mask=ok, other=0).to(tl.float32)
    out = d * tl.sum(acc, axis=1) - dm * tl.sum(accm, axis=1)
    tl.atomic_add(y_ptr + pid_p * stride_y + rows, out, mask=ok)


_BKEY = ["N", "K", "BITS", "GROUP", "P_BUCKET", "MODE"]
_grouped_gemv_bytes = triton.autotune(configs=_bconfigs(False), key=_BKEY, restore_value=["y_ptr"],
                                      prune_configs_by={"early_config_prune": _bprune})(_gemv_bytes)
_grouped_gemv_bytes_host = triton.autotune(configs=_bconfigs(True), key=_BKEY, restore_value=["y_ptr"],
                                           prune_configs_by={"early_config_prune": _bprune})(_gemv_bytes)
BYTEWISE = os.environ.get("RAMOG_BYTEWISE_GEMV", "1") != "0"


def _bytes_fit(k: int, bits: int, group: int) -> bool:
    """ramog.gemv1.bytes_fit (kept here: kernels does not import gemv1)."""
    if bits not in (4, 8):
        return False
    per = 32 // bits
    return any(k % (w * per) == 0 and (w * per) % group == 0 for w in (8, 16, 32, 64))


def grouped_gemv(x: torch.Tensor, token: torch.Tensor, slot: torch.Tensor, qweight: torch.Tensor,
                 scale: torch.Tensor, minimum: torch.Tensor | None, bits: int, group: int, *,
                 mode: int = 0, expert: torch.Tensor | None = None, out: torch.Tensor | None = None,
                 fmt: int = 0, d: torch.Tensor | None = None, dmin: torch.Tensor | None = None,
                 config: dict | None = None) -> torch.Tensor:
    """x: [T, K]; token/slot: [P] int32; qweight: [S, N, K*bits/8] uint8 (or its int32 view).
    fmt 0 (rtn): scale/minimum fp16 [S, N, K/group].
    fmt 1 (k4row): scale = uint8 [S, N, K/group] codes (sc | mc << 4), minimum unused (None), d/dmin fp16 [S, N].
    Returns y: [P, N] float32, y[p] = dequant(W[slot[p]]) @ x[token[p]].

    mode=1 computes only pairs with slot >= 0; mode=2 only pairs with slot < 0, using W[expert[p]]. Both add into
    `out` when given, so a hot launch and a cold launch together fill every row exactly once. The weight pool may
    live in pinned host memory: the GPU then reads it over the host link (zero-copy). Launch parameters are
    autotuned per (shape, bits, group, format, pairs, mode) and pool location; `config` ({"BLOCK_N", "SPLIT_K",
    "WORDS", "num_warps"}) launches one fixed configuration instead (tests, benchmarks; `out` is not zeroed).
    """
    n_pairs = token.numel()
    n = qweight.shape[1]
    k = x.shape[1]
    y = out if out is not None else torch.zeros(n_pairs, n, device=x.device, dtype=torch.float32)
    if n_pairs == 0:
        return y
    qw32 = qweight if qweight.dtype == torch.int32 else qweight.view(torch.int32)
    if fmt == 0:
        assert minimum is not None and minimum.stride() == scale.stride(), "fmt 0 needs scale/minimum alike"
        mn, dd, dm, stride_d = minimum, scale, scale, 0  # d/dmin unused
    elif fmt == 1:
        assert scale.dtype == torch.uint8, "fmt 1: scale is the uint8 (sc | mc << 4) code per group"
        assert d is not None and dmin is not None and d.dtype == dmin.dtype == torch.float16
        assert d.stride() == dmin.stride() and d.stride(-1) == 1, "fmt 1: d/dmin [S, N] with equal strides"
        mn, dd, dm, stride_d = scale, d, dmin, d.stride(0)  # minimum unused
    else:
        raise ValueError(f"unknown weight format {fmt}")
    bucket = 1 << max(0, (n_pairs - 1).bit_length())  # autotune per power-of-two number of pairs
    if fmt == 1 and config is None and BYTEWISE and _bytes_fit(k, bits, group):
        qb = qweight if qweight.dtype == torch.uint8 else qweight.view(torch.uint8)
        kern = _grouped_gemv_bytes_host if qweight.device.type == "cpu" else _grouped_gemv_bytes
        kern[lambda meta: (n_pairs, triton.cdiv(n, meta["BLOCK_N"]))](
            x, token, slot, expert if expert is not None else slot, qb, scale, d, dmin, y, n, k, bucket,
            x.stride(0), qb.stride(0), qb.stride(1), scale.stride(0), scale.stride(1), d.stride(0), y.stride(0),
            BITS=bits, GROUP=group, MODE=mode)
        return y
    exp = expert if expert is not None else slot
    args = (x, token, slot, exp, qw32, scale, mn, dd, dm, y,
            n, k, bucket,
            x.stride(0), qw32.stride(0), qw32.stride(1), scale.stride(0), scale.stride(1), stride_d, y.stride(0))

    if config is not None:
        meta = {key: config[key] for key in ("BLOCK_N", "SPLIT_K", "WORDS")}
        words = row_words(k, bits)
        assert words % meta["WORDS"] == 0, f"WORDS {meta['WORDS']} does not divide the {words} words of a row"
        grid = (n_pairs, triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])
        _gemv_body[grid](*args, BITS=bits, GROUP=group, FMT=fmt, MODE=mode, num_warps=config.get("num_warps", 4),
                         **meta)
        return y

    def grid(meta):
        return (n_pairs, triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

    kernel = _grouped_gemv_host if qweight.device.type == "cpu" else _grouped_gemv
    kernel[grid](*args, BITS=bits, GROUP=group, FMT=fmt, MODE=mode)
    return y


# -- prefill: grouped dequant-GEMM over (token, expert) pairs ---------------------------------------------------
# A prompt routes many tokens to each expert, so instead of one weight read per pair (the GEMV above) the pairs
# are grouped by expert (moe_blocks) and every program computes a [BLOCK_M pairs, BLOCK_N rows] tile of one
# expert: the packed weight tile is read once per K step, dequantized in registers (the float32 arithmetic of
# BlobLayout.dequant, rounded to bf16 like the dequantize-then-matmul path it replaces) and multiplied with tl.dot.
# Nothing waits for the host: the block table is built on the GPU and has a fixed upper bound on its length, so
# the launch grid is static.

def moe_blocks(expert_of_pair: torch.Tensor, num_experts: int, block_m: int):
    """Group (token, expert) pairs by expert into blocks of at most block_m pairs, on the GPU without a sync.

    expert_of_pair: [P] expert of every pair. Returns (order, blocks, counts):
      order  int32 [P]          pair ids sorted by expert (stable);
      blocks int32 [nb_max, 3]  per block (expert, first position in order, number of pairs), with
                                nb_max = ceil(P / block_m) + num_experts; unused blocks have expert = num_experts;
      counts int64 [E]          pairs per expert.
    """
    flat = expert_of_pair.reshape(-1).long()
    p = flat.numel()
    dev = flat.device
    e = num_experts
    counts = torch.zeros(e, dtype=torch.long, device=dev).index_add_(0, flat, torch.ones_like(flat))
    order = torch.argsort(flat, stable=True).to(torch.int32)
    nblk = (counts + block_m - 1) // block_m
    blk_end = torch.cumsum(nblk, 0)
    seg_end = torch.cumsum(counts, 0)
    nb_max = (p + block_m - 1) // block_m + e
    b = torch.arange(nb_max, device=dev)
    be = torch.searchsorted(blk_end, b, right=True)  # expert of block b (e for blocks past the last)
    ec = be.clamp(max=e - 1)
    row0 = seg_end[ec] - counts[ec] + (b - (blk_end[ec] - nblk[ec])) * block_m
    rows = (seg_end[ec] - row0).clamp(0, block_m)
    used = be < e
    blocks = torch.stack([be, torch.where(used, row0, 0), torch.where(used, rows, 0)], 1).to(torch.int32)
    return order, blocks, counts


def block_m_for(pairs: int, experts: int, host: bool = False) -> int:
    """Pairs per block from the average number of pairs per expert (host-known sizes only, no sync). Weights read
    over the host link (host=True) cost a link transfer per block, so those launches take bigger blocks: an expert
    with up to block_m pairs is read once."""
    avg = pairs / max(1, experts)
    if host:
        return 16 if avg <= 16 else 64 if avg <= 48 else 128
    return 16 if avg <= 16 else 32 if avg <= 48 else 64


def _gemm_configs():
    # (BLOCK_N, BLOCK_K, warps, stages); configs too big for the block_m of a call are pruned (_gemm_prune)
    out = []
    for block_n, block_k, warps, stages in ((32, 128, 4, 2), (64, 64, 4, 2), (64, 128, 4, 2), (128, 64, 4, 2),
                                            (64, 256, 8, 2), (128, 128, 8, 2), (64, 128, 4, 1), (32, 64, 4, 1)):
        out.append(triton.Config({"BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps, num_stages=stages))
    return out


def _gemm_host_configs():
    # zero-copy weights: at least 64 contiguous bytes per row and K step, narrow row blocks, many reads in flight;
    # one stage (no software pipelining) leaves the whole LDS to the dot operands. The BLOCK_K 64 configs are for
    # big blocks with float32 operands: on pre-Ampere NVIDIA a 128-pair block with BLOCK_K 128 alone needs 72 KiB.
    out = []
    for block_n, block_k, warps in ((16, 256, 4), (32, 256, 4), (16, 512, 4), (32, 512, 8), (64, 256, 8),
                                    (16, 128, 4), (16, 64, 4), (32, 64, 4)):
        out.append(triton.Config({"BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps, num_stages=1))
    return out


LDS_BYTES = 65536  # cap on the modelled shared memory of a config (MI250 and Turing allow exactly 64 KiB per block)


def _gemm_smem(block_m: int, block_n: int, block_k: int, stages: int, elem: int = 2, bits: int = 4,
               backend: str = "hip", arch=None) -> int:
    """Shared memory of a _gemm_body config, as Triton 3.6 compiles it (tests/test_gemm_smem.py compiles every
    config for each target and checks this model). elem: bytes per dot operand value (2: bf16, 4: DOT_F32).
    HIP (gfx90a, gfx942, gfx1100): without software pipelining the two operand tiles share one buffer, with it they
    get one each. NVIDIA: one buffer per operand in either case; on sm_80+ (cp.async), 2+ stages also buffer the
    packed weight tile of the next K step(s)."""
    if backend != "cuda":
        if stages <= 1:
            return max(block_m, block_n) * block_k * elem
        return (block_m + block_n) * block_k * elem
    smem = (block_m + block_n) * block_k * elem
    if stages >= 2 and arch is not None and int(arch) >= 80:
        smem += (stages - 1) * block_n * block_k * bits // 8
    return smem


def dot_f32(device=None) -> bool:
    """Feed tl.dot float32 instead of bf16: NVIDIA GPUs before sm_80 (Turing, e.g. the GTX 1650) have no bf16
    matrix units, so the prefill GEMM takes exact float32 FMA products there. ENGINE_DOT_F32=1/0 forces it."""
    env = os.environ.get("ENGINE_DOT_F32")
    if env is not None:
        return env == "1"
    if torch.version.hip:
        return False
    return torch.cuda.get_device_capability(device)[0] < 8


_DEVICE_INFO: dict = {}


def _device_target():
    """(backend, arch, max shared memory per block in bytes) of the GPU Triton launches on."""
    drv = triton.runtime.driver.active
    dev = drv.get_current_device()
    if dev not in _DEVICE_INFO:
        t = drv.get_current_target()
        _DEVICE_INFO[dev] = (t.backend, t.arch, int(drv.utils.get_device_properties(dev)["max_shared_mem"]))
    return _DEVICE_INFO[dev]


def gemm_candidates(configs, k: int, group: int, bits: int, block_m: int, f32: bool, backend: str, arch,
                    limit: int) -> list:
    """The configs of the prefill GEMM the autotuner benchmarks for one key on a GPU (backend, arch) with `limit`
    bytes of shared memory per block: legal K steps whose modelled shared memory fits min(limit, LDS_BYTES), plus
    always one minimal config (BLOCK_N 16, BLOCK_K 64 or the smallest legal step, one stage; at most 36 KiB even for
    128-pair float32 blocks). Configs that turn out not to fit are only skipped by the autotuner (OutOfResources is
    timed as inf); the minimal one guarantees it still has one to launch, on GPUs or Triton versions this model
    has not been checked on."""
    per = 32 // bits
    elem = 4 if f32 else 2
    cap = min(limit, LDS_BYTES)

    def legal(bk):
        return k % bk == 0 and bk % group == 0 and bk % per == 0

    ok = [c for c in configs if legal(c.kwargs["BLOCK_K"])
          and _gemm_smem(block_m, c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"], c.num_stages, elem, bits, backend,
                         arch) <= cap]
    bk = 64 if legal(64) else max(group, per, 16)  # the layouts guarantee K % group == 0
    least = triton.Config({"BLOCK_N": 16, "BLOCK_K": bk}, num_warps=4, num_stages=1)
    if not any(c.kwargs == least.kwargs and c.num_warps == least.num_warps and c.num_stages == least.num_stages
               for c in ok):
        ok.append(least)
    return ok


def _gemm_prune(configs, named_args, **kwargs):
    backend, arch, limit = _device_target()
    return gemm_candidates(configs, named_args["K"], kwargs["GROUP"], kwargs["BITS"], kwargs["BLOCK_M"],
                           bool(kwargs.get("DOT_F32")), backend, arch, limit)


@triton.jit
def _gemm_body(
    x_ptr, pair_ptr, blk_ptr, slot_ptr, qw_ptr, sc_ptr, mn_ptr, d_ptr, dm_ptr, y_ptr,
    N, K, E, IN_DIV,
    stride_x, stride_qw_slot, stride_qw_row, stride_sc_slot, stride_sc_row, stride_d_slot, stride_y,
    BITS: tl.constexpr, GROUP: tl.constexpr, FMT: tl.constexpr, MODE: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DOT_F32: tl.constexpr,
):
    """One [BLOCK_M, BLOCK_N] tile of y[pair] = dequant(W) @ x[pair // IN_DIV] for the pairs of one block.
    blk_ptr: per block (expert, first position in pair_ptr, number of pairs). MODE 0: weights at slot = expert.
    MODE 1 (hot): slot = slot_ptr[expert]; blocks of uncached experts are skipped. MODE 2 (cold): blocks of cached
    experts are skipped; weights at index expert. Weight formats (FMT) and pointers as in _gemv_body."""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(blk_ptr + pid_b * 3)
    if e >= E:
        return
    r0 = tl.load(blk_ptr + pid_b * 3 + 1)
    nr = tl.load(blk_ptr + pid_b * 3 + 2)
    slot = e.to(tl.int64)
    if MODE == 1:
        s_hot = tl.load(slot_ptr + e)
        if s_hot < 0:
            return
        slot = s_hot.to(tl.int64)
    if MODE == 2:
        if tl.load(slot_ptr + e) >= 0:
            return
    PER: tl.constexpr = 32 // BITS
    KW: tl.constexpr = BLOCK_K // PER  # int32 words per row and K step
    GS: tl.constexpr = BLOCK_K // GROUP  # groups per K step
    MASK: tl.constexpr = (1 << BITS) - 1

    rm = tl.arange(0, BLOCK_M)
    mok = rm < nr
    pair = tl.load(pair_ptr + r0 + rm, mask=mok, other=0)
    src = (pair // IN_DIV).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nok = rn < N
    rn64 = rn.to(tl.int64)
    cw = tl.arange(0, KW)
    cg = tl.arange(0, GS)
    ck = tl.arange(0, BLOCK_K)
    shifts = tl.arange(0, PER) * BITS
    qb = qw_ptr + slot * stride_qw_slot + rn64[:, None] * stride_qw_row + cw[None, :]
    sb = sc_ptr + slot * stride_sc_slot + rn64[:, None] * stride_sc_row + cg[None, :]
    xb = x_ptr + src[:, None] * stride_x + ck[None, :]
    if FMT == 0:
        mb = mn_ptr + slot * stride_sc_slot + rn64[:, None] * stride_sc_row + cg[None, :]
    else:
        dd = tl.load(d_ptr + slot * stride_d_slot + rn64, mask=nok, other=0).to(tl.float32)
        dmm = tl.load(dm_ptr + slot * stride_d_slot + rn64, mask=nok, other=0).to(tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        xt = tl.load(xb + k0, mask=mok[:, None], other=0.0).to(tl.bfloat16)
        words = tl.load(qb + k0 // PER, mask=nok[:, None], other=0)
        q = tl.reshape((words[:, :, None] >> shifts[None, None, :]) & MASK, (BLOCK_N, BLOCK_K)).to(tl.float32)
        if FMT == 0:
            s = tl.load(sb + k0 // GROUP, mask=nok[:, None], other=0).to(tl.float32)
            m = tl.load(mb + k0 // GROUP, mask=nok[:, None], other=0).to(tl.float32)
            s = tl.reshape(tl.broadcast_to(s[:, :, None], (BLOCK_N, GS, GROUP)), (BLOCK_N, BLOCK_K))
            m = tl.reshape(tl.broadcast_to(m[:, :, None], (BLOCK_N, GS, GROUP)), (BLOCK_N, BLOCK_K))
            w = q * s + m
        else:
            code = tl.load(sb + k0 // GROUP, mask=nok[:, None], other=0).to(tl.int32)
            st = dd[:, None] * (code & 15).to(tl.float32)  # step = d * sc, per group
            mn = dmm[:, None] * (code >> 4).to(tl.float32)  # -min = dmin * mc, per group
            st = tl.reshape(tl.broadcast_to(st[:, :, None], (BLOCK_N, GS, GROUP)), (BLOCK_N, BLOCK_K))
            mn = tl.reshape(tl.broadcast_to(mn[:, :, None], (BLOCK_N, GS, GROUP)), (BLOCK_N, BLOCK_K))
            w = st * q - mn
        if DOT_F32:  # no bf16 matrix units (pre-Ampere NVIDIA): exact float32 products
            acc += tl.dot(xt.to(tl.float32), tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(xt, tl.trans(w.to(tl.bfloat16)))
    yb = y_ptr + pair.to(tl.int64)[:, None] * stride_y + rn64[None, :]
    tl.store(yb, acc, mask=mok[:, None] & nok[None, :])


_GEMM_KEY = ["N", "K", "BITS", "GROUP", "FMT", "MODE", "BLOCK_M", "DOT_F32"]
_moe_gemm = triton.autotune(configs=_gemm_configs(), key=_GEMM_KEY,
                            prune_configs_by={"early_config_prune": _gemm_prune})(_gemm_body)
_moe_gemm_host = triton.autotune(configs=_gemm_host_configs(), key=_GEMM_KEY,
                                 prune_configs_by={"early_config_prune": _gemm_prune})(_gemm_body)


def moe_gemm(x: torch.Tensor, order: torch.Tensor, blocks: torch.Tensor, slot: torch.Tensor | None,
             qweight: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor | None, bits: int, group: int, *,
             num_experts: int, out: torch.Tensor, block_m: int, mode: int = 0, in_div: int = 1, fmt: int = 0,
             d: torch.Tensor | None = None, dmin: torch.Tensor | None = None,
             config: dict | None = None) -> torch.Tensor:
    """Prefill counterpart of grouped_gemv: out[pair] = dequant(W[expert]) @ x[pair // in_div] in float32, for every
    pair of every block of (order, blocks) = moe_blocks(..., num_experts, block_m). Weight arguments as in
    grouped_gemv. mode 0: W[expert] (one pool with every expert; a QuantLinear is num_experts = 1); mode 1: only
    experts with slot[expert] >= 0, from W[slot[expert]] (the VRAM cache); mode 2: only experts with
    slot[expert] < 0, from W[expert] (the cold blobs, usually pinned host memory read zero-copy). Rows of the pairs
    a mode skips are left untouched, so a mode-1 and a mode-2 launch into one `out` write every row once.
    `config` ({"BLOCK_N", "BLOCK_K", "num_warps", "num_stages"}) launches one fixed configuration instead (tests)."""
    n = qweight.shape[1]
    k = x.shape[1]
    if order.numel() == 0:
        return out
    qw32 = qweight if qweight.dtype == torch.int32 else qweight.view(torch.int32)
    if fmt == 0:
        assert minimum is not None and minimum.stride() == scale.stride(), "fmt 0 needs scale/minimum alike"
        mn, dd, dm, stride_d = minimum, scale, scale, 0
    elif fmt == 1:
        assert scale.dtype == torch.uint8 and d is not None and dmin is not None and d.stride() == dmin.stride()
        mn, dd, dm, stride_d = scale, d, dmin, d.stride(0)
    else:
        raise ValueError(f"unknown weight format {fmt}")
    assert k % group == 0 and out.dtype == torch.float32 and x.stride(1) == 1 and mode in (0, 1, 2)
    sl = slot if slot is not None else order  # not read in mode 0
    f32 = dot_f32(x.device)
    args = (x, order, blocks, sl, qw32, scale, mn, dd, dm, out, n, k, num_experts, in_div,
            x.stride(0), qw32.stride(0), qw32.stride(1), scale.stride(0), scale.stride(1), stride_d, out.stride(0))
    if config is not None:
        grid = (blocks.shape[0], triton.cdiv(n, config["BLOCK_N"]))
        _gemm_body[grid](*args, BITS=bits, GROUP=group, FMT=fmt, MODE=mode, BLOCK_M=block_m, DOT_F32=f32,
                         BLOCK_N=config["BLOCK_N"], BLOCK_K=config["BLOCK_K"], num_warps=config.get("num_warps", 4),
                         num_stages=config.get("num_stages", 2))
        return out

    def grid(meta):
        return (blocks.shape[0], triton.cdiv(n, meta["BLOCK_N"]))

    kernel = _moe_gemm_host if qweight.device.type == "cpu" else _moe_gemm
    kernel[grid](*args, BITS=bits, GROUP=group, FMT=fmt, MODE=mode, BLOCK_M=block_m, DOT_F32=f32)
    return out
