"""Triton kernels for trellis-coded (TCQ) expert weights (format and reference decoder: moe_tiering/tcq.py). NVIDIA and
AMD; every launch is graph-capturable (no host synchronization, no data-dependent Python).

    rotate(x, signs, b)        x_rot = H (D x) per row: the input rotation (signs, then the normalised block
                               Walsh-Hadamard transform), float32 out. Once per (token, layer) for gate_up's input,
                               once per (token, expert) pair for down's input.
    tcq_gemv(...)              decode: grouped GEMV over (token, expert) pairs, moe_tiering.kernels.grouped_gemv's
                               calling convention (token/slot per pair, modes 0/1/2, adds into `out`); the packed
                               words are read from VRAM or zero-copy from pinned host memory.
    tcq_dequant(...)           tiles -> W_rot (rotated domain, row scale applied) in bf16 or float32.
    tcq_moe_gemm(...)          prefill: grouped GEMM with moe_tiering.kernels.moe_gemm's convention, the weight tile
                               decoded in registers per K step and multiplied with tl.dot.
All of them compute in the rotated domain: y = s * (Q @ x_rot); the output side is not rotated.

Decode (tcq_gemv). Small T (2-8 tokens) needs no other kernel: the pairs of all T tokens go into one launch
(token[p] picks the input row), as in the engine's decode path; a variant that grouped the pairs by expert and
decoded each expert's tiles once per group was 3.5-5x slower at T = 1..8 on MI250 and was dropped
(results/qint/qw1/variants/). A program covers RT row tiles x CT column tiles (NT = RT * CT tiles, one per
"lane" of the [NT, ...] tensors) and walks K in steps of CT tiles. Per step of the walk (_decode_tiles):
  * the NT tiles' words are loaded as one coalesced [NT, WPAD] block (rows of a row tile are contiguous in memory,
    so a zero-copy read over the host link fetches every byte once);
  * the 12-bit states of all 128 trellis steps of every tile, [NT, 128], come from two tl.gather on that block
    (word w0(t) and w0(t) + 1, compile-time positions) and a 64-bit funnel shift;
  * the LUT (int32 [4096], two fp16 per entry, loaded once per program) is read with one tl.gather of all
    NT * 128 states. With num_warps >= 2 Triton lowers these gathers through shared memory (LDS), i.e. the LUT
    sits in shared memory: on MI250 the random LUT reads through L1 (one cache line per lane) were the bottleneck
    of every register/L1 variant tried (80+ us vs 59 us for gate_up, results/qint/qw1/). With num_warps = 1 the
    gather is "warp-local" and lowered to shuffles, ~15x slower: the wrappers refuse it;
  * the [NT, 16 rows, 8 steps] values times the rotated input give per-row partial sums, kept per lane.
The epilogue sums each row over the CT lanes (tl.sum), multiplies by the row scale and adds into `out` (one
atomic add per element). There is no split-K across programs, so the result is the same, bit for bit, run to run,
eager or graph-replayed, VRAM or pinned host, for a given launch config. The default config is a fixed rule per GPU
vendor (default_gemv_config), never tuned at run time, so it is the same for VRAM and host weights.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import tcq

MIN_WARPS = 2  # see the module docstring: one warp makes the LUT gather warp-local (shuffles), ~15x slower

# ============================================================================================ shared device code


@triton.jit
def _unpack2(u):
    """int32 / uint32 LUT entries -> the two fp16 values (low half first) as float32."""
    u = u.to(tl.uint32, bitcast=True)
    v0 = (u & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    v1 = (u >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return v0, v1


@triton.jit
def _step_windows(B0: tl.constexpr, B1: tl.constexpr, WPT: tl.constexpr):
    """Per step t of a tile: (word holding the window's first bit, the next word (circular), right shift of the
    64-bit word pair that puts the 12-bit window at the bottom). Same formula as tcq.step_window."""
    TOT: tl.constexpr = 64 * (B0 + B1)
    t = tl.arange(0, 128)
    pos = (t // 2) * (B0 + B1) + B0 + (t % 2) * B1
    start = (pos - 12 + TOT) % TOT
    w0 = start // 32
    w1 = (w0 + 1) % WPT
    sh = (52 - start % 32).to(tl.uint64)
    return w0, w1, sh


@triton.jit
def _decode_tiles(tb, ok, lutv, w0, w1, sh, NT: tl.constexpr, WPT: tl.constexpr, WPAD: tl.constexpr):
    """Unscaled values of NT tiles: (v0, v1) float32 [NT, 16 rows, 8 steps] (weights at columns 2q and 2q + 1).
    tb [NT]: pointer to each tile's first word; ok [NT]: tiles to read (others decode as words 0)."""
    wj = tl.arange(0, WPAD)
    words = tl.load(tb[:, None] + wj[None, :], mask=ok[:, None] & (wj < WPT)[None, :], other=0)
    a = tl.gather(words, tl.broadcast_to(w0[None, :], (NT, 128)), axis=1).to(tl.uint32, bitcast=True)
    b = tl.gather(words, tl.broadcast_to(w1[None, :], (NT, 128)), axis=1).to(tl.uint32, bitcast=True)
    st = ((((a.to(tl.uint64) << 32) | b.to(tl.uint64)) >> sh[None, :]) & 0xFFF).to(tl.int32)
    u = tl.gather(lutv, tl.reshape(st, (NT * 128,)), axis=0)
    return _unpack2(tl.reshape(u, (NT, 16, 8)))


@triton.jit
def _row_partials(v0, v1, xb, cok, q):
    """sum over a tile row's 8 steps of v0 * x[2q] + v1 * x[2q + 1] -> [NT, 16]; xb [NT]: x at the tile's first
    column (masked by cok)."""
    xe = tl.load(xb[:, None] + 2 * q[None, :], mask=cok[:, None], other=0.0).to(tl.float32)
    xo = tl.load(xb[:, None] + 2 * q[None, :] + 1, mask=cok[:, None], other=0.0).to(tl.float32)
    return tl.sum(v0 * xe[:, None, :] + v1 * xo[:, None, :], axis=2)


def _check_weights(qweight: torch.Tensor, scale: torch.Tensor, pattern):
    b0, b1 = tcq.parse_pattern(pattern)
    wpt = tcq.words_per_tile(b0, b1)
    assert qweight.dim() == 4, "qweight: int32 [S, N/16, K/16, WPT]"
    _, nrt, nct, w_ = qweight.shape
    assert w_ == wpt, f"qweight has {w_} words per tile, pattern {tcq.pattern_name(b0, b1)} needs {wpt}"
    assert qweight.dtype == torch.int32 and qweight.stride(3) == 1 and qweight.stride(2) == wpt \
        and qweight.stride(1) == nct * wpt, "qweight: int32 [S, N/16, K/16, WPT], contiguous per expert"
    assert scale.dtype == torch.float16 and scale.shape[-1] == nrt * 16 and scale.stride(-1) == 1, \
        "scale: fp16 [S, N]"
    return b0, b1, wpt, nrt * 16, nct * 16


def default_gemv_config() -> dict:
    """Launch shape of the decode kernels when none is given: a fixed rule per GPU vendor (never tuned at run
    time, so VRAM and host launches of one shape use the same arithmetic). Measured best on MI250 (RT 1) and on a
    GTX 1650 (RT 2) for Qwen3-30B's shapes (tools/bench_tcq.py, results/qint/qw1/); 8 warps on MI250 since the decode
    step's other kernels were fused (agents/speed/tcq_sweep.py: gate_up 50.3 -> 46.3 us, down 28.0 -> 26.8 us)."""
    return {"RT": 1 if torch.version.hip else 2, "CT": 16, "num_warps": 8 if torch.version.hip else 4}


def _cfg(config):
    cfg = {**default_gemv_config(), **(config or {})}
    if cfg["num_warps"] < MIN_WARPS:
        raise ValueError(f"num_warps={cfg['num_warps']}: the TCQ decode kernels need >= {MIN_WARPS} (LUT gather "
                         "through shared memory)")
    return cfg


# ============================================================================================ decode GEMV


@triton.jit
def _gemv_tcq(x_ptr, tok_ptr, slot_ptr, exp_ptr, qw_ptr, s_ptr, lut_ptr, y_ptr,
              N, K, stride_x, stride_qw_slot, stride_s_slot, stride_y,
              B0: tl.constexpr, B1: tl.constexpr, WPT: tl.constexpr, WPAD: tl.constexpr, RT: tl.constexpr,
              CT: tl.constexpr, MODE: tl.constexpr):
    """y[p, rows] += s[rows] * (Q[slot[p]] @ x[token[p]]) for the RT row tiles of program (p, pid_n).
    MODE 0: weights at slot[p]; 1: skip pairs with slot[p] < 0; 2: skip pairs with slot[p] >= 0 and read expert
    exp[p] instead (moe_tiering.kernels._gemv_body's modes)."""
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
    NT: tl.constexpr = RT * CT
    nct = K // 16
    nrt = N // 16
    i = tl.arange(0, NT)
    rt = pid_n * RT + i // CT
    ct = i % CT
    rok = rt < nrt
    w0, w1, sh = _step_windows(B0, B1, WPT)
    lutv = tl.load(lut_ptr + tl.arange(0, 4096))
    q = tl.arange(0, 8)
    wbase = qw_ptr + slot * stride_qw_slot + (rt.to(tl.int64) * nct + ct) * WPT
    xbase = x_ptr + tok * stride_x + ct * 16
    acc = tl.zeros((NT, 16), dtype=tl.float32)
    for c0 in range(0, nct, CT):
        cok = (c0 + ct) < nct
        v0, v1 = _decode_tiles(wbase + c0 * WPT, rok & cok, lutv, w0, w1, sh, NT, WPT, WPAD)
        acc += _row_partials(v0, v1, xbase + c0 * 16, cok, q)
    tot = tl.sum(tl.reshape(acc, (RT, CT, 16)), axis=1)  # [RT, 16]
    rr = pid_n * RT + tl.arange(0, RT)
    rows = rr[:, None] * 16 + tl.arange(0, 16)[None, :]
    rmask = (rr < nrt)[:, None]
    sc = tl.load(s_ptr + slot * stride_s_slot + rows, mask=rmask, other=0).to(tl.float32)
    tl.atomic_add(y_ptr + pid_p * stride_y + rows, tot * sc, mask=rmask)


def tcq_gemv(x: torch.Tensor, token: torch.Tensor, slot: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
             pattern, *, mode: int = 0, expert: torch.Tensor | None = None, out: torch.Tensor | None = None,
             config: dict | None = None, lut32: torch.Tensor | None = None) -> torch.Tensor:
    """Grouped decode GEMV over (token, expert) pairs for TCQ weights; moe_tiering.kernels.grouped_gemv's convention.

    x: [T, K] ROTATED input (rotate()), any float dtype (rotate() gives float32), unit stride along K.
    token / slot (and expert): [P] int32 on the GPU. qweight: int32 [S, N/16, K/16, WPT] (a strided [S, ...] view
    of a blob pool is fine: only stride(0) is free), in VRAM or pinned host memory (read zero-copy); scale: fp16
    [S, N], same placement; pattern: (b0, b1) or a name ('b4-5').
    Returns y [P, N] float32: y[p] = s * (Q[slot[p]] @ x[token[p]]), added into `out` when given (else zeros).
    mode 1: only pairs with slot >= 0; mode 2: only pairs with slot < 0, from W[expert[p]]; rows of the other
    pairs are not touched, so one launch per weight source into one `out` fills every row once.
    lut32: the pattern's packed LUT on the GPU (tcq.lut_packed; looked up and cached when None - do that once
    before a graph capture). config: {"RT", "CT", "num_warps"} (default_gemv_config)."""
    b0, b1, wpt, n, k = _check_weights(qweight, scale, pattern)
    n_pairs = token.numel()
    assert x.shape[1] == k and x.stride(1) == 1, (tuple(x.shape), k)
    y = out if out is not None else torch.zeros(n_pairs, n, device=x.device, dtype=torch.float32)
    if n_pairs == 0:
        return y
    assert y.dtype == torch.float32 and y.stride(1) == 1 and y.shape[1] == n
    lut32 = lut32 if lut32 is not None else tcq.lut_packed((b0, b1), x.device)
    cfg = _cfg(config)
    exp = expert if expert is not None else slot
    grid = (n_pairs, triton.cdiv(n // 16, cfg["RT"]))
    _gemv_tcq[grid](x, token, slot, exp, qweight, scale, lut32, y, n, k, x.stride(0), qweight.stride(0),
                    scale.stride(0), y.stride(0), B0=b0, B1=b1, WPT=wpt, WPAD=triton.next_power_of_2(wpt),
                    RT=cfg["RT"], CT=cfg["CT"], MODE=mode, num_warps=cfg["num_warps"])
    return y


# ============================================================================================ prefill


@triton.jit
def _tile_block(wb, rt0, ct0, nrt, nct, lutv, w0, w1, sh, RT: tl.constexpr, CT: tl.constexpr, WPT: tl.constexpr,
                WPAD: tl.constexpr):
    """The unscaled [RT * 16, CT * 16] weight block of row tiles rt0.. and column tiles ct0.. of one expert (wb: its
    first word), decoded with _decode_tiles and rearranged from [RT, CT, 16 rows, 16 cols] to rows x columns.
    Tiles past the matrix decode as words 0 (callers mask them)."""
    NT: tl.constexpr = RT * CT
    i = tl.arange(0, NT)
    rt = rt0 + i // CT
    ct = ct0 + i % CT
    v0, v1 = _decode_tiles(wb + (rt.to(tl.int64) * nct + ct) * WPT, (rt < nrt) & (ct < nct), lutv, w0, w1, sh,
                           NT, WPT, WPAD)
    v = tl.reshape(tl.join(v0, v1), (RT, CT, 16, 16))
    return tl.reshape(tl.permute(v, (0, 2, 1, 3)), (RT * 16, CT * 16))


@triton.jit
def _dequant_tcq(qw_ptr, s_ptr, ids_ptr, lut_ptr, out_ptr, N, K, stride_qw_slot, stride_s_slot, stride_o,
                 stride_on, B0: tl.constexpr, B1: tl.constexpr, WPT: tl.constexpr, WPAD: tl.constexpr,
                 RT: tl.constexpr, CT: tl.constexpr, OUT_F32: tl.constexpr):
    """out[m, rows, cols] = s * LUT values of expert ids[m] (ids < 0: skipped), bf16 or float32."""
    m = tl.program_id(0)
    slot = tl.load(ids_ptr + m).to(tl.int64)
    if slot < 0:
        return
    w0, w1, sh = _step_windows(B0, B1, WPT)
    lutv = tl.load(lut_ptr + tl.arange(0, 4096))
    rt0 = tl.program_id(1) * RT
    ct0 = tl.program_id(2) * CT
    w = _tile_block(qw_ptr + slot * stride_qw_slot, rt0, ct0, N // 16, K // 16, lutv, w0, w1, sh, RT, CT, WPT,
                    WPAD)
    rn = rt0 * 16 + tl.arange(0, RT * 16)
    cols = ct0 * 16 + tl.arange(0, CT * 16)
    sc = tl.load(s_ptr + slot * stride_s_slot + rn, mask=rn < N, other=0).to(tl.float32)[:, None]
    w = w * sc
    optr = out_ptr + m.to(tl.int64) * stride_o + rn[:, None].to(tl.int64) * stride_on + cols[None, :]
    omask = (rn < N)[:, None] & (cols < K)[None, :]
    if OUT_F32:
        tl.store(optr, w, mask=omask)
    else:
        tl.store(optr, w.to(tl.bfloat16), mask=omask)


def tcq_dequant(qweight: torch.Tensor, scale: torch.Tensor, pattern, ids: torch.Tensor,
                out: torch.Tensor | None = None, dtype=torch.bfloat16, row_tiles: int = 2,
                col_tiles: int = 8, lut32: torch.Tensor | None = None) -> torch.Tensor:
    """Decode whole experts to W_rot (rotated domain, row scale applied): out[m] = s * Q of expert ids[m]
    ([M, N, K] bf16 or float32; ids [M] int32 on the GPU; entries < 0 leave out[m] untouched). Each value is the
    float32 product LUT * s (bit-identical to tcq.decode) rounded once to the output dtype. qweight / scale as in
    tcq_gemv (VRAM or pinned host); lut32 as in tcq_gemv. No host synchronization."""
    b0, b1, wpt, n, k = _check_weights(qweight, scale, pattern)
    m = ids.numel()
    dev = ids.device
    if out is None:
        out = torch.empty(m, n, k, device=dev, dtype=dtype)
    assert out.dtype in (torch.bfloat16, torch.float32) and out.stride(2) == 1 and tuple(out.shape[1:]) == (n, k)
    if m == 0:
        return out
    lut32 = lut32 if lut32 is not None else tcq.lut_packed((b0, b1), dev)
    grid = (m, triton.cdiv(n // 16, row_tiles), triton.cdiv(k // 16, col_tiles))
    _dequant_tcq[grid](qweight, scale, ids, lut32, out, n, k, qweight.stride(0), scale.stride(0), out.stride(0),
                       out.stride(1), B0=b0, B1=b1, WPT=wpt, WPAD=triton.next_power_of_2(wpt), RT=row_tiles,
                       CT=col_tiles, OUT_F32=out.dtype == torch.float32, num_warps=4)
    return out


@triton.jit
def _gemm_tcq(x_ptr, pair_ptr, blk_ptr, slot_ptr, qw_ptr, s_ptr, lut_ptr, y_ptr, N, K, E, IN_DIV,
              stride_x, stride_qw_slot, stride_s_slot, stride_y,
              B0: tl.constexpr, B1: tl.constexpr, WPT: tl.constexpr, WPAD: tl.constexpr, MODE: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DOT_F32: tl.constexpr):
    """moe_tiering.kernels._gemm_body for TCQ weights: one [BLOCK_M pairs, BLOCK_N rows] tile of
    y[pair] = W_rot[expert] @ x_rot[pair // IN_DIV] for the pairs of one block (moe_blocks). Per K step the
    (BLOCK_N / 16) x (BLOCK_K / 16) tiles are decoded as in the GEMV (_decode_tiles: coalesced words, LUT gather
    through shared memory), rearranged into the [BLOCK_N, BLOCK_K] weight tile, scaled (s * LUT value in float32,
    rounded to bf16 like the engine's dequant GEMM; float32 on GPUs without bf16 matrix units) and multiplied.
    Modes as _gemm_body (1: hot, slot_ptr[expert] >= 0; 2: cold, experts with slot_ptr[expert] < 0, weights at
    index expert)."""
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
    RT: tl.constexpr = BLOCK_N // 16
    CT: tl.constexpr = BLOCK_K // 16
    rm = tl.arange(0, BLOCK_M)
    mok = rm < nr
    pair = tl.load(pair_ptr + r0 + rm, mask=mok, other=0)
    src = (pair // IN_DIV).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nok = rn < N
    ck = tl.arange(0, BLOCK_K)
    wb = qw_ptr + slot * stride_qw_slot
    w0, w1, sh = _step_windows(B0, B1, WPT)
    lutv = tl.load(lut_ptr + tl.arange(0, 4096))
    sc = tl.load(s_ptr + slot * stride_s_slot + rn, mask=nok, other=0).to(tl.float32)[:, None]
    xb = x_ptr + src[:, None] * stride_x + ck[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        xt = tl.load(xb + k0, mask=mok[:, None], other=0.0)
        w = _tile_block(wb, pid_n * RT, k0 // 16, N // 16, K // 16, lutv, w0, w1, sh, RT, CT, WPT, WPAD) * sc
        if DOT_F32:
            acc += tl.dot(xt.to(tl.float32), tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(xt.to(tl.bfloat16), tl.trans(w.to(tl.bfloat16)))
    yb = y_ptr + pair.to(tl.int64)[:, None] * stride_y + rn.to(tl.int64)[None, :]
    tl.store(yb, acc, mask=mok[:, None] & nok[None, :])


def tcq_moe_gemm(x: torch.Tensor, order: torch.Tensor, blocks: torch.Tensor, slot: torch.Tensor | None,
                 qweight: torch.Tensor, scale: torch.Tensor, pattern, *, num_experts: int, out: torch.Tensor,
                 block_m: int, mode: int = 0, in_div: int = 1, config: dict | None = None,
                 lut32: torch.Tensor | None = None, f32: bool | None = None) -> torch.Tensor:
    """Prefill grouped GEMM for TCQ weights, moe_tiering.kernels.moe_gemm's convention: out[pair] = W_rot[expert] @
    x_rot[pair // in_div] (float32, stored) for every pair of (order, blocks) = kernels.moe_blocks(...). x: the
    ROTATED input [rows, K]. mode 0: W[expert]; 1: experts with slot[expert] >= 0, from W[slot[expert]]; 2: experts
    with slot[expert] < 0, from W[expert] (cold, VRAM or pinned host). Rows of skipped pairs are left untouched.
    Operands: bf16 (x_rot and s * Q each rounded once), float32 on GPUs without bf16 dot (kernels.dot_f32);
    f32=True forces float32 operands (IEEE tl.dot) on every GPU, f32=False bf16: the engine's TieredMoE passes True
    (design-quality amendment 8: bf16 rounding of x_rot breaks attention-sink tokens).
    config: {"BLOCK_N", "BLOCK_K", "num_warps"} (multiples of 16; default 64, 64 and 4 warps, 8 on NVIDIA for
    block_m >= 64; num_warps >= 2). lut32 as in tcq_gemv. No host synchronization."""
    from .kernels import dot_f32

    b0, b1, wpt, n, k = _check_weights(qweight, scale, pattern)
    assert x.shape[1] == k and x.stride(1) == 1 and out.dtype == torch.float32 and mode in (0, 1, 2)
    if order.numel() == 0:
        return out
    # BLOCK_N 64, BLOCK_K 64 measured best or within 2% on MI250 and the GTX 1650 (variants/*prefill*); 8 warps for
    # block_m >= 64 on NVIDIA (GTX 1650, host weights: 156 ms vs 520 ms with 4 warps), 4 elsewhere
    nw = 8 if (block_m >= 64 and not torch.version.hip) else 4
    cfg = {"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": nw, **(config or {})}
    assert cfg["BLOCK_N"] % 16 == 0 and cfg["BLOCK_K"] % 16 == 0 and cfg["num_warps"] >= MIN_WARPS
    bk = cfg["BLOCK_K"] if k % cfg["BLOCK_K"] == 0 else 16
    sl = slot if slot is not None else order
    lut32 = lut32 if lut32 is not None else tcq.lut_packed((b0, b1), x.device)
    grid = (blocks.shape[0], triton.cdiv(n, cfg["BLOCK_N"]))
    _gemm_tcq[grid](x, order, blocks, sl, qweight, scale, lut32, out, n, k, num_experts, in_div, x.stride(0),
                    qweight.stride(0), scale.stride(0), out.stride(0), B0=b0, B1=b1, WPT=wpt,
                    WPAD=triton.next_power_of_2(wpt), MODE=mode,
                    BLOCK_M=block_m, BLOCK_N=cfg["BLOCK_N"], BLOCK_K=bk,
                    DOT_F32=dot_f32(x.device) if f32 is None else bool(f32),
                    num_warps=cfg["num_warps"], num_stages=1)
    return out


# ============================================================================================ input rotation


@triton.jit
def _hadamard(n: tl.constexpr):
    """Unnormalised Sylvester Hadamard matrix [n, n] (n <= 4096) in float32: H[i, j] = (-1)^popcount(i & j)."""
    i = tl.arange(0, n)[:, None]
    j = tl.arange(0, n)[None, :]
    p = i & j
    p = p ^ (p >> 8)
    p = p ^ (p >> 4)
    p = p ^ (p >> 2)
    p = p ^ (p >> 1)
    return 1.0 - 2.0 * (p & 1).to(tl.float32)


@triton.jit
def _fwht(x_ptr, sg_ptr, y_ptr, stride_x, stride_y, NORM, B: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr,
          NS: tl.constexpr):
    """y = H_B (signs * x) for one block of B values of one row (H_B normalised: times NORM = 1/sqrt(B)).
    B1 == 1 (B <= 128): y = sum_i x_i H[i, :] in registers (no tl.dot: a 128 x 128 float32 dot operand needs
    64 KiB of shared memory, over Turing's limit together with the other operand).
    B1 > 1: X [B1, B2] row-major, Y = (H_B1 X) H_B2 (H_B = H_B1 kron H_B2), two float32 tl.dot ("ieee"); program
    axis 2 splits Y's columns into NS slices of B2 / NS (one row of 2048 alone: 8.7 -> 5.1 us on MI250)."""
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    if B1 == 1:
        offs = blk * B + tl.arange(0, B)
        xv = tl.load(x_ptr + row * stride_x + offs).to(tl.float32) * tl.load(sg_ptr + offs)
        yv = tl.sum(xv[:, None] * _hadamard(B), axis=0) * NORM
        tl.store(y_ptr + row * stride_y + offs, yv)
    else:
        W: tl.constexpr = B2 // NS
        sp = tl.program_id(2)
        offs = blk * B + tl.arange(0, B1)[:, None] * B2 + tl.arange(0, B2)[None, :]
        xv = tl.load(x_ptr + row * stride_x + offs).to(tl.float32) * tl.load(sg_ptr + offs)
        t1 = tl.dot(_hadamard(B1), xv, input_precision="ieee")
        j = sp * W + tl.arange(0, W)[None, :]
        p = tl.arange(0, B2)[:, None] & j  # columns sp*W.. of H_B2
        p = p ^ (p >> 8)
        p = p ^ (p >> 4)
        p = p ^ (p >> 2)
        p = p ^ (p >> 1)
        yv = tl.dot(t1, 1.0 - 2.0 * (p & 1).to(tl.float32), input_precision="ieee") * NORM
        tl.store(y_ptr + row * stride_y + blk * B + tl.arange(0, B1)[:, None] * B2 + j, yv)


def rotate(x: torch.Tensor, signs: torch.Tensor, b: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [R, K] (any float dtype, unit stride along K) -> float32 [R, K]: every block of b values of a row times
    the signs, then the normalised Walsh-Hadamard transform H_b; x_rot = H (D x), equal to tcq.rotate_act_ref up
    to float32 rounding. (signs, b) from tcq.layer_rotation(K, layer, device) (create them before a graph
    capture); b a power of two 16..4096 (tcq.hadamard_factors). Programs per (row, block, column slice); no host
    sync."""
    r, k = x.shape
    assert k % b == 0 and x.stride(1) == 1 and signs.dtype == torch.float32 and signs.numel() == k
    y = out if out is not None else torch.empty(r, k, device=x.device, dtype=torch.float32)
    if r == 0:
        return y
    b1, b2 = tcq.hadamard_factors(b)
    ns = 1 if b1 == 1 else b2 // 16
    _fwht[(r, k // b, ns)](x, signs, y, x.stride(0), y.stride(0), float(b) ** -0.5, B=b, B1=b1, B2=b2, NS=ns,
                           num_warps=4 if b1 == 1 or b >= 4096 else 2)
    return y
