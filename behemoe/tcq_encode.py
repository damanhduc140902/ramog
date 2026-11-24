"""Production trellis-coded quantization (TCQ) of MoE experts: on-the-fly Hessians, Viterbi, BlockLDLQ.

The quality sprint's config A (results/quality/REPORT.md, track T4: quant_methods/t4_tcq.py, t4_quant.py,
agents/quality/t4/hessians.py) as a converter can run it on models whose Hessians do not fit on disk: each MoE
layer's Hessians are accumulated while the calibration pass streams through that layer, the layer is encoded, and
everything of it is freed. Credit: the T4 track. The code is copied here so that the engine does not import
quant_methods; the arithmetic is kept operation for operation (the same float ops in the same order, hence the same
codes: the saved A2577 codes are reproduced from the same Hessians bit for bit).

Format (T4's). Per layer, one rotation per input size, shared by all experts: W' = W D H_b (random signs D from
seed layer * 1000, block Walsh-Hadamard H_b along the input dim; behemoe.quantizers.rotation). Every row of W' gets
an fp16 scale s = its RMS; W' / s is coded in tiles of tr x tc = 16 x 16 weights (row-major, 256 weights) by a
tail-biting bitshift trellis: L = 12 bits of state, V = 2 weights per step, b_t bits per step from a repeating
pattern of one or two values (b5 = 2.5 bits/weight, b4,5 = 2.25, ...), values LUT[state] from a 4096 x 2 fp16 table
per pattern (trained once for N(0, 1) by T4, stored with its sha256). A tile is sum(b_t) bits, nothing else.

Error feedback: BlockLDLQ (QuIP#) with block = tc columns, under the rotated, damped Hessian
    H' = R^T H R (symmetrised) + damp * mean(diag) * I,   H^-1 = U^T U (float64 Cholesky, kept float32)
of the expert's routed inputs. The tiles of a column block are independent Viterbi sequences, then
E_J = (W_J - Q_J) U_JJ^-1 and W_rest -= E_J U_J,rest.

Hessians (HessianAccumulator; agents/quality/t4/hessians.py's semantics exactly):
    G[e] = sum x x^T over every (token, expert) pair routed to e, x = the MoE block input (bf16 -> float32),
    D[e] = sum a a^T, a = act(x Wg^T) * (x Wu^T) recomputed per expert from the layer's bf16 weights (F.linear),
every position of every window (no position-0 exclusion), unweighted (optionally weighted by the routing weight
squared), float32 sums folded in with one addmm per expert every `flush` (32) forwards. Finished as
    H = (S / count) / scale  in fp16,  scale = mean diag of S / count (fp32),  H = 0 where count = 0
which is exactly the content of T4's hess-* files; experts with fewer than min_tokens (8) routed tokens get H = I.

Stored per layer and matrix (gate, up, down) exactly as T4's quantize(save=DIR), so that
quant_methods/t4_quant.py:quantize_from_codes reads them:
    {"spec", "layer", "matrix", "experts" (ids), "rot": "layer", "words": int32 [n/tc, E * R/tr, words per tile]
     (column block major, then expert, then row tile), "scale": fp16 [E, R], "shape": (R, n), "lut_sha256",
     "lut_share_bits", "bits_per_expert"}
A tile's words are its bitstream: the chunks c_t = s_t & (2^b_t - 1) back to back, most significant bit first; step
t's state is the L-bit window ending after its chunk (circular).

Determinism: the arithmetic is run-to-run bit-reproducible on one GPU type (measured on MI250 / ROCm torch 2.11 /
Triton 3.6, results/qint/qw3/). The Viterbi kernel has no float reduction across threads (min is exact, ties go
to the lowest index); the LDLQ GEMMs have K = 16 and fixed shapes; the Hessian GEMMs fold fixed groups of `flush`
windows in window order with one addmm per expert. While other processes use the same GPU, results of single ops
occasionally come out different. rocSOLVER's batched float64 Cholesky: 0 alone, 3-5% of batches under typical load,
up to most batches under the heaviest load, and in bursts of minutes no repeated factor in 8 runs at all. Also seen
once each: a whole-layer job's gate/up factors (cause not isolated) and a 235B encode (4 tiles). So every stage
whose output decides the codes is checked or computed twice and compared bitwise. Each guard settles a mismatch by
more runs (a value must come out twice) or refuses with LayerFailed: LayerJob first re-runs the chunk's encode in
place after a pause, then the converter converts the layer again in another streaming pass:
  Hessian sums   exact symmetry (fp32 GEMM sums X^T X are bitwise symmetric when computed right; checked per
                 platform by gemm_symmetric) - HessianAccumulator.check_symmetric
  factors        hessian_factors computed twice end to end (H -> rotation -> damping -> Cholesky -> inverse) per
                 Cholesky batch (FACTOR_VOTES = 2), and its Cholesky voted inside (cholesky_voted)
  encode         rotation + Viterbi + LDLQ run twice per expert stack (ldlq_encode_voted, ENCODE_VOTES = 2), the
                 Viterbi traceback checked (forced s_0 found again)
  copies         every host <-> device copy that feeds the codes, and every result copied back, is checked against
                 an exact fingerprint taken where the data came from (LayerJob, to_device_verified, to_host_checked,
                 checked_rotation)
Remaining exposure (results/qint/qw3/fix2/): a fault that changes a Hessian GEMM's result while keeping it exactly
symmetric, and an identical wrong result coming out of two independent runs. T4's 99.94% repeat rate (layer 5 gate,
results/quality/t4/checks.txt) is most likely the Cholesky effect (T4 ran on a shared GCD).

Expert groups (chunk invariance): BLAS libraries pick their GEMM kernel by problem size, batch count included, so a
batched op over the experts of a stack gives an expert different last bits when the stack has a different number
of experts (measured: the LDLQ feedback bmm's at batch 8 vs 32 or 128 move 1.0-1.4% of tiles; the rotation GEMM
depends on its M at small shapes; results/qint/verify/qw3/chunk_inv.json). Every batched op of the encoder therefore
runs on fixed groups of `group` experts (EXPERT_GROUP = 8): the input rotation, the row-scale reduction, the 16 x 16
block inverses, both feedback bmm's and the error bookkeeping. The Viterbi kernel is per sequence, the Hessian sums
are per expert and the Cholesky batches are chol_chunk(n) experts, so an expert's codes are the same bits whatever
expert chunk (a multiple of 8) the layer is split into. group = 0 is T4's arithmetic (whole-stack batches): it
reproduces T4's saved A2577 codes bit for bit, but only unchunked; with group 8 about 0.3% of the code words differ
from them (same rounding-level effect, same quality; results/qint/qw3/fix/).
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from behemoe.quantizers import rotate_in, rotation

ROT_SEED = 1000  # rotation seed = layer * ROT_SEED (T4's rot="layer")
SEQ_CHUNK = 1 << 14  # Viterbi sequences per launch
FP16_MAX, FP16_TINY = 65504.0, 2.0 ** -24  # largest fp16, smallest positive (subnormal) fp16
MAX_REL_ERR = 0.5  # a stack or expert whose ||W' - sQ||^2 / ||W'||^2 exceeds this is refused (normal: 0.01-0.08)
REL_ERR_FLOOR = 1e-6  # per-expert check: ||W'_e||^2 is floored at this fraction of the stack's mean (dead experts)
CHOL_CHUNK = 8  # Hessians factored per batched Cholesky (fixed: the result must not depend on the expert chunking)
EXPERT_GROUP = 8  # experts per batched op of the encoder (fixed: the codes must not depend on the expert chunking)
TRSM_ALIGN = 128  # triangular inverses are solved at a multiple of this size (inv_upper: rocBLAS trsm failures)
REPLAY_TRIES = 3  # attempts per verified copy, per repaired flush group and per replayed expert chunk


class LayerFailed(RuntimeError):
    """A layer could not be converted because a device result or copy kept coming out wrong (a transient fault:
    the data are fine). tools/tcq_convert.py converts such a layer again in a later pass instead of aborting."""


class ReplayCorrupt(LayerFailed):
    """A replayed input does not match the fingerprint of its capture. bad = positions in the flush group."""

    def __init__(self, msg: str, bad=()):
        super().__init__(msg)
        self.bad = list(bad)


class CaptureFailed(LayerFailed):
    """A window could not be copied to the host with a matching fingerprint while its original was on the device."""


class HostCaptureCorrupt(LayerFailed):
    """A host capture no longer matches the fingerprint it had right after its verified copy (cannot be repaired:
    the original is gone)."""


class AccumulationFlake(LayerFailed):
    """Hessian sums that a correct GEMM leaves exactly symmetric are not."""


class NoMajority(LayerFailed):
    """Repeated runs of a computation never gave the same result twice."""


def chol_chunk(n: int) -> int:
    """Matrices per batched Cholesky for n x n Hessians: T4's 8 up to n = 4096 (Qwen3-30B / 235B bits), 1 above.
    Above 4096 a batch of 8 needs ~2.4 GB per float64 copy (about seven are alive), and rocSOLVER's batched potrf
    is slow there: 0.42 s for 2 x 6144^2 vs 0.04 s per single matrix (MI250, steady state; results/qint/qw3/fix/
    chol_time.txt). A function of the shape only, so the bits do not depend on the expert chunking (expert chunks
    are multiples of CHOL_CHUNK, hence of chol_chunk(n))."""
    return CHOL_CHUNK if n <= 4096 else 1


def expert_groups(n_experts: int, group: int = EXPERT_GROUP) -> list[tuple[int, int]]:
    """[e0, e1) ranges of `group` experts (the last one shorter if group does not divide n_experts); group 0 = one
    range over the whole stack (T4's batches)."""
    g = group or n_experts
    return [(e, min(n_experts, e + g)) for e in range(0, n_experts, g)]


MATRICES = ("gate", "up", "down")


def lut_dirs() -> list[str]:
    """Where the encoder's trellis codebooks are looked up: $MOE_TCQ_LUT_DIR, then T4's $T4_CBDIR, then the ones
    shipped with the engine (behemoe/data/codebooks: T4's tables, the same as agents/quality/t4/codebooks)."""
    out = [os.environ.get("MOE_TCQ_LUT_DIR"), os.environ.get("T4_CBDIR"),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "codebooks")]
    return [d for d in out if d]


# ============================================================================================ spec

def parse_spec(spec: str) -> dict:
    """tcq:L{L}:V{V}:b{b1[,b2..]}[:T{T}][:c{tc}] -> dict (T4's keys). E.g. tcq:L12:V2:b5 = 2.5 bits/weight."""
    parts = spec.split(":")
    if parts[0] != "tcq":
        raise ValueError(f"{spec}: not a tcq spec")
    sp = {"kind": "tcq", "L": 12, "V": 2, "T": 256, "tc": 16, "spec": spec, "b": None, "k": None}
    for p in parts[1:]:
        if p[0] == "b":
            sp["b"] = tuple(int(x) for x in p[1:].split(","))
        elif p[0] == "k":
            sp["k"] = int(p[1:])
        else:
            sp[{"L": "L", "V": "V", "T": "T", "c": "tc"}[p[0]]] = int(p[1:])
    if sp["b"] is None:
        sp["b"] = (sp["k"] * sp["V"],)
    N = sp["T"] // sp["V"]
    if sp["T"] % sp["tc"] or sp["T"] % sp["V"] or N % len(sp["b"]):
        raise ValueError(f"{spec}: T must be a multiple of tc and V, and T/V of the pattern length")
    if max(sp["b"]) > min(8, sp["L"]):
        raise ValueError(f"{spec}: at most min(8, L) bits per step")
    sp["tr"] = sp["T"] // sp["tc"]
    sp["N"] = N
    sp["bits_per_tile"] = sum(sp["b"]) * (N // len(sp["b"]))
    sp["bpw_codes"] = sum(sp["b"]) / len(sp["b"]) / sp["V"]
    sp["lut_key"] = (sp["L"], sp["V"], sp["b"])
    return sp


def table_bits(sp: dict) -> int:
    return (1 << sp["L"]) * sp["V"] * 16


def lut_name(sp: dict) -> str:
    return f"tcq-L{sp['L']}-V{sp['V']}-b{'-'.join(str(b) for b in sp['b'])}.pt"


# ============================================================================================ LUT

def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


_LUT: dict = {}


def lut_file(sp: dict, lut_dir: str | None = None) -> str:
    for d in ([lut_dir] if lut_dir else lut_dirs()):
        p = os.path.join(d, lut_name(sp))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{lut_name(sp)} not found in {[lut_dir] if lut_dir else lut_dirs()}")


def load_lut(sp: dict, device, lut_dir: str | None = None) -> tuple[torch.Tensor, str]:
    """(LUT float32 [2^L, V] on `device` = the stored fp16 values, its sha256). The file's sha256 is checked."""
    p = lut_file(sp, lut_dir)
    key = (p, str(device))
    if key not in _LUT:
        o = torch.load(p, map_location="cpu")
        if _sha(o["LUT"]) != o["sha256"] or tuple(o["LUT"].shape) != (1 << sp["L"], sp["V"]):
            raise RuntimeError(f"{p}: corrupt LUT")
        lut = o["LUT"].float()
        if torch.device(device).type != "cpu":
            lut = to_device_verified(lut[None], device)[0]  # synchronized, checked upload
        if _sha(lut.half()) != o["sha256"]:
            raise RuntimeError(f"{p}: LUT changed in the upload")
        _LUT[key] = (lut, o["sha256"])
    return _LUT[key]


# ============================================================================================ Viterbi

def _fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """float32 fma(a, b, c) = round(a * b + c): the product is exact in float64, the sum is rounded to float64
    and then to float32 (double rounding differs from one rounding only on exact float32 midpoints)."""
    return (a.double() * b.double() + c.double()).float()


def _metric(xt: torch.Tensor, LUT: torch.Tensor, lutn: torch.Tensor) -> torch.Tensor:
    """||LUT[s]||^2 - 2 <LUT[s], x_t> [B, 2^L] with the Triton kernel's float32 roundings: the compiler contracts
    `lut0 * x0 + lut1 * x1` into fma(lut0, x0, lut1 * x1) (measured on MI250 / Triton 3.6: this reference then
    gives the kernel's paths on 12288 of 12288 Gaussian sequences, a separately rounded one on 12287, T4's addmm
    on 12284), and `lutn - 2.0 * lut0 * x` (V = 1) into fma(-2 lut0, x, lutn)."""
    if LUT.shape[1] == 2:
        s = _fma32(LUT[None, :, 0], xt[:, 0:1], LUT[None, :, 1] * xt[:, 1:2])
        return lutn[None] - 2.0 * s
    return _fma32(-2.0 * LUT[None, :, 0], xt[:, 0:1], lutn[None].expand(xt.shape[0], -1))


@torch.no_grad()
def viterbi_ref(x: torch.Tensor, LUT: torch.Tensor, L: int, shifts, start: torch.Tensor | None = None,
                end: torch.Tensor | None = None):
    """Torch reference (t4_tcq.viterbi with the kernel's metric roundings, _metric). x [B, N, V] targets;
    LUT [2^L, V]; shifts: bits per step (pattern, repeated); start/end: [B] forced s_0 / s_N (None = free).
    Returns (states [B, N] int64 = s_1..s_N, s_0 [B]).
    cost_t[s] = min over the 2^b_t predecessors (c << (L - b_t)) | (s >> b_t) of cost_{t-1} + ||LUT[s] - x_t||^2
    (minus the constant ||x_t||^2); ties go to the lowest c (torch.min returns the first minimum, as the kernel's
    tl.min with return_indices_tie_break_left)."""
    B, N, V = x.shape
    S = 1 << L
    dev = x.device
    lutn = (LUT.float() ** 2).sum(1)
    pat = list(shifts)
    bp = torch.empty(N, B, S >> min(pat), dtype=torch.uint8, device=dev)
    if start is None:
        cost = torch.zeros(B, S, device=dev)
    else:
        cost = torch.full((B, S), float("inf"), device=dev)
        cost.scatter_(1, start.view(B, 1), 0.0)
    for t in range(N):
        b = pat[t % len(pat)]
        C, P = 1 << b, S >> b
        m, a = cost.view(B, C, P).min(1)
        bp[t, :, :P] = a.to(torch.uint8)
        d = _metric(x[:, t].float(), LUT.float(), lutn)
        cost = (d.view(B, P, C) + m.unsqueeze(-1)).view(B, S)
    s = cost.argmin(1) if end is None else end.clone()
    states = torch.empty(B, N, dtype=torch.long, device=dev)
    for t in range(N - 1, -1, -1):
        b = pat[t % len(pat)]
        states[:, t] = s
        p = s >> b
        c_old = bp[t].gather(1, p.view(B, 1)).view(B).long()
        s = (c_old << (L - b)) | p
    return states, s


try:  # one program per sequence: the 2^L costs stay in registers for all N steps, traceback in the same kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _vit_fwd(x_ptr, lut0_ptr, lut1_ptr, lutn_ptr, start_ptr, end_ptr, bp_ptr, st_ptr, s0_ptr, N: tl.constexpr,
                 L: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr, V: tl.constexpr, HAS_START: tl.constexpr,
                 HAS_END: tl.constexpr, PMAX: tl.constexpr):
        """t4_tcq._vit_fwd, unchanged. Forward pass with the 2^L costs in registers (back-pointers to global
        memory), then the traceback. Steps alternate B0 (even t) and B1 (odd t) bits."""
        seq = tl.program_id(0).to(tl.int64)
        S: tl.constexpr = 1 << L
        C0: tl.constexpr = 1 << B0
        P0: tl.constexpr = S >> B0
        C1: tl.constexpr = 1 << B1
        P1: tl.constexpr = S >> B1
        offs = tl.arange(0, S)
        lut0 = tl.load(lut0_ptr + offs)
        lutn = tl.load(lutn_ptr + offs)
        if V == 2:
            lut1 = tl.load(lut1_ptr + offs)
        if HAS_START:
            st = tl.load(start_ptr + seq).to(tl.int32)
            cost = tl.where(offs == st, 0.0, float("inf"))
        else:
            cost = tl.zeros([S], dtype=tl.float32)
        xb = x_ptr + seq * N * V
        bb = bp_ptr + seq * N * PMAX
        for t2 in range(N // 2):
            t = 2 * t2
            cv = tl.reshape(cost, (C0, P0))
            m, a = tl.min(cv, axis=0, return_indices=True)
            tl.store(bb + t * PMAX + tl.arange(0, P0), a.to(tl.uint8))
            if V == 2:
                d = lutn - 2.0 * (lut0 * tl.load(xb + t * 2) + lut1 * tl.load(xb + t * 2 + 1))
            else:
                d = lutn - 2.0 * lut0 * tl.load(xb + t)
            cost = tl.reshape(tl.reshape(d, (P0, C0)) + m[:, None], (S,))
            t = t + 1
            cv = tl.reshape(cost, (C1, P1))
            m, a = tl.min(cv, axis=0, return_indices=True)
            tl.store(bb + t * PMAX + tl.arange(0, P1), a.to(tl.uint8))
            if V == 2:
                d = lutn - 2.0 * (lut0 * tl.load(xb + t * 2) + lut1 * tl.load(xb + t * 2 + 1))
            else:
                d = lutn - 2.0 * lut0 * tl.load(xb + t)
            cost = tl.reshape(tl.reshape(d, (P1, C1)) + m[:, None], (S,))
        if HAS_END:
            s = tl.load(end_ptr + seq).to(tl.int32)
        else:
            s = tl.argmin(cost, axis=0).to(tl.int32)
        sb = st_ptr + seq * N
        for i in range(N // 2):
            t = N - 1 - 2 * i  # odd step: B1 bits
            tl.store(sb + t, s)
            p = s >> B1
            c = tl.load(bb + t * PMAX + p).to(tl.int32)
            s = (c << (L - B1)) | p
            t = t - 1  # even step: B0 bits
            tl.store(sb + t, s)
            p = s >> B0
            c = tl.load(bb + t * PMAX + p).to(tl.int32)
            s = (c << (L - B0)) | p
        tl.store(s0_ptr + seq, s)

    HAVE_TRITON = True
except Exception:  # noqa: BLE001 - no Triton: viterbi() runs the torch reference instead (same results, slower)
    HAVE_TRITON = False


def _sync(dev) -> None:
    """Wait for the device's queued work (timings); nothing to wait for on the CPU. (A CPU encode also needs a torch
    build with CPU LAPACK for the float64 Cholesky; the server's ROCm torch has none.)"""
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)

NUM_WARPS = 1  # T4's setting (the result does not depend on it: min is exact and ties go to the lowest index)


def kernel_ok(x: torch.Tensor, shifts) -> bool:
    B, N, V = x.shape
    return HAVE_TRITON and x.is_cuda and len(shifts) <= 2 and N % 2 == 0 and V in (1, 2)


@torch.no_grad()
def viterbi(x: torch.Tensor, LUT: torch.Tensor, L: int, shifts, start: torch.Tensor | None = None,
            end: torch.Tensor | None = None, force_ref: bool = False):
    """viterbi_ref() with the whole pass in one Triton kernel (patterns of length 1 or 2, N even); same outputs."""
    B, N, V = x.shape
    pat = list(shifts)
    if force_ref or not kernel_ok(x, pat):
        return viterbi_ref(x, LUT, L, shifts, start, end)
    b0, b1 = pat[0], pat[-1]
    S = 1 << L
    PMAX = S >> min(pat)
    dev = x.device
    xc = x.contiguous().float()
    lut0 = LUT[:, 0].contiguous().float()
    lut1 = LUT[:, 1].contiguous().float() if V == 2 else lut0
    lutn = (LUT.float() ** 2).sum(1).contiguous()
    bp = torch.empty(B, N, PMAX, dtype=torch.uint8, device=dev)
    states = torch.empty(B, N, dtype=torch.int32, device=dev)
    s0 = torch.empty(B, dtype=torch.int32, device=dev)
    dummy = torch.zeros(1, dtype=torch.long, device=dev)
    _vit_fwd[(B,)](xc, lut0, lut1, lutn, start.contiguous() if start is not None else dummy,
                   end.contiguous() if end is not None else dummy, bp, states, s0, N=N, L=L, B0=b0, B1=b1, V=V,
                   HAS_START=start is not None, HAS_END=end is not None, PMAX=PMAX, num_warps=NUM_WARPS)
    return states.long(), s0.long()


@torch.no_grad()
def tcq_seq(x: torch.Tensor, LUT: torch.Tensor, L: int, shifts, tailbite: bool = True,
            force_ref: bool = False) -> torch.Tensor:
    """Quantize sequences x [B, T] -> states [B, N] of the tail-biting (circular) code (values: LUT[states]).
    QTIP's two passes: the sequence rotated by about N/2 steps (pattern-aligned) estimates the boundary state, then
    the sequence itself with s_0 = s_N forced to it. The second pass's traceback must end in the forced s_0 (every
    state is reachable from it, so the forced end has a finite cost): a free check of the kernel's back-pointers,
    re-run on a mismatch (ENCODE_STATS["traceback_retries"])."""
    B, T = x.shape
    V = LUT.shape[1]
    N = T // V
    xv = x.view(B, N, V)
    if not tailbite:
        return viterbi(xv, LUT, L, shifts, force_ref=force_ref)[0]
    h = (N // 2) // len(shifts) * len(shifts)
    rot = torch.cat([xv[:, h:], xv[:, :h]], 1)  # original step N-1 sits at rotated position N-1-h
    st1, _ = viterbi(rot, LUT, L, shifts, force_ref=force_ref)
    sb = st1[:, N - 1 - h]
    for _ in range(REPLAY_TRIES):
        st2, s0 = viterbi(xv, LUT, L, shifts, start=sb, end=sb, force_ref=force_ref)
        if torch.equal(s0, sb):
            return st2
        ENCODE_STATS["traceback_retries"] += 1
    raise LayerFailed(f"Viterbi traceback did not return to the forced start in {REPLAY_TRIES} runs")


def check_circular(states: torch.Tensor, L: int, shifts) -> bool:
    """Every state is the L-bit window ending after its chunk in the circular stream of chunks c_t = s_t & (2^b_t-1)
    (i.e. the stored words alone decode to these states)."""
    B, N = states.shape
    pat = [shifts[t % len(shifts)] for t in range(N)]
    bits = []
    for t in range(N):
        c = states[:, t] & ((1 << pat[t]) - 1)
        bits += [(c >> (pat[t] - 1 - i)) & 1 for i in range(pat[t])]
    stream = torch.stack(bits, 1)
    total = stream.shape[1]
    pos = 0
    for t in range(N):
        pos += pat[t]
        idx = torch.arange(pos - L, pos, device=states.device) % total
        val = (stream[:, idx] * (1 << torch.arange(L - 1, -1, -1, device=states.device))).sum(1)
        if not bool((val == states[:, t]).all()):
            return False
    return True


# ============================================================================================ packing

def _pattern(shifts, N: int, device) -> torch.Tensor:
    """Bits per step [N] int64 on `device` (shifts repeated), built there from scalars: no host -> device copy (such
    uploads are not ordered after queued kernels on this ROCm build under load; LayerJob docstring)."""
    r = torch.arange(N, device=device) % len(shifts)
    pat = torch.zeros(N, dtype=torch.int64, device=device)
    for j, b in enumerate(shifts):
        pat += (r == j).long() * int(b)
    return pat


def pack_states(states: torch.Tensor, shifts, chunk: int = 1 << 16) -> torch.Tensor:
    """states [tiles, N] -> int32 words [tiles, bits per tile / 32] (t4_tcq.pack_states)."""
    if states.shape[0] > chunk:
        return torch.cat([pack_states(states[i:i + chunk], shifts, chunk) for i in range(0, states.shape[0], chunk)])
    T, N = states.shape
    dev = states.device
    pat = _pattern(shifts, N, dev)
    tot = int(pat.sum())
    if tot % 32:
        raise ValueError(f"{tot} bits per tile is not a whole number of words")
    sh = pat[:, None] - 1 - torch.arange(int(pat.max()), device=dev)[None, :]  # [N, bmax], MSB first
    valid = sh >= 0
    c = states & ((1 << pat) - 1)[None, :]
    bits = ((c[:, :, None] >> sh.clamp(min=0)[None]) & 1)[:, valid]  # [T, tot], t-major then bit
    w = (bits.view(T, tot // 32, 32) << torch.arange(31, -1, -1, device=dev)).sum(-1)
    return (w - ((w >> 31) << 32)).to(torch.int32)


def unpack_states(words: torch.Tensor, L: int, shifts, N: int, chunk: int = 1 << 15) -> torch.Tensor:
    """int32 words [tiles, WPT] -> states [tiles, N] (t4_tcq.unpack_states): reads nothing but the words."""
    if words.shape[0] > chunk:
        return torch.cat([unpack_states(words[i:i + chunk], L, shifts, N, chunk)
                          for i in range(0, words.shape[0], chunk)])
    T, wpt = words.shape
    dev = words.device
    u = words.long() & 0xFFFFFFFF
    bits = ((u[:, :, None] >> torch.arange(31, -1, -1, device=dev)) & 1).reshape(T, wpt * 32)
    tot = wpt * 32
    pos = torch.cumsum(_pattern(shifts, N, dev), 0)
    idx = (pos[:, None] - L + torch.arange(L, device=dev)[None, :]) % tot
    weights = 1 << torch.arange(L - 1, -1, -1, device=dev)
    return (bits[:, idx] * weights).sum(-1)


def decode(words: torch.Tensor, scale: torch.Tensor, sp: dict, LUT: torch.Tensor) -> torch.Tensor:
    """Stored words [n/tc, E*R/tr, WPT] + fp16 scales [E, R] -> rotated-domain weights W' [E, R, n] (float32)."""
    nb, tiles, _ = words.shape
    E, R = scale.shape
    tr, tc = sp["tr"], sp["tc"]
    st = unpack_states(words.reshape(-1, words.shape[-1]), sp["L"], sp["b"], sp["N"])
    vals = LUT[st].reshape(nb, E, R // tr, tr, tc).permute(1, 2, 3, 0, 4).reshape(E, R, nb * tc)
    return vals * scale.float()[:, :, None]


# ============================================================================================ Hessians

@dataclass
class LayerHessians:
    """Routed-input second moments of one input kind ("gate_up" or "down") of one MoE layer, for experts `ids`,
    in T4's file semantics: H fp16 [n_ids, n, n] = (S / count) / scale, scale fp32 [n_ids], count int64 [n_ids]."""
    kind: str
    ids: list
    H: torch.Tensor
    scale: torch.Tensor
    count: torch.Tensor


_SYM: dict = {}


@torch.no_grad()
def gemm_symmetric(n: int, device) -> bool:
    """Whether this device's float32 addmm leaves S += X^T X bitwise symmetric at size n (tested once per device and
    size on random data, token counts 37 / 300 / 2311, two accumulations each; a second try before concluding "no",
    so that a fault during the test cannot switch the guard off): the precondition of
    HessianAccumulator.check_symmetric."""
    key = (n, str(torch.device(device)))
    if key not in _SYM:
        def one(seed):
            g = torch.Generator(device=device).manual_seed(seed)
            for k in (37, 300, 2311):
                X = torch.randn(k, n, generator=g, device=device).to(torch.bfloat16).float()
                S = torch.zeros(n, n, device=device)
                S.addmm_(X.t(), X)
                S.addmm_(X.t(), X)
                if not torch.equal(S, S.t()):
                    return False
            return True
        _SYM[key] = one(1000 + n) or one(2000 + n)
    return _SYM[key]


def silu_glu(inter: int, act=F.silu):
    """The down input of a gated expert from its gate/up output h [n, 2I]: act(gate) * up (Qwen3, T4's recompute)."""
    return lambda h: act(h[:, :inter]) * h[:, inter:]


class HessianAccumulator:
    """Per-expert E[x x^T] of the gate/up input and of the down input for experts [e0, e1) of one MoE layer.

    add(x [T, H] bf16, idx [T, k] int64, wts [T, k] or None, W_gu [E or e1-e0, 2I, H] bf16) once per forward
    (window), in order; the forwards are buffered and folded in every `flush` forwards and at the n_windows-th
    (agents/quality/t4/hessians.py HessHook, exactly: the same concatenation, stable sort by expert, float32 addmm
    per expert, down input recomputed with F.linear on the expert's own bf16 weights, made contiguous so that the
    GEMM does not depend on how the caller's weight tensor is strided). `w_offset` = the expert id of W_gu[0] (0 when
    the whole layer's weights are passed). Peak extra memory: the float32 sums of the chunk,
    (e1 - e0) * (H^2 + I^2) * 4 bytes, plus `flush` forwards of inputs.
    check(tags, rows, x, idx, wts, inputs), optional: called on every concatenated flush group before it is folded in
    (tags = the `tag` of each add, rows = its token count, inputs = the buffered (x, idx, wts) of each add); LayerJob
    uses it to verify replayed inputs (raises ReplayCorrupt, whose .bad lists the positions that differ).
    reload(tag) -> (x, idx, wts), optional: with it, the windows a check found bad are fetched again and the group is
    concatenated and checked again (at most REPLAY_TRIES times) instead of failing.
    gu_bias, optional: the layer's gate/up expert biases [E, 2I] (gpt-oss; indexed by the absolute expert id), added
    inside the down-input recompute (F.linear(X, W_e, b_e)) before glu, as the experts add them before their gate.
    None (every architecture without expert biases) is the unchanged computation."""

    def __init__(self, n_experts: int, hidden: int, inter: int, device, e0: int = 0, e1: int | None = None,
                 flush: int = 32, weighted: bool = False, n_windows: int = 0, glu=None, check=None, reload=None,
                 gu_bias: torch.Tensor | None = None):
        self.E, self.Hd, self.I = n_experts, hidden, inter
        self.gu_bias = gu_bias
        self.check, self.reload = check, reload
        self.e0, self.e1 = e0, n_experts if e1 is None else e1
        self.flush_every, self.weighted, self.n_windows = flush, weighted, n_windows
        self.glu = glu or silu_glu(inter)
        n = self.e1 - self.e0
        self.G = torch.zeros(n, hidden, hidden, device=device)
        self.D = torch.zeros(n, inter, inter, device=device)
        self.cnt = torch.zeros(n_experts, dtype=torch.int64, device=device)
        self.buf: list = []
        self.seen = 0
        self.flushes = 0
        self.repairs = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor | None, W_gu: torch.Tensor,
            w_offset: int = 0, tag=None) -> None:
        self.buf.append((x, idx, wts if self.weighted else None, tag))
        self.seen += 1
        if len(self.buf) >= self.flush_every or self.seen == self.n_windows:
            self.flush(W_gu, w_offset)

    @torch.no_grad()
    def flush(self, W_gu: torch.Tensor, w_offset: int = 0) -> None:
        if not self.buf:
            return
        for attempt in range(REPLAY_TRIES):
            x = torch.cat([b[0] for b in self.buf])
            idx = torch.cat([b[1] for b in self.buf])
            wts = torch.cat([b[2] for b in self.buf]) if self.weighted else None
            if self.check is None:
                break
            try:
                self.check([b[3] for b in self.buf], [b[0].shape[0] for b in self.buf], x, idx, wts,
                           [b[:3] for b in self.buf])
                break
            except ReplayCorrupt as err:
                if self.reload is None or not err.bad or attempt == REPLAY_TRIES - 1:
                    raise
                del x, idx, wts
                for p in err.bad:  # fetch the bad windows again, then concatenate and check the group again
                    tag = self.buf[p][3]
                    xx, ii, ww = self.reload(tag)
                    self.buf[p] = (xx, ii, ww if self.weighted else None, tag)
                self.repairs += 1
        self.buf = []
        self.flushes += 1
        flat = idx.reshape(-1)
        k = idx.shape[1]
        order = torch.argsort(flat, stable=True)
        gw = wts.reshape(-1)[order].float() if wts is not None else None
        counts = torch.bincount(flat, minlength=self.E)
        self.cnt += counts
        tok_all = order // k
        start = 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            s0 = start
            start += c
            if not (self.e0 <= e < self.e1):
                continue
            tok = tok_all[s0:s0 + c]
            g = None if gw is None else gw[s0:s0 + c, None]
            X = x[tok]
            Xf = X.float()
            j = e - self.e0
            self.G[j].addmm_((Xf if g is None else Xf * g * g).t(), Xf)
            if self.gu_bias is None:
                a = self.glu(F.linear(X, W_gu[e - w_offset].contiguous())).float()
            else:
                a = self.glu(F.linear(X, W_gu[e - w_offset].contiguous(), self.gu_bias[e])).float()
            self.D[j].addmm_((a if g is None else a * g * g).t(), a)

    @torch.no_grad()
    def check_symmetric(self) -> None:
        """Raise AccumulationFlake unless every expert's float32 sums are bitwise symmetric. Sums of X^T X computed
        right are (each output element is the same dot product in the same order as its mirror; measured on MI250
        for K = 290 .. 4096 tokens, and every T4 hess-t2v1 matrix is symmetric), while a GEMM fault (e.g. a race in
        one output tile) breaks the symmetry. Checked only where gemm_symmetric() confirmed the property on this
        device and size, and only for unweighted sums ((X g^2)^T X is not symmetric in rounding)."""
        if self.weighted or self.G is None:
            return
        for name, M, n in (("gate_up", self.G, self.Hd), ("down", self.D, self.I)):
            if not gemm_symmetric(n, M.device):
                continue
            bad = []
            for a in range(0, M.shape[0], 8):  # groups of 8: bounded temporary
                m = M[a:a + 8]
                bad += [a + j for j in (m != m.transpose(1, 2)).flatten(1).any(1).nonzero().flatten().tolist()]
            if bad:
                raise AccumulationFlake(f"{name} Hessian sums of experts {[self.e0 + j for j in bad][:16]} are not "
                                        "exactly symmetric (GEMM fault)")

    @torch.no_grad()
    def result(self) -> tuple[LayerHessians, LayerHessians]:
        """(gate_up, down) in T4's file semantics, on the accumulator's device. The float32 sums are checked for
        exact symmetry (check_symmetric), then normalised in place (elementwise division gives the same bits in
        place or not) and freed."""
        if self.buf:
            raise RuntimeError(f"{len(self.buf)} forwards not folded in (n_windows {self.n_windows}, seen "
                               f"{self.seen})")
        self.check_symmetric()
        ids = list(range(self.e0, self.e1))
        cnt = self.cnt[self.e0:self.e1]
        out = []
        for key in ("gate_up", "down"):
            M = self.G if key == "gate_up" else self.D
            M.div_(cnt.clamp(min=1).float()[:, None, None])
            scale = M.diagonal(dim1=1, dim2=2).mean(1)
            M.div_(scale.clamp(min=1e-30)[:, None, None])
            M[cnt == 0] = 0
            H16 = M.half()
            if key == "gate_up":
                self.G = None
            else:
                self.D = None
            del M
            if not bool(torch.isfinite(H16).all()):
                raise RuntimeError(f"{key}: fp16 overflow")
            out.append(LayerHessians(key, ids, H16, scale, cnt.clone()))
        return out[0], out[1]


CHOL_STATS = {"batches": 0, "disagreeing_batches": 0, "extra_runs": 0, "matrices_fixed": 0}
CHOL_MAX_RUNS = 8


def cholesky_voted(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.cholesky_ex of a float64 batch, run until every matrix's factor came out bit-identical twice.

    Why: on MI250 (ROCm torch 2.11, rocSOLVER batched potrf, 2048 x 2048 float64) the factor of one matrix of a
    batch occasionally differs in the last bits from run to run while another process uses the same GPU (measured:
    agents/qint/qw3/det_chol.py, 2 flakes in 12 x 16 batches; never alone; solve_triangular, the rotation GEMMs,
    the Hessian GEMMs, Viterbi and LDLQ repeat exactly). Two runs that agree are the reproducible value (the T4
    codes are reproduced bit for bit through this function); a matrix whose two runs disagree is re-run until a
    value repeats. Cost: one extra Cholesky per batch (measured in results/qint/qw3/).
    Results are compared bitwise (NaN-safe, -0.0 != +0.0), and a matrix whose factorization failed (info != 0: the
    caller adds damping and factors it again, its L is discarded) only needs the same info: its partial factor may
    hold NaN or leftovers that never repeat (the float comparison of the previous version could then raise
    NoMajority instead of letting inv_chol_upper's damping escalation run). Every earlier run of a disagreeing
    matrix stays a candidate (the previous version compared with the first and the last two runs only)."""
    def same(La, ia, Lb, ib):
        return (ia == ib) & ((_bits(La) == _bits(Lb)).flatten(1).all(1) | (ia != 0))

    L1, i1 = torch.linalg.cholesky_ex(A)
    L2, i2 = torch.linalg.cholesky_ex(A)
    CHOL_STATS["batches"] += 1
    CHOL_STATS["extra_runs"] += 1
    agree = same(L1, i1, L2, i2)
    if bool(agree.all()):
        return L1, i1
    CHOL_STATS["disagreeing_batches"] += 1
    todo = (~agree).nonzero().flatten().tolist()
    cands = {j: [(L1[j].clone(), i1[j].clone()), (L2[j].clone(), i2[j].clone())] for j in todo}
    for _ in range(CHOL_MAX_RUNS - 2):
        L3, i3 = torch.linalg.cholesky_ex(A)
        CHOL_STATS["extra_runs"] += 1
        left = []
        for j in todo:
            hit = any(bool(same(L3[j:j + 1], i3[j:j + 1], c[None], ci[None])[0]) for c, ci in cands[j])
            if hit:
                L1[j], i1[j] = L3[j], i3[j]
                CHOL_STATS["matrices_fixed"] += 1
            else:
                cands[j].append((L3[j].clone(), i3[j].clone()))
                left.append(j)
        todo = left
        if not todo:
            return L1, i1
    raise NoMajority(f"cholesky_ex gave no repeated factor in {CHOL_MAX_RUNS} runs for matrices {todo}")


def trsm_size(n: int) -> int:
    """The size inv_upper solves an n x n triangular inverse at: n rounded up to a multiple of TRSM_ALIGN."""
    return -(-n // TRSM_ALIGN) * TRSM_ALIGN


def inv_upper(R: torch.Tensor) -> torch.Tensor:
    """R [b, n, n] upper triangular, invertible -> R^-1 [b, n, n] (same dtype), one batched triangular solve against
    the identity at size m = trsm_size(n). For m > n, R is padded with an identity block: blockdiag(R, I)^-1 =
    blockdiag(R^-1, I), and the zero coupling block makes the leading n x n block of the padded solve an ordinary
    triangular solve of R (residual max |R U - I| of the same order as unpadded: results/qint/qw3/fix/).
    Why: on MI250 / ROCm torch 2.11, rocBLAS trsm fails with HIPBLAS_STATUS_ALLOC_FAILED for every float64 size
    n >= 2000 that is not a multiple of 128 (float32: n >= 2848), at batch 1, 2 and 8 alike, while every multiple of
    128 up to 8192 works (agents/qint/qw3/trsm_probe.py; results/qint/qw3/fix/trsm_sweep.json): gpt-oss's 2880 was
    the target size that crashed. Sizes that already are multiples of 128 (Qwen3-30B / 235B: 768, 1536, 2048, 4096)
    take the unpadded path, the same call as before (T4's codes are unchanged)."""
    b, n, _ = R.shape
    m = trsm_size(n)
    eye = torch.eye(m, dtype=R.dtype, device=R.device).expand(b, m, m).contiguous()
    if m == n:
        return torch.linalg.solve_triangular(R, eye, upper=True)
    Rp = eye.clone()
    Rp[:, :n, :n] = R
    return torch.linalg.solve_triangular(Rp, eye, upper=True)[:, :n, :n]


def inv_chol_upper(H: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
    """H [E, n, n] SPD -> U [E, n, n] float32, upper, H^-1 = U^T U, in float64 `chunk` matrices at a time
    (t4_quant.inv_chol_upper_batched): with the reversal P, P H P = L L^T, H = R R^T, R = P L P, U = R^-1.
    Matrices that fail to factor get more damping. The Cholesky is cholesky_voted (reproducible), the inverse
    inv_upper (any n)."""
    E, n, _ = H.shape
    chunk = chunk or chol_chunk(n)
    eye = torch.eye(n, dtype=torch.float64, device=H.device)
    U = torch.empty(E, n, n, dtype=torch.float32, device=H.device)
    for c0 in range(0, E, chunk):
        Hc = H[c0:c0 + chunk].double()
        md = Hc.diagonal(dim1=1, dim2=2).mean(1)
        todo = torch.arange(Hc.shape[0], device=H.device)
        for extra in (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
            Hd = Hc[todo] + (extra * md[todo])[:, None, None] * eye
            Lc, info = cholesky_voted(Hd.flip(1, 2))
            ok = info == 0
            if bool(ok.any()):
                R = Lc[ok].flip(1, 2)
                U[c0 + todo[ok]] = inv_upper(R).float()
            todo = todo[~ok]
            if todo.numel() == 0:
                break
        if todo.numel():
            raise RuntimeError(f"{todo.numel()} Hessians failed to factor")
    return U


FACTOR_VOTES = 2  # end-to-end computations of every Cholesky batch's factors (hessian_factors); 1 = once
FACTOR_MAX_RUNS = 5
FACTOR_STATS = {"batches": 0, "extra_runs": 0, "disagreeing_batches": 0, "experts_fixed": 0}


def _factor_batch(hs: LayerHessians, rot, damp: float, min_tokens: int, c0: int, c1: int, chunk: int,
                  counts: list) -> torch.Tensor:
    """U [c1 - c0, n, n] of experts [c0, c1): t4_quant.hessians_file's rotation / symmetrisation / damping, then
    inv_chol_upper (voted Cholesky, triangular inverse)."""
    n = hs.H.shape[-1]
    dev = hs.H.device
    Hc = torch.zeros(c1 - c0, n, n, dtype=torch.float32, device=dev)
    for i in range(c0, c1):
        if counts[i] < min_tokens:
            Hc[i - c0].diagonal().fill_(1.0)
            continue
        Hi = hs.H[i].to(dev).float()
        Hi = rotate_in(rotate_in(Hi, *rot).t().contiguous(), *rot)  # R^T H R with R = D H_b
        Hi = (Hi + Hi.t()) / 2
        Hi.diagonal().add_(damp * Hi.diagonal().mean().clamp(min=1e-12))
        Hc[i - c0] = Hi
    return inv_chol_upper(Hc, chunk)


@torch.no_grad()
def hessian_factors(hs: LayerHessians, rot, damp: float, min_tokens: int = 8,
                    chunk: int | None = None, votes: int | None = None) -> torch.Tensor:
    """U factors [E, n, n] float32 of the rotated, symmetrised, damped Hessians (t4_quant.hessians_file + the
    batched Cholesky). Built `chunk` experts at a time, so the float32 H of the whole stack never exists.
    votes (default FACTOR_VOTES = 2): every batch is computed that many times end to end and compared bitwise per
    expert; an expert whose runs disagree is settled by more runs until one of its results has come out twice
    (NoMajority after FACTOR_MAX_RUNS). Why: a whole-layer job under concurrent GPU load once gave different
    gate/up factors although its Cholesky was voted (results/qint/verify/qw3/r2/, probe_adv_2); the rotation GEMMs,
    the damping and the triangular inverse had no check. Counters: FACTOR_STATS."""
    votes = FACTOR_VOTES if votes is None else votes
    E, n = hs.H.shape[0], hs.H.shape[-1]
    chunk = chunk or chol_chunk(n)
    dev = hs.H.device
    U = torch.empty(E, n, n, dtype=torch.float32, device=dev)
    counts = hs.count.tolist()
    for c0 in range(0, E, chunk):
        c1 = min(E, c0 + chunk)
        runs = [_factor_batch(hs, rot, damp, min_tokens, c0, c1, chunk, counts)]
        if votes >= 2:
            FACTOR_STATS["batches"] += 1
            runs.append(_factor_batch(hs, rot, damp, min_tokens, c0, c1, chunk, counts))
            FACTOR_STATS["extra_runs"] += 1
            same = [torch.equal(_bits(runs[0][j]), _bits(runs[1][j])) for j in range(c1 - c0)]
            if not all(same):
                FACTOR_STATS["disagreeing_batches"] += 1
                todo = [j for j, s in enumerate(same) if not s]
                while todo:
                    if len(runs) >= FACTOR_MAX_RUNS:
                        raise NoMajority(f"hessian_factors: no repeated factor for experts "
                                         f"{[hs.ids[c0 + j] for j in todo]} in {len(runs)} runs")
                    runs.append(_factor_batch(hs, rot, damp, min_tokens, c0, c1, chunk, counts))
                    FACTOR_STATS["extra_runs"] += 1
                    left = []
                    for j in todo:
                        hit = next((k for k in range(len(runs) - 1)
                                    if torch.equal(_bits(runs[k][j]), _bits(runs[-1][j]))), None)
                        if hit is None:
                            left.append(j)
                        else:
                            runs[0][j] = runs[hit][j]
                            FACTOR_STATS["experts_fixed"] += 1
                    todo = left
        U[c0:c1] = runs[0]
        del runs
    return U


# ============================================================================================ BlockLDLQ

@torch.no_grad()
def ldlq_encode(Wr: torch.Tensor, U: torch.Tensor | None, sp: dict, LUT: torch.Tensor, scale_mult: float = 1.0,
                seq_chunk: int = SEQ_CHUNK, force_ref: bool = False, group: int = EXPERT_GROUP):
    """Wr [E, R, n] float32 rotated weights (not modified), U [E, n, n] or None (no feedback). Returns
    (s [E, R] fp16 row scales, words int32 [n/tc, E * R/tr, WPT], err2 [E], w2 [E]): per expert, float64 on Wr's
    device, the sum of squared errors ||W'_e - s Q_e||^2 and of squares ||W'_e||^2 (bookkeeping only).
    t4_tcq.quantize_stack operation for operation, except that each column block's states are packed at once and
    that the batched ops (row scales, block inverses, feedback bmm's) run on `group` experts at a time (module
    docstring: an expert's bits then do not depend on the stack size; group 0 = whole stack = T4's batches)."""
    E, R, n = Wr.shape
    L, tr, tc, shifts = sp["L"], sp["tr"], sp["tc"], sp["b"]
    if R % tr or n % tc:
        raise ValueError(f"{sp['spec']}: {R}x{n} does not tile into {tr}x{tc}")
    grp = expert_groups(E, group)
    s = torch.cat([(Wr[a:c].pow(2).mean(-1, keepdim=True).sqrt() * scale_mult).clamp(min=1e-12, max=FP16_MAX).half()
                   for a, c in grp])
    # fp16 underflow guard (not in T4): a row whose RMS is below ~3e-8 (dead neurons: 30% of Qwen3-235B layer 0's
    # gate rows) would get s = 0, W'/s = inf/NaN, and NaN path costs corrupt the other 15 rows of its tiles. Such
    # rows get the smallest fp16 subnormal instead; rows with a nonzero fp16 scale are unchanged (no Qwen3-30B row
    # is affected, so T4's codes are still reproduced).
    s = torch.where(s == 0, torch.full_like(s, FP16_TINY), s).float()
    Wn = Wr / s
    if not bool(torch.isfinite(Wn).all()):
        raise RuntimeError(f"{sp['spec']}: non-finite normalised weights (non-finite input?)")
    nb = n // tc
    Minv = []  # per group: the 16 x 16 diagonal-block inverses [c - a, nb, tc, tc], as solve_triangular returns them
    if U is not None:
        ar = torch.arange(nb, device=U.device)
        for a, c in grp:
            Ud = U[a:c].view(c - a, nb, tc, nb, tc)[:, ar, :, ar, :].permute(1, 0, 2, 3).contiguous()
            Minv.append(torch.linalg.solve_triangular(
                Ud, torch.eye(tc, device=U.device).expand(c - a, nb, tc, tc).contiguous(), upper=True))
            del Ud
    words = torch.empty(nb, E * (R // tr), sp["bits_per_tile"] // 32, dtype=torch.int32, device=Wr.device)
    err2 = torch.zeros(E, dtype=torch.float64, device=Wr.device)
    w2 = torch.zeros(E, dtype=torch.float64, device=Wr.device)
    for b in range(nb):
        j0, j1 = b * tc, (b + 1) * tc
        blk = Wn[:, :, j0:j1]
        seqs = blk.reshape(E, R // tr, tr, tc).reshape(-1, tr * tc)
        q = torch.empty_like(seqs)
        for c0 in range(0, seqs.shape[0], seq_chunk):  # per sequence: independent of how many are batched
            st = tcq_seq(seqs[c0:c0 + seq_chunk].contiguous(), LUT, L, shifts, force_ref=force_ref)
            if b == 0 and c0 == 0 and not check_circular(st[:32], L, shifts):
                raise RuntimeError(f"{sp['spec']}: Viterbi states are not a circular bitstream")
            words[b, c0:c0 + st.shape[0]] = pack_states(st, shifts)
            q[c0:c0 + seq_chunk] = LUT[st].reshape(st.shape[0], -1)
        qb = q.view(E, R // tr, tr, tc).reshape(E, R, tc)
        wb = Wr[:, :, j0:j1]
        e2b, w2b = (wb - qb * s).pow(2), wb.pow(2)  # bookkeeping only (elementwise, then per-group sums)
        err2 += torch.cat([e2b[a:c].sum((1, 2), dtype=torch.float64) for a, c in grp])
        w2 += torch.cat([w2b[a:c].sum((1, 2), dtype=torch.float64) for a, c in grp])
        if U is not None and j1 < n:
            d = blk - qb
            for gi, (a, c) in enumerate(grp):
                err = torch.bmm(d[a:c], Minv[gi][:, b])
                Wn[a:c, :, j1:].sub_(torch.bmm(err, U[a:c, j0:j1, j1:]))
    return s.view(E, R).half(), words, err2, w2


ENCODE_VOTES = 2  # encodes per stack (ldlq_encode_voted); 1 = once
ENCODE_MAX_RUNS = 5
ENCODE_STATS = {"stacks": 0, "extra_runs": 0, "disagreeing_stacks": 0, "experts_fixed": 0, "traceback_retries": 0}


def ldlq_encode_voted(Wr: torch.Tensor | None, U: torch.Tensor | None, sp: dict, LUT: torch.Tensor,
                      votes: int = ENCODE_VOTES, force_ref: bool = False, group: int = EXPERT_GROUP, rotate=None):
    """ldlq_encode, run twice (votes = 2) and compared expert by expert (words and fp16 row scales, bitwise); an
    expert on which the runs disagree is settled by more runs of the whole stack until one of its results has come
    out twice (the groups stay the same, so an expert's result can be taken from any run; NoMajority after
    ENCODE_MAX_RUNS). votes = 1: one run. rotate, optional: a function returning the rotated stack W', called
    for every run instead of passing Wr (so the input rotation is inside the vote, and only one W' is alive).
    Why: a transient wrong result in the encode was seen once on MI250 (Qwen3-235B layer 0 gate, unchunked: 4 tiles
    of 4 experts from column block 230 on, while six repeats and the chunked run agreed; results/qint/qw3/fix/).
    Returns ldlq_encode's (s, words, err2, w2). Counters: ENCODE_STATS."""
    def run():
        return ldlq_encode(Wr if rotate is None else rotate(), U, sp, LUT, force_ref=force_ref, group=group)

    first = run()
    if votes < 2:
        return first
    ENCODE_STATS["stacks"] += 1
    E = first[0].shape[0]
    nb, wpt = first[1].shape[0], first[1].shape[-1]

    def key(res, e):  # one expert's result, as raw integers
        return res[0][e].view(torch.int16), res[1].view(nb, E, -1, wpt)[:, e]

    def same(a, b, e):
        ka, kb = key(a, e), key(b, e)
        return torch.equal(ka[0], kb[0]) and torch.equal(ka[1], kb[1])

    runs = [first, run()]
    ENCODE_STATS["extra_runs"] += 1
    s0, s1 = first[0].view(torch.int16), runs[1][0].view(torch.int16)
    w0, w1 = first[1].view(nb, E, -1, wpt), runs[1][1].view(nb, E, -1, wpt)
    agree = (s0 == s1).all(1) & (w0 == w1).permute(1, 0, 2, 3).reshape(E, -1).all(1)
    if bool(agree.all()):
        return first
    ENCODE_STATS["disagreeing_stacks"] += 1
    todo = (~agree).nonzero().flatten().tolist()
    choice = {}
    while todo and len(runs) < ENCODE_MAX_RUNS:
        runs.append(run())
        ENCODE_STATS["extra_runs"] += 1
        left = []
        for e in todo:
            hit = next((k for k in range(len(runs) - 1) if same(runs[k], runs[-1], e)), None)
            if hit is None:
                left.append(e)
            else:
                choice[e] = hit
        todo = left
    if todo:
        raise NoMajority(f"{sp['spec']}: no repeated encode for experts {todo} in {len(runs)} runs")
    s, words, err2, w2 = (t.clone() for t in first)
    wv = words.view(nb, E, -1, wpt)
    for e, k in choice.items():
        if k:
            s[e], err2[e], w2[e] = runs[k][0][e], runs[k][2][e], runs[k][3][e]
            wv[:, e] = runs[k][1].view(nb, E, -1, wpt)[:, e]
        ENCODE_STATS["experts_fixed"] += 1
    return s, words, err2, w2


def matrix_bits(sp: dict, out: int, n_in: int, users: int) -> float:
    """Stored bits of one expert matrix: codes + fp16 row scales + its share of the LUT (t4_quant.matrix_bits)."""
    codes = out * n_in * sp["bpw_codes"]
    if abs(codes - round(codes)) > 1e-6:
        raise ValueError(f"{sp['spec']}: {out}x{n_in} codes are not a whole number of bits")
    return round(codes) + 16 * out + table_bits(sp) / users


_ROT: dict = {}     # (n, seed, device) -> (signs, b); the signs are n floats per layer
_ROT_HB: dict = {}  # (b, device) -> Hadamard block; it does not depend on the seed, so one copy serves every layer


def checked_rotation(n: int, seed: int, device):
    """behemoe.quantizers.rotation(n, seed, device) with its device tensors uploaded by to_device_verified from the
    host-built values (rotation() builds the signs and the Hadamard block on the host and copies them unchecked;
    the block is lru-cached on the device). The same values, hence the same bits. The signs are cached per
    (n, seed, device) and the block per (b, device), so GPU memory does not grow with the layer count."""
    dev = str(torch.device(device))
    key = (n, seed, dev)
    if key not in _ROT:
        signs, hb, b = rotation(n, seed, "cpu")
        on_gpu = torch.device(device).type != "cpu"
        if on_gpu:
            signs = to_device_verified(signs[None], device)[0]
        if (b, dev) not in _ROT_HB:
            _ROT_HB[(b, dev)] = to_device_verified(hb[None], device)[0] if on_gpu else hb
        _ROT[key] = (signs, b)
    signs, b = _ROT[key]
    return signs, _ROT_HB[(b, dev)], b


def rotate_stack(W: torch.Tensor, rot, device, group: int = EXPERT_GROUP) -> torch.Tensor:
    """W [E, R, n] (bf16, any device) -> W' = W D H_b [E, R, n] float32 on `device`, one rotation GEMM per expert
    group (an expert's bits do not depend on E; group 0 = one GEMM over the whole stack, as T4)."""
    E, R, n = W.shape
    grp = expert_groups(E, group)
    if len(grp) == 1:
        return rotate_in(W.to(device).contiguous().float().reshape(E * R, n), *rot).view(E, R, n)
    Wr = torch.empty(E, R, n, dtype=torch.float32, device=device)
    for a, c in grp:
        Wr[a:c] = rotate_in(W[a:c].to(device).contiguous().float().reshape((c - a) * R, n), *rot).view(c - a, R, n)
    return Wr


@torch.no_grad()
def encode_matrix(W: torch.Tensor, U: torch.Tensor | None, spec: str, layer: int, rot, device=None,
                  lut_dir: str | None = None, users: int = 1, matrix: str = "", ids=None,
                  force_ref: bool = False, group: int = EXPERT_GROUP, votes: int = ENCODE_VOTES) -> dict:
    """One expert stack W [E, R, n] (bf16, any device) -> T4's saved-code record (tensors on the CPU), encoded with
    ldlq_encode_voted (the input rotation inside each voted run). Bookkeeping (not stored): _err2 / _w2 the stack's
    squared error / squares (floats), _err2_e / _w2_e per expert (float64)."""
    dev = device or (U.device if U is not None else W.device)
    sp = parse_spec(spec)
    LUT, lsha = load_lut(sp, dev, lut_dir)
    E, R, n = W.shape
    if W.device != torch.device(dev):  # one synchronized, checked upload (no copies under queued kernels)
        W = to_device_verified(W, dev)
    s, words, e2, w2 = ldlq_encode_voted(None, U, sp, LUT, votes, force_ref=force_ref, group=group,
                                         rotate=lambda: rotate_stack(W, rot, dev, group))
    e2, w2 = to_host_checked(e2), to_host_checked(w2)  # fingerprint-checked device -> host copies
    return {"spec": spec, "layer": layer, "matrix": matrix, "experts": list(range(E)) if ids is None else list(ids),
            "rot": "layer", "words": to_host_checked(words), "scale": to_host_checked(s), "shape": (R, n),
            "lut_sha256": lsha,
            "lut_share_bits": table_bits(sp) / users, "bits_per_expert": matrix_bits(sp, R, n, users),
            "_err2": float(e2.sum()), "_w2": float(w2.sum()), "_err2_e": e2, "_w2_e": w2}


@torch.no_grad()
def encode_layer(W_gate_up: torch.Tensor, W_down: torch.Tensor, H_gu: LayerHessians, H_dn: LayerHessians,
                 plan_entry: dict, layer: int, users: dict | None = None, device=None, lut_dir: str | None = None,
                 timings: dict | None = None, group: int = EXPERT_GROUP, votes: int = ENCODE_VOTES) -> dict:
    """Encode the experts of one MoE layer (or a chunk of its experts: H_gu.ids). group: experts per batched op
    (module docstring; 0 = T4's whole-stack batches); votes: encodes per stack (ldlq_encode_voted).

    W_gate_up [E, 2I, H] (gate rows first, transformers' fused layout), W_down [E, H, I], bf16 on any device;
    H_gu / H_dn: their LayerHessians (HessianAccumulator.result()); plan_entry: {"gu": spec, "dn": spec, "damp",
    "min_tokens"} (behemoe.tcq_plan.entry); users: {lut_key: expert matrices of the model decoding with that LUT}
    (behemoe.tcq_plan.validate()["users"]), for the bit account. Returns {"gate"|"up"|"down": record}: words int32
    [n/tc, E * R/tr, WPT] + fp16 row scales [E, R] + T4's metadata, on the CPU. Gate and up are separate stacks
    (as T4 saved them) sharing the gate/up input's Hessian factors."""
    dev = device or H_gu.H.device
    I = W_down.shape[2]
    if W_gate_up.shape[1] != 2 * I or W_gate_up.shape[0] != len(H_gu.ids) or W_down.shape[0] != len(H_dn.ids):
        raise ValueError(f"shapes gate_up {tuple(W_gate_up.shape)} down {tuple(W_down.shape)} vs {len(H_gu.ids)} "
                         "experts")
    damp, min_tok = float(plan_entry.get("damp", 0.3)), int(plan_entry.get("min_tokens", 8))
    out = {}
    tm = timings if timings is not None else {}
    for hs, mats in ((H_gu, (("gate", W_gate_up[:, :I], plan_entry["gu"]), ("up", W_gate_up[:, I:],
                                                                           plan_entry["gu"]))),
                     (H_dn, (("down", W_down, plan_entry["dn"]),))):
        n = hs.H.shape[-1]
        rot = checked_rotation(n, layer * ROT_SEED, dev)
        t0 = time.time()
        U = hessian_factors(hs, rot, damp, min_tok)
        _sync(dev)
        tm["factor_s"] = tm.get("factor_s", 0.0) + time.time() - t0
        for m, W, spec in mats:
            t0 = time.time()
            sp = parse_spec(spec)
            u = (users or {}).get(sp["lut_key"], 1)
            out[m] = encode_matrix(W, U, spec, layer, rot, dev, lut_dir, u, m, hs.ids, group=group, votes=votes)
            tm[f"encode_{m}_s"] = tm.get(f"encode_{m}_s", 0.0) + time.time() - t0
        del U
    return out


def concat_records(parts: list[dict]) -> dict:
    """Records of consecutive expert chunks of one matrix -> one record (words along the expert axis)."""
    if len(parts) == 1:
        return parts[0]
    r = dict(parts[0])
    sp = parse_spec(r["spec"])
    R, n = r["shape"]
    nb, per = n // sp["tc"], R // sp["tr"]
    r["words"] = torch.cat([p["words"].view(nb, len(p["experts"]), per, -1) for p in parts], 1).view(nb, -1,
                                                                                                    sp["bits_per_tile"] // 32)
    r["scale"] = torch.cat([p["scale"] for p in parts])
    r["experts"] = [e for p in parts for e in p["experts"]]
    if all("_err2_e" in p for p in parts):  # the same totals as one record over all experts
        r["_err2_e"] = torch.cat([p["_err2_e"] for p in parts])
        r["_w2_e"] = torch.cat([p["_w2_e"] for p in parts])
        r["_err2"], r["_w2"] = float(r["_err2_e"].sum()), float(r["_w2_e"].sum())
    else:
        r["_err2"] = sum(p["_err2"] for p in parts)
        r["_w2"] = sum(p["_w2"] for p in parts)
    for p in parts[1:]:
        if (p["spec"], p["shape"], p["lut_sha256"], p["bits_per_expert"]) != (r["spec"], r["shape"], r["lut_sha256"],
                                                                                r["bits_per_expert"]):
            raise ValueError("chunks of one matrix disagree on spec / shape / LUT / bits")
    return r


def save_record(rec: dict, out_dir: str) -> str:
    """T4's file layer{L:03d}.{matrix}.pt (tmp + rename); keys starting with '_' are not stored."""
    obj = {k: v for k, v in rec.items() if not k.startswith("_")}
    path = os.path.join(out_dir, f"layer{rec['layer']:03d}.{rec['matrix']}.pt")
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def save_hessians(hs_gu: LayerHessians, hs_dn: LayerHessians, path: str) -> None:
    """agents/quality/t4/hessians.py's file format (for comparisons; the converter does not need it)."""
    obj = {h.kind: {"H": h.H.cpu(), "scale": h.scale.cpu(), "count": h.count.cpu()} for h in (hs_gu, hs_dn)}
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ============================================================================================ one layer, streamed

REPLAY_STATS = {"captured_windows": 0, "capture_retries": 0, "replayed_chunks": 0, "replay_retries": 0,
                "upload_retries": 0, "group_repairs": 0, "download_retries": 0, "finish_retries": 0}
FINISH_PAUSES_S = (30, 120)  # LayerJob: pauses before re-running a chunk's encode that failed (e.g. NoMajority)
FP_CHUNK = 1 << 24  # elements per partial sum of fingerprint() (bounded temporaries)
_FP_W: dict = {}


def release_fp_buffers() -> int:
    """Drop the device position tables of fingerprint() (128 MiB of int64 per device; rebuilt on the next device
    fingerprint). The loader calls it after its verified uploads, so the tables are not counted as part of the dense
    part when the cache is sized (design-quality WP13). Returns the bytes released."""
    n = sum(t.numel() * t.element_size() for t in _FP_W.values())
    _FP_W.clear()
    return n


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The raw integers of a tensor (same element size), so that comparisons are bitwise (NaN-safe)."""
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _fp_positions(device) -> torch.Tensor:
    key = str(torch.device(device))
    if key not in _FP_W:  # built on the device itself (no copy)
        _FP_W[key] = torch.arange(1, FP_CHUNK + 1, dtype=torch.int64, device=device)
    return _FP_W[key]


def _fp_words(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bytes as a flat integer tensor: 4-byte words when the byte count allows (fewer elements to sum),
    otherwise its raw elements. Depends on the bytes only (a misaligned view is copied first)."""
    v = _bits(t).reshape(-1)
    es = v.element_size()
    if es < 4 and (v.numel() * es) % 4 == 0 and v.numel():
        if v.storage_offset() % (4 // es):
            v = v.clone()
        v = v.view(torch.int32)
    return v


def _wrap64(n: int) -> int:
    n %= 1 << 64
    return n - (1 << 64) if n >= 1 << 63 else n


_NP_POS: list = []
_NP_BUF: list = []  # the one int64 work buffer of the CPU fingerprint (FP_CHUNK elements, reused by every call)
_NP_LOCK = threading.Lock()


def fingerprint(t: torch.Tensor) -> torch.Tensor:
    """Exact integer fingerprint of a tensor's bytes, computed where the tensor is: [sum, position-weighted sum] of
    its raw integers (4-byte little-endian words, _fp_words) modulo 2^64 as int64, the weight of word i being i + 1
    (not periodic: a swap of two different words always changes the weighted sum, by (v_i - v_j)(j - i)). The CPU
    and the GPU give the same value for the same bytes (integer sums are exact modulo 2^64 in any order). On the CPU
    it is computed single-threaded with numpy (about 1.3 ms per 4.5 MiB window on the server, also under heavy load;
    torch's CPU thread pool took 216 ms there when the machine was oversubscribed), widening each chunk into ONE
    reused int64 buffer (design-quality amendment 9: a fresh 128 MiB int64 array per chunk stalled under memory
    pressure; same values). Catches changed, lost and moved data. (The previous weights, (i + 1) mod 65521, were 0
    at every 65521st element and blind to swaps 65521 elements apart.)"""
    v = _fp_words(t)
    if v.device.type == "cpu":
        with _NP_LOCK:  # one shared work buffer
            if not _NP_POS:
                _NP_POS.append(np.arange(1, FP_CHUNK + 1, dtype=np.int64))
                _NP_BUF.append(np.empty(FP_CHUNK, dtype=np.int64))
            pos, buf, a_all = _NP_POS[0], _NP_BUF[0], v.numpy()
            s = ws = 0
            for o in range(0, a_all.size, FP_CHUNK):
                n = min(FP_CHUNK, a_all.size - o)
                a = buf[:n]
                a[...] = a_all[o:o + n]  # widened in place (the values astype(np.int64) gives), no allocation
                cs = int(a.sum())
                s += cs
                ws += int(np.dot(a, pos[:n])) + cs * o  # sum_k a_k (o + k + 1)
        return torch.tensor([_wrap64(s), _wrap64(ws)], dtype=torch.int64)
    P = _fp_positions(v.device)
    s = torch.zeros((), dtype=torch.int64, device=v.device)
    ws = torch.zeros((), dtype=torch.int64, device=v.device)
    for o in range(0, v.numel(), FP_CHUNK):
        c = v[o:o + FP_CHUNK].to(torch.int64)
        cs = c.sum()
        s += cs
        ws += (c * P[:c.numel()]).sum() + cs * o  # sum_k c_k (o + k + 1)
    return torch.stack([s, ws])


def window_fingerprint(x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor | None) -> torch.Tensor:
    """[x sum, x weighted, idx sum, idx weighted, wts sum, wts weighted] (zeros for wts None), on x's device."""
    z = torch.zeros(2, dtype=torch.int64, device=x.device)
    return torch.cat([fingerprint(x), fingerprint(idx), z if wts is None else fingerprint(wts)])


def _parts_differ(a: torch.Tensor, b: torch.Tensor) -> str:
    return "+".join(n for k, n in enumerate(("x", "idx", "wts")) if not torch.equal(a[2 * k:2 * k + 2],
                                                                                    b[2 * k:2 * k + 2]))


def to_host_verified(t: torch.Tensor) -> torch.Tensor:
    """t.to("cpu") (pageable), checked by copying it back and comparing bit for bit with t (retried, then refused).
    Kept for callers of the previous version; LayerJob now captures into pinned memory (HostCapture)."""
    for _ in range(REPLAY_TRIES):
        h = t.to("cpu")
        if torch.equal(_bits(h.to(t.device)), _bits(t)):
            return h
        REPLAY_STATS["capture_retries"] += 1
    raise CaptureFailed(f"device -> host copy of a {tuple(t.shape)} tensor does not round-trip")


def to_host_checked(t: torch.Tensor) -> torch.Tensor:
    """t.cpu() checked against the fingerprint of the device original (computed on the device, the copy's on the
    CPU); re-copied on a mismatch, LayerFailed after REPLAY_TRIES. For results that leave the device (code words,
    scales, Hessian copies). t itself when it is on the CPU."""
    if t.device.type == "cpu":
        return t
    want = fingerprint(t).cpu()
    for _ in range(REPLAY_TRIES):
        h = t.cpu()
        if torch.equal(fingerprint(h), want):
            return h
        REPLAY_STATS["download_retries"] += 1
    raise LayerFailed(f"device -> host copy of a {tuple(t.shape)} tensor kept differing from the device bytes")


def to_device_verified(t: torch.Tensor, device) -> torch.Tensor:
    """t.to(device) for a host tensor, checked against the fingerprint of the host bytes (computed on the CPU), with
    the device synchronized before the copy (nothing queued while it lands); re-copied on a mismatch, LayerFailed
    after REPLAY_TRIES. t itself when it already is on the device."""
    if t.device == torch.device(device):
        return t
    want = fingerprint(t)
    for _ in range(REPLAY_TRIES):
        _sync(device)
        d = t.to(device)
        if torch.equal(fingerprint(d).cpu(), want):
            return d
        REPLAY_STATS["upload_retries"] += 1
        del d
    raise LayerFailed(f"host -> device copy of a {tuple(t.shape)} tensor kept differing from the host bytes")


class PinnedPool:
    """Pinned host blocks for LayerJob captures: taken while a layer captures, given back at its end and reused by
    the next layer (pinned allocation is slow; the blocks are never freed during a conversion, so no address is
    reused and no copy goes through the runtime's pageable staging path). release() drops the free blocks."""

    def __init__(self, block_bytes: int = 1 << 28):
        self.block_bytes, self.free, self.allocated = block_bytes, [], 0

    def take(self, nbytes: int) -> torch.Tensor:
        for i, b in enumerate(self.free):
            if b.numel() >= nbytes:
                return self.free.pop(i)
        n = max(self.block_bytes, nbytes)
        b = torch.empty(n, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
        self.allocated += n
        return b

    def give(self, blocks: list) -> None:
        self.free.extend(blocks)

    def release(self) -> None:
        self.allocated -= sum(b.numel() for b in self.free)
        self.free = []


PINNED = PinnedPool()


class HostCapture:
    """Slots for a layer's captured windows in pinned blocks of a PinnedPool (4 KiB aligned)."""
    ALIGN = 4096

    def __init__(self, pool: PinnedPool):
        self.pool, self.blocks, self.used = pool, [], 0

    def slot(self, t: torch.Tensor) -> torch.Tensor:
        n = t.numel() * t.element_size()
        if not self.blocks or self.used + n > self.blocks[-1].numel():
            self.blocks.append(self.pool.take(n))
            self.used = 0
        v = self.blocks[-1][self.used:self.used + n].view(t.dtype).view(t.shape)
        self.used += -(-n // self.ALIGN) * self.ALIGN
        return v

    def bytes(self) -> int:
        return sum(b.numel() for b in self.blocks)

    def close(self) -> None:
        self.pool.give(self.blocks)
        self.blocks = []


def expert_chunks(n_experts: int, chunk: int | None, group: int = EXPERT_GROUP) -> list[tuple[int, int]]:
    """[e0, e1) expert chunks of a LayerJob. A chunk must be a multiple of CHOL_CHUNK and of the expert group, so
    that chunk boundaries are Cholesky-batch and group boundaries (the codes are then those of the whole layer)."""
    if not chunk or chunk >= n_experts:
        return [(0, n_experts)]
    if not group:
        raise ValueError("expert chunking needs a fixed expert group: group 0 (T4's whole-stack batches) would make "
                         "the codes depend on the chunk")
    mult = math.lcm(CHOL_CHUNK, group)
    if chunk % mult:
        raise ValueError(f"expert chunk {chunk} must be a multiple of {mult}")
    return [(e, min(n_experts, e + chunk)) for e in range(0, n_experts, chunk)]


class LayerJob:
    """One MoE layer through the converter: accumulate its Hessians during the calibration pass, then encode and free.

    A streamer drives it with two calls:
        job.add(x, idx, wts, W_gu)   once per calibration forward, in window order, while the layer's weights are
                                     bound (x = MoE block input [T, H], idx / wts = top-k ids / weights [T, k],
                                     W_gu = the bound [E, 2I, H] weights, for the down-input recompute)
        recs = job.finish(W_gu, W_dn)  after the last window (weights on the GPU or in host memory), returns
                                     {"gate"|"up"|"down": record} (see encode_layer)
    Both raise LayerFailed (a subclass) when a device result or copy kept coming out wrong; the job is then dead
    (abort() frees it) and the layer has to be converted again (tools/tcq_convert.py does that in a later pass).
    Before that, a chunk whose encode raised LayerFailed (e.g. a Cholesky that gave no repeated factor in 8 runs:
    seen once in N2 under heavy load, results/qint/qw3/fix2/) is encoded again in place after the pauses
    FINISH_PAUSES_S (30 s, then 120 s): its Hessians are still there, and a layer redo costs a re-stream.
    Expert chunking (expert_chunk < E): only the first chunk's Hessians are accumulated live; every forward's
    (x, idx, wts) is also kept in host memory and replayed for the later chunks (same flush grouping, so every
    expert's sums are the same bits as without chunking). Each chunk is accumulated, encoded and freed before the
    next: GPU memory = one chunk's Hessians + factors + one chunk's weights, host memory = the layer's routed
    inputs (T x H bf16, pinned) while chunking. With the fixed expert group (group = EXPERT_GROUP) the records are
    the same bits for every chunk (a multiple of 8) as for the whole layer; group 0 (T4's batches) refuses chunking.
    Copies (seen faulty on MI250 / ROCm torch 2.11 next to other GPU jobs: replayed windows that did not match their
    capture, transient and once persistent; results/qint/verify/qw3/r2/, results/qint/qw3/fix2/):
      capture  into pinned slots (HostCapture, PINNED pool); per window: synchronize, fingerprint the original on
               the device and copy it (both queued), synchronize, fingerprint the host copy ON THE CPU and require
               both fingerprints to agree (exact for the bytes the CPU sees: stronger than a round trip, which
               reads the copy back through the same path); re-captured while the original is still on the device
               (CaptureFailed after REPLAY_TRIES). The CPU fingerprint is the window's reference (self.refs): it
               never crosses the bus. Two synchronizations per window (each costs milliseconds on a busy GPU).
      replay   per window: synchronize, upload, device fingerprint vs the reference, re-upload on a mismatch; if
               the host copy itself no longer matches its reference, HostCaptureCorrupt. Per flush group, after the
               concatenation (= the data folded in): every window vs its reference; bad windows are uploaded again
               and the group re-checked (HessianAccumulator reload); a chunk whose repairs fail, or whose sums are
               not exactly symmetric, is replayed again (REPLAY_TRIES). Weights: to_device_verified.
    Counters: self.stats (this layer) and REPLAY_STATS (process)."""

    def __init__(self, layer: int, n_experts: int, hidden: int, inter: int, plan_entry: dict, device,
                 n_windows: int, flush: int = 32, weighted: bool = False, expert_chunk: int | None = None,
                 users: dict | None = None, glu=None, lut_dir: str | None = None, keep_hessians: bool = False,
                 timed: bool = True, group: int = EXPERT_GROUP, votes: int = ENCODE_VOTES,
                 pool: PinnedPool | None = None, gu_bias: torch.Tensor | None = None):
        self.layer, self.E, self.Hd, self.I = layer, n_experts, hidden, inter
        self.gu_bias = gu_bias  # [E, 2I] on the device, or None (HessianAccumulator)
        self.votes = votes
        self.entry, self.dev, self.n_windows = plan_entry, torch.device(device), n_windows
        self.flush, self.weighted, self.users, self.lut_dir = flush, weighted, users, lut_dir
        self.glu = glu or silu_glu(inter)
        self.group = group
        self.chunks = expert_chunks(n_experts, expert_chunk, group)
        self.keep_hessians = keep_hessians
        self.timed = timed and self.dev.type == "cuda"
        e0, e1 = self.chunks[0]
        self.acc = HessianAccumulator(n_experts, hidden, inter, self.dev, e0, e1, flush, weighted, n_windows,
                                      self.glu, gu_bias=gu_bias)
        chunked = len(self.chunks) > 1
        self.store = HostCapture(pool or PINNED) if chunked else None
        self.captured: list | None = [] if chunked else None
        self.refs = torch.zeros(n_windows if chunked else 0, 6, dtype=torch.int64)  # CPU fingerprints of captures
        self.replay_errors: list = []
        self.stats = {"captured_windows": 0, "capture_retries": 0, "upload_retries": 0, "group_repairs": 0,
                      "replayed_chunks": 0, "replay_retries": 0, "finish_retries": 0, "capture_gib": 0.0}
        self.timings: dict = {"accumulate_s": 0.0}
        self.hessians: list = []
        self.dead = False

    def abort(self) -> None:
        """Free everything of a job that will not finish (pinned blocks back to the pool)."""
        self.dead = True
        self.acc = None
        self.captured, self.refs = None, torch.zeros(0, 6, dtype=torch.int64)
        if self.store is not None:
            self.store.close()

    @torch.no_grad()
    def add(self, x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor | None, W_gu: torch.Tensor) -> None:
        if self.dead:
            raise LayerFailed(f"layer {self.layer}: job aborted")
        if self.timed:  # exact accumulation time (the forward's queued work is waited for outside the timer)
            torch.cuda.synchronize(self.dev)
        t0 = time.time()
        try:
            if self.captured is not None:
                self._capture(x, idx, wts if self.weighted else None)
            self.acc.add(x, idx, wts, W_gu)
        except LayerFailed:
            self.abort()
            raise
        if self.timed:
            torch.cuda.synchronize(self.dev)
        self.timings["accumulate_s"] += time.time() - t0

    def _count(self, key: str, msg: str | None = None) -> None:
        self.stats[key] += 1
        REPLAY_STATS[key] += 1
        if msg:
            self.replay_errors.append(msg)

    @torch.no_grad()
    def _capture(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor | None) -> None:
        i = len(self.captured)
        parts = [x, idx] + ([] if w is None else [w])
        slots = [self.store.slot(p) for p in parts]
        why = ""
        for attempt in range(REPLAY_TRIES):
            _sync(self.dev)
            ref = window_fingerprint(x, idx, w)  # the originals, on the device (queued)
            for s, p in zip(slots, parts):
                s.copy_(p, non_blocking=self.dev.type == "cuda")  # device -> pinned host (queued)
            _sync(self.dev)  # one wait for all of it
            ref = ref.cpu()
            cpu = window_fingerprint(slots[0], slots[1], slots[2] if w is not None else None)  # the copy, on the CPU
            if torch.equal(cpu, ref):
                self.refs[i] = cpu
                self.captured.append((slots[0], slots[1], slots[2] if w is not None else None))
                self._count("captured_windows")
                self.stats["capture_gib"] = round(self.store.bytes() / 2**30, 3)
                return
            why = f"host copy fingerprint != device fingerprint ({_parts_differ(cpu, ref)})"
            self._count("capture_retries", f"layer {self.layer} window {i} capture attempt {attempt}: {why}")
        raise CaptureFailed(f"layer {self.layer}: window {i} not captured in {REPLAY_TRIES} attempts: {why}")

    @torch.no_grad()
    def _upload(self, i: int):
        """Window i's capture on the device, checked against its reference; re-uploaded on a mismatch."""
        slots, want = self.captured[i], self.refs[i]
        nb = self.dev.type == "cuda"
        for attempt in range(REPLAY_TRIES):
            _sync(self.dev)  # nothing queued while the host -> device copies land
            d = tuple(None if s is None else s.to(self.dev, non_blocking=nb) for s in slots)
            got = window_fingerprint(*d)
            _sync(self.dev)
            got = got.cpu()
            if torch.equal(got, want):
                return d
            host = window_fingerprint(*slots)
            if not torch.equal(host, want):
                raise HostCaptureCorrupt(f"layer {self.layer}: the host copy of window {i} changed after its verified "
                                         f"capture ({_parts_differ(host, want)})")
            self._count("upload_retries", f"layer {self.layer} window {i} upload attempt {attempt}: upload differs "
                        f"from the capture ({_parts_differ(got, want)})")
            del d
        raise ReplayCorrupt(f"layer {self.layer}: window {i}: {REPLAY_TRIES} uploads differ from the capture")

    def _reload(self, tag: int):
        self._count("group_repairs")
        return self._upload(tag)

    def _replay_check(self, tags, rows, x, idx, wts, inputs) -> None:
        """HessianAccumulator check: every window of a concatenated flush group against its reference (raises
        ReplayCorrupt with the positions that differ). The message says which part differs (x / idx / wts) and
        whether the window's uploaded copy, before the concatenation, still matched (diagnostics)."""
        got, off = [], 0
        for r in rows:
            got.append(window_fingerprint(x[off:off + r], idx[off:off + r], None if wts is None else wts[off:off + r]))
            off += r
        got = torch.stack(got).cpu()
        want = self.refs[tags]
        if not torch.equal(got, want):
            rows_bad = (got != want).any(1).nonzero().flatten().tolist()
            what = []
            for i in rows_bad[:8]:
                upload_ok = torch.equal(window_fingerprint(*inputs[i]).cpu(), want[i])
                what.append(f"{tags[i]}:{_parts_differ(got[i], want[i])}:upload {'ok' if upload_ok else 'BAD'}")
            raise ReplayCorrupt(f"replayed windows differ from their capture ({len(rows_bad)} of {len(rows)} in the "
                                f"flush group; window:parts:uploaded copy) {what}", bad=rows_bad)

    def _replay(self, ci: int, e0: int, e1: int, Wg: torch.Tensor) -> HessianAccumulator:
        """Accumulate chunk [e0, e1) from the captured inputs, verified; replayed again when a flush group cannot be
        repaired or the sums are not exactly symmetric. HostCaptureCorrupt is not retried (it cannot change)."""
        for attempt in range(REPLAY_TRIES):
            acc = HessianAccumulator(self.E, self.Hd, self.I, self.dev, e0, e1, self.flush, self.weighted,
                                     self.n_windows, self.glu, check=self._replay_check, reload=self._reload,
                                     gu_bias=self.gu_bias)
            try:
                for i in range(len(self.captured)):
                    x, idx, wts = self._upload(i)
                    acc.add(x, idx, wts, Wg, e0, tag=i)
                    del x, idx, wts
                acc.check_symmetric()
                self._count("replayed_chunks")
                return acc
            except (ReplayCorrupt, AccumulationFlake) as err:
                self._count("replay_retries", f"layer {self.layer} chunk {ci} attempt {attempt}: {err}")
                del acc
        raise ReplayCorrupt(f"layer {self.layer}: chunk {ci} replay failed {REPLAY_TRIES} times: "
                            f"{self.replay_errors[-REPLAY_TRIES:]}")

    @torch.no_grad()
    def finish(self, W_gu: torch.Tensor, W_dn: torch.Tensor) -> dict:
        if self.dead:
            raise LayerFailed(f"layer {self.layer}: job aborted")
        if self.acc.seen != self.n_windows:
            raise RuntimeError(f"layer {self.layer}: {self.acc.seen} forwards seen, {self.n_windows} expected")
        try:
            return self._finish(W_gu, W_dn)
        except LayerFailed:
            self.abort()
            raise
        finally:
            if self.store is not None:
                self.store.close()
            self.captured = None

    def _finish(self, W_gu: torch.Tensor, W_dn: torch.Tensor) -> dict:
        parts: dict = {m: [] for m in MATRICES}
        counts = None
        for ci, (e0, e1) in enumerate(self.chunks):
            t0 = time.time()
            Wg = to_device_verified(W_gu[e0:e1], self.dev)
            if ci == 0:
                acc = self.acc
                self.acc = None
            else:
                acc = self._replay(ci, e0, e1, Wg)
                if counts is not None and not torch.equal(acc.cnt, counts):
                    raise LayerFailed(f"layer {self.layer}: replayed routing differs from the live pass")
            try:
                h_gu, h_dn = acc.result()  # raises AccumulationFlake before touching the sums
            except AccumulationFlake as err:
                if ci or self.captured is None:  # the live, unchunked sums cannot be rebuilt: convert the layer again
                    raise
                self._count("replay_retries", f"layer {self.layer} chunk 0 (live): {err}; replayed from the captures")
                acc = self._replay(0, e0, e1, Wg)
                h_gu, h_dn = acc.result()
            counts = acc.cnt
            del acc
            _sync(self.dev)
            self.timings["hessian_finish_s"] = self.timings.get("hessian_finish_s", 0.0) + time.time() - t0
            if self.keep_hessians:  # host copies (for comparisons and analysis only)
                self.hessians.append(tuple(LayerHessians(h.kind, h.ids, to_host_checked(h.H),
                                                         to_host_checked(h.scale), to_host_checked(h.count))
                                           for h in (h_gu, h_dn)))
            for attempt in range(len(FINISH_PAUSES_S) + 1):  # the chunk's Hessians stay: retry in place first
                try:
                    recs = encode_layer(Wg, to_device_verified(W_dn[e0:e1], self.dev), h_gu, h_dn, self.entry,
                                        self.layer, self.users, self.dev, self.lut_dir, self.timings, self.group,
                                        self.votes)
                    break
                except LayerFailed as err:
                    if attempt == len(FINISH_PAUSES_S):
                        raise
                    self._count("finish_retries", f"layer {self.layer} chunk {ci} encode attempt {attempt}: {err}; "
                                f"again in {FINISH_PAUSES_S[attempt]} s")
                    time.sleep(FINISH_PAUSES_S[attempt])
            for m in MATRICES:
                parts[m].append(recs[m])
            del Wg, h_gu, h_dn, recs
        self.refs = torch.zeros(0, 6, dtype=torch.int64)
        self.counts = to_host_checked(counts)
        out = {m: concat_records(parts[m]) for m in MATRICES}
        check_records(out, f"layer {self.layer}")
        return out


def relative_error(rec: dict) -> float:
    """||W' - s Q||^2 / ||W'||^2 of a record (T4's 'rel w-err')."""
    return rec["_err2"] / max(rec["_w2"], 1e-30)


def expert_relative_errors(rec: dict) -> torch.Tensor:
    """Per expert ||W'_e - s Q_e||^2 / max(||W'_e||^2, REL_ERR_FLOOR * mean_e ||W'_e||^2) (float64 [E]): an expert
    whose weights are dead (a millionth of the stack's mean energy or less) cannot fail the check on its own."""
    w2 = rec["_w2_e"]
    return rec["_err2_e"] / w2.clamp(min=max(REL_ERR_FLOOR * float(w2.mean()), 1e-30))


def check_records(recs: dict, what: str = "") -> None:
    """Refuse a broken encoding: any stack, or any single expert of it, with relative error above MAX_REL_ERR (or
    NaN). Normal values are 0.01-0.08 (results/qint/qw3/)."""
    bad = {}
    for m, r in recs.items():
        e = relative_error(r)
        if not e <= MAX_REL_ERR:
            bad[m] = round(e, 4)
        if "_err2_e" in r:
            ee = expert_relative_errors(r)
            worst = int(torch.nan_to_num(ee, nan=float("inf")).argmax())
            if not float(ee[worst]) <= MAX_REL_ERR:
                bad[f"{m} expert {r['experts'][worst]}"] = round(float(ee[worst]), 4)
    if bad:
        raise RuntimeError(f"{what}: relative weight error {bad} > {MAX_REL_ERR}: broken encoding")


with open(__file__, "rb") as _f:
    SOURCE_SHA = hashlib.sha256(_f.read()).hexdigest()[:16]  # at import: what this process executes


def source_sha() -> str:
    return SOURCE_SHA


def bytes_per_expert(hidden: int, inter: int) -> dict:
    """GPU bytes one expert costs a LayerJob chunk, by phase (from the code; checked against measured peaks in
    results/qint/qw3/): accumulate = float32 sums G + D; finish = the fp16 Hessians of both kinds + the gate/up
    float32 factors + the chunk's bf16 weights + encode_matrix's working copies of one gate/up stack (bf16
    contiguous, float32, rotated W', row-normalised W'/s: 14 bytes per weight)."""
    H, I = hidden, inter
    return {"accumulate": 4 * (H * H + I * I), "finish": 2 * (H * H + I * I) + 4 * H * H + 6 * H * I + 14 * H * I}


def chol_bytes(hidden: int) -> int:
    """Transient working set of the gate/up factors, independent of the expert chunk: one batched Cholesky +
    triangular inverse (chol_chunk(H) matrices, about seven float64 copies alive at once with cholesky_voted;
    inv_upper's padded size) + the factor vote's two float32 candidates of the batch (hessian_factors)."""
    return 7 * chol_chunk(hidden) * trsm_size(hidden) ** 2 * 8 + 2 * chol_chunk(hidden) * hidden ** 2 * 4


def pick_expert_chunk(n_experts: int, hidden: int, inter: int, budget_bytes: int,
                      group: int = EXPERT_GROUP) -> int:
    """Expert chunk for a LayerJob whose peak must fit in budget_bytes: the fewest chunks that fit, split evenly
    (multiples of lcm(CHOL_CHUNK, group), as expert_chunks requires; the whole layer when it fits). With a fixed
    expert group the codes do not depend on the chunk, so the budget may be the memory free when the layer starts."""
    mult = math.lcm(CHOL_CHUNK, group or CHOL_CHUNK)
    per = max(bytes_per_expert(hidden, inter).values())
    c_max = max(mult, int((budget_bytes - chol_bytes(hidden)) // per) // mult * mult)
    if c_max >= n_experts:
        return n_experts
    n = math.ceil(n_experts / c_max)
    return math.ceil(n_experts / n / mult) * mult
