"""Trellis-coded (TCQ) expert weights for the engine: the format, the LUT registry, a pure-torch reference decoder
and the input rotation. Kernels: ramog/tcq_kernels.py.

The codes are those of the quality sprint's track T4 (quant_methods/t4_tcq.py, t4_quant.py; QTIP / EXL3-style
bitshift trellis, results/quality/REPORT.md). This module only decodes them; it never encodes.

Weight of one expert matrix W [n, k] (nn.Linear layout, out x in):
    W ~= W_rot (D H)^T = W_rot H^T D       W_rot = diag(s) Q   (the "rotated domain")
    D = diag(signs), H = block-diagonal normalised Walsh-Hadamard along `in` (ramog.quantizers.rotation with
    seed = layer * 1000, shared by every expert of a layer and by gate and up; see layer_rotation()). The output side
    (rows) is NOT rotated: t4_quant.quantize rotates only along `in` (_rot_w -> rotate_in) and the row scale s is
    per output row. So
        y = W x = W_rot x_rot,   x_rot = H^T D x = H (D x)          (H is symmetric)
    i.e. rotate the input once (per token and layer for gate_up, per (token, expert) pair for down), then a plain
    GEMV with the decoded rotated weight; the epilogue multiplies each output row by s.

Stored arrays per expert matrix (engine layout):
    words  int32 [n/16, k/16, WPT]  one tail-biting bitstream per 16x16 tile, tiles row-tile-major then column tile
                                     (tile (R, C) covers rows 16R..16R+15, cols 16C..16C+15); WPT = 64 (b0 + b1) / 32
    scale  fp16  [n]                 row scales s
    LUT    fp16  [4096, 2]           one table per rate pattern for the whole model (ramog/data/tcq/)
Inside a tile the 256 weights are row-major (weight (i, j) = element 16 i + j) and read as 128 steps of V = 2
weights: step t covers row t // 8, columns 2 (t % 8) and 2 (t % 8) + 1. The bitstream is MSB first (bit 0 = bit 31
of word 0); step t appends b_t bits (b0 for even t, b1 for odd t), so it ends at
    pos_t = (t // 2) (b0 + b1) + b0 + (t % 2) b1
and its 12-bit state is the window of bits [pos_t - 12, pos_t) taken circularly (mod 64 (b0 + b1)): the stream
wraps, no start state is stored. The two weights of step t are LUT[state, 0] and LUT[state, 1] (times s).
Gate and up are separate T4 matrices with the same pattern and rotation; the engine fuses them as one gate_up
matrix [2I, H] by stacking the row tiles (gate first, as ramog.store's gate_up) and the row scales.

Rate patterns (b0, b1) -> bits per weight (b0 + b1) / 4 (codes only; + 16 / k for the row scales):
    b3-4 1.75  b4 2.0  b4-5 2.25  b5 2.5  b5-6 2.75  b6 3.0  b6-7 3.25  b7 3.5  b7-8 3.75  b8 4.0
T4's config A (A2577) uses the 8 patterns from b4 to b7-8 (results/quality/t4/plans/alloc-2.577.json).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from .quantizers import hadamard_block, rotate_in, rotation, unrotate_in

L = 12  # trellis state bits
V = 2  # weights per step
TILE = 16  # tile side (tile = 16 x 16 weights)
STEPS = TILE * TILE // V  # 128 steps per tile
LUT_ROWS = 1 << L
DATA_DIR = Path(__file__).resolve().parent / "data" / "tcq"

# name -> (b0, b1); bits per weight = (b0 + b1) / (2 V)
PATTERNS: dict[str, tuple[int, int]] = {
    "b3-4": (3, 4), "b4": (4, 4), "b4-5": (4, 5), "b5": (5, 5), "b5-6": (5, 6),
    "b6": (6, 6), "b6-7": (6, 7), "b7": (7, 7), "b7-8": (7, 8), "b8": (8, 8),
}
A2577_PATTERNS = ("b4", "b4-5", "b5", "b5-6", "b6", "b6-7", "b7", "b7-8")


def pattern_name(b0: int, b1: int) -> str:
    return f"b{b0}" if b0 == b1 else f"b{b0}-{b1}"


def parse_pattern(p) -> tuple[int, int]:
    """'b4-5' / (4, 5) / a T4 spec 'tcq:L12:V2:b4,5' -> (4, 5). Only the 10 engine patterns are accepted."""
    if isinstance(p, (tuple, list)):
        b = tuple(int(v) for v in p)
        b = (b[0], b[-1])
    elif p.startswith("tcq:"):
        parts = {q[0]: q[1:] for q in p.split(":")[1:]}
        if int(parts.get("L", 12)) != L or int(parts.get("V", 2)) != V or int(parts.get("T", 256)) != 256 \
                or int(parts.get("c", 16)) != TILE:
            raise ValueError(f"{p}: the engine decodes L12, V2, 16x16 tiles only")
        b = tuple(int(v) for v in parts["b"].split(","))
        if len(b) not in (1, 2):
            raise ValueError(f"{p}: patterns of one or two step sizes only")
        b = (b[0], b[-1])
    else:
        b = PATTERNS[p] if p in PATTERNS else None
        if b is None:
            raise ValueError(f"unknown TCQ pattern {p!r} (known: {', '.join(PATTERNS)})")
    if pattern_name(*b) not in PATTERNS:
        raise ValueError(f"TCQ pattern {b} is not one of the engine's {list(PATTERNS.values())}")
    return b


def words_per_tile(b0: int, b1: int) -> int:
    return 64 * (b0 + b1) // 32


def bpw(b0: int, b1: int, k: int | None = None) -> float:
    """Stored bits per weight: codes, plus the fp16 row scale when k (the row length) is given."""
    return (b0 + b1) / 4 + (16 / k if k else 0.0)


def step_window(t: int, b0: int, b1: int) -> tuple[int, int]:
    """(first bit of step t's 12-bit window, mod the tile's bits; tile bits). The kernels use the same formula."""
    tot = 64 * (b0 + b1)
    pos = (t // 2) * (b0 + b1) + b0 + (t % 2) * b1
    return (pos - L) % tot, tot


# ============================================================================================ LUT registry

def lut_file(b0: int, b1: int) -> Path:
    return DATA_DIR / f"tcq-L12-V2-{pattern_name(b0, b1)}.bin"


@lru_cache(maxsize=None)
def manifest() -> dict:
    with open(DATA_DIR / "manifest.json") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _lut_cpu(b0: int, b1: int) -> torch.Tensor:
    """fp16 [4096, 2] on the CPU, checked against the manifest's sha256 (raw little-endian bytes)."""
    name = lut_file(b0, b1).name
    ent = manifest()["luts"].get(name)
    if ent is None:
        raise FileNotFoundError(f"{name}: not in {DATA_DIR / 'manifest.json'}")
    raw = lut_file(b0, b1).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != ent["sha256"]:
        raise RuntimeError(f"{lut_file(b0, b1)}: sha256 {sha} != manifest {ent['sha256']} (corrupt or replaced LUT)")
    if len(raw) != LUT_ROWS * V * 2:
        raise RuntimeError(f"{lut_file(b0, b1)}: {len(raw)} bytes, expected {LUT_ROWS * V * 2}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float16).view(LUT_ROWS, V).clone()


def lut_sha256(pattern) -> str:
    """sha256 of the LUT's raw fp16 bytes: equals T4's lut_sha256 recorded in its saved codes."""
    b0, b1 = parse_pattern(pattern)
    return manifest()["luts"][lut_file(b0, b1).name]["sha256"]


_LUT_DEV: dict = {}


def lut(pattern, device="cpu") -> torch.Tensor:
    """The pattern's LUT, fp16 [4096, 2], on `device` (checked once per process)."""
    b0, b1 = parse_pattern(pattern)
    key = (b0, b1, str(torch.device(device)))
    if key not in _LUT_DEV:
        _LUT_DEV[key] = _lut_cpu(b0, b1).to(device)
    return _LUT_DEV[key]


def lut_packed(pattern, device) -> torch.Tensor:
    """The LUT as int32 [4096] (entry s = LUT[s, 0] in the low 16 bits, LUT[s, 1] in the high 16): what the kernels
    read, one 32-bit load per step."""
    b0, b1 = parse_pattern(pattern)
    key = (b0, b1, str(torch.device(device)), "i32")
    if key not in _LUT_DEV:
        _LUT_DEV[key] = lut((b0, b1), device).contiguous().view(torch.int32).view(LUT_ROWS).contiguous()
    return _LUT_DEV[key]


# ============================================================================================ reference decoder

def unpack_states(words: torch.Tensor, b0: int, b1: int) -> torch.Tensor:
    """words int32 [..., WPT] -> int64 states [..., 128]: step t's state is the 12-bit window ending at pos_t
    (circular, MSB first). Pure torch, any device."""
    wpt = words.shape[-1]
    if wpt != words_per_tile(b0, b1):
        raise ValueError(f"{wpt} words per tile, pattern {pattern_name(b0, b1)} has {words_per_tile(b0, b1)}")
    lead = words.shape[:-1]
    w = words.reshape(-1, wpt).long() & 0xFFFFFFFF
    dev = w.device
    starts = torch.tensor([step_window(t, b0, b1)[0] for t in range(STEPS)], device=dev)
    w0 = starts // 32
    off = starts % 32
    w1 = (w0 + 1) % wpt
    pair = (w[:, w0] << 32) | w[:, w1]  # 64-bit window [word w0, word w0+1]
    st = (pair >> (52 - off)) & 0xFFF  # 64 - 12 - off
    return st.view(*lead, STEPS)


def decode_tiles(words: torch.Tensor, pattern, lut16: torch.Tensor | None = None) -> torch.Tensor:
    """words int32 [..., WPT] -> float32 [..., 16, 16] unit-domain values (LUT entries, no row scale)."""
    b0, b1 = parse_pattern(pattern)
    tab = (lut16 if lut16 is not None else lut((b0, b1), words.device)).float()
    st = unpack_states(words, b0, b1)
    return tab[st].reshape(*words.shape[:-1], TILE, TILE)


def decode(words: torch.Tensor, scale: torch.Tensor, pattern) -> torch.Tensor:
    """Reference decoder. words int32 [n/16, k/16, WPT] (or [E, n/16, k/16, WPT]), scale fp16 [n] (or [E, n]) ->
    float32 W_rot [n, k] (or [E, n, k]) in the ROTATED domain: LUT value (fp16 -> float32) times the fp16 row scale
    (-> float32), one float32 multiply, exactly as T4's quantize_from_codes forms Q before unrotating."""
    vals = decode_tiles(words, pattern)  # [..., R, C, 16, 16]
    *lead, nr, nc, _, _ = vals.shape
    w = vals.transpose(-3, -2).reshape(*lead, nr * TILE, nc * TILE)
    return w * scale.float()[..., :, None]


def decode_unrotated(words, scale, pattern, layer: int, device=None) -> torch.Tensor:
    """decode() mapped back to the original basis (W = W_rot H^T D), as t4_quant.quantize_from_codes returns it."""
    w = decode(words, scale, pattern)
    rot = layer_rotation(w.shape[-1], layer, device or w.device)
    lead = w.shape[:-2]
    return unrotate_in(w.reshape(-1, w.shape[-1]), *rot).view(*lead, *w.shape[-2:])


# ============================================================================================ rotation

def rotation_seed(layer: int) -> int:
    return layer * 1000


def layer_rotation(n_in: int, layer: int, device):
    """(signs float32 [n_in], h float32 [b, b], b): T4's rotation for one layer and input size, exactly
    ramog.quantizers.rotation(n_in, layer * 1000, device) (t4_quant._rotations with rot="layer")."""
    return rotation(n_in, rotation_seed(layer), device)


def block_structure(n_in: int) -> tuple[int, int]:
    """(Hadamard block b, number of blocks n_in / b) of ramog.quantizers.hadamard_block: the largest power of two
    <= 4096 dividing n_in. Target dims (hidden: gate/up input; expert intermediate: down input):
        704  -> 64 x 11     768  -> 256 x 3     1024 -> 1024 x 1    1408 -> 128 x 11    1536 -> 512 x 3
        2048 -> 2048 x 1    2880 -> 64 x 45     3072 -> 1024 x 3    4096 -> 4096 x 1    5120 -> 1024 x 5
        6144 -> 2048 x 3    7168 -> 1024 x 7
    (a dim divisible only by small powers of two gets a weak rotation: 2880 and 704 mix only 64 channels)."""
    b = hadamard_block(n_in)
    return b, n_in // b


def rotate_act_ref(x: torch.Tensor, signs: torch.Tensor, h: torch.Tensor, b: int) -> torch.Tensor:
    """x [T, n_in] -> x_rot = H (D x) per row, with T4's own arithmetic (rotate_in on the activations: the row
    vector (D x)^T times the symmetric H). The kernels' FWHT computes the same map (tcq_kernels.rotate)."""
    return rotate_in(x, signs, h, b)


# ============================================================================================ T4 codes -> engine layout

def from_t4(obj: dict, experts: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """One matrix stack saved by t4_quant.quantize(save=DIR) (words [k/16, E * n/16, WPT], column block major, then
    expert, then row tile; scale [E, n]) -> (words int32 [E, n/16, k/16, WPT] contiguous, scale fp16 [E, n],
    pattern), experts in the order of obj["experts"] (T4 saves 0..E-1). `experts` selects a subset (indices into
    obj["experts"])."""
    n, k = obj["shape"]
    b0, b1 = parse_pattern(obj["spec"])
    if obj.get("rot", "layer") != "layer":
        raise ValueError(f"rot={obj.get('rot')}: the engine rotates per layer (seed = layer * 1000) only")
    if obj["lut_sha256"] != lut_sha256((b0, b1)):
        raise ValueError(f"{obj['spec']}: codes were made with LUT {obj['lut_sha256'][:16]}, engine has "
                         f"{lut_sha256((b0, b1))[:16]}")
    w = obj["words"]
    nb, rows, wpt = w.shape
    e = len(obj["experts"])
    if nb != k // TILE or rows != e * (n // TILE) or wpt != words_per_tile(b0, b1):
        raise ValueError(f"words {tuple(w.shape)} do not match shape {n}x{k}, {e} experts, {obj['spec']}")
    w = w.view(nb, e, n // TILE, wpt)
    s = obj["scale"]
    if experts is not None:
        idx = torch.as_tensor(experts, dtype=torch.long)
        w, s = w[:, idx], s[idx]
    return w.permute(1, 2, 0, 3).contiguous(), s.contiguous(), (b0, b1)


def fuse_gate_up(gate: tuple, up: tuple) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """from_t4(gate), from_t4(up) -> the fused gate_up matrix [2I, H]: row tiles and scales stacked, gate first."""
    (wg, sg, pg), (wu, su, pu) = gate, up
    if pg != pu:
        raise ValueError(f"gate {pg} and up {pu} use different patterns: they cannot share one fused matrix")
    return torch.cat([wg, wu], dim=-3).contiguous(), torch.cat([sg, su], dim=-1).contiguous(), pg


# ============================================================================================ blob layout

def _round_up(n: int, a: int) -> int:
    return -(-n // a) * a


ALIGN = 16


@dataclass(frozen=True)
class TcqLayout:
    """Byte layout of one expert's TCQ blob (gate_up, then down), like ramog.store.BlobLayout:
        gu_w  int32 [2I/16, H/16, WPT_gu]   gu_s fp16 [2I]   dn_w int32 [H/16, I/16, WPT_dn]   dn_s fp16 [H]
    each array padded to 16 bytes. A layer has one pattern for gate_up and one for down (T4's per-layer plan)."""

    gu_shape: tuple[int, int]  # (2I, H)
    dn_shape: tuple[int, int]  # (H, I)
    gu_pattern: tuple[int, int]
    dn_pattern: tuple[int, int]

    def __post_init__(self):
        for (n, k), p in ((self.gu_shape, self.gu_pattern), (self.dn_shape, self.dn_pattern)):
            parse_pattern(p)
            if n % TILE or k % TILE:
                raise ValueError(f"TCQ needs both dims divisible by 16, got [{n}, {k}]")

    def _arrays(self):
        out = []
        for m, (n, k), p in (("gu", self.gu_shape, self.gu_pattern), ("dn", self.dn_shape, self.dn_pattern)):
            wpt = words_per_tile(*p)
            out.append((f"{m}_w", (n // TILE, k // TILE, wpt), torch.int32, n * k // 256 * wpt * 4))
            out.append((f"{m}_s", (n,), torch.float16, 2 * n))
        return [(*a, _round_up(a[3], ALIGN)) for a in out]

    @property
    def nbytes(self) -> int:
        return sum(a[4] for a in self._arrays())

    def offsets(self) -> list[int]:
        offs, o = [], 0
        for a in self._arrays():
            offs.append(o)
            o += a[4]
        return offs

    def views(self, pool: torch.Tensor) -> dict:
        """pool uint8 [S, nbytes] (VRAM or pinned host) -> {gu_w, gu_s, dn_w, dn_s} strided [S, ...] views."""
        s = pool.shape[0]
        out = {}
        for (name, shape, dtype, size, _), off in zip(self._arrays(), self.offsets(), strict=True):
            out[name] = pool[:, off:off + size].view(dtype).view(s, *shape)
        return out

    def assemble(self, arrays: dict, e: int) -> torch.Tensor:
        """{gu_w, gu_s, dn_w, dn_s} with a leading expert dim -> uint8 blobs [e, nbytes] (zero padding)."""
        parts = []
        for name, _shape, _dtype, size, padded in self._arrays():
            t = arrays[name].contiguous().reshape(e, -1).view(torch.uint8)
            assert t.shape[1] == size, (name, t.shape, size)
            parts.append(t)
            if padded > size:
                parts.append(t.new_zeros(e, padded - size))
        return torch.cat(parts, dim=1)

    def gemv_args(self, views: dict, which: str) -> dict:
        """Keyword arguments of tcq_kernels.tcq_gemv / tcq_moe_gemm / tcq_dequant for matrix "gu" or "dn"."""
        return {"qweight": views[f"{which}_w"], "scale": views[f"{which}_s"],
                "pattern": self.gu_pattern if which == "gu" else self.dn_pattern}

    def describe(self) -> dict:
        return {"format": "tcq", "gu_pattern": pattern_name(*self.gu_pattern),
                "dn_pattern": pattern_name(*self.dn_pattern), "nbytes": self.nbytes,
                "bpw": 8 * self.nbytes / (self.gu_shape[0] * self.gu_shape[1] + self.dn_shape[0] * self.dn_shape[1])}


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """max |a - b| / max |b| in float64 (the tests' error measure)."""
    a64, b64 = a.double(), b.double()
    return float((a64 - b64).abs().max() / b64.abs().max().clamp(min=1e-300))


def hadamard_factors(b: int) -> tuple[int, int]:
    """FWHT kernel split of a block b = b1 * b2 (H_b = H_b1 kron H_b2): two tl.dot (H_b1 X H_b2) for b >= 256 with
    b1 = max(16, b / 64); b1 = 1 (one [b, b] product in registers) for 16 <= b <= 128."""
    if b < 16 or b & (b - 1) or b > 4096:
        raise ValueError(f"Hadamard block {b}: the FWHT kernel handles powers of two 16..4096")
    if b <= 128:
        return 1, b
    b1 = max(16, b // 64)
    return b1, b // b1

