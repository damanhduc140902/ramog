"""moe_tiering.dequant on synthetic tensors: float64 definitions, transformers' own decoders, round trips (CPU, seconds).

Checks
  formats      torch's float8_e4m3fn / float8_e5m2 -> float and the E2M1 table equal the OCP bit-field definitions
               (all codes), so the float64 references below do not lean on the code under test.
  e8m0         all 256 E8M0 bytes: bitwise equal to transformers' formula (b - 127).exp2(), to math.ldexp, and to
               torch's float8_e8m0fnu (bytes 0..254; 255 is NaN there and +inf in transformers and here).
  fp8 full     e4m3fn / e5m2 codes (every finite byte value occurs), scales f32 / bf16 / f16 / E8M0 (uint8 and
               float8_e8m0fnu), outputs bf16 / f32 / f16, blocks 128x128, 32x32 and 64x32, with and without leading
               dims. Bitwise equal to the float64 definition fp32(q * s[i // bm, j // bn]) -> out, and to
               transformers' Fp8Dequantize._dequantize_one(q, s, out).
  fp8 partial  ceil grids ([576, 384] in 128-blocks -> scale [5, 3], [100, 200], [129, 257], [3, 130, 70] in 64x32):
               the float64 definition, and zero-pad -> transformers -> crop.
  fp8 trip     codes survive decode -> divide by the block scale -> re-cast, for arbitrary f32 scales (amax / 448)
               and for power-of-two E8M0 scales (DeepSeek's ue8m0); with power-of-two scales the decode is exact.
  mxfp4        random blocks and exponents: the float64 definition lut[nibble] * 2^(e - 127) (signed zeros
               included) and transformers' convert_moe_packed_tensors(b, s).transpose(1, 2); an OCP MX encoder
               round trip (encode -> decode -> encode gives the same bytes).
  fp4_i8_e8m0  int8 / uint8 / float4_e2m1fn_x2 codes with E8M0 scales: the float64 definition, transformers'
               _dequantize_one(int8 codes, uint8 scales, out), a partial last block.
  layout       non-contiguous inputs (dim-0 slices of stacked experts, transposed and strided views) decode like
               contiguous ones.
  errors       wrong grids, fnuz / integer FP8 inputs, bad scale dtypes and bad blocks raise.
  registry     the four format names; duplicate registration raises.
  device       (--cuda, or DEQUANT_TEST_CUDA=1 under pytest) every decoder on the GPU equals its CPU result bit for bit.
The transformers comparisons are skipped, and reported as skipped, when transformers is not importable.

python tests/test_dequant.py [--cuda]        (exit code 1 on any failure; also runs under pytest)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_tiering import dequant as dq  # noqa: E402

E8M0 = getattr(torch, "float8_e8m0fnu", None)
FP4X2 = getattr(torch, "float4_e2m1fn_x2", None)
FNUZ = [d for d in (getattr(torch, "float8_e4m3fnuz", None), getattr(torch, "float8_e5m2fnuz", None)) if d is not None]
SKIPPED: list[str] = []


# ---------------------------------------------------------------- helpers

def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def assert_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """Same dtype, same shape, same bytes (so -0.0 != +0.0 and NaN payloads count)."""
    assert got.dtype == want.dtype, f"{what}: dtype {got.dtype} != {want.dtype}"
    assert tuple(got.shape) == tuple(want.shape), f"{what}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    ne = bits(got) != bits(want)
    n = int(ne.sum())
    if n:
        i = int(ne.flatten().nonzero()[0])
        raise AssertionError(f"{what}: {n} of {got.numel()} elements differ; first at flat {i}: "
                             f"{got.flatten()[i].item()!r} vs {want.flatten()[i].item()!r}")


def fp8_table(dtype: torch.dtype) -> torch.Tensor:
    """float64 value of every byte of an OCP FP8 format, from its bit fields (NaN / inf where the format has them)."""
    e_bits, m_bits, bias = (4, 3, 7) if dtype == torch.float8_e4m3fn else (5, 2, 15)
    vals = []
    for b in range(256):
        s, e, m = b >> 7, (b >> m_bits) & ((1 << e_bits) - 1), b & ((1 << m_bits) - 1)
        if dtype == torch.float8_e4m3fn and e == 15 and m == 7:
            v = math.nan
        elif dtype == torch.float8_e5m2 and e == 31:
            v = math.inf if m == 0 else math.nan
        elif e == 0:
            v = math.ldexp(m / (1 << m_bits), 1 - bias)
        else:
            v = math.ldexp(1 + m / (1 << m_bits), e - bias)
        vals.append(-v if s else v)
    return torch.tensor(vals, dtype=torch.float64)


def e2m1_table() -> torch.Tensor:
    """float64 value of the 16 E2M1 codes from the bit fields (1 sign, 2 exponent with bias 1, 1 mantissa)."""
    vals = []
    for c in range(16):
        s, e, m = c >> 3, (c >> 1) & 3, c & 1
        v = m * 0.5 if e == 0 else math.ldexp(1 + m / 2, e - 1)
        vals.append(-v if s else v)  # code 8 is -0.0
    return torch.tensor(vals, dtype=torch.float64)


E8M0_F64 = torch.tensor([math.ldexp(1.0, b - 127) for b in range(255)] + [math.inf], dtype=torch.float64)


def scale_f64(s: torch.Tensor) -> torch.Tensor:
    if s.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return s.to(torch.float64)
    return E8M0_F64[s.view(torch.uint8).long()]


def ref_blocks(x64: torch.Tensor, s64: torch.Tensor, block, out: torch.dtype) -> torch.Tensor:
    """The definition: out[..., i, j] = fp32(x[..., i, j] * s[..., i // bm, j // bn]) cast to `out`. The float64
    product is exact (at most 4 + 24 significant bits), so .float() rounds it once, as a float32 multiply does."""
    bm, bn = block
    rows, cols = x64.shape[-2:]
    full = s64[..., torch.arange(rows) // bm, :][..., torch.arange(cols) // bn]
    return (x64 * full).to(torch.float32).to(out)


def rand_fp8(shape, dtype, g) -> torch.Tensor:
    """Random finite FP8 codes; the first 256 elements run through every byte value (non-finite ones -> 0)."""
    b = torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)
    flat = b.view(-1)
    k = min(256, flat.numel())
    flat[:k] = torch.arange(k, dtype=torch.uint8)
    bad = ~torch.isfinite(fp8_table(dtype))[b.long()]
    b[bad] = 0
    return b.view(dtype)


def rand_scale(grid, kind: str, g) -> torch.Tensor:
    if kind in ("u8", "e8m0"):
        s = torch.randint(100, 150, grid, dtype=torch.uint8, generator=g)
        return s if kind == "u8" else s.view(E8M0)
    s = torch.exp(torch.randn(grid, generator=g) * 2 - 6)
    return s.to({"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[kind])


def scale_kinds() -> list[str]:
    return ["f32", "bf16", "f16", "u8"] + (["e8m0"] if E8M0 is not None else [])


def hf_fp8():
    try:
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return Fp8Dequantize(None), None


def hf_mxfp4():
    try:
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return convert_moe_packed_tensors, None


def skip(what: str, why: str) -> None:
    SKIPPED.append(f"{what}: {why}")


# ---------------------------------------------------------------- tests

def test_formats():
    for dtype in dq.FP8_DTYPES:
        got = torch.arange(256, dtype=torch.uint8).view(dtype).to(torch.float64)
        want = fp8_table(dtype)
        same = (got == want) | (torch.isnan(got) & torch.isnan(want))
        assert bool(same.all()), f"torch {dtype} -> float disagrees with the OCP definition"
        assert_bits(torch.signbit(got), torch.signbit(want), f"{dtype} signs")
    assert_bits(torch.tensor(dq.FP4_E2M1, dtype=torch.float64), e2m1_table(), "FP4_E2M1 vs E2M1 bit fields")


def test_e8m0():
    b = torch.arange(256, dtype=torch.uint8)
    got = dq.e8m0_to_f32(b)
    assert got.dtype == torch.float32
    assert_bits(got, (b.to(torch.float32) - 127.0).exp2(), "e8m0 vs transformers' (b - 127).exp2()")
    assert_bits(got, E8M0_F64.to(torch.float32), "e8m0 vs math.ldexp")
    assert got[0].item() == 2.0 ** -127 and got[127].item() == 1.0 and got[255].item() == math.inf
    if E8M0 is not None:
        tv = b.view(E8M0).to(torch.float32)
        assert_bits(got[:255], tv[:255], "e8m0 vs torch float8_e8m0fnu (0..254)")
        assert bool(torch.isnan(tv[255])), "torch float8_e8m0fnu 0xFF should be NaN"
        assert_bits(dq.e8m0_to_f32(b.view(E8M0)), got, "e8m0 from float8_e8m0fnu input")
        assert_bits(dq.scale_to_f32(b.view(E8M0)), got, "scale_to_f32(float8_e8m0fnu)")
    else:
        skip("e8m0 float8_e8m0fnu", "torch has no float8_e8m0fnu")
    assert_bits(dq.scale_to_f32(b), got, "scale_to_f32(uint8)")


def test_fp8_full_grid():
    g = torch.Generator().manual_seed(1)
    hf, why = hf_fp8()
    if hf is None:
        skip("fp8 full vs transformers", why)
    cases = [((256, 384), (128, 128)), ((3, 256, 128), (128, 128)), ((96, 64), (32, 32)), ((2, 128, 96), (64, 32))]
    n = 0
    for dtype in dq.FP8_DTYPES:
        for shape, block in cases:
            q = rand_fp8(shape, dtype, g)
            grid = (*shape[:-2], *dq.ceil_grid(shape[-2], shape[-1], block))
            for kind in scale_kinds():
                s = rand_scale(grid, kind, g)
                for out in (torch.bfloat16, torch.float32, torch.float16):
                    what = f"fp8 {dtype} {shape} block {block} scale {kind} -> {out}"
                    got = dq.fp8_block(q, s, block, out)
                    assert_bits(got, ref_blocks(fp8_table(dtype)[q.view(torch.uint8).long()], scale_f64(s), block,
                                                out), what + " vs float64")
                    if hf is not None:
                        assert_bits(got, hf._dequantize_one(q, s, out), what + " vs transformers")
                    n += 1
    assert n == 2 * len(cases) * len(scale_kinds()) * 3


def test_fp8_partial_grid():
    g = torch.Generator().manual_seed(2)
    hf, why = hf_fp8()
    if hf is None:
        skip("fp8 partial vs transformers", why)
    cases = [((576, 384), (128, 128)), ((100, 200), (128, 128)), ((129, 257), (128, 128)), ((3, 130, 70), (64, 32))]
    for shape, block in cases:
        q = rand_fp8(shape, torch.float8_e4m3fn, g)
        grid = (*shape[:-2], *dq.ceil_grid(shape[-2], shape[-1], block))
        for kind in ("f32", "bf16", "u8"):
            s = rand_scale(grid, kind, g)
            for out in (torch.bfloat16, torch.float32):
                what = f"fp8 partial {shape} block {block} scale {kind} -> {out}"
                got = dq.fp8_block(q, s, block, out)
                assert got.is_contiguous(), what + ": result not contiguous"
                assert_bits(got, ref_blocks(fp8_table(torch.float8_e4m3fn)[q.view(torch.uint8).long()],
                                            scale_f64(s), block, out), what + " vs float64")
                if hf is not None:
                    rows, cols = shape[-2:]
                    pr, pc = grid[-2] * block[0], grid[-1] * block[1]
                    qp = torch.zeros((*shape[:-2], pr, pc), dtype=torch.uint8)
                    qp[..., :rows, :cols] = q.view(torch.uint8)
                    want = hf._dequantize_one(qp.view(q.dtype), s, out)[..., :rows, :cols]
                    assert_bits(got, want, what + " vs zero-pad -> transformers -> crop")


def test_fp8_roundtrip():
    g = torch.Generator().manual_seed(3)
    for shape in ((512, 768), (2, 300, 260)):
        w = torch.randn(shape, generator=g) * 0.02
        w.view(-1)[:: 997] *= 40  # outliers, so blocks differ in range
        block = (128, 128)
        grid = (*shape[:-2], *dq.ceil_grid(shape[-2], shape[-1], block))
        rows, cols = shape[-2:]
        pad = torch.zeros((*shape[:-2], grid[-2] * 128, grid[-1] * 128))
        pad[..., :rows, :cols] = w.abs()
        amax = pad.view(*shape[:-2], grid[-2], 128, grid[-1], 128).amax(dim=(-3, -1))
        ri, ci = torch.arange(rows) // 128, torch.arange(cols) // 128
        for kind in ("f32", "pow2"):
            if kind == "f32":
                s = amax / 448.0
                s_store = s
            else:  # DeepSeek ue8m0: the smallest power of two >= amax / 448, stored as an E8M0 byte
                k = torch.ceil(torch.log2(amax / 448.0))
                s = torch.pow(2.0, k)
                s_store = (k + 127).to(torch.uint8)
            sfull = s[..., ri, :][..., ci]
            q = (w / sfull).clamp(-448, 448).to(torch.float8_e4m3fn)
            w_hat = dq.fp8_block(q, s_store, block, torch.float32)
            q2 = (w_hat / sfull).to(torch.float8_e4m3fn)
            assert_bits(q2, q, f"fp8 round trip {shape} {kind}")
            if kind == "pow2":
                assert_bits(w_hat, (q.to(torch.float64) * sfull.to(torch.float64)).to(torch.float32),
                            f"fp8 power-of-two decode exact {shape}")
            err = ((w_hat - w).abs() / sfull).max().item()
            assert err <= 16.0, f"fp8 round trip {shape} {kind}: error {err} block-scale units"


def mx_encode(v: torch.Tensor):
    """Reference OCP MX encoder: [..., n, k] -> blocks [..., n, k/32, 16] uint8 (low nibble = even column), scales
    [..., n, k/32] uint8. Shared exponent floor(log2(amax)) - 2; elements to the nearest E2M1 value, saturating."""
    *lead, n, k = v.shape
    vb = v.to(torch.float64).reshape(*lead, n, k // 32, 32)
    amax = vb.abs().amax(-1)
    e = (torch.floor(torch.log2(amax.clamp_min(2.0 ** -127))) - 2).clamp(-127, 127)
    scaled = vb / torch.pow(2.0, e).unsqueeze(-1)
    mags = e2m1_table()[:8]
    code = (scaled.abs().clamp(max=6.0).unsqueeze(-1) - mags).abs().argmin(-1)
    code = code + 8 * torch.signbit(scaled).long()
    code = code.reshape(*lead, n, k // 32, 16, 2)
    blocks = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return blocks, (e + 127).to(torch.uint8)


def test_mxfp4():
    g = torch.Generator().manual_seed(4)
    hf, why = hf_mxfp4()
    if hf is None:
        skip("mxfp4 vs transformers", why)
    e2m1 = e2m1_table()
    for shape in ((3, 40, 5), (2, 7, 3), (6, 4)):  # [..., n, G]
        blocks = torch.randint(0, 256, (*shape, 16), dtype=torch.uint8, generator=g)
        blocks.view(-1)[:256] = torch.arange(256, dtype=torch.uint8)
        scales = torch.randint(100, 150, shape, dtype=torch.uint8, generator=g)
        scales.view(-1)[:2] = torch.tensor([0, 1], dtype=torch.uint8)  # subnormal and smallest normal exponents
        for out in (torch.bfloat16, torch.float32):
            got = dq.mxfp4(blocks, scales, out)
            assert tuple(got.shape) == (*shape[:-1], shape[-1] * 32)
            vals = torch.stack([e2m1[(blocks & 15).long()], e2m1[(blocks >> 4).long()]], -1)  # [..., n, G, 16, 2]
            vals = vals.reshape(*shape, 32) * E8M0_F64[scales.long()].unsqueeze(-1)
            want = vals.reshape(*shape[:-1], shape[-1] * 32).to(torch.float32).to(out)
            assert_bits(got, want, f"mxfp4 {shape} -> {out} vs float64")
            if hf is not None and len(shape) == 3:
                assert_bits(got, hf(blocks, scales, dtype=out).transpose(1, 2),
                            f"mxfp4 {shape} -> {out} vs convert_moe_packed_tensors(b, s).transpose(1, 2)")
            if E8M0 is not None:
                assert_bits(dq.mxfp4(blocks, scales.view(E8M0), out), got, "mxfp4 float8_e8m0fnu scales")
    # round trip through the reference encoder
    v = torch.randn(4, 24, 96, generator=g) * torch.exp(torch.randn(4, 24, 1, generator=g))
    b1, s1 = mx_encode(v)
    v_hat = dq.mxfp4(b1, s1, torch.float32)
    b2, s2 = mx_encode(v_hat)
    assert_bits(s2, s1, "mxfp4 round trip scales")
    assert_bits(b2, b1, "mxfp4 round trip blocks")


def test_fp4_i8_e8m0():
    g = torch.Generator().manual_seed(5)
    hf, why = hf_fp8()
    if hf is None:
        skip("fp4_i8_e8m0 vs transformers", why)
    e2m1 = e2m1_table()
    for shape, block in (((64, 48), (1, 32)), ((2, 32, 64), (1, 32)), ((64, 40), (1, 32)), ((96, 64), (32, 32))):
        q8 = torch.randint(-128, 128, shape, dtype=torch.int8, generator=g)
        q8.view(-1)[:256] = torch.arange(-128, 128, dtype=torch.int8)
        rows, cols = shape[-2], 2 * shape[-1]
        grid = (*shape[:-2], *dq.ceil_grid(rows, cols, block))
        s = torch.randint(100, 150, grid, dtype=torch.uint8, generator=g)
        u = q8.view(torch.uint8).long()
        x64 = torch.stack([e2m1[u & 15], e2m1[u >> 4]], -1).reshape(*shape[:-1], cols)
        for out in (torch.bfloat16, torch.float32):
            what = f"fp4_i8_e8m0 {shape} block {block} -> {out}"
            got = dq.fp4_i8_e8m0(q8, s, block, out)
            assert_bits(got, ref_blocks(x64, scale_f64(s), block, out), what + " vs float64")
            assert_bits(dq.fp4_i8_e8m0(q8.view(torch.uint8), s, block, out), got, what + " (uint8 codes)")
            if FP4X2 is not None:
                assert_bits(dq.fp4_i8_e8m0(q8.view(FP4X2), s, block, out), got, what + " (float4_e2m1fn_x2)")
            if E8M0 is not None:
                assert_bits(dq.fp4_i8_e8m0(q8, s.view(E8M0), block, out), got, what + " (float8_e8m0fnu scales)")
            if hf is not None and cols % block[1] == 0 and rows % block[0] == 0:
                assert_bits(got, hf._dequantize_one(q8, s, out), what + " vs transformers")
    if FP4X2 is None:
        skip("fp4_i8_e8m0 float4_e2m1fn_x2", "torch has no float4_e2m1fn_x2")


def test_layout():
    g = torch.Generator().manual_seed(6)
    q = rand_fp8((4, 256, 384), torch.float8_e4m3fn, g)
    s = rand_scale((4, 2, 3), "f32", g)
    full = dq.fp8_block(q, s)
    assert_bits(dq.fp8_block(q[1], s[1]), full[1], "fp8 dim-0 slice")
    assert_bits(dq.fp8_block(q[1:3], s[1:3]), full[1:3], "fp8 dim-0 range")
    qt, st = q[2].t(), s[2].t()  # non-contiguous [384, 256] with scale [3, 2]
    assert not qt.is_contiguous()
    assert_bits(dq.fp8_block(qt, st), dq.fp8_block(qt.contiguous(), st.contiguous()), "fp8 transposed view")
    assert_bits(dq.fp8_block(qt, st), full[2].t().contiguous(), "fp8 transposed == transpose of decode")
    qs = q[:, ::2]  # strided rows: [4, 128, 384]; every other row of each expert
    assert_bits(dq.fp8_block(qs, s[:, :1]), dq.fp8_block(qs.contiguous(), s[:, :1].contiguous()), "fp8 strided rows")
    blocks = torch.randint(0, 256, (4, 40, 6, 16), dtype=torch.uint8, generator=g)
    scales = torch.randint(110, 140, (4, 40, 6), dtype=torch.uint8, generator=g)
    mfull = dq.mxfp4(blocks, scales)
    assert_bits(dq.mxfp4(blocks[1:3], scales[1:3]), mfull[1:3], "mxfp4 dim-0 range")
    assert_bits(dq.mxfp4(blocks[:, ::2], scales[:, ::2]), mfull[:, ::2], "mxfp4 strided rows")
    q8 = torch.randint(-128, 128, (3, 64, 48), dtype=torch.int8, generator=g)
    s8 = torch.randint(110, 140, (3, 64, 3), dtype=torch.uint8, generator=g)
    ffull = dq.fp4_i8_e8m0(q8, s8)
    assert_bits(dq.fp4_i8_e8m0(q8[1:], s8[1:]), ffull[1:], "fp4_i8_e8m0 dim-0 range")
    assert_bits(dq.fp4_i8_e8m0(q8[:, ::2], s8[:, ::2]), ffull[:, ::2], "fp4_i8_e8m0 strided rows")
    w = torch.randn(8, 16, generator=g).to(torch.bfloat16)
    assert dq.plain(w) is w or torch.equal(dq.plain(w), w)
    assert_bits(dq.plain(w, torch.float32), w.to(torch.float32), "plain bf16 -> f32")


def raises(exc, fn, *a, **k) -> None:
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


def test_errors():
    q = torch.zeros(256, 384, dtype=torch.float8_e4m3fn)
    s = torch.ones(2, 3)
    raises(ValueError, dq.fp8_block, q, torch.ones(2, 2))          # wrong grid
    raises(ValueError, dq.fp8_block, q, torch.ones(3, 3))          # floor/ceil confusion would show here
    raises(ValueError, dq.fp8_block, q, s, (64, 64))               # block does not match the grid
    raises(ValueError, dq.fp8_block, q, s, (0, 128))
    raises(ValueError, dq.fp8_block, q[0], torch.ones(3))          # 1-D
    raises(ValueError, dq.fp8_block, q.expand(2, 256, 384), s)     # leading dims must match
    raises(TypeError, dq.fp8_block, q.view(torch.uint8), s)        # raw bytes: format unknown
    raises(TypeError, dq.fp8_block, q, s.to(torch.int32))
    raises(TypeError, dq.fp8_block, q, s.to(torch.float64))
    for d in FNUZ:
        raises(TypeError, dq.fp8_block, q.view(d), s)
    blocks = torch.zeros(4, 6, 16, dtype=torch.uint8)
    raises(ValueError, dq.mxfp4, blocks, torch.zeros(4, 5, dtype=torch.uint8))
    raises(TypeError, dq.mxfp4, blocks.view(torch.int8), torch.zeros(4, 6, dtype=torch.uint8))
    raises(TypeError, dq.mxfp4, blocks, torch.zeros(4, 6))
    raises(TypeError, dq.fp4_i8_e8m0, torch.zeros(4, 16, dtype=torch.int8), torch.ones(4, 1))  # E8M0 only
    raises(TypeError, dq.fp4_i8_e8m0, torch.zeros(4, 16, dtype=torch.int16), torch.zeros(4, 1, dtype=torch.uint8))
    raises(ValueError, dq.fp4_i8_e8m0, torch.zeros(4, 16, dtype=torch.int8), torch.zeros(4, 2, dtype=torch.uint8))
    raises(TypeError, dq.e8m0_to_f32, torch.zeros(3, dtype=torch.int32))
    raises(TypeError, dq.plain, torch.zeros(3, dtype=torch.int8))


def test_registry():
    for name, fn in (("plain", dq.plain), ("fp8_block", dq.fp8_block), ("mxfp4", dq.mxfp4),
                     ("fp4_i8_e8m0", dq.fp4_i8_e8m0)):
        assert dq.DECODERS[name] is fn, name
    raises(ValueError, dq.register_decoder("fp8_block"), lambda *a: None)
    try:
        dq.register_decoder("_test_dummy")(lambda w: w)
        assert "_test_dummy" in dq.DECODERS
    finally:
        dq.DECODERS.pop("_test_dummy", None)


def test_device_parity():
    if os.environ.get("DEQUANT_TEST_CUDA") != "1":
        skip("device parity", "not requested (--cuda / DEQUANT_TEST_CUDA=1)")
        return
    assert torch.cuda.is_available(), "DEQUANT_TEST_CUDA=1 but no GPU is visible"
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(7)
    for shape, block in (((3, 256, 384), (128, 128)), ((576, 384), (128, 128)), ((96, 64), (32, 32))):
        grid = (*shape[:-2], *dq.ceil_grid(shape[-2], shape[-1], block))
        for dtype in dq.FP8_DTYPES:
            q = rand_fp8(shape, dtype, g)
            for kind in scale_kinds():
                s = rand_scale(grid, kind, g)
                if kind == "u8":
                    s.view(-1)[:3] = torch.tensor([0, 1, 254], dtype=torch.uint8)
                for out in (torch.bfloat16, torch.float32):
                    assert_bits(dq.fp8_block(q.to(dev), s.to(dev), block, out).cpu(), dq.fp8_block(q, s, block, out),
                                f"device fp8 {dtype} {shape} {kind} {out}")
    blocks = torch.randint(0, 256, (4, 64, 9, 16), dtype=torch.uint8, generator=g)
    scales = torch.randint(100, 150, (4, 64, 9), dtype=torch.uint8, generator=g)
    scales.view(-1)[:3] = torch.tensor([0, 1, 254], dtype=torch.uint8)
    for out in (torch.bfloat16, torch.float32):
        assert_bits(dq.mxfp4(blocks.to(dev), scales.to(dev), out).cpu(), dq.mxfp4(blocks, scales, out),
                    f"device mxfp4 {out}")
    q8 = torch.randint(-128, 128, (2, 64, 40), dtype=torch.int8, generator=g)
    s8 = torch.randint(100, 150, (2, 64, 3), dtype=torch.uint8, generator=g)
    for out in (torch.bfloat16, torch.float32):
        assert_bits(dq.fp4_i8_e8m0(q8.to(dev), s8.to(dev), (1, 32), out).cpu(), dq.fp4_i8_e8m0(q8, s8, (1, 32), out),
                    f"device fp4_i8_e8m0 {out}")
    b = torch.arange(256, dtype=torch.uint8)
    assert_bits(dq.e8m0_to_f32(b.to(dev)).cpu(), dq.e8m0_to_f32(b), "device e8m0")


TESTS = [test_formats, test_e8m0, test_fp8_full_grid, test_fp8_partial_grid, test_fp8_roundtrip, test_mxfp4,
         test_fp4_i8_e8m0, test_layout, test_errors, test_registry, test_device_parity]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", help="also compare every decoder on the GPU with the CPU")
    args = ap.parse_args()
    if args.cuda:
        os.environ["DEQUANT_TEST_CUDA"] = "1"
    try:
        import transformers
        tf = transformers.__version__
    except Exception:  # noqa: BLE001
        tf = None
    print(f"torch {torch.__version__}  transformers {tf}  python {sys.version.split()[0]}")
    fails = 0
    for t in TESTS:
        t0 = time.perf_counter()
        try:
            t()
            status = "PASS"
        except Exception:  # noqa: BLE001
            fails += 1
            status = "FAIL"
            traceback.print_exc()
        print(f"{status} {t.__name__} ({time.perf_counter() - t0:.2f} s)")
    for s in SKIPPED:
        print(f"SKIPPED {s}")
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
