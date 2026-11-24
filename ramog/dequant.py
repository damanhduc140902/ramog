"""Decoders for the release formats of the source checkpoints: block-scaled FP8, MXFP4, and FP4 packed in int8.

The converter, the reference streamer and the store check all read releases through these functions. The engine,
its quantizers and the reference therefore see the same weights, and a decoding bug would hide from KL. That is
why tools/check_dequant_real.py compares them bit for bit with transformers' own decoders on real tensors, and
tests/test_dequant.py checks them against float64 definitions on synthetic ones.

Formats ("out-major" = [..., out_features, in_features], nn.Linear layout):

  fp8_block    q [..., rows, cols] float8_e4m3fn or float8_e5m2, scale [..., ceil(rows/bm), ceil(cols/bn)].
               The scale is float32, bfloat16 or float16, or E8M0 bytes (uint8 or float8_e8m0fnu) worth 2^(b-127).
               out = (float32(q) * float32(scale of q's block)).to(out): one float32 product per element, then a
               cast. On full grids this is bitwise transformers' Fp8Dequantize._dequantize_one(q, scale, out).
               transformers derives the block from the shapes: it raises when the scale grid does not divide the
               weight, and it would silently pick a wrong block when a ceil grid happens to divide it. GLM-5.3 and
               DeepSeek-V3.2 kv_a_proj_with_mqa ([576, H], scale [5, H/128]) have a partial last block (it raises).
               Here the caller gives the block, the grid is the ceiling, and a partial grid gives what
               zero-pad -> transformers -> crop gives. The block is the config's weight_block_size unless a
               tensor's scale grid says otherwise: DeepSeek-V4.1's engram.embed tables use 1x32 under a 32x32
               config. The survey in tools/check_dequant_real.py lists such tensors per checkpoint.
  mxfp4        blocks [..., n, G, 16] uint8 (two E2M1 codes per byte, low nibble first), scales [..., n, G] E8M0
               -> [..., n, 32 G]. transformers' LUT and ldexp, without the final transpose(1, 2) of
               convert_moe_packed_tensors: the result stays out-major (gpt-oss gate_up [E, 2I, H], down [E, H, I];
               gate_up rows remain interleaved gate/up, as transformers' _apply_gate expects).
  fp4_i8_e8m0  q [..., rows, cols/2] int8 (or uint8 / float4_e2m1fn_x2), two E2M1 codes per byte along the last
               dim, low nibble first; scale [..., ceil(rows/bm), ceil(cols/bn)] E8M0, block (1, 32) by default.
               This is DeepSeek-V4's expert format and the FP4 path of Fp8Dequantize._dequantize_one (int8 input).
  plain        bf16 / f16 / f32 tensors: a cast.

E8M0 bytes are decoded by bit construction, exact on every device: b << 23 is the float32 2^(b-127) for
b = 1..254, b = 0 is the subnormal 2^-127, and b = 255 gives +inf. That is what transformers computes for uint8
scales, (b - 127).exp2(); the E8M0 NaN encoding (0xFF) never occurs in a valid checkpoint.

Only the OCP FP8 dtypes are accepted. The fnuz variants have a different exponent bias, so a reader that maps
F8_E4M3 to float8_e4m3fnuz would silently halve every weight; it fails here instead.

DECODERS maps a format name to its function. Converters record these names in meta.json (source_formats), and
a new release format plugs in with @register_decoder("name").
"""

from __future__ import annotations

from typing import Callable

import torch

FP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)  # transformers' LUT
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
SCALE_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
PLAIN_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
E8M0 = getattr(torch, "float8_e8m0fnu", None)
FP4X2 = getattr(torch, "float4_e2m1fn_x2", None)

DECODERS: dict[str, Callable[..., torch.Tensor]] = {}


def register_decoder(name: str):
    """Decorator: DECODERS[name] = fn. A name is registered once."""
    def deco(fn):
        if name in DECODERS:
            raise ValueError(f"decoder {name!r} is already registered")
        DECODERS[name] = fn
        return fn
    return deco


def ceil_grid(rows: int, cols: int, block: tuple[int, int]) -> tuple[int, int]:
    """Scale grid of a [rows, cols] weight in (bm, bn) blocks, partial last blocks included."""
    bm, bn = block
    return -(-rows // bm), -(-cols // bn)


def _e8m0_bytes(t: torch.Tensor) -> torch.Tensor:
    if E8M0 is not None and t.dtype == E8M0:
        return t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise TypeError(f"E8M0 scales must be uint8 or float8_e8m0fnu, got {t.dtype}")
    return t


def e8m0_to_f32(b: torch.Tensor) -> torch.Tensor:
    """E8M0 exponent bytes (uint8 or float8_e8m0fnu) -> float32 2^(b-127), exact: bitwise transformers'
    (b.float() - 127).exp2() for all 256 byte values (255 -> +inf)."""
    b = _e8m0_bytes(b)
    bits = torch.where(b == 0, 0x00400000, b.to(torch.int32) << 23)  # 2^-127 is a float32 subnormal
    return bits.view(torch.float32)


def scale_to_f32(scale: torch.Tensor) -> torch.Tensor:
    """float32 / bfloat16 / float16 scales -> float32 (exact); E8M0 bytes -> 2^(b-127)."""
    if scale.dtype in SCALE_FLOAT_DTYPES:
        return scale.to(torch.float32)
    return e8m0_to_f32(scale)


def _check_block(block) -> tuple[int, int]:
    bm, bn = (int(v) for v in block)
    if bm < 1 or bn < 1:
        raise ValueError(f"block must be positive, got {tuple(block)}")
    return bm, bn


def _scale_blocks(x: torch.Tensor, s: torch.Tensor, block: tuple[int, int], out: torch.dtype,
                  what: str) -> torch.Tensor:
    """x [..., rows, cols] float32, contiguous, owned by the caller (it is scaled in place); s [..., gr, gc] float32
    with (gr, gc) = ceil_grid. Returns x * s (each block's scale) cast to `out`: one float32 multiply per element."""
    bm, bn = block
    *lead, rows, cols = x.shape
    gr, gc = ceil_grid(rows, cols, block)
    if tuple(s.shape) != (*lead, gr, gc):
        raise ValueError(f"{what}: scale shape {tuple(s.shape)} != {(*lead, gr, gc)}, the ceil grid of weight "
                         f"{tuple(x.shape)} in {bm}x{bn} blocks")
    sv = s.reshape(*lead, gr, 1, gc, 1)
    if rows % bm == 0 and cols % bn == 0:
        x.view(*lead, gr, bm, gc, bn).mul_(sv)
        return x.to(out)
    # Partial last block(s): zero-pad to the full grid, scale, crop (the padding never reaches the result).
    xp = x.new_zeros(*lead, gr * bm, gc * bn)
    xp[..., :rows, :cols] = x
    xp.view(*lead, gr, bm, gc, bn).mul_(sv)
    return xp[..., :rows, :cols].to(out).contiguous()


@register_decoder("plain")
def plain(w: torch.Tensor, out: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """bf16 / f16 / f32 release tensor -> `out` (a cast; a no-op when the dtype already matches)."""
    if w.dtype not in PLAIN_DTYPES:
        raise TypeError(f"plain: expected one of {PLAIN_DTYPES}, got {w.dtype}")
    return w.to(out)


@register_decoder("fp8_block")
def fp8_block(q: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] = (128, 128),
              out: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Block-scaled FP8 (weight + weight_scale_inv, or DeepSeek-V4's weight + scale) -> `out` [..., rows, cols].

    q: [..., rows, cols] float8_e4m3fn or float8_e5m2. scale: [..., ceil(rows/bm), ceil(cols/bn)], float32, bfloat16,
    float16, or E8M0 bytes (uint8 / float8_e8m0fnu). block: the checkpoint's weight_block_size (128, 128 for every
    FP8 release in the model set; DeepSeek-V4's dense layers use 32, 32)."""
    if q.dtype not in FP8_DTYPES:
        raise TypeError(f"fp8_block: q must be float8_e4m3fn or float8_e5m2 (OCP), got {q.dtype}")
    if q.dim() < 2:
        raise ValueError(f"fp8_block: q must be at least 2-D, got {tuple(q.shape)}")
    x = q.to(torch.float32, memory_format=torch.contiguous_format)
    return _scale_blocks(x, scale_to_f32(scale), _check_block(block), out, "fp8_block")


def _fp4_pairs(dtype: torch.dtype, device) -> torch.Tensor:
    """[256, 2] table: byte -> (E2M1 value of its low nibble, of its high nibble), in `dtype`."""
    lut = torch.tensor(FP4_E2M1, dtype=dtype, device=device)
    b = torch.arange(256, device=device)
    return torch.stack([lut[b & 0x0F], lut[b >> 4]], dim=-1)


def _unpack_fp4(codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[..., m] uint8, two E2M1 codes per byte (low nibble first) -> [..., 2m] in `dtype` (exact)."""
    pairs = _fp4_pairs(dtype, codes.device)[codes.to(torch.int32)]  # [..., m, 2]
    return pairs.view(*codes.shape[:-1], 2 * codes.shape[-1])


@register_decoder("mxfp4")
def mxfp4(blocks: torch.Tensor, scales: torch.Tensor, out: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """MXFP4 (OCP microscaling: 32 E2M1 values share one E8M0 exponent) -> [..., n, 32 G] in `out`, out-major.

    blocks: [..., n, G, 16] uint8 (gpt-oss *_blocks); scales: [..., n, G] uint8 or float8_e8m0fnu (*_scales).
    Bitwise equal to transformers' convert_moe_packed_tensors(blocks, scales, dtype=out).transpose(1, 2): the same
    LUT in `out`, the same ldexp (a float32 product of exact values, then the cast)."""
    if blocks.dtype != torch.uint8:
        raise TypeError(f"mxfp4: blocks must be uint8, got {blocks.dtype}")
    scales = _e8m0_bytes(scales)
    if blocks.dim() < 2 or tuple(scales.shape) != tuple(blocks.shape[:-1]):
        raise ValueError(f"mxfp4: blocks {tuple(blocks.shape)} need scales {tuple(blocks.shape[:-1])}, "
                         f"got {tuple(scales.shape)}")
    y = _unpack_fp4(blocks, out)  # [..., n, G, 32]: lut[low], lut[high] alternate, as transformers writes them
    torch.ldexp(y, (scales.to(torch.int32) - 127).unsqueeze(-1), out=y)
    return y.view(*blocks.shape[:-2], blocks.shape[-2] * y.shape[-1])


@register_decoder("fp4_i8_e8m0")
def fp4_i8_e8m0(q: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] = (1, 32),
                out: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """FP4 packed two per byte along the last dim (low nibble first) with E8M0 block scales -> [..., rows, cols].

    q: [..., rows, cols/2] int8, uint8 or float4_e2m1fn_x2 (DeepSeek-V4 experts: I8). scale:
    [..., ceil(rows/bm), ceil(cols/bn)] E8M0 bytes (F8_E8M0 read as uint8, or float8_e8m0fnu). Bitwise equal to
    transformers' Fp8Dequantize._dequantize_one(q_int8, scale, out) on full grids."""
    if q.dtype in (torch.int8, torch.uint8) or (FP4X2 is not None and q.dtype == FP4X2):
        codes = q.view(torch.uint8)
    else:
        raise TypeError(f"fp4_i8_e8m0: q must be int8, uint8 or float4_e2m1fn_x2, got {q.dtype}")
    if q.dim() < 2:
        raise ValueError(f"fp4_i8_e8m0: q must be at least 2-D, got {tuple(q.shape)}")
    x = _unpack_fp4(codes, torch.float32)
    return _scale_blocks(x, e8m0_to_f32(scale), _check_block(block), out, "fp4_i8_e8m0")
