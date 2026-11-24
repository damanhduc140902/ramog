"""KLinear: a weight-only k4row nn.Linear for the dense part (dense-c: attention, lm_head, router).

Weights are k4row-packed (behemoe.kquant): codes q (2/4/8 bits), one byte per group sb = sc | mc << 4, fp16 d and
dmin per row; w = d * sc * q - dmin * mc, group 16 or 32 along `in`. The format is behemoe.kernels' FMT 1, so both
paths run the existing kernels (imported, unchanged) on a one-slot pool:
  decode  (<= DECODE_MAX_TOKENS rows): the grouped dequant-GEMV (kernels._gemv_body), float32 throughout. With
          deterministic=True (default) it is launched through an autotuner restricted to SPLIT_K <= 2 configs (at
          most two atomic adds onto zero, which commute), so every call (eager or graph replay) gives the same bits;
          deterministic=False uses kernels.grouped_gemv (all configs, split-K atomics allowed).
  prefill (more rows): the grouped dequant-GEMM (kernels._gemm_body) with a single "expert". prefill="f32" (default)
          launches it with DOT_F32 = True on every GPU: the dequantized weights stay float32 and tl.dot runs in IEEE
          float32 (the kernel reads x as bf16, which is exact for bf16 activations); prefill="bf16" is the engine's
          MoE numerics (kernels.moe_gemm: weights rounded to bf16 on GPUs with bf16 matrix units). Inputs that are
          not bf16 take the exact fallback (dequantize row blocks to float32, then matmul) so nothing is rounded.
Both paths return float32 sums, cast to x.dtype after the (float32) bias; matmul_f32() returns them uncast.
Graph capture: call warmup() (or one eager forward per token count) first so the autotuners have picked their
configurations; afterwards a forward launches only kernels and allocates from the graph pool, no host sync.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton

from . import gptq as _gptq
from . import kernels, kquant
from .gemv1 import bytes_fit, gemv1b

DECODE_MAX_TOKENS = 8
PREFILL_CHUNK_BYTES = 256 << 20  # float32 output per prefill pass (as behemoe.qlinear)
SUPPORTED_BITS = (2, 4, 8)
SUPPORTED_GROUPS = (16, 32)


# ------------------------------------------------------------------------------------------- deterministic decode

def _det_prune(configs, named_args, **kwargs):
    words = named_args["K"] * kwargs["BITS"] // 32
    ok = [c for c in configs if words % c.kwargs["WORDS"] == 0]
    if not ok:
        raise ValueError(f"no SPLIT_K<=2 GEMV config divides {words} words per row")
    return ok


# SPLIT_K <= 2 is deterministic: y starts at exact zero and receives at most two atomic adds, 0 + a + b == 0 + b + a
# (IEEE addition commutes; -0 and inf/nan behave the same in either order). Only partial sums below the float32
# normal range (< 1.2e-38, where the GPU's float atomics may flush) could break the tie. SPLIT_K >= 3 is not.
_DET_CONFIGS = [c for c in kernels._configs() if c.kwargs["SPLIT_K"] <= 2]
_gemv_det = triton.autotune(configs=_DET_CONFIGS, key=list(kernels._KEY), restore_value=["y_ptr"],
                            prune_configs_by={"early_config_prune": _det_prune})(kernels._gemv_body)


def gemv_supported(k: int, bits: int) -> bool:
    words = k * bits // 32
    return k * bits % 32 == 0 and any(words % c.kwargs["WORDS"] == 0 for c in _DET_CONFIGS)


_BLOCKS: dict = {}


def _one_expert_blocks(n: int, block_m: int, device):
    """(order, blocks) of moe_gemm for n rows of a single expert (cached per device, n and block_m)."""
    key = (str(device), n, block_m)
    if key not in _BLOCKS:
        b = torch.arange(-(-n // block_m), device=device)
        blocks = torch.stack([torch.zeros_like(b), b * block_m, (n - b * block_m).clamp(max=block_m)], 1)
        _BLOCKS[key] = (torch.arange(n, device=device, dtype=torch.int32), blocks.to(torch.int32))
    return _BLOCKS[key]


class KLinear(nn.Module):
    def __init__(self, qweight: torch.Tensor, sb: torch.Tensor, d: torch.Tensor, dmin: torch.Tensor, bits: int,
                 group: int, bias: torch.Tensor | None = None, *, prefill: str = "f32", deterministic: bool = True,
                 device=None):
        """qweight uint8 [out, in * bits / 8], sb uint8 [out, in / group], d / dmin fp16 [out] (kquant.KPacked
        fields), moved to `device` (default: qweight's)."""
        super().__init__()
        if bits not in SUPPORTED_BITS or group not in SUPPORTED_GROUPS:
            raise ValueError(f"KLinear: bits {bits} / group {group} not in {SUPPORTED_BITS} / {SUPPORTED_GROUPS}")
        if prefill not in ("f32", "bf16"):
            raise ValueError(f"KLinear: prefill must be 'f32' or 'bf16', got {prefill!r}")
        out_f, nb = qweight.shape
        in_f = nb * 8 // bits
        if sb.shape != (out_f, in_f // group) or d.shape != (out_f,) or dmin.shape != (out_f,):
            raise ValueError(f"KLinear: inconsistent packed shapes {tuple(qweight.shape)} {tuple(sb.shape)} "
                             f"{tuple(d.shape)} {tuple(dmin.shape)} for bits {bits} group {group}")
        if qweight.dtype != torch.uint8 or sb.dtype != torch.uint8 or d.dtype != torch.float16 \
                or dmin.dtype != torch.float16:
            raise TypeError("KLinear: qweight/sb uint8, d/dmin float16")
        dev = qweight.device if device is None else torch.device(device)
        self.register_buffer("qweight", qweight.to(dev).contiguous().unsqueeze(0))  # [1, out, in*b/8]: a 1-slot pool
        self.register_buffer("sb", sb.to(dev).contiguous().unsqueeze(0))
        self.register_buffer("d", d.to(dev).contiguous().unsqueeze(0))
        self.register_buffer("dmin", dmin.to(dev).contiguous().unsqueeze(0))
        self.register_buffer("bias", None if bias is None else bias.to(dev).clone())
        self.bits, self.group = bits, group
        self.in_features, self.out_features = in_f, out_f
        self.prefill, self.deterministic = prefill, deterministic
        self.gemv_ok = gemv_supported(in_f, bits)
        self.register_buffer("_slot", torch.zeros(DECODE_MAX_TOKENS, dtype=torch.int32, device=dev), persistent=False)
        self.register_buffer("_tok", torch.arange(DECODE_MAX_TOKENS, dtype=torch.int32, device=dev), persistent=False)

    # ------------------------------------------------------------------------------------------- constructors
    @classmethod
    def from_packed(cls, p: kquant.KPacked, bias: torch.Tensor | None = None, **kw) -> KLinear:
        return cls(p.qweight, p.sb, p.d, p.dmin, p.bits, p.group, bias, **kw)

    @classmethod
    @torch.no_grad()
    def from_float(cls, W: torch.Tensor, bits: int, group: int, importance: torch.Tensor | None = None,
                   gptq_H: torch.Tensor | dict | None = None, *, bias: torch.Tensor | None = None, device=None,
                   damp: float = _gptq.DAMP, act_order: bool = True, gptq_path: str = "cpu",
                   info: dict | None = None, **kw) -> KLinear:
        """Quantize W [out, in] (nn.Linear layout) on `device` (default: W's device; a GPU for real sizes).
        importance [in]: the grid search's per-input-channel weights (gptq.importance(entry, ...)), None = MSE.
        gptq_H: None = plain k4row rounding (behemoe.kquant.quantize); else GPTQ (behemoe.gptq) with H = X^T X / n
        [in, in] or a tools/dense_stats.py entry {xtx, count, ...}; damp, act_order and the factorization path
        (cpu = T1 bit for bit, gpu = faster) as in behemoe.gptq.quantize."""
        dev = W.device if device is None else torch.device(device)
        kp = _gptq.quantize(W.to(dev), bits, group, importance, gptq_H, damp=damp, act_order=act_order,
                            path=gptq_path, info=info)
        return cls.from_packed(kp, bias, **kw)

    # ------------------------------------------------------------------------------------------- weights
    def packed(self) -> kquant.KPacked:
        return kquant.KPacked(self.qweight[0], self.sb[0], self.d[0], self.dmin[0], self.bits, self.group)

    def dequantize(self, dtype=torch.float32, rows: slice | None = None) -> torch.Tensor:
        """The dequantized weight [out, in] (rows: a block of rows), behemoe.kquant.dequantize's float32 math."""
        r = slice(None) if rows is None else rows
        return kquant.dequantize(self.qweight[0, r], self.sb[0, r], self.d[0, r], self.dmin[0, r], self.bits,
                                 self.group, dtype)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized bf16 copy, for code that reads .weight (tied heads, probes). Costly: avoid in hot paths."""
        return self.dequantize(torch.bfloat16)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.qweight, self.sb, self.d, self.dmin))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, k4row {self.bits}-bit "
                f"g{self.group}, prefill={self.prefill}, deterministic={self.deterministic}, "
                f"bias={self.bias is not None}")

    # ------------------------------------------------------------------------------------------- compute
    def _gemv(self, x2: torch.Tensor) -> torch.Tensor:
        t = x2.shape[0]
        if t == 1 and x2.is_cuda and bytes_fit(self.in_features, self.bits, self.group):
            # the same kernel as forward()'s bf16 fast path, float32 out: forward stays this result's cast
            return gemv1b(x2, self.qweight[0], self.sb[0], self.d[0], self.dmin[0], self.bits, self.group,
                          out_dtype=torch.float32).view(1, self.out_features)
        if not self.deterministic:
            return kernels.grouped_gemv(x2, self._tok[:t], self._slot[:t], self.qweight, self.sb, None, self.bits,
                                        self.group, fmt=1, d=self.d, dmin=self.dmin)
        n, k = self.out_features, self.in_features
        y = torch.zeros(t, n, device=x2.device, dtype=torch.float32)
        qw32 = self.qweight.view(torch.int32)
        bucket = 1 << max(0, (t - 1).bit_length())
        tok, slot = self._tok[:t], self._slot[:t]
        args = (x2, tok, slot, slot, qw32, self.sb, self.sb, self.d, self.dmin, y, n, k, bucket,
                x2.stride(0), qw32.stride(0), qw32.stride(1), self.sb.stride(0), self.sb.stride(1),
                self.d.stride(0), y.stride(0))

        def grid(meta):
            return (t, triton.cdiv(n, meta["BLOCK_N"]), meta["SPLIT_K"])

        _gemv_det[grid](*args, BITS=self.bits, GROUP=self.group, FMT=1, MODE=0)
        return y

    def _gemm(self, x2: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        t = x2.shape[0]
        bm = kernels.block_m_for(t, 1)
        order, blocks = _one_expert_blocks(t, bm, x2.device)
        if self.prefill == "bf16":
            return kernels.moe_gemm(x2, order, blocks, None, self.qweight, self.sb, None, self.bits, self.group,
                                    num_experts=1, out=out, block_m=bm, mode=0, fmt=1, d=self.d, dmin=self.dmin)
        n, k = self.out_features, self.in_features
        qw32 = self.qweight.view(torch.int32)
        args = (x2, order, blocks, order, qw32, self.sb, self.sb, self.d, self.dmin, out, n, k, 1, 1,
                x2.stride(0), qw32.stride(0), qw32.stride(1), self.sb.stride(0), self.sb.stride(1), self.d.stride(0),
                out.stride(0))

        def grid(meta):
            return (blocks.shape[0], triton.cdiv(n, meta["BLOCK_N"]))

        kernels._moe_gemm[grid](*args, BITS=self.bits, GROUP=self.group, FMT=1, MODE=0, BLOCK_M=bm, DOT_F32=True)
        return out

    def _dequant_matmul(self, x2: torch.Tensor) -> torch.Tensor:
        """Exact fallback: float32 row blocks of the dequantized weight times float32 x."""
        y = torch.empty(x2.shape[0], self.out_features, device=x2.device, dtype=torch.float32)
        xf = x2.float()
        step = 8192
        for r0 in range(0, self.out_features, step):
            r1 = min(self.out_features, r0 + step)
            y[:, r0:r1] = xf @ self.dequantize(torch.float32, slice(r0, r1)).t()
        return y

    def matmul_f32(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., in] -> float32 [..., out] = x @ dequant(W)^T (+ bias), before the cast to x.dtype."""
        shape = x.shape
        if shape[-1] != self.in_features:
            raise ValueError(f"KLinear: input has {shape[-1]} features, expected {self.in_features}")
        x2 = x.reshape(-1, shape[-1]).contiguous()
        t = x2.shape[0]
        if t == 0:
            y = torch.empty(0, self.out_features, device=x.device, dtype=torch.float32)
        elif not self.gemv_ok:
            y = self._dequant_matmul(x2)
        elif t <= DECODE_MAX_TOKENS:
            y = self._gemv(x2)
        elif x2.dtype != torch.bfloat16:  # the GEMM kernel reads x as bf16: keep other dtypes exact
            y = self._dequant_matmul(x2)
        else:
            y = torch.empty(t, self.out_features, device=x.device, dtype=torch.float32)
            step = max(64, PREFILL_CHUNK_BYTES // (4 * self.out_features))
            for r0 in range(0, t, step):
                r1 = min(t, r0 + step)
                if r1 - r0 <= DECODE_MAX_TOKENS:  # a short tail chunk: same math, GEMV kernel
                    y[r0:r1] = self._gemv(x2[r0:r1])
                else:
                    self._gemm(x2[r0:r1], y[r0:r1])
        if self.bias is not None:
            y = y + self.bias.float()
        return y.reshape(*shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dtype == torch.bfloat16 and self.bias is None and self.gemv_ok and x.numel() == self.in_features
                and x.is_cuda and bytes_fit(self.in_features, self.bits, self.group)):
            # one token: behemoe.gemv1's byte-wise kernel writes bf16 directly (no fill / cast kernels, deterministic);
            # matmul_f32 runs its float32 twin, so forward() is its cast within one bf16 ulp (two compiled kernels)
            return gemv1b(x, self.qweight[0], self.sb[0], self.d[0], self.dmin[0], self.bits,
                          self.group).view(*x.shape[:-1], self.out_features)
        return self.matmul_f32(x).to(x.dtype)

    @torch.no_grad()
    def warmup(self, token_counts=(1, 2, 3, 4, 5, 6, 7, 8), dtype=torch.bfloat16) -> None:
        """Run once per token count so the autotuners pick their configurations before a graph capture."""
        for t in token_counts:
            self.forward(torch.zeros(t, self.in_features, device=self.qweight.device, dtype=dtype))
        torch.cuda.synchronize(self.qweight.device)
