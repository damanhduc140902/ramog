"""QEmbedding: a k4row-quantized token embedding (dense-c: 4-bit codes, group 32), dequantized per token on the GPU.

Every row of the table is one self-contained record of R bytes (R a multiple of 4):
    [ d fp16 | dmin fp16 | codes: H * bits / 8 bytes | sb: H / group bytes | zero padding ]
(w = d * sc * q - dmin * mc as behemoe.kquant; d and dmin first so their fp16 views are always 2-byte aligned).
A lookup reads the R-byte records of the requested ids and dequantizes them with behemoe.kquant.dequantize's float32
arithmetic (min + step * q, step = d * sc, min = -dmin * mc), then casts to the output dtype (bf16 by default, the
dtype the model's embedding returns). The table lives either
  on the GPU   (host=False), or
  in host RAM  (host=True):  exact-size page-locked memory (behemoe.hostmem), read zero-copy over the host link like
                             the cold expert blobs and behemoe.hostemb.HostEmbedding's bf16 table.
fused=True (default): one Triton kernel per lookup reads the records (GPU or pinned host memory) and writes the
dequantized rows. step * q is exact in float32 (d has 11 significant bits, sc 4, q at most 8), so a fused
multiply-add gives the same bits as torch's separate multiply and add. fused=False: the same arithmetic in torch ops
(index_select, or behemoe.kernels.copy_blobs from host memory, then kquant.dequantize), about 25 small kernels.
Neither path synchronizes with the host, so both run inside a captured graph. Ids may be int32 or int64, on the CPU or
the GPU, of any shape and any layout (strided or expanded views are made contiguous before the gather kernels).
Either way the output equals nn.Embedding on the dequantized table (kquant.dequantize(...).to(dtype)) bit for bit,
times embed_scale.to(dtype) when one is given (Gemma-style scaled embeddings, as HostEmbedding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from . import hostmem, kquant


@triton.jit
def _qembed_rows(rows_ptr, rows16_ptr, ids_ptr, out_ptr, R, H, OFF_SB, stride_out,
                 BITS: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    """out[t, j] = dequantized element j of record ids[t]: rows_ptr = uint8 [V, R] records, rows16_ptr = the same
    memory as fp16 (d = element 0, dmin = element 1 of each record)."""
    t = tl.program_id(0)
    pb = tl.program_id(1)
    rid = tl.load(ids_ptr + t).to(tl.int64)
    base = rows_ptr + rid * R
    d = tl.load(rows16_ptr + rid * (R // 2)).to(tl.float32)
    dm = tl.load(rows16_ptr + rid * (R // 2) + 1).to(tl.float32)
    PER: tl.constexpr = 8 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    j = pb * BLOCK + tl.arange(0, BLOCK)
    ok = j < H
    byte = tl.load(base + 4 + j // PER, mask=ok, other=0).to(tl.int32)
    q = ((byte >> ((j % PER) * BITS)) & MASK).to(tl.float32)
    code = tl.load(base + OFF_SB + j // GROUP, mask=ok, other=0).to(tl.int32)
    st = d * (code & 15).to(tl.float32)
    mn = -dm * (code >> 4).to(tl.float32)
    w = mn + st * q
    tl.store(out_ptr + t.to(tl.int64) * stride_out + j, w.to(out_ptr.dtype.element_ty), mask=ok)


def row_layout(dim: int, bits: int, group: int) -> dict:
    """Byte offsets of one record: d, dmin, codes, sb, and the padded record size R."""
    if 8 % bits or dim % group or (dim * bits) % 8:
        raise ValueError(f"QEmbedding: bits {bits} / group {group} do not fit dim {dim}")
    qb, G = dim * bits // 8, dim // group
    off_q = 4
    off_sb = off_q + qb
    R = -(-(off_sb + G) // 4) * 4
    return {"d": 0, "dmin": 2, "q": off_q, "q_bytes": qb, "sb": off_sb, "groups": G, "row_bytes": R}


def pack_rows(qweight: torch.Tensor, sb: torch.Tensor, d: torch.Tensor, dmin: torch.Tensor, bits: int,
              group: int) -> torch.Tensor:
    """KPacked fields [V, ...] -> uint8 records [V, R] (on the inputs' device)."""
    V = qweight.shape[0]
    dim = qweight.shape[1] * 8 // bits
    L = row_layout(dim, bits, group)
    rows = torch.zeros(V, L["row_bytes"], dtype=torch.uint8, device=qweight.device)
    rows[:, 0:2] = d.contiguous().view(torch.uint8).view(V, 2)
    rows[:, 2:4] = dmin.contiguous().view(torch.uint8).view(V, 2)
    rows[:, L["q"]:L["q"] + L["q_bytes"]] = qweight
    rows[:, L["sb"]:L["sb"] + L["groups"]] = sb
    return rows


def unpack_rows(rows: torch.Tensor, dim: int, bits: int, group: int):
    """uint8 records [T, R] -> (codes, sb, d, dmin) views/tensors in the KPacked layout."""
    L = row_layout(dim, bits, group)
    T = rows.shape[0]
    d = rows[:, 0:2].contiguous().view(torch.float16).view(T)
    dmin = rows[:, 2:4].contiguous().view(torch.float16).view(T)
    q = rows[:, L["q"]:L["q"] + L["q_bytes"]]
    sb = rows[:, L["sb"]:L["sb"] + L["groups"]]
    return q, sb, d, dmin


class QEmbedding(nn.Module):
    def __init__(self, qweight: torch.Tensor, sb: torch.Tensor, d: torch.Tensor, dmin: torch.Tensor, bits: int = 4,
                 group: int = 32, *, host: bool = False, device=None, dtype=torch.bfloat16,
                 embed_scale: torch.Tensor | None = None, padding_idx: int | None = None, fused: bool = True):
        """qweight uint8 [V, H * bits / 8], sb uint8 [V, H / group], d / dmin fp16 [V] (kquant.KPacked fields, any
        device); device: the GPU the outputs go to (and the table, unless host=True); dtype: output dtype."""
        super().__init__()
        V = qweight.shape[0]
        self.num_embeddings, self.embedding_dim = V, qweight.shape[1] * 8 // bits
        self.bits, self.group, self.host, self.dtype, self.fused = bits, group, host, dtype, fused
        self.padding_idx = padding_idx
        self.device = torch.device(device) if device is not None else (
            qweight.device if qweight.device.type == "cuda" else torch.device("cuda", torch.cuda.current_device()))
        self.layout = row_layout(self.embedding_dim, bits, group)
        if host:
            rows = pack_rows(qweight.cpu(), sb.cpu(), d.cpu(), dmin.cpu(), bits, group)
            object.__setattr__(self, "rows", hostmem.pin_copy(rows))  # unregistered: model.to() never moves it
            del rows
        else:
            self.register_buffer("rows", pack_rows(qweight.to(self.device), sb.to(self.device), d.to(self.device),
                                                   dmin.to(self.device), bits, group))
        if embed_scale is not None:
            self.register_buffer("embed_scale", embed_scale.detach().clone().to(self.device), persistent=False)
        else:
            self.embed_scale = None
        self._want = torch.ones(1, dtype=torch.int32, device=self.device)

    # ------------------------------------------------------------------------------------------- constructors
    @classmethod
    def from_packed(cls, p: kquant.KPacked, **kw) -> QEmbedding:
        return cls(p.qweight, p.sb, p.d, p.dmin, p.bits, p.group, **kw)

    @classmethod
    @torch.no_grad()
    def from_float(cls, table: torch.Tensor, bits: int = 4, group: int = 32, importance=None, *, device=None,
                   chunk_rows: int = 32768, **kw) -> QEmbedding:
        """Quantize a [V, H] table with behemoe.kquant (dense-c's embedding: 4-bit g32, plain MSE, importance None)
        on `device`, chunk_rows rows at a time (per-row decisions: chunking changes nothing)."""
        dev = torch.device(device) if device is not None else (
            table.device if table.device.type == "cuda" else torch.device("cuda", torch.cuda.current_device()))
        parts = [kquant.quantize(table[r0:r0 + chunk_rows].to(dev).float(), bits, group, importance)
                 for r0 in range(0, table.shape[0], chunk_rows)]
        cat = (lambda ts: ts[0]) if len(parts) == 1 else torch.cat
        return cls(cat([p.qweight for p in parts]), cat([p.sb for p in parts]), cat([p.d for p in parts]),
                   cat([p.dmin for p in parts]), bits, group, device=dev, **kw)

    # ------------------------------------------------------------------------------------------- table
    def packed(self) -> kquant.KPacked:
        q, sb, d, dmin = unpack_rows(self.rows, self.embedding_dim, self.bits, self.group)
        return kquant.KPacked(q.contiguous(), sb.contiguous(), d, dmin, self.bits, self.group)

    def dequantize(self, dtype=torch.float32, chunk_rows: int = 32768) -> torch.Tensor:
        """The whole dequantized table [V, H] on the GPU (for tied heads and tests)."""
        out = torch.empty(self.num_embeddings, self.embedding_dim, dtype=dtype, device=self.device)
        for r0 in range(0, self.num_embeddings, chunk_rows):
            r1 = min(self.num_embeddings, r0 + chunk_rows)
            rows = self.rows[r0:r1].to(self.device)
            out[r0:r1] = self._decode(rows, dtype)
        return out

    @property
    def weight(self) -> torch.Tensor:
        """The dequantized table in the output dtype (costly: for code that reads .weight once, e.g. a tied head)."""
        return self.dequantize(self.dtype)

    def nbytes(self) -> int:
        return self.rows.numel() * self.rows.element_size()

    # ------------------------------------------------------------------------------------------- lookup
    def _ones(self, n: int) -> torch.Tensor:
        if self._want.numel() < n:
            self._want = torch.ones(max(n, 2 * self._want.numel()), dtype=torch.int32, device=self.device)
        return self._want[:n]

    def _gather(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.host:
            return self.rows.index_select(0, ids)
        from .kernels import copy_blobs  # Triton; imported on first use

        n = ids.numel()
        out = torch.empty(n, self.layout["row_bytes"], dtype=torch.uint8, device=self.device)
        if n:
            copy_blobs(self.rows, out, ids.to(torch.int32), self._ones(n))
        return out

    def _decode(self, rows: torch.Tensor, dtype) -> torch.Tensor:
        q, sb, d, dmin = unpack_rows(rows, self.embedding_dim, self.bits, self.group)
        return kquant.dequantize(q, sb, d, dmin, self.bits, self.group, torch.float32).to(dtype)

    def _lookup_fused(self, ids: torch.Tensor) -> torch.Tensor:
        n, H = ids.numel(), self.embedding_dim
        out = torch.empty(n, H, dtype=self.dtype, device=self.device)
        if n:
            L = self.layout
            block = min(1024, triton.next_power_of_2(H))
            _qembed_rows[(n, triton.cdiv(H, block))](self.rows, self.rows.view(torch.float16), ids, out,
                                                     L["row_bytes"], H, L["sb"], out.stride(0), BITS=self.bits,
                                                     GROUP=self.group, BLOCK=block, num_warps=4)
        return out

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.reshape(-1)
        if ids.device != self.device:
            ids = ids.to(self.device)
        # Both gather kernels (_qembed_rows, kernels.copy_blobs) read ids[t] at ids_ptr + t, i.e. they need unit
        # stride. reshape(-1) keeps a view's stride when it can (x[:, ::3] -> stride 3, a column of [T, 2] -> 2, an
        # expanded id -> 0), and .to(int32) on an int32 view is a no-op, so make it contiguous here (a no-op for the
        # usual contiguous ids; one small copy otherwise, also inside a captured graph).
        ids = ids.contiguous()
        if self.fused:
            out = self._lookup_fused(ids)
        else:
            out = self._decode(self._gather(ids), self.dtype)
        out = out.view(*input_ids.shape, self.embedding_dim)
        if self.embed_scale is not None:
            out = out * self.embed_scale.to(out.dtype)
        return out

    def extra_repr(self) -> str:
        return (f"{self.num_embeddings}, {self.embedding_dim}, k4row {self.bits}-bit g{self.group}, "
                f"{'pinned host' if self.host else 'GPU'} table of {self.layout['row_bytes']} B rows, "
                f"out {self.dtype}, scaled={self.embed_scale is not None}, fused={self.fused}")
