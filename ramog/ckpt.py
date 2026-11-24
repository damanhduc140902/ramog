"""Checkpoint access for the generic converter and the reference streamer (design-core B4; work package WP3).

  CheckpointIndex  every shard header of a safetensors checkpoint (dtype, shape, byte range per key), validated the way
                   safetensors validates it; dim-0 byte-range reads straight into one (optionally pinned) buffer, with
                   adjacent ranges of a shard coalesced into a single read.
  KeyMap           how every model tensor is made from checkpoint tensors. Names come from transformers itself: the arch
                   prefix rules, then transformers' conversion mapping (ramog.hfcompat.KeyRenamer, the 5.19 loader's
                   naming). Side tensors attach to their base (FP8 `*_scale_inv`, MXFP4 `*_blocks`/`*_scales`, and
                   `.scale` only on an FP8/int8 weight); skips are named rules (ramog.arch.SkipRules: tower, mtp,
                   class_ignore); anything else is a problem, and build() raises on any problem. The map is checked both
                   ways: every checkpoint key is consumed once (mapped, a side tensor, or skipped by a named rule) and
                   every model parameter / persistent buffer is made exactly once with its exact shape (a tied lm_head
                   excepted). KeyMap.load(key) gives the tensor from_pretrained(dtype=bf16) would hold, decoded with
                   ramog.dequant (bitwise transformers' decoders, WP1) and cast with the dtype rule of a plain bf16
                   load (design-core A7: the strict-fp32 plan, else the meta parameter's dtype). Note: transformers'
                   pre-quantized path (an FP8 / MXFP4 config, also with dequantize=True) instead keeps the checkpoint
                   dtype of every key it renames (core_model_loading 5.19 :1680-1692), so from_pretrained of the FP8
                   releases holds MiniMax-M2's router weight and bias in F32 and GLM-5.3-Flash's hc base/scale in F32
                   and its conv1d in BF16; the values agree, only those dtypes differ (tests/test_keymap.py and
                   tools/keymap_check.py --parity check exactly this).
  ExpertSource     experts of one MoE layer in chunks [e0, e1): (gate_up [n, 2I, H], down [n, H, I], gate_up_bias,
                   down_bias) in the [out, in] layout a blob stores, rows in transformers' output order (gpt-oss stays
                   interleaved), in the dtype transformers loads (bf16). Sources are detected per checkpoint and layer,
                   never per model type: per_expert (a WeightConverter merge of per-expert tensors, role order from its
                   source_patterns: gate before up, mixtral w1 before w3), fused (dim-0 slices of [E, ...] tensors),
                   mxfp4
                   (dim-0 slices of out-major blocks/scales).
  StreamingWriter  a safetensors file written header first, then one tensor at a time (any order, each exactly once),
                   so peak RAM is one tensor; atomically renamed into place on close. write_dense() streams a KeyMap's
                   dense (non-expert) tensors into it under the model's key names.

Host tensors (ArchEntry.host_keys: DeepSeek-V4.1's engram n-gram tables) are mapped as "host_raw" targets: the stored
bytes in their release dtype (viewed, never decoded or cast), served zero-copy by KeyMap.host_view as a memory map of
the checkpoint (CheckpointIndex.mmap); KeyMap.load refuses them and dense_keys() leaves them out. A store keeps them
in host.safetensors: copy_host() copies the stored bytes (release dtype, model key names) in chunks, mmap_file() maps
that file back zero-copy (ramog.loader binds the views; nothing is decoded, pinned or put on a device).

Source formats are a registry (SOURCE_FORMATS, @register_source_format) over ramog.dequant's DECODERS: the format of a
checkpoint tensor is decided per key from its dtype and side tensors (GLM-5.3-Flash's o_proj is BF16 in KDA layers and
FP8 in MLA layers), and KeyMap.report() counts them for meta.json (source_formats). Nothing here quantizes: the store's
own expert / dense formats (k4row, TCQ, ...) are the converter's pluggable step on top of ExpertSource and
StreamingWriter, which carry plain bf16 tensors or any packed arrays.

FP8 blocks are taken per tensor: the config's weight_block_size when its ceil grid equals the stored scale grid, else
the unique power-of-two block that gives that grid (DeepSeek-V4.1's engram tables use 1x32 under a 32x32 config); a
tensor whose grid fits neither raises. Partial last blocks (GLM-5.3 / DeepSeek-V3.2 kv_a_proj_with_mqa) are decoded on
the ceil grid, as ramog.dequant documents.

Runs on any CPU with torch, transformers (via ramog.hfcompat) and safetensors; no GPU kernels are imported.
"""

from __future__ import annotations

import json
import math
import mmap
import os
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from . import arch, dequant, hfcompat

if sys.byteorder != "little":  # safetensors data is little-endian; reads are raw byte copies
    raise RuntimeError("ramog.ckpt needs a little-endian host")

# ---------------------------------------------------------------------------------------------------------------------
# safetensors dtypes

# name -> (torch dtype the bytes are read as, element size). F8_E8M0 is read as uint8 (design-core B4).
ST_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "BF16": (torch.bfloat16, 2), "F16": (torch.float16, 2), "F32": (torch.float32, 4), "F64": (torch.float64, 8),
    "F8_E4M3": (torch.float8_e4m3fn, 1), "F8_E5M2": (torch.float8_e5m2, 1), "F8_E8M0": (torch.uint8, 1),
    "I8": (torch.int8, 1), "U8": (torch.uint8, 1), "I16": (torch.int16, 2), "I32": (torch.int32, 4),
    "I64": (torch.int64, 8), "BOOL": (torch.bool, 1),
}
# torch dtype -> safetensors name (the writer's direction)
TORCH_TO_ST: dict[torch.dtype, str] = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2", torch.int8: "I8", torch.uint8: "U8",
    torch.int16: "I16", torch.int32: "I32", torch.int64: "I64", torch.bool: "BOOL",
}
if dequant.E8M0 is not None:
    TORCH_TO_ST[dequant.E8M0] = "F8_E8M0"
FP8_ST = ("F8_E4M3", "F8_E5M2")
FLOAT_ST = ("BF16", "F16", "F32", "F64")
E8M0_ST = ("U8", "F8_E8M0")
MAX_HEADER = 100 << 20  # safetensors' own limit on the JSON header


def st_dtype_name(dtype: torch.dtype) -> str:
    try:
        return TORCH_TO_ST[dtype]
    except KeyError:
        raise TypeError(f"no safetensors dtype for {dtype}") from None


# ---------------------------------------------------------------------------------------------------------------------
# CheckpointIndex


@dataclass(frozen=True)
class TensorInfo:
    """One stored tensor: shard file (relative to the checkpoint dir), safetensors dtype name, shape, absolute byte
    offset in the file and byte count."""

    key: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    nbytes: int

    @property
    def supported(self) -> bool:
        return self.dtype in ST_DTYPES

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype not in ST_DTYPES:
            raise TypeError(f"{self.key}: safetensors dtype {self.dtype} is not supported ({sorted(ST_DTYPES)})")
        return ST_DTYPES[self.dtype][0]

    @property
    def itemsize(self) -> int:
        return ST_DTYPES[self.dtype][1] if self.dtype in ST_DTYPES else 0

    def row_bytes(self) -> int:
        """Bytes per index of dim 0."""
        if not self.shape:
            raise ValueError(f"{self.key}: a 0-d tensor has no rows")
        return self.nbytes // self.shape[0] if self.shape[0] else 0


def parse_safetensors_header(path: str) -> tuple[dict[str, tuple[str, tuple[int, ...], int, int]], dict, int]:
    """{key: (dtype, shape, absolute start, nbytes)}, __metadata__, data start of one .safetensors file. Validated as
    safetensors validates it: data offsets contiguous from 0 to the end of the file, without overlap, and (for known
    dtypes) nbytes == numel * element size."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (too short)")
        n = struct.unpack("<Q", raw)[0]
        if n > MAX_HEADER or 8 + n > size:
            raise ValueError(f"{path}: header length {n} is invalid (file {size} bytes)")
        header = json.loads(fh.read(n))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header is not a JSON object")
    data_start = 8 + n
    meta = header.pop("__metadata__", None) or {}
    out = {}
    spans = []
    for k, v in header.items():
        dt, shape, (b, e) = v["dtype"], tuple(int(d) for d in v["shape"]), v["data_offsets"]
        if any(d < 0 for d in shape) or not 0 <= b <= e:
            raise ValueError(f"{path}: {k}: bad shape {shape} or offsets {(b, e)}")
        if dt in ST_DTYPES and e - b != math.prod(shape) * ST_DTYPES[dt][1]:
            raise ValueError(f"{path}: {k}: {e - b} bytes for {dt} {list(shape)}")
        out[k] = (dt, shape, data_start + b, e - b)
        spans.append((b, e, k))
    spans.sort()
    pos = 0
    for b, e, k in spans:
        if b != pos:
            raise ValueError(f"{path}: {k} starts at {b}, expected {pos} (gap or overlap in data offsets)")
        pos = e
    if data_start + pos != size:
        raise ValueError(f"{path}: data ends at {data_start + pos}, file has {size} bytes")
    return out, meta, data_start


def _alloc(nbytes: int, pin: bool) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)


def _can_pin() -> bool:
    return torch.cuda.is_available()


class CheckpointIndex:
    """All shard headers of a safetensors checkpoint directory and byte-range reads of its tensors.

    Shards: those named by model.safetensors.index.json (whose key list must equal the shards' headers, key for key
    and file for file), else every top-level *.safetensors. A key in two shards raises. config.json, when present, is
    kept raw (`config`) for the quantization block size. Keys with a dtype outside ST_DTYPES are listed in
    `unsupported`; reading one raises. Reads to a GPU reuse one pinned staging buffer per index, so one index must not
    be read from several threads at once (give each reader thread its own CheckpointIndex)."""

    def __init__(self, model_dir: str | os.PathLike):
        self.dir = str(model_dir)
        idx = os.path.join(self.dir, "model.safetensors.index.json")
        wm = None
        if os.path.exists(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
            files = sorted(set(wm.values()))
        else:
            files = sorted(f for f in os.listdir(self.dir) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"{self.dir}: no safetensors shards")
        self.files = files
        self.tensors: dict[str, TensorInfo] = {}
        self.file_meta: dict[str, dict] = {}
        for fname in files:
            hdr, meta, _ = parse_safetensors_header(os.path.join(self.dir, fname))
            self.file_meta[fname] = meta
            for k, (dt, shape, start, nb) in hdr.items():
                if k in self.tensors:
                    raise ValueError(f"{k} appears in {self.tensors[k].file} and {fname}")
                self.tensors[k] = TensorInfo(k, fname, dt, shape, start, nb)
        if wm is not None:
            only_index = set(wm) - set(self.tensors)
            only_shards = set(self.tensors) - set(wm)
            wrong = [k for k, f in wm.items() if k in self.tensors and self.tensors[k].file != f]
            if only_index or only_shards or wrong:
                raise ValueError(f"{idx} disagrees with the shard headers: {len(only_index)} keys only in the index "
                                 f"({sorted(only_index)[:3]}), {len(only_shards)} only in shards "
                                 f"({sorted(only_shards)[:3]}), {len(wrong)} in another shard ({wrong[:3]})")
        self.unsupported = sorted(k for k, t in self.tensors.items() if not t.supported)
        cfg_path = os.path.join(self.dir, "config.json")
        self.config: dict = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        self._fh: dict[str, object] = {}
        self._stage: torch.Tensor | None = None
        self._mm: dict[str, mmap.mmap] = {}  # copy-on-write maps for mmap(); never closed (tensors may view them)

    # -- header queries -----------------------------------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def __getitem__(self, key: str) -> TensorInfo:
        return self.tensors[key]

    def __len__(self) -> int:
        return len(self.tensors)

    def keys(self):
        return self.tensors.keys()

    @property
    def quantization_config(self) -> dict:
        c = self.config
        return c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config") or {}

    @property
    def weight_block_size(self) -> tuple[int, int] | None:
        b = self.quantization_config.get("weight_block_size")
        return tuple(int(x) for x in b) if b else None

    def dtype_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(t.dtype for t in self.tensors.values()).items()))

    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())

    def mmap(self, key: str) -> torch.Tensor:
        """Stored tensor `key` as a zero-copy CPU tensor over a private (copy-on-write) memory map of its shard: pages
        are read on first touch, writes never reach the file. Dtype as read (ST_DTYPES: F8_E8M0 is uint8). The map
        stays open while any tensor views it (close() leaves it alone)."""
        t = self.tensors[key]
        dt = t.torch_dtype
        mm = self._mm.get(t.file)
        if mm is None:
            with open(os.path.join(self.dir, t.file), "rb") as fh:
                mm = self._mm[t.file] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
        flat = torch.frombuffer(mm, dtype=torch.uint8, count=t.nbytes, offset=t.start)
        return flat.view(dt).view(t.shape)

    # -- reads --------------------------------------------------------------------------------------------------------

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()
        self._stage = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _file(self, fname: str):
        fh = self._fh.get(fname)
        if fh is None:
            fh = self._fh[fname] = open(os.path.join(self.dir, fname), "rb", buffering=0)
        return fh

    def _read_into(self, fname: str, offset: int, view: memoryview) -> None:
        fh = self._file(fname)
        done, n = 0, len(view)
        preadv = getattr(os, "preadv", None)
        while done < n:
            if preadv is not None:
                got = preadv(fh.fileno(), [view[done:]], offset + done)
            else:
                fh.seek(offset + done)
                got = fh.readinto(view[done:])
            if not got:
                raise OSError(f"{fname}: short read at {offset + done} ({done} of {n} bytes)")
            done += got

    def _span(self, key: str, rows: tuple[int, int] | None) -> tuple[TensorInfo, int, int, tuple[int, ...]]:
        t = self.tensors[key]
        t.torch_dtype  # noqa: B018 - raises on an unsupported dtype
        if rows is None:
            return t, t.start, t.nbytes, t.shape
        r0, r1 = rows
        if not t.shape or not 0 <= r0 <= r1 <= t.shape[0]:
            raise IndexError(f"{key}: rows {rows} outside dim 0 of {list(t.shape)}")
        rb = t.row_bytes()
        return t, t.start + r0 * rb, (r1 - r0) * rb, (r1 - r0, *t.shape[1:])

    def read_many(self, requests: Sequence[tuple[str, tuple[int, int] | None]], device: str | torch.device = "cpu",
                  pin: bool | None = None) -> list[torch.Tensor]:
        """Tensors for [(key, rows or None)], rows = a [r0, r1) slice of dim 0, read as raw bytes (no safetensors
        call, no mmap). Adjacent byte ranges of one shard become one read into one buffer; for a CUDA `device` that
        buffer is a reused pinned staging buffer followed by one host-to-device copy, and the results are views of the
        device copy. pin=True also pins the CPU result buffer (default: only when staging for a GPU)."""
        device = torch.device(device)
        spans = []  # (file, start, nbytes, request index, info, shape)
        for i, (key, rows) in enumerate(requests):
            t, start, nb, shape = self._span(key, rows)
            spans.append((t.file, start, nb, i, t, shape))
        order = sorted(range(len(spans)), key=lambda j: (spans[j][0], spans[j][1]))
        runs: list[list[int]] = []  # each: request indices whose byte ranges are adjacent in one file
        for j in order:
            f, s, nb = spans[j][:3]
            if runs:
                pf, ps, pnb = spans[runs[-1][-1]][:3]
                if pf == f and ps + pnb == s:
                    runs[-1].append(j)
                    continue
            runs.append([j])
        ALIGN = 64
        pos, run_pos = 0, []
        for run in runs:
            run_pos.append(pos)
            nb = sum(spans[j][2] for j in run)
            pos += -(-nb // ALIGN) * ALIGN
        total = pos
        to_gpu = device.type != "cpu"
        if to_gpu:
            if self._stage is None or self._stage.numel() < total:
                self._stage = _alloc(max(total, 1 << 20), _can_pin())
            buf = self._stage[:total]
        else:
            buf = _alloc(total, bool(pin) and _can_pin())
        mv = memoryview(buf.numpy()) if total else None
        offs = [0] * len(spans)
        for run, rp in zip(runs, run_pos):
            f, s0 = spans[run[0]][:2]
            nb = sum(spans[j][2] for j in run)
            if nb:
                self._read_into(f, s0, mv[rp:rp + nb])
            for j in run:
                offs[j] = rp + spans[j][1] - s0
        src = buf.to(device) if to_gpu else buf
        out = []
        for i, (_f, _s, nb, _i, t, shape) in enumerate(spans):
            raw = src[offs[i]:offs[i] + nb]
            if offs[i] % max(t.itemsize, 1):
                raw = raw.clone()  # unaligned inside a coalesced run: copy to fresh storage before the dtype view
            out.append(raw.view(t.torch_dtype).view(shape))
        return out

    def read(self, key: str, rows: tuple[int, int] | None = None, device: str | torch.device = "cpu",
             pin: bool | None = None) -> torch.Tensor:
        return self.read_many([(key, rows)], device, pin)[0]


# ---------------------------------------------------------------------------------------------------------------------
# source formats


@dataclass(frozen=True)
class SourceFormat:
    """How one stored tensor (plus its side tensors) decodes.
    decode(base, sides, part, dtype) -> tensor, from the raw stored tensors (sides in Part.sides order);
    shape(stored shape) -> decoded shape; out_major: the decoded tensor is [..., out, in] whatever the model's layout
    (MXFP4), else it is in the model's layout."""

    name: str
    decode: Callable[[torch.Tensor, list[torch.Tensor], Part, torch.dtype], torch.Tensor]
    shape: Callable[[tuple[int, ...]], tuple[int, ...]]
    out_major: bool = False


SOURCE_FORMATS: dict[str, SourceFormat] = {}


def register_source_format(fmt: SourceFormat) -> SourceFormat:
    if fmt.name in SOURCE_FORMATS:
        raise ValueError(f"source format {fmt.name!r} is already registered")
    SOURCE_FORMATS[fmt.name] = fmt
    return fmt


def _plain(base, sides, part, dtype):
    return dequant.plain(base, dtype) if base.dtype in dequant.PLAIN_DTYPES else base.to(dtype)


register_source_format(SourceFormat("plain", _plain, lambda s: s))
register_source_format(SourceFormat("raw", lambda b, s, p, d: b.to(d), lambda s: s))  # integer / bool tensors
register_source_format(SourceFormat(
    "fp8_block", lambda b, s, p, d: dequant.fp8_block(b, s[0], p.block, d), lambda s: s))
register_source_format(SourceFormat(
    "fp4_i8_e8m0", lambda b, s, p, d: dequant.fp4_i8_e8m0(b, s[0], p.block, d), lambda s: (*s[:-1], 2 * s[-1])))
register_source_format(SourceFormat(
    "mxfp4", lambda b, s, p, d: dequant.mxfp4(b, s[0], d), lambda s: (*s[:-2], s[-2] * s[-1] * 2), out_major=True))
# host tensors (ArchEntry.host_keys): the stored bytes reinterpreted as the model tensor's dtype (same element size)
register_source_format(SourceFormat("host_raw", lambda b, s, p, d: b.view(d), lambda s: s))


def resolve_block(rows: int, cols: int, grid: tuple[int, int], preferred: tuple[int, int] | None) -> tuple[int, int]:
    """The (bm, bn) whose ceil grid over [rows, cols] is `grid`. Per dim: the preferred (config) block when it gives
    that count; else, when one block covers the whole dim, the smallest power of two >= the dim (every block that
    large decodes identically); else the unique power-of-two block that gives the count. Raises when none fits."""
    if preferred is not None and dequant.ceil_grid(rows, cols, preferred) == tuple(grid):
        return tuple(preferred)
    out = []
    for i, (n, g) in enumerate(((rows, grid[0]), (cols, grid[1]))):
        pref = None if preferred is None else int(preferred[i])
        if pref and -(-n // pref) == g:
            out.append(pref)
            continue
        if g == 1 and n >= 1:
            out.append(1 << (n - 1).bit_length())
            continue
        cands = [1 << p for p in range(0, 16) if -(-n // (1 << p)) == g]
        if len(cands) != 1:
            raise ValueError(f"scale grid {list(grid)} does not fit [{rows}, {cols}] with block {preferred} and no "
                             f"unique power-of-two block gives it (candidates for dim {n}: {cands})")
        out.append(cands[0])
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------------------
# KeyMap


@dataclass(frozen=True)
class Part:
    """One stored tensor feeding a model tensor: checkpoint key (MXFP4: the _blocks key), source format, side tensor
    keys (FP8/FP4: the scale; MXFP4: the _scales), block (FP8/FP4), stored shape (the decoded shape is
    SOURCE_FORMATS[fmt].shape(shape)), expert index (per-expert merges)."""

    key: str
    fmt: str
    sides: tuple[str, ...] = ()
    block: tuple[int, int] | None = None
    shape: tuple[int, ...] = ()
    expert: int | None = None


@dataclass
class Target:
    """How one model tensor (state_dict key) is made.
    kind: "direct" (one stored tensor), "converter" (transformers' WeightConverter: ops on per-pattern lists),
    "mxfp4" (blocks + scales). parts: {source pattern ("" unless converter): [Part, in loader order]}. dtype: what a
    plain from_pretrained(dtype=bf16) gives (the strict-fp32 plan, else the meta parameter's dtype). transposed: the
    model tensor is [..., in, out] (gpt-oss experts). site/role: the MoE layer and "gate_up" | "down" | "gate_up_bias" |
    "down_bias" for expert tensors."""

    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    kind: str
    parts: dict[str, list[Part]]
    ops: tuple[tuple[str, int], ...] = ()
    source_patterns: tuple[str, ...] = ()
    transposed: bool = False
    site: int | None = None
    role: str | None = None
    host: bool = False  # ArchEntry.host_keys: stays in host memory in its release dtype (host_view), never loaded

    def all_parts(self) -> list[Part]:
        return [p for v in self.parts.values() for p in v]

    @property
    def decoded_out_major(self) -> bool:
        return any(SOURCE_FORMATS[p.fmt].out_major for p in self.all_parts())

    @property
    def is_expert(self) -> bool:
        return self.site is not None


class KeyMapError(ValueError):
    def __init__(self, msg: str, report: dict):
        super().__init__(msg)
        self.report = report


_EXPERT_IDX = re.compile(r"(?:^|\.)experts\.(\d+)\.")
EXPERT_ROLES = (("gate_up_proj", "gate_up"), ("up_proj", "gate_up"), ("down_proj", "down"),
                ("gate_up_proj_bias", "gate_up_bias"), ("up_proj_bias", "gate_up_bias"),
                ("down_proj_bias", "down_bias"))


def tied_keys(model: nn.Module) -> dict[str, str]:
    """{head key: embedding key} when the model ties its output head to the input embedding (the head is not
    loaded)."""
    out = {}
    cfg = getattr(model, "config", None)
    if getattr(cfg, "tie_word_embeddings", False):
        head, emb = model.get_output_embeddings(), model.get_input_embeddings()
        if head is not None and emb is not None and head.weight is emb.weight:
            sd = model.state_dict(keep_vars=True)
            hk = next((k for k, v in sd.items() if v is head.weight and k.startswith("lm_head")), "lm_head.weight")
            ek = next(k for k, v in sd.items() if v is emb.weight and not k.startswith("lm_head"))
            out[hk] = ek
    return out


def _side_kind(key: str) -> tuple[str, str] | None:
    """(kind, base) of a would-be side tensor: ("scale_inv", base checkpoint key), ("mx_blocks"|"mx_scales", stored
    name of the MXFP4 tensor), or ("scale", base checkpoint key `X.weight` of an `X.scale`)."""
    if key.endswith("_scale_inv"):
        return "scale_inv", key[: -len("_scale_inv")]
    if key.endswith("_blocks"):
        return "mx_blocks", key[: -len("_blocks")]
    if key.endswith("_scales"):
        return "mx_scales", key[: -len("_scales")]
    if key.endswith(".scale"):
        return "scale", key[: -len("scale")] + "weight"
    return None


class KeyMap:
    """Checkpoint keys -> model tensors for one (meta model, checkpoint) pair; see the module docstring.
    Attributes: targets {model key: Target} (state_dict order), sites (ramog.arch.MoESite), tied {head: emb},
    skipped Counter by rule (and skipped_keys), side Counter by kind, problems (empty after a strict build)."""

    def __init__(self):
        self.index: CheckpointIndex | None = None
        self.entry: arch.ArchEntry | None = None
        self.targets: dict[str, Target] = {}
        self.sites: list[arch.MoESite] = []
        self.tied: dict[str, str] = {}
        self.skipped: Counter = Counter()
        self.skipped_keys: dict[str, list[str]] = defaultdict(list)
        self.side: Counter = Counter()
        # checkpoint key -> (category, model key or skip rule); category: "direct", "converter", "side:<kind>",
        # "skip"; every key of a complete map has exactly one
        self.consumed: dict[str, tuple[str, str]] = {}
        self.problems: list[str] = []
        self.block: tuple[int, int] | None = None
        self.block_inferred: Counter = Counter()
        self.seconds = 0.0

    # -- build --------------------------------------------------------------------------------------------------------

    @classmethod
    def build(cls, meta: nn.Module, entry: arch.ArchEntry, index: CheckpointIndex, *,
              block: tuple[int, int] | None = None, strict: bool = True) -> KeyMap:
        """Map `index` onto the meta model `meta` (ramog.arch.build_meta) of architecture `entry`. block: FP8 block
        size (default: the checkpoint's weight_block_size, else 128x128). strict: raise KeyMapError (with .report)
        on any problem; otherwise the problems are listed in .problems."""
        t0 = time.time()
        km = cls()
        km.index, km.entry = index, entry
        km.block = tuple(block) if block else (index.weight_block_size or (128, 128))
        cfg = meta.config
        sd = meta.state_dict()
        shapes = {k: tuple(v.shape) for k, v in sd.items()}
        meta_dtype = {k: v.dtype for k, v in sd.items()}
        renamer = hfcompat.KeyRenamer(meta, shapes)
        skip = arch.SkipRules(meta, cfg)
        plan = hfcompat.DtypePlan(meta, torch.bfloat16)
        km.sites = arch.find_moe_sites(meta, cfg)
        km.tied = tied_keys(meta)
        host_rx = [re.compile(p) for p in entry.host_keys]
        km._renamer_desc = renamer.describe()
        expert_of: dict[str, tuple[arch.MoESite, str]] = {}
        for s in km.sites:
            for attr, role in EXPERT_ROLES:
                k = f"{s.name}.{attr}"
                if k in shapes:
                    expert_of[k] = (s, role)
        P = km.problems

        # pass 1: keys that transformers names as model keys (directly or through a converter)
        direct: dict[str, list[str]] = defaultdict(list)
        conv: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        pending: list[tuple[str, str, str]] = []
        for k in hfcompat.sort_keys(index.keys()):
            k1 = arch.apply_prefix_rules(k, entry)
            tgt, pat = renamer(k1)
            if tgt in shapes:
                (direct[tgt] if pat is None else conv[tgt][pat]).append(k)
                km.consumed[k] = ("direct" if pat is None else "converter", tgt)
            else:
                pending.append((k, k1, tgt))

        # pass 2: named skip rules for keys that look like no side tensor; then side tensors, which attach to a mapped
        # base (or share the skip rule of a skipped base); anything else is unmapped
        scale_of: dict[str, tuple[str, str]] = {}  # base checkpoint key -> (side key, kind)
        mx: dict[str, dict[str, str]] = defaultdict(dict)  # stored MXFP4 name -> {"blocks"|"scales": key}
        unmapped = []

        def skip_or_unmapped(k, rule):
            if rule is None:
                unmapped.append(k)
                return
            km.skipped[rule] += 1
            km.skipped_keys[rule].append(k)
            km.consumed[k] = ("skip", rule)

        sides = []
        for k, k1, tgt in pending:
            if _side_kind(k) is None:
                skip_or_unmapped(k, skip(k, k1, tgt))
            else:
                sides.append((k, k1, tgt))
        for k, k1, tgt in sides:
            kind, base = _side_kind(k)
            bcat, bwhat = km.consumed.get(base, ("", ""))
            if kind in ("scale_inv", "scale") and bcat in ("direct", "converter"):
                bdt = index[base].dtype
                # `.scale` attaches only to FP8 / int8 weights: Gemma's router.scale is a parameter (mapped directly)
                if kind == "scale_inv" or bdt in FP8_ST + ("I8", "U8"):
                    name = "fp8_scale" if bdt in FP8_ST else "fp4_scale" if bdt in ("I8", "U8") else kind
                    if base in scale_of:
                        P.append(f"{base}: two scales ({scale_of[base][0]}, {k})")
                    else:
                        scale_of[base] = (k, kind)
                    km.consumed[k] = (f"side:{name}", bwhat)
                    km.side[name] += 1
                    continue
            elif kind in ("mx_blocks", "mx_scales"):
                mtgt, _ = renamer(arch.apply_prefix_rules(base, entry))
                if mtgt in shapes:
                    which = kind[3:]
                    if which in mx[base]:
                        P.append(f"{base}: two MXFP4 {which} tensors")
                    mx[base][which] = k
                    mx[base]["target"] = mtgt
                    km.consumed[k] = (f"side:mxfp4_{which}", mtgt)
                    km.side[f"mxfp4_{which}"] += 1
                    continue
            elif bcat == "skip":
                skip_or_unmapped(k, bwhat)  # the scale of a skipped tensor goes with it
                continue
            skip_or_unmapped(k, skip(k, k1, tgt))
        for k in unmapped:
            P.append(f"unmapped checkpoint key {k} ({index[k].dtype} {list(index[k].shape)})")
        mx_by_target: dict[str, list[str]] = defaultdict(list)
        for base, d in mx.items():
            mx_by_target[d["target"]].append(base)

        # pass 3: every model key made exactly once, with its shape
        def make_part(ck: str, want_float: bool, expert: int | None = None) -> Part | None:
            t = index[ck]
            if not t.supported:
                P.append(f"{ck}: unsupported dtype {t.dtype}")
                return None
            side = scale_of.get(ck)
            if t.dtype in FP8_ST:
                if side is None:
                    P.append(f"{ck}: {t.dtype} without a scale")
                    return None
                sk = index[side[0]]
                if sk.dtype not in FLOAT_ST + E8M0_ST or len(t.shape) < 2 or sk.shape[:-2] != t.shape[:-2] \
                        or len(sk.shape) != len(t.shape):
                    P.append(f"{ck}: scale {side[0]} {sk.dtype} {list(sk.shape)} does not fit {list(t.shape)}")
                    return None
                try:
                    blk = resolve_block(t.shape[-2], t.shape[-1], sk.shape[-2:], km.block)
                except ValueError as e:
                    P.append(f"{ck}: {e}")
                    return None
                if blk != km.block:
                    km.block_inferred[f"{blk[0]}x{blk[1]}"] += 1
                return Part(ck, "fp8_block", (side[0],), blk, t.shape, expert)
            if side is not None:
                sk = index[side[0]]
                if t.dtype in ("I8", "U8") and sk.dtype in E8M0_ST and len(t.shape) >= 2 and \
                        len(sk.shape) == len(t.shape) and sk.shape[:-2] == t.shape[:-2]:
                    shape = SOURCE_FORMATS["fp4_i8_e8m0"].shape(t.shape)
                    try:
                        blk = resolve_block(shape[-2], shape[-1], sk.shape[-2:], (1, 32))
                    except ValueError as e:
                        P.append(f"{ck}: {e}")
                        return None
                    return Part(ck, "fp4_i8_e8m0", (side[0],), blk, t.shape, expert)
                P.append(f"{ck}: {t.dtype} tensor with a scale {side[0]} ({sk.dtype})")
                return None
            if t.dtype not in FLOAT_ST and want_float:
                P.append(f"{ck}: {t.dtype} without a scale feeds a floating-point model tensor")
                return None
            return Part(ck, "plain" if t.dtype in FLOAT_ST else "raw", (), None, t.shape, expert)

        for mk, mshape in shapes.items():
            srcs, groups, mxs = direct.get(mk, []), conv.get(mk), mx_by_target.get(mk, [])
            ways = bool(srcs) + bool(groups) + bool(mxs)
            if ways == 0:
                if mk not in km.tied:
                    P.append(f"model key {mk} {list(mshape)} is made by no checkpoint key")
                continue
            if ways > 1 or len(srcs) > 1 or len(mxs) > 1:
                P.append(f"model key {mk} has several sources: direct {srcs[:3]}, converter "
                         f"{sorted(groups or {})}, mxfp4 {mxs}")
                continue
            site, role = expert_of.get(mk, (None, None))
            dtype = hfcompat.load_dtype(plan, mk, meta_dtype[mk])
            transposed = bool(site is not None and site.layout.transposed and role in ("gate_up", "down"))
            if srcs and any(rx.search(mk) for rx in host_rx):
                t = index[srcs[0]]
                size_ok = t.supported and t.itemsize == torch.empty((), dtype=dtype).element_size()
                if site is not None or not size_ok or tuple(t.shape) != mshape:
                    P.append(f"host key {mk} {dtype} {list(mshape)}: stored {srcs[0]} is {t.dtype} {list(t.shape)}")
                    continue
                km.targets[mk] = Target(mk, mshape, dtype, "direct", {"": [Part(srcs[0], "host_raw", (), None,
                                                                                 t.shape)]}, host=True)
            elif srcs:
                part = make_part(srcs[0], dtype.is_floating_point)
                if part is None:
                    continue
                got = SOURCE_FORMATS[part.fmt].shape(part.shape)
                if got != mshape:
                    P.append(f"{srcs[0]} decodes to {list(got)}, model key {mk} is {list(mshape)}")
                    continue
                # a fused expert weight stored as the model holds it: [E, in, out] on gpt-oss (BF16 releases,
                # save_pretrained), so load(out_major=True) and ExpertSource must transpose it like the other kinds
                km.targets[mk] = Target(mk, mshape, dtype, "direct", {"": [part]}, transposed=transposed,
                                        site=_li(site), role=role)
            elif groups:
                pats = list(groups)
                c = renamer.converter(pats[0])
                if any(renamer.converter(p) is not c for p in pats):
                    P.append(f"{mk}: sources from different converters {pats}")
                    continue
                try:
                    ops = tuple(hfcompat.converter_ops(c))
                except NotImplementedError as e:
                    P.append(f"{mk}: {e}")
                    continue
                if site is not None:
                    if not ops or ops not in ((("merge", 0),), (("merge", 0), ("concat", ops[-1][1]))):
                        P.append(f"{mk}: expert converter ops {list(ops)}; supported: merge(0) [+ concat]")
                        continue
                elif not ops or any(op != "concat" for op, _ in ops):
                    P.append(f"{mk}: dense converter ops {list(ops)}; only Concatenate is supported")
                    continue
                src_pats = tuple(c.source_patterns)
                parts: dict[str, list[Part]] = {}
                bad = False
                for p in [q for q in src_pats if q in groups] + [q for q in pats if q not in src_pats]:
                    plist = []
                    for ck in groups[p]:
                        m = _EXPERT_IDX.search(ck)
                        part = make_part(ck, dtype.is_floating_point, int(m.group(1)) if m else None)
                        bad |= part is None
                        if part is not None:
                            plist.append(part)
                    if ops[0][0] == "merge":
                        idx = [q.expert for q in plist]
                        if None in idx or idx != list(range(len(plist))):
                            P.append(f"{mk}: {p} experts are {idx[:8]}..., not exactly 0..{len(plist) - 1} in order")
                            bad = True
                    parts[p] = plist
                if bad:
                    continue
                if ops[0][0] == "merge" and len({len(v) for v in parts.values()}) != 1:
                    P.append(f"{mk}: unequal expert counts per pattern {[len(v) for v in parts.values()]}")
                    continue
                try:
                    got = hfcompat.converted_shape(c, {p: [SOURCE_FORMATS[q.fmt].shape(q.shape) for q in v]
                                                       for p, v in parts.items()})
                except ValueError as e:
                    P.append(f"{mk}: {e}")
                    continue
                if got != mshape:
                    P.append(f"{mk}: converter gives {list(got)}, model is {list(mshape)}")
                    continue
                km.targets[mk] = Target(mk, mshape, dtype, "converter", parts, ops, src_pats, transposed,
                                        _li(site), role)
            else:
                base = mxs[0]
                d = mx[base]
                if "blocks" not in d or "scales" not in d:
                    P.append(f"{base}: MXFP4 needs both _blocks and _scales, found {sorted(set(d) - {'target'})}")
                    continue
                if site is None or role not in ("gate_up", "down"):
                    P.append(f"{mk}: MXFP4 source outside an experts weight (only gpt-oss experts are MXFP4)")
                    continue
                bt, st = index[d["blocks"]], index[d["scales"]]
                if bt.dtype != "U8" or st.dtype not in E8M0_ST or len(bt.shape) < 3 or bt.shape[-1] != 16 \
                        or st.shape != bt.shape[:-1]:
                    P.append(f"{mk}: MXFP4 blocks {bt.dtype} {list(bt.shape)}, scales {st.dtype} {list(st.shape)}")
                    continue
                dec = SOURCE_FORMATS["mxfp4"].shape(bt.shape)
                model_shape = (*dec[:-2], dec[-1], dec[-2]) if transposed else dec
                if model_shape != mshape:
                    P.append(f"{mk}: MXFP4 decodes to {list(dec)} (out-major; transposed={transposed}), model is "
                             f"{list(mshape)}")
                    continue
                part = Part(d["blocks"], "mxfp4", (d["scales"],), None, bt.shape, None)
                km.targets[mk] = Target(mk, mshape, dtype, "mxfp4", {"": [part]}, transposed=transposed,
                                        site=_li(site), role=role)
        # scales whose base was mapped but never used as an FP8/FP4 part (e.g. a scale next to a BF16 tensor) were
        # reported by make_part; every side key is consumed by construction.
        for s in km.sites:
            need = ["gate_up", "down"] + (["gate_up_bias", "down_bias"] if s.layout.bias else [])
            have = {t.role for t in km.targets.values() if t.site == s.layer_idx}
            if not set(need) <= have:
                P.append(f"MoE layer {s.layer_idx}: expert tensors {sorted(set(need) - have)} have no source")
        km.seconds = time.time() - t0
        if strict and P:
            raise KeyMapError(f"{len(P)} key-map problems, first: {P[:5]}", km.report())
        return km

    # -- queries ------------------------------------------------------------------------------------------------------

    def expert_keys(self) -> list[str]:
        return [k for k, t in self.targets.items() if t.is_expert]

    def dense_keys(self) -> list[str]:
        """Model keys that are not expert tensors of a MoE site (what the dense writer stores), in state_dict order;
        a tied head and host tensors are not among them."""
        return [k for k, t in self.targets.items() if not t.is_expert and not t.host]

    def host_keys(self) -> list[str]:
        """Model keys of host tensors (ArchEntry.host_keys), in state_dict order."""
        return [k for k, t in self.targets.items() if t.host]

    def host_view(self, key: str) -> torch.Tensor:
        """Host tensor `key` as the model holds it (its release dtype and shape), zero-copy over a memory map of the
        checkpoint (CheckpointIndex.mmap)."""
        t = self.targets[key]
        if not t.host:
            raise ValueError(f"{key} is not a host tensor (ArchEntry.host_keys)")
        part = t.parts[""][0]
        return self.index.mmap(part.key).view(t.dtype).view(t.shape)

    def keys_by_layer(self) -> dict[int | None, list[str]]:
        """{decoder layer index: its model keys; None: the rest (embedding, norm, head, ...)}, in state_dict order."""
        rx = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
        out: dict[int | None, list[str]] = defaultdict(list)
        for k in self.targets:
            m = rx.match(k)
            out[int(m.group(1)) if m else None].append(k)
        return dict(out)

    def specs(self, keys: Iterable[str] | None = None) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
        keys = self.targets if keys is None else keys
        return {k: (self.targets[k].dtype, self.targets[k].shape) for k in keys}

    def source_formats(self) -> dict[str, int]:
        """{source format: number of stored tensors decoded with it} (meta.json v2 source_formats)."""
        return dict(sorted(Counter(p.fmt for t in self.targets.values() for p in t.all_parts()).items()))

    def report(self) -> dict:
        """Counts for logs, tests and meta.json v2. Checkpoint keys: mapped_direct + mapped_converter + side + skipped
        + unmapped == checkpoint_keys (balance_ok); model keys: targets + tied == model_keys."""
        idx = self.index
        cat = Counter(c for c, _ in self.consumed.values())
        n_unmapped = sum(1 for p in self.problems if p.startswith("unmapped checkpoint key"))
        kinds = Counter(t.kind for t in self.targets.values())
        moves = Counter()
        for t in self.targets.values():
            fm = "/".join(sorted({p.fmt for p in t.all_parts()}))
            moves[f"{t.kind} {fm} x{len(t.all_parts())} -> {str(t.dtype).replace('torch.', '')}"] += 1
        n_side = sum(v for c, v in cat.items() if c.startswith("side:"))
        return {
            "checkpoint_keys": len(idx) if idx is not None else None,
            "mapped_direct": cat.get("direct", 0),
            "mapped_converter": cat.get("converter", 0),
            "side": dict(sorted(self.side.items())),
            "skipped": dict(sorted(self.skipped.items())),
            "skipped_examples": {r: v[:3] for r, v in sorted(self.skipped_keys.items())},
            "unmapped": n_unmapped,
            "balance_ok": idx is not None and cat.get("direct", 0) + cat.get("converter", 0) + n_side
            + cat.get("skip", 0) + n_unmapped == len(idx) and len(self.consumed) + n_unmapped == len(idx),
            "model_keys": len(self.targets) + len(self.tied) + sum(
                1 for p in self.problems if p.startswith("model key")),
            "targets": len(self.targets),
            "target_kinds": dict(sorted(kinds.items())),
            "tied": dict(self.tied),
            "expert_targets": len(self.expert_keys()),
            "dense_targets": len(self.dense_keys()),
            "host_targets": len(self.host_keys()),
            "source_formats": self.source_formats(),
            "fp8_block": list(self.block) if self.block else None,
            "fp8_block_inferred": dict(self.block_inferred),
            "dtype_moves": dict(sorted(moves.items())),
            "moe_layers": [s.layer_idx for s in self.sites],
            "transforms": list(getattr(self, "_renamer_desc", [])),
            "problems": len(self.problems),
            "problem_examples": self.problems[:20],
            "complete": not self.problems,
            "seconds": round(self.seconds, 2),
        }

    # -- loading ------------------------------------------------------------------------------------------------------

    def _decode(self, parts: Sequence[Part], device, dtype: torch.dtype,
                rows: tuple[int, int] | None = None) -> list[torch.Tensor]:
        reqs = []
        for p in parts:
            reqs.append((p.key, rows))
            reqs.extend((s, rows) for s in p.sides)
        raw = self.index.read_many(reqs, device)
        out, i = [], 0
        for p in parts:
            base, sides = raw[i], raw[i + 1:i + 1 + len(p.sides)]
            i += 1 + len(p.sides)
            out.append(SOURCE_FORMATS[p.fmt].decode(base, list(sides), p, dtype))
        return out

    def load(self, key: str, device: str | torch.device = "cpu", rows: tuple[int, int] | None = None,
             dtype: torch.dtype | None = None, out_major: bool = False) -> torch.Tensor:
        """Model tensor `key` as from_pretrained(dtype=bf16) would hold it (dtype: override the load dtype).
        rows: experts [e0, e1) of an expert tensor (dim 0). out_major: an expert weight in [..., out, in] even when the
        model stores it transposed (gpt-oss)."""
        t = self.targets[key]
        if t.host:
            raise ValueError(f"{key} is a host tensor: use host_view() (it is never loaded or decoded)")
        dtype = dtype or t.dtype
        if rows is not None:
            e0, e1 = rows
            n = t.shape[0] if t.shape else 0
            if not t.is_expert or not 0 <= e0 <= e1 <= n:
                raise IndexError(f"{key}: rows {rows} (only expert tensors, within dim 0 = {n})")
        if t.kind == "converter":
            parts = t.parts if rows is None else {p: v[rows[0]:rows[1]] for p, v in t.parts.items()}
            flat = [q for v in parts.values() for q in v]
            dec = self._decode(flat, device, dtype)
            state: dict[str, object] = {}
            i = 0
            for p, v in parts.items():
                state[p] = dec[i:i + len(v)]
                i += len(v)
            for op, dim in t.ops:  # transformers' MergeModulelist / Concatenate (core_model_loading :150-260)
                if op == "merge":
                    state = {p: torch.stack(v, dim=dim) for p, v in state.items()}
                else:
                    cat = []
                    for p in t.source_patterns:
                        if p in state:
                            v = state.pop(p)
                            cat.extend(v if isinstance(v, list) else [v])
                    state = {"": torch.cat(cat, dim=dim)}
            (out,) = state.values()
            if isinstance(out, list):
                (out,) = out
        else:
            part = t.parts[""][0]
            if rows is not None and part.fmt in ("fp8_block", "fp4_i8_e8m0") and len(part.shape) < 3:
                raise ValueError(f"{key}: dim-0 rows of a 2-D block-scaled tensor")
            (out,) = self._decode([part], device, dtype, rows)
        if t.transposed and (t.decoded_out_major != out_major):
            out = out.transpose(-1, -2).contiguous()
        return out


def _li(site) -> int | None:
    return None if site is None else site.layer_idx


def build_keymap(model_dir: str, *, strict: bool = True) -> tuple[KeyMap, nn.Module, CheckpointIndex]:
    """CheckpointIndex(model_dir) + ramog.arch.build_meta(config) + KeyMap.build: the usual entry point."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    entry = arch.arch_for(cfg)
    meta = arch.build_meta(cfg)
    index = CheckpointIndex(model_dir)
    return KeyMap.build(meta, entry, index, strict=strict), meta, index


# ---------------------------------------------------------------------------------------------------------------------
# ExpertSource


@dataclass
class _SiteSource:
    layer: int
    layout: arch.ExpertsLayout
    gu: Target
    dn: Target
    gub: Target | None = None
    dnb: Target | None = None
    kind: str = ""
    formats: dict = field(default_factory=dict)


class ExpertSource:
    """Experts of every MoE layer of a checkpoint, read in chunks (see the module docstring). kind(layer):
    "per_expert" | "fused" | "mxfp4" (or "mixed:<gu>/<dn>")."""

    def __init__(self, keymap: KeyMap):
        self.keymap = keymap
        self._sites: dict[int, _SiteSource] = {}
        for s in keymap.sites:
            lay = s.layout
            by_role = {t.role: t for t in keymap.targets.values() if t.site == s.layer_idx}
            ss = _SiteSource(s.layer_idx, lay, by_role["gate_up"], by_role["down"], by_role.get("gate_up_bias"),
                             by_role.get("down_bias"))
            k = {"converter": "per_expert", "direct": "fused", "mxfp4": "mxfp4"}
            kg, kd = k[ss.gu.kind], k[ss.dn.kind]
            ss.kind = kg if kg == kd else f"mixed:{kg}/{kd}"
            ss.formats = {r: sorted({p.fmt for p in t.all_parts()}) for r, t in
                          (("gate_up", ss.gu), ("down", ss.dn), ("gate_up_bias", ss.gub), ("down_bias", ss.dnb))
                          if t is not None}
            self._sites[s.layer_idx] = ss

    @property
    def layers(self) -> list[int]:
        return list(self._sites)

    def layout(self, layer: int) -> arch.ExpertsLayout:
        return self._sites[layer].layout

    def num_experts(self, layer: int) -> int:
        return self._sites[layer].layout.E

    def kind(self, layer: int) -> str:
        return self._sites[layer].kind

    def formats(self, layer: int) -> dict:
        return self._sites[layer].formats

    def kinds(self) -> dict[str, int]:
        return dict(Counter(s.kind for s in self._sites.values()))

    def read(self, layer: int, e0: int, e1: int, device: str | torch.device = "cpu",
             dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
                                                        torch.Tensor | None]:
        """Experts [e0, e1) of MoE layer `layer`: gate_up [n, gu_out, H], down [n, H, I] ([out, in], rows in
        transformers' output order), gate_up_bias [n, gu_out] and down_bias [n, H] (None without biases), decoded
        on `device` in `dtype` (default: the dtype transformers loads them in, bf16)."""
        ss = self._sites[layer]
        lay = ss.layout
        if not 0 <= e0 < e1 <= lay.E:
            raise IndexError(f"layer {layer}: experts [{e0}, {e1}) outside 0..{lay.E}")
        km = self.keymap
        gu = km.load(ss.gu.key, device, (e0, e1), dtype, out_major=True)
        dn = km.load(ss.dn.key, device, (e0, e1), dtype, out_major=True)
        n = e1 - e0
        if tuple(gu.shape) != (n, *lay.gu_shape) or tuple(dn.shape) != (n, *lay.dn_shape):
            raise RuntimeError(f"layer {layer}: read gate_up {list(gu.shape)}, down {list(dn.shape)}; expected "
                               f"{[n, *lay.gu_shape]}, {[n, *lay.dn_shape]}")
        gub = km.load(ss.gub.key, device, (e0, e1), dtype) if ss.gub is not None else None
        dnb = km.load(ss.dnb.key, device, (e0, e1), dtype) if ss.dnb is not None else None
        return gu, dn, gub, dnb

    def iter_chunks(self, layer: int, chunk: int = 16, device: str | torch.device = "cpu",
                    dtype: torch.dtype | None = None):
        """(e0, e1, gu, dn, gu_bias, dn_bias) for consecutive chunks of `chunk` experts of MoE layer `layer`."""
        E = self.num_experts(layer)
        for e0 in range(0, E, chunk):
            e1 = min(E, e0 + chunk)
            yield (e0, e1, *self.read(layer, e0, e1, device, dtype))


# ---------------------------------------------------------------------------------------------------------------------
# StreamingWriter


class StreamingWriter:
    """A safetensors file written header first, then one tensor at a time.

    specs: {name: (torch dtype or safetensors dtype name, shape)}; the header is written at construction (tensors laid
    out by element size, largest first, then in the given order, so every tensor starts aligned to its element size,
    as safetensors' own writer does). write(name, tensor) puts each tensor at its offset, in any order, exactly once;
    the tensor may live on any device. close() checks every tensor was written and renames the temporary file into
    place; an exception inside `with` (or abort()) removes it. Peak memory: one tensor."""

    def __init__(self, path: str | os.PathLike, specs: Mapping[str, tuple[torch.dtype | str, Sequence[int]]],
                 metadata: Mapping[str, str] | None = None):
        self.path = Path(path)
        self.tmp = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
        self.specs: dict[str, tuple[str, tuple[int, ...]]] = {}
        for name, (dt, shape) in specs.items():
            dts = dt if isinstance(dt, str) else st_dtype_name(dt)
            if dts not in ST_DTYPES:
                raise TypeError(f"{name}: dtype {dt} not writable")
            self.specs[name] = (dts, tuple(int(d) for d in shape))
        order = sorted(self.specs, key=lambda n: -ST_DTYPES[self.specs[n][0]][1])  # stable: given order within a size
        header: dict[str, object] = {}
        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        pos = 0
        self.offsets: dict[str, tuple[int, int]] = {}
        for name in order:
            dts, shape = self.specs[name]
            nb = math.prod(shape) * ST_DTYPES[dts][1]
            header[name] = {"dtype": dts, "shape": list(shape), "data_offsets": [pos, pos + nb]}
            self.offsets[name] = (pos, nb)
            pos += nb
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        self.data_start = 8 + len(hb)
        self.size = self.data_start + pos
        self.written: set[str] = set()
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.tmp, "wb")
        self._fh.write(struct.pack("<Q", len(hb)))
        self._fh.write(hb)
        self._fh.truncate(self.size)
        self.closed = False

    def write(self, name: str, tensor: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError(f"{self.path}: already closed")
        if name not in self.specs:
            raise KeyError(f"{name} is not in the header of {self.path}")
        if name in self.written:
            raise ValueError(f"{name} written twice")
        dts, shape = self.specs[name]
        if st_dtype_name(tensor.dtype) != dts or tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: got {tensor.dtype} {list(tensor.shape)}, header says {dts} {list(shape)}")
        off, nb = self.offsets[name]
        data = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
        if data.numel() != nb:
            raise AssertionError((name, data.numel(), nb))
        if nb:
            self._fh.seek(self.data_start + off)
            self._fh.write(memoryview(data.numpy()))
        self.written.add(name)

    def write_stream(self, name: str, chunks: Iterable) -> None:
        """Tensor `name` from raw byte chunks (bytes-like, in order), for tensors larger than RAM (host tensors).
        The chunks must add up to exactly the tensor's byte count."""
        if self.closed:
            raise RuntimeError(f"{self.path}: already closed")
        if name not in self.specs:
            raise KeyError(f"{name} is not in the header of {self.path}")
        if name in self.written:
            raise ValueError(f"{name} written twice")
        off, nb = self.offsets[name]
        self._fh.seek(self.data_start + off)
        got = 0
        for c in chunks:
            n = len(memoryview(c).cast("B"))
            if got + n > nb:
                raise ValueError(f"{name}: more than {nb} bytes")
            self._fh.write(c)
            got += n
        if got != nb:
            raise ValueError(f"{name}: {got} bytes written, header says {nb}")
        self.written.add(name)

    def missing(self) -> list[str]:
        return [n for n in self.specs if n not in self.written]

    def close(self) -> Path:
        if self.closed:
            return self.path
        miss = self.missing()
        if miss:
            self.abort()
            raise RuntimeError(f"{self.path}: {len(miss)} tensors never written, e.g. {miss[:5]}")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        os.replace(self.tmp, self.path)
        self.closed = True
        return self.path

    def abort(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        if self.tmp.exists():
            self.tmp.unlink()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
            return False
        self.close()
        return False


def write_dense(keymap: KeyMap, path: str | os.PathLike, keys: Sequence[str] | None = None,
                device: str | torch.device = "cpu", metadata: Mapping[str, str] | None = None,
                log: Callable[[str], None] | None = None) -> dict:
    """Stream `keys` (default: keymap.dense_keys()) into a safetensors file at `path` under their model key names, each
    decoded and cast as from_pretrained(dtype=bf16) would hold it. Returns {tensors, bytes, seconds}."""
    t0 = time.time()
    keys = keymap.dense_keys() if keys is None else list(keys)
    meta = {"format": "pt", **(metadata or {})}
    nbytes = 0
    with StreamingWriter(path, keymap.specs(keys), meta) as w:
        for i, k in enumerate(keys):
            t = keymap.load(k, device)
            w.write(k, t)
            nbytes += t.numel() * t.element_size()
            del t
            if log is not None and (i + 1) % 500 == 0:
                log(f"dense {i + 1}/{len(keys)} ({nbytes / 2**30:.2f} GiB)")
    return {"tensors": len(keys), "bytes": nbytes, "seconds": round(time.time() - t0, 2)}


def _read_chunks(path: str, start: int, nbytes: int, chunk: int):
    """Raw bytes [start, start + nbytes) of a file in chunks; each chunk's pages are dropped from the page cache
    after use (posix_fadvise DONTNEED where available), so copying a table larger than RAM does not fill it."""
    fd = os.open(path, os.O_RDONLY)
    try:
        pos, end = start, start + nbytes
        while pos < end:
            n = min(chunk, end - pos)
            b = os.pread(fd, n, pos)
            if len(b) != n:
                raise OSError(f"{path}: short read at {pos} ({len(b)} of {n} bytes)")
            yield b
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, pos, n, os.POSIX_FADV_DONTNEED)
            pos += n
    finally:
        os.close(fd)


def host_specs(keymap: KeyMap) -> dict[str, tuple[str, tuple[int, ...]]]:
    """{model key: (safetensors dtype name, shape)} of the host tensors as the model holds them (release dtype)."""
    out = {}
    for k in keymap.host_keys():
        t = keymap.targets[k]
        out[k] = (st_dtype_name(t.dtype), tuple(t.shape))
    return out


def copy_host(keymap: KeyMap, path: str | os.PathLike, metadata: Mapping[str, str] | None = None,
              chunk: int = 256 << 20, log: Callable[[str], None] | None = None) -> dict:
    """Copy every host tensor (KeyMap.host_keys) into one safetensors file at `path` under its model key name: the
    stored bytes, unchanged (release dtype, never decoded), read and written in `chunk`-byte pieces (peak RAM: one
    chunk). Returns {tensors, bytes, seconds, dtypes}."""
    t0 = time.time()
    specs = host_specs(keymap)
    meta = {"format": "pt", **(metadata or {})}
    nbytes = 0
    with StreamingWriter(path, specs, meta) as w:
        for k in specs:
            part = keymap.targets[k].parts[""][0]
            info = keymap.index.tensors[part.key]
            want = math.prod(specs[k][1]) * ST_DTYPES[specs[k][0]][1]
            if info.nbytes != want:
                raise ValueError(f"{k}: stored {part.key} has {info.nbytes} bytes, the model tensor {want}")
            w.write_stream(k, _read_chunks(os.path.join(keymap.index.dir, info.file), info.start, info.nbytes, chunk))
            nbytes += info.nbytes
            if log is not None:
                log(f"host {k}: {info.nbytes / 2**30:.2f} GiB ({specs[k][0]} {list(specs[k][1])}), "
                    f"{time.time() - t0:.0f}s")
    return {"tensors": len(specs), "bytes": nbytes, "seconds": round(time.time() - t0, 2),
            "dtypes": dict(sorted(Counter(dt for dt, _ in specs.values()).items()))}


def mmap_file(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Every tensor of one safetensors file as a zero-copy CPU tensor over a private (copy-on-write) memory map of the
    file (CheckpointIndex.mmap's mapping): pages are read on first touch, writes never reach the file, the map stays
    open while any tensor views it. Dtype as read (ST_DTYPES: F8_E8M0 is uint8; callers view it as their dtype)."""
    hdr, _meta, _start = parse_safetensors_header(str(path))
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    out = {}
    for k, (dt, shape, start, nb) in hdr.items():
        if dt not in ST_DTYPES:
            raise TypeError(f"{path}: {k}: safetensors dtype {dt} is not supported")
        flat = torch.frombuffer(mm, dtype=torch.uint8, count=nb, offset=start) if nb else torch.empty(0, dtype=torch.uint8)
        out[k] = flat.view(ST_DTYPES[dt][0]).view(shape)
    return out
