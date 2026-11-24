"""One-pass converter for any architecture: trellis rungs + H4 + dense parts + counts (design-quality WP16).

One streaming pass of the BF16 (or FP8 / MXFP4, decoded as transformers does) checkpoint over the calibration corpus
writes everything a model's ladder needs (design-quality sections 2 and 5):
  - the trellis rungs (any number of tcq-plan/1 plans, e.g. TQ2.57 and TQ4.0): per MoE layer, the per-expert
    Hessians of the routed inputs are accumulated while the layer's windows stream through it (ramog.tcq_encode
    LayerJob: T4's semantics, the encode / factor votes, verified copies, expert chunking), then every rung is
    encoded from the same Hessians (the Hessian factors are shared between rungs with the same damping) and the
    Hessians are freed: they are never written out. Codes are written in T4's saved-code format
    (<out>/codes/<rung>/layer{L:03d}.{gate,up,down}.pt, the format quant_methods/t4_quant.py and qbench read);
  - H4 = k4row 4-bit g16 of the unrotated weights with Hessian-diagonal importance (sumsq = diag(H_e scale_e)
    count_e, exactly agents/qint/qd/hot_rot.py rot=0 imp=hess, i.e. WP11's statistics; ramog.store.BlobLayout's
    quantize, i.e. core's convert_any cold blobs): <out>/h4/layer{L:03d}.cold.bin. Without any rung, the importance
    is the routed inputs' sum of squares accumulated directly (ramog.stream.ExpertStats: WP11's path for new
    models), so no Hessian is built;
  - the packed dense parts, dense-hi (QD: every dense linear and the router k4row 8-bit g32 with importance,
    embeddings 4-bit g32) and dense-c (T1: q / o / lm_head / other linears 4-bit g16 GPTQ, k / v / router 8-bit g32,
    embeddings 4-bit g32), from per-module input statistics accumulated in the same pass (tools/dense_stats.py's Acc
    arithmetic; q / k / v share theirs), ramog.dense_recipe's quantizer and file format ("moe-tiering/dense-q" v1)
    under the model's module names: <out>/dense-hi/, <out>/dense-c/;
  - per-expert calibration counts (tokens routed to each expert), <out>/counts.safetensors;
  - store v2 directories (ramog/build_store_v2.py), one per variant: <out>/stores/<rung>/ and <out>/stores/H4/, each
    with the cold blobs, both dense directories, dense.safetensors (every dense tensor unquantized: ramog.loader
    loads the store today; WP10 then reads the packed parts), counts, plan, SHA256SUMS and the encoder's sha256
    (design-quality amendment 7).

The calibration forward is core's ramog/stream.py (WP6) for every Phase-A architecture: transformers' own base
forward with each decoder layer streamed in through ramog.ckpt.KeyMap (dtype plan with fp32_keys, stable router
top-k). LockstepStreamer adds what one weight pass over a real calibration corpus needs:
  - windows of several lengths (T2's corpus: 1024 / 2048 / 4096 tokens) in ONE weight pass: one base forward per
    length runs in its own thread; the threads hand over strictly (only one runs at a time) and the main thread runs
    every decoder layer once, binding its weights once, over all windows IN FILE ORDER (one window per call, as
    calib_stream.py), so the Hessian sums see the forwards in the order and flush grouping T4 / tools/tcq_convert.py
    use and the codes are T4's (with --expert-group 0: bit for bit);
  - verified binds (design-quality amendment 6): every tensor bound for a layer (and the resident stage) is also
    decoded on the CPU from the same read-ahead bytes and their exact fingerprints (tcq_encode.fingerprint) must
    agree, else the layer is bound again (LayerFailed after REPLAY_TRIES);
  - layer hooks after the bind (weights on the device): the converter finishes each layer there, before the unbind,
    so no host snapshot of the expert weights is needed;
  - with --offload-hidden (design-quality amendment 10, HostOffload) the hidden states stay in host memory between
    layer calls and only one window is on the device at a time; the output bits are the same.

Faults and resume (tools/tcq_convert.py's rules, with design-quality amendment 9): a layer that raised LayerFailed is
converted again in another pass (--layer-retries). <out>/layers/layer{L:03d}.done.json records every file of the layer
with its sha256 and the run key; a run started again with the same settings skips the layers whose done.json parses
and matches (a corrupt done.json counts as not done; the forward still streams through them). The run key names the
model, the tokens' sha256, every plan, the H4 / dense settings, the encoder settings, the sha256 of every source file
that decides the output (the encoder, ramog/quantizers.py, the streamer, the key map, the dequantizers, the k4row /
GPTQ quantizers, this file) and every LUT's sha256.

    python -m ramog.convert_tcq --model models/Qwen3.6-35B-A3B --tokens WINDOWS.pt --out DIR \
        --rung TQ2.57=PLAN.json [--rung TQ4.0=plan2.json] [--h4] [--dense dense-hi,dense-c] \
        [--expert-group 8] [--expert-chunk 0|N|auto] [--max-layers N] [--layer-retries 2] [--no-resume] [--no-stores]
        [--offload-hidden]
--rung NAME=depth16 makes the plan with ramog.tcq_rate (section 6 rule 2) at the ladder rung's bpw cap.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import queue
import sys
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from . import arch, ckpt, hfcompat, kquant, stream, tcq_plan
from . import dense_recipe as dr
from . import tcq_encode as tq
from .store import BlobLayout

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = "ramog.convert_tcq"
FORMAT = "moe-tiering/convert-tcq/1"
H4_BITS, H4_GROUP = 4, 16
H4_CHUNK = 16  # experts per H4 BlobLayout.quantize call (convert_any's default chunk)
EXIT_LAYERS_MISSING = 3

# dense-hi (QD, agents/qint/qd/cfg/dense-hi.json; the q235 lane's agents/q235/encode/plan/dense_parts.py): every dense
# linear (q/k/v/o, other linears, lm_head) and the router k4row 8-bit g32 with importance, no GPTQ; embeddings k4row
# 4-bit g32 plain MSE; norms bf16. Both recipes are ramog.dense_recipe's registered ones (WP10), applied through
# dr.recipe_spec(recipe, model_type) (per-architecture floors: gpt-oss attention and input embedding stay bf16).
DENSE_HI = dr.RECIPES["dense-hi"]
RECIPES = {"dense-hi": DENSE_HI, "dense-c": dr.RECIPES["dense-c"]}
DENSE_DIRS = {"dense-hi": "dense", "dense-c": "dense-c"}  # directory names inside a store (design-quality 2.1)
PACKED_ROLES = ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "router", "embed", "default")

# source files whose code decides a conversion's output (run key, amendment 9)
CODE_FILES = ("ramog/tcq_encode.py", "ramog/quantizers.py", "ramog/stream.py", "ramog/ckpt.py", "ramog/dequant.py",
              "ramog/arch.py", "ramog/hfcompat.py", "ramog/kquant.py", "ramog/gptq.py", "ramog/dense_recipe.py",
              "ramog/store.py", "ramog/tcq_plan.py", "ramog/convert_tcq.py")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def file_sha(path: str | os.PathLike, n: int = 64) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:n]


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def read_json_or_none(path: Path):
    """The JSON object in `path`, or None when the file is missing or corrupt (amendment 9: corrupt = not done)."""
    try:
        with open(path) as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def write_bytes(path: Path, data: torch.Tensor) -> None:
    """uint8 CPU tensor -> file (tmp, fsync, rename)."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(memoryview(data.contiguous().numpy()))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_safetensors(tensors: dict, path: Path, metadata: dict | None = None) -> None:
    """A safetensors file with reproducible bytes: ramog.ckpt.StreamingWriter (header written by json in the given
    order, tensors in the given order within an element size) instead of safetensors' own writer, whose header lists a
    multi-key metadata dict in a different order in every process (the same tensors gave files with different sha256
    in a conversion and its twin)."""
    specs = {k: (v.dtype, tuple(v.shape)) for k, v in tensors.items()}
    with ckpt.StreamingWriter(path, specs, metadata) as w:
        for k, v in tensors.items():
            w.write(k, v)


def host_mem() -> dict:
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS:", "VmHWM:")):
                k, v = line.split(":")
                out[{"VmRSS": "rss_gib", "VmHWM": "peak_rss_gib"}[k]] = round(int(v.split()[0]) / 2**20, 3)
    except OSError:
        pass
    return out


# =====================================================================================================================
# lockstep streamer: windows of several lengths in one weight pass, in file order, verified binds


class PassHooks:
    """What a LockstepStreamer pass calls (main thread, in this order per decoder layer):
    layer_begin(li, layer, site) after the layer's weights are bound (and verified); window(li, ids) before each call
    of the layer (ids = the global window indices of the call, in file order); layer_end(li, layer, site) after the
    layer's last call, weights still bound; layer_unbound(li) after the unbind. site: the MoESite or None."""

    def layer_begin(self, li, layer, site):
        pass

    def window(self, li, ids):
        pass

    def layer_end(self, li, layer, site):
        pass

    def layer_unbound(self, li):
        pass


class _Abort(RuntimeError):
    pass


def _fillable(out, n: int) -> bool:
    if isinstance(out, torch.Tensor):
        return out.dim() >= 1 and out.shape[0] == n
    if isinstance(out, tuple) and not hasattr(out, "_fields"):
        return all(o is None or _fillable(o, n) for o in out)
    return False


def _alloc_like(out, B: int):
    if isinstance(out, torch.Tensor):
        return torch.empty((B, *out.shape[1:]), dtype=out.dtype, device=out.device)
    return tuple(None if o is None else _alloc_like(o, B) for o in out)


def _fill(buf, out, s: int, e: int) -> None:
    if isinstance(buf, torch.Tensor):
        if not isinstance(out, torch.Tensor) or out.shape[0] != e - s or tuple(out.shape[1:]) != tuple(buf.shape[1:]):
            raise RuntimeError(f"micro-batch output {getattr(out, 'shape', type(out))} does not fit [{e - s}, "
                               f"{list(buf.shape[1:])}]")
        buf[s:e].copy_(out)
        return
    if not isinstance(out, tuple) or len(out) != len(buf):
        raise RuntimeError("micro-batch outputs change structure between calls")
    for b, o in zip(buf, out):
        if (b is None) != (o is None):
            raise RuntimeError("micro-batch outputs mix None and values")
        if b is not None:
            _fill(b, o, s, e)


class _Gather:
    """The outputs of a group's consecutive micro-batch calls, concatenated along dim 0 into preallocated tensors as
    they arrive (bitwise the same as ramog.stream.concat_batch's torch.cat; peak memory one output instead of two).
    Other output types fall back to concat_batch."""

    def __init__(self, B: int, spans: list):
        self.B, self.spans = B, spans
        self.buf, self.items, self.mode = None, [None] * len(spans), None

    def add(self, si: int, out) -> None:
        s, e = self.spans[si]
        if self.mode is None:
            self.mode = "fill" if _fillable(out, e - s) else "list"
            if self.mode == "fill":
                self.buf = _alloc_like(out, self.B)
        if self.mode == "fill":
            _fill(self.buf, out, s, e)
        else:
            self.items[si] = out

    def result(self):
        return self.buf if self.mode == "fill" else stream.concat_batch(self.items, self.B)


def _run_span(fn, module, args, kwargs, B: int, micro: int, maps: dict, spans: list, si: int):
    """ramog.stream.micro_call's computation for ONE micro-batch span (the same slicing and mapping copies)."""
    if micro >= B:
        return fn(module, *args, **kwargs)
    for v in list(args) + list(kwargs.values()):
        if stream._is_cache(v):
            raise RuntimeError(f"{type(module).__name__}: a cache cannot be shared by {len(spans)} micro-batches")
    s, e = spans[si]

    def one(x):
        return stream._span_mapping(x, spans, B, maps, si) if isinstance(x, Mapping) else stream.slice_batch(x, s, e, B)

    return fn(module, *(one(x) for x in args), **{k: one(v) for k, v in kwargs.items()})


class HostOffload:
    """The hidden states of a run_ordered pass in host memory instead of on the device (design-quality amendment 10).

    Without it every calibration window's hidden state stays on the device for the whole pass, input and output of a
    layer both alive: tokens x hidden x 2 B per copy, 31 GiB at hidden 6144 over T2's 2.5M tokens (GLM-5.3), more
    than half of a 64 GiB GCD. With it (convert --offload-hidden):
      - the input embedding runs one window (micro-batch span) at a time on the device and each output is copied into
        one host tensor [B, S, H] per window length;
      - the base forward (its own thread, as before) receives a PLACEHOLDER instead: a zero-stride device tensor of
        the same shape, dtype and device (no memory; NaN for float dtypes), registered to the host tensor. Everything the base forward
        derives from the hidden state's shape, dtype and device (position ids, rotary tables, masks) is computed
        exactly as before, on the device;
      - every decoder layer call (and every tail module: final norm, hc_head, ...) resolves the placeholders among
        its arguments to their host rows, uploads one span at a time, runs the same module call on the same values
        as the device path, and copies each span's outputs back into host tensors, which reach the base forward as
        new placeholders. Outputs reuse the host tensors of the call's consumed placeholder inputs of the same shape
        and dtype (in place: a span reads its rows before it writes them, and no row of another span is touched), so
        the host holds about one copy of the stream (plus the indexer top-k on GLM-5.3);
      - a consumed placeholder is unregistered after its call. The base forward may not read an offloaded hidden
        state again (e.g. a long skip connection) or transform one between two calls: the hidden-state argument of
        every layer / tail call must be the placeholder the previous step returned (identity), and an unregistered
        zero-stride device tensor reaching a call raises. tests/test_convert_offload.py checks the outputs against
        the device path bit for bit per architecture;
      - copies are checked (amendment 6): device -> host against the device bytes' fingerprint (tcq_encode.fingerprint
        on the device and on the host copy, which is recorded), host -> device against the recorded fingerprint (else
        computed on the CPU); re-copied on a mismatch, LayerFailed after REPLAY_TRIES. Counters: self.stats.
    The values every device computation sees are the device path's, so the outputs are the same bits; peak device
    memory is one window's activations."""

    def __init__(self, device):
        self.dev = torch.device(device)
        self._ph: dict = {}  # id(placeholder) -> (weakref(placeholder), host tensor)
        self._fp: dict = {}  # (id(host base), byte offset, nbytes) -> (weakref(host base), CPU fingerprint)
        self._lock = threading.RLock()
        self.cur: dict = {}  # group -> the placeholder its next layer / tail call must receive
        self.stats = {"d2h": 0, "h2d": 0, "d2h_gib": 0.0, "h2d_gib": 0.0, "d2h_retries": 0, "h2d_retries": 0,
                      "fp_hits": 0, "fp_misses": 0, "host_allocs": 0, "host_alloc_gib": 0.0, "reused": 0,
                      "d2h_s": 0.0, "h2d_s": 0.0}

    # -- placeholders -------------------------------------------------------------------------------------------------

    def placeholder(self, host: torch.Tensor) -> torch.Tensor:
        fill = float("nan") if host.dtype.is_floating_point else 0  # poison: a value read from it shows as NaN
        ph = torch.full((), fill, dtype=host.dtype, device=self.dev).expand(host.shape)
        k = id(ph)

        def gone(r, k=k):
            with self._lock:
                e = self._ph.get(k)
                if e is not None and e[0] is r:
                    del self._ph[k]
        with self._lock:
            self._ph[k] = (weakref.ref(ph, gone), host)
        return ph

    def host_of(self, x) -> torch.Tensor | None:
        """The host tensor of a registered placeholder; None for any other value. An unregistered zero-stride device
        tensor (an offloaded hidden state used again, or a view of one) raises."""
        if not isinstance(x, torch.Tensor):
            return None
        with self._lock:
            e = self._ph.get(id(x))
        if e is not None and e[0]() is x:
            return e[1]
        if (x.device == self.dev and x.dim() >= 1 and x.numel() > 1 and all(s == 0 for s in x.stride())):
            raise RuntimeError(f"host offload: an offloaded hidden state {list(x.shape)} was used again or transformed "
                               "outside a decoder layer / tail module call; this architecture needs the device path "
                               "(no --offload-hidden)")
        return None

    def _consumed(self, args, kwargs) -> list:
        """(placeholder, host) of every registered placeholder among the arguments (tuples / lists / mappings)."""
        found: dict = {}

        def walk(x):
            if isinstance(x, torch.Tensor):
                h = self.host_of(x)
                if h is not None:
                    found[id(x)] = (x, h)
            elif isinstance(x, (tuple, list)):
                for v in x:
                    walk(v)
            elif isinstance(x, Mapping):
                for v in x.values():
                    if isinstance(v, torch.Tensor) and self.host_of(v) is not None:
                        raise RuntimeError("host offload: a mapping argument holds an offloaded hidden state")
        for v in list(args) + list(kwargs.values()):
            walk(v)
        return list(found.values())

    def release(self, consumed: list) -> None:
        with self._lock:
            for ph, _h in consumed:
                e = self._ph.get(id(ph))
                if e is not None and e[0]() is ph:
                    del self._ph[id(ph)]
        self.prune()

    def prune(self) -> None:
        with self._lock:
            for k in [k for k, (r, _fp) in self._fp.items() if r() is None]:
                del self._fp[k]

    # -- checked copies -----------------------------------------------------------------------------------------------

    @staticmethod
    def _key(t: torch.Tensor):
        b = t._base if t._base is not None else t
        return (id(b), t.data_ptr() - b.data_ptr(), t.numel() * t.element_size()), b

    def d2h(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype:
            raise RuntimeError(f"host offload: {list(src.shape)} {src.dtype} into {list(dst.shape)} {dst.dtype}")
        t0 = time.time()
        want = tq.fingerprint(src).cpu()
        for _ in range(tq.REPLAY_TRIES):
            dst.copy_(src)
            got = tq.fingerprint(dst)
            if torch.equal(got, want):
                k, b = self._key(dst)
                with self._lock:
                    self._fp[k] = (weakref.ref(b), got)
                self.stats["d2h"] += 1
                self.stats["d2h_gib"] += dst.numel() * dst.element_size() / 2**30
                self.stats["d2h_s"] += time.time() - t0
                return
            self.stats["d2h_retries"] += 1
        raise tq.LayerFailed(f"device -> host copy of a {tuple(src.shape)} hidden state kept differing from the device")

    def h2d(self, t: torch.Tensor) -> torch.Tensor:
        k, b = self._key(t)
        with self._lock:
            rec = self._fp.get(k)
        if rec is not None and rec[0]() is b:
            want = rec[1]
            self.stats["fp_hits"] += 1
        else:
            want = tq.fingerprint(t)
            self.stats["fp_misses"] += 1
        t0 = time.time()
        for _ in range(tq.REPLAY_TRIES):
            tq._sync(self.dev)
            d = t.to(self.dev)
            if torch.equal(tq.fingerprint(d).cpu(), want):
                self.stats["h2d"] += 1
                self.stats["h2d_gib"] += t.numel() * t.element_size() / 2**30
                self.stats["h2d_s"] += time.time() - t0
                return d
            self.stats["h2d_retries"] += 1
            del d
        raise tq.LayerFailed(f"host -> device copy of a {tuple(t.shape)} hidden state kept differing from the host bytes")

    # -- calls --------------------------------------------------------------------------------------------------------

    def _span_args(self, args, kwargs, B: int, spans: list, si: int, maps: dict):
        """_run_span's per-span arguments with every placeholder replaced by its uploaded host rows."""
        s, e = spans[si]

        def one(x):
            if isinstance(x, torch.Tensor):
                h = self.host_of(x)
                if h is None:
                    return stream.slice_batch(x, s, e, B)
                if h.shape[0] != B:
                    raise RuntimeError(f"host offload: placeholder {list(h.shape)} in a batch of {B}")
                return self.h2d(h[s:e])
            if isinstance(x, tuple) and not hasattr(x, "_fields"):
                return tuple(one(v) for v in x)
            if isinstance(x, list):
                return [one(v) for v in x]
            if isinstance(x, Mapping):
                return stream._span_mapping(x, spans, B, maps, si)
            return x

        return tuple(one(x) for x in args), {k: one(v) for k, v in kwargs.items()}

    def call(self, fn, module, args, kwargs, g, B: int, spans: list, maps: dict, what: str):
        """fn(module, *args, **kwargs) span by span with host-memory inputs and outputs (see the class docstring);
        returns the outputs as placeholders (run_ordered's decoder layers use begin / gather / finish directly, to
        interleave the groups' spans in file order)."""
        consumed = self.begin(args, kwargs, g, what)
        gat = self.gather(B, spans, consumed)
        for si in range(len(spans)):
            a, kw = self._span_args(args, kwargs, B, spans, si, maps)
            out = fn(module, *a, **kw)
            del a, kw
            gat.add(si, out)
            del out
        return self.finish(gat, consumed, g)

    def begin(self, args, kwargs, g, what: str) -> list:
        hid = args[0] if args else kwargs.get("hidden_states")
        if hid is not self.cur.get(g):
            raise RuntimeError(f"host offload: the hidden-state argument of {what} is not the offloaded output of the "
                               "previous step (the base forward transformed it); this architecture needs the device "
                               "path (no --offload-hidden)")
        return self._consumed(args, kwargs)

    def gather(self, B: int, spans: list, consumed: list) -> "_HostGather":
        return _HostGather(self, B, spans, [h for _ph, h in consumed])

    def finish(self, gat: "_HostGather", consumed: list, g):
        res = gat.result()
        self.release(consumed)
        self.cur[g] = res if isinstance(res, torch.Tensor) else next(
            (o for o in res if isinstance(o, torch.Tensor)), None)
        return res

    def embed(self, fn, module, args, kwargs, g, B: int, spans: list):
        gat = self.gather(B, spans, [])
        for si, (s, e) in enumerate(spans):
            a = tuple(stream.slice_batch(x, s, e, B) for x in args)
            kw = {k: stream.slice_batch(v, s, e, B) for k, v in kwargs.items()}
            out = fn(module, *a, **kw)
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"host offload: the input embedding returned {type(out).__name__}")
            gat.add(si, out)
            del out
        res = gat.result()
        self.cur[g] = res
        return res


class _HostGather:
    """The span outputs of one call in host tensors [B, ...] (checked copies), reusing the call's consumed host inputs
    of the same shape and dtype; result() gives them back as placeholders in the outputs' structure."""

    def __init__(self, off: HostOffload, B: int, spans: list, reuse: list):
        self.off, self.B, self.spans, self.pool, self.buf = off, B, spans, list(reuse), None

    def _alloc(self, out):
        if isinstance(out, torch.Tensor):
            shape = (self.B, *out.shape[1:])
            for i, h in enumerate(self.pool):
                if tuple(h.shape) == shape and h.dtype == out.dtype and h.is_contiguous():
                    self.off.stats["reused"] += 1
                    return self.pool.pop(i)
            h = torch.empty(shape, dtype=out.dtype)
            self.off.stats["host_allocs"] += 1
            self.off.stats["host_alloc_gib"] += h.numel() * h.element_size() / 2**30
            return h
        return tuple(None if o is None else self._alloc(o) for o in out)

    def _fill(self, buf, out, s: int, e: int) -> None:
        if isinstance(buf, torch.Tensor):
            if not isinstance(out, torch.Tensor) or out.shape[0] != e - s or tuple(out.shape[1:]) != tuple(buf.shape[1:]):
                raise RuntimeError(f"micro-batch output {getattr(out, 'shape', type(out))} does not fit [{e - s}, "
                                   f"{list(buf.shape[1:])}]")
            self.off.d2h(buf[s:e], out)
            return
        if not isinstance(out, tuple) or len(out) != len(buf):
            raise RuntimeError("micro-batch outputs change structure between calls")
        for b, o in zip(buf, out):
            if (b is None) != (o is None):
                raise RuntimeError("micro-batch outputs mix None and values")
            if b is not None:
                self._fill(b, o, s, e)

    def add(self, si: int, out) -> None:
        s, e = self.spans[si]
        if self.buf is None:
            if not _fillable(out, e - s):
                raise RuntimeError(f"host offload: a call output {type(out).__name__} is not batch-shaped (dim 0 = "
                                   "windows); this architecture needs the device path (no --offload-hidden)")
            self.buf = self._alloc(out)
            self.pool = []
        self._fill(self.buf, out, s, e)

    def result(self):
        def wrap(b):
            if isinstance(b, torch.Tensor):
                return self.off.placeholder(b)
            return tuple(None if v is None else wrap(v) for v in b)
        return wrap(self.buf)


class LockstepStreamer(stream.Streamer):
    """ramog.stream.Streamer with run_ordered(): see the module docstring. verify: fingerprint-checked binds.
    offload_hidden: the hidden states in host memory (HostOffload, design-quality amendment 10)."""

    def __init__(self, model_dir, device="cuda:0", *, verify: bool = True, offload_hidden: bool = False, **kw):
        super().__init__(model_dir, device, **kw)
        self.offload = HostOffload(self.dev) if offload_hidden else None
        self.verify = verify
        self.verify_info = {"layers_verified": 0, "rebinds": 0, "bytes_verified": 0, "resident_reuploads": 0,
                            "seconds": 0.0}
        if verify:
            self.verify_resident()

    # -- verification -------------------------------------------------------------------------------------------------

    def _tensor(self, key: str) -> torch.Tensor:
        mod, leaf = self._owner[key]
        return mod._parameters[leaf] if leaf in mod._parameters else mod._buffers[leaf]

    def _cpu_fingerprint(self, km: ckpt.KeyMap, key: str) -> torch.Tensor:
        """The fingerprint of model tensor `key` decoded on the CPU (the same KeyMap decode as the device bind)."""
        t = km.targets[key]
        v = km.load(key, "cpu")
        if tuple(v.shape) != tuple(t.shape) or v.dtype != t.dtype:
            raise RuntimeError(f"{key}: CPU decode {v.dtype} {list(v.shape)}, expected {t.dtype} {list(t.shape)}")
        self.verify_info["bytes_verified"] += v.numel() * v.element_size()
        return tq.fingerprint(v)

    def _retie(self) -> None:
        for head_key, emb_key in self.keymap.tied.items():
            hmod, hleaf = stream._split(head_key)
            emod, eleaf = self._owner[emb_key]
            self.model.get_submodule(hmod)._parameters[hleaf] = emod._parameters[eleaf]

    @torch.no_grad()
    def verify_resident(self) -> None:
        """Every resident tensor (embedding, final norm, head, ...) against its CPU decode; re-uploaded on a mismatch."""
        t0 = time.time()
        bad = []
        for k in self.resident_keys:
            ref = self._cpu_fingerprint(self.keymap, k)
            for _ in range(tq.REPLAY_TRIES):
                if torch.equal(tq.fingerprint(self._tensor(k)).cpu(), ref):
                    break
                self.verify_info["resident_reuploads"] += 1
                self.log(f"resident tensor {k}: device bytes differ from the checkpoint decode; uploading again")
                self._set(k, self._value(self.keymap, k))
                self._retie()
            else:
                bad.append(k)
        self.index.close()
        self.verify_info["seconds"] += time.time() - t0
        if bad:
            raise tq.LayerFailed(f"resident tensors {bad[:5]} kept differing from their checkpoint decode")

    def bind_verified(self, keys, raw) -> list:
        """bind(), checked: each bound tensor's device fingerprint must equal the fingerprint of the same key decoded on
        the CPU from the same read-ahead bytes; bound again on a mismatch (LayerFailed after REPLAY_TRIES)."""
        if not self.verify:
            return self.bind(keys, raw)
        t0 = time.time()
        km = copy.copy(self.keymap)
        km.index = stream.RawIndex(raw)
        refs = {k: self._cpu_fingerprint(km, k) for k in keys}
        bad: list = []
        for _ in range(tq.REPLAY_TRIES):
            saved = self.bind(keys, raw)
            bad = [k for k in keys if not torch.equal(tq.fingerprint(self._tensor(k)).cpu(), refs[k])]
            if not bad:
                self.verify_info["layers_verified"] += 1
                self.verify_info["seconds"] += time.time() - t0
                return saved
            self.unbind(saved)
            self.verify_info["rebinds"] += 1
            self.log(f"bind: {len(bad)} tensors differ from their CPU decode ({bad[:3]}); binding again")
        raise tq.LayerFailed(f"bound tensors {bad[:5]} kept differing from their CPU decode")

    # -- the pass -----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def run_ordered(self, windows, *, micro: int = 1, n_layers: int | None = None, hooks: PassHooks | None = None,
                    keep_hidden: bool = True) -> dict:
        """One weight pass over `windows` (1-D token tensors of any lengths, in file order). Each length is one base
        forward in its own thread; the main thread runs every decoder layer once over all windows in file order
        (micro windows per call within a length). Returns {"hidden": [last hidden state [S, H] per window] (views; with
        keep_hidden), "per_layer": [...], "seconds", "groups": {length: windows}, "verify": ...}."""
        hooks = hooks or PassHooks()
        if micro < 1:
            raise ValueError("micro must be >= 1")
        rows = [w.reshape(-1) for w in windows]
        lens = sorted({int(w.numel()) for w in rows})
        gidx = [[i for i, w in enumerate(rows) if int(w.numel()) == L] for L in lens]
        G = list(range(len(lens)))
        ids = {g: torch.stack([rows[i] for i in gidx[g]]) for g in G}
        B = {g: ids[g].shape[0] for g in G}
        spans = {g: [(s, min(B[g], s + micro)) for s in range(0, B[g], micro)] for g in G}
        order = sorted((gidx[g][s], g, si) for g in G for si, (s, _e) in enumerate(spans[g]))
        n_all = len(self.layers)
        n = n_all if n_layers is None else int(n_layers)
        if not 1 <= n <= n_all:
            raise ValueError(f"n_layers {n} outside 1..{n_all}")
        plans = [(li, self.checkpoint_keys(self.layer_keys[li])) for li in range(n)]
        reader = stream.LayerReader(self.index, plans, depth=self.depth, threads=self.threads, drop_cache=self.drop,
                                    pin=self.dev.type == "cuda")
        tls = threading.local()
        to_main = {g: queue.Queue() for g in G}
        to_worker = {g: queue.Queue() for g in G}
        maps = {g: {} for g in G}
        dev_sync = (lambda: torch.cuda.synchronize(self.dev)) if self.dev.type == "cuda" else (lambda: None)
        off = self.offload
        if off is not None:
            off.cur = {}

        def make_layer_forward(li):
            def forward(*args, **kwargs):
                g = getattr(tls, "group", None)
                if g is None:
                    raise RuntimeError(f"decoder layer {li} called outside a lockstep worker thread")
                to_main[g].put(("layer", li, args, kwargs))
                rep = to_worker[g].get()
                if isinstance(rep, BaseException):
                    raise rep
                return rep
            return forward

        def make_tail_forward(m):
            orig = type(m).forward

            def forward(*args, **kwargs):
                g = tls.group
                if off is not None and off._consumed(args, kwargs):
                    return off.call(orig, m, args, kwargs, g, B[g], spans[g], maps[g], type(m).__name__)
                return stream.micro_call(orig, m, args, kwargs, B[g], micro, maps[g])
            return forward

        def make_embed_forward(m):
            orig = type(m).forward

            def forward(*args, **kwargs):
                g = tls.group
                return off.embed(orig, m, args, kwargs, g, B[g], spans[g])
            return forward

        def worker(g):
            tls.group = g
            try:
                if self.dev.type == "cuda":
                    torch.cuda.set_device(self.dev)
                with torch.no_grad():
                    out = self.base(input_ids=ids[g].to(self.dev), use_cache=False)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                to_main[g].put(("done", hidden))
            except BaseException as e:  # noqa: BLE001 - handed to the main thread
                to_main[g].put(("error", e))

        def recv(g, thread):
            while True:
                try:
                    msg = to_main[g].get(timeout=10)
                except queue.Empty:
                    if not thread.is_alive():
                        raise RuntimeError(f"lockstep worker {g} died without a message") from None
                    continue
                if msg[0] == "error":
                    raise msg[1]
                return msg

        t_start = time.time()
        installed, threads, per_layer = [], {}, []
        layers_all = self.layers
        hidden = {}
        try:
            if n < n_all:
                self.base.layers = nn.ModuleList(list(layers_all)[:n])
            for li in range(n):
                layers_all[li].forward = make_layer_forward(li)
                installed.append(layers_all[li])
            emb = self.model.get_input_embeddings() if off is not None else None
            for m in self._tail_modules():
                if m is emb:
                    continue
                m.forward = make_tail_forward(m)
                installed.append(m)
            if emb is not None:
                emb.forward = make_embed_forward(emb)
                installed.append(emb)
            pending = {}
            for g in G:  # strict handoff: each worker runs alone until it reaches the first decoder layer
                threads[g] = threading.Thread(target=worker, args=(g,), name=f"lockstep-{lens[g]}", daemon=True)
                threads[g].start()
                pending[g] = recv(g, threads[g])
            for li in range(n):
                layer = layers_all[li]
                for g in G:
                    if pending[g][0] != "layer" or pending[g][1] != li:
                        raise RuntimeError(f"group {lens[g]}: expected decoder layer {li}, got {pending[g][:2]}")
                tw = time.time()
                key, raw, buf, rsec, rbytes = reader.get()
                t_wait = time.time() - tw
                if key != li:
                    raise RuntimeError(f"read-ahead delivered layer {key}, the pass is at layer {li}")
                t1 = time.time()
                try:
                    saved = self.bind_verified(self.layer_keys[li], raw)
                finally:
                    reader.release(buf)
                del raw
                left = [nm for nm, t in list(layer.named_parameters()) + list(layer.named_buffers())
                        if t.device.type == "meta"]
                if left:
                    self.unbind(saved)
                    raise RuntimeError(f"layer {li}: tensors left on meta after binding: {left[:5]}")
                dev_sync()
                t_bind = time.time() - t1
                site = self.sites.get(li)
                orig = type(layer).forward
                outs = {}
                try:
                    hooks.layer_begin(li, layer, site)
                    t2 = time.time()
                    if off is None:
                        gathers = {g: _Gather(B[g], spans[g]) for g in G}
                    else:
                        consumed = {g: off.begin(pending[g][2], pending[g][3], g, f"decoder layer {li}") for g in G}
                        gathers = {g: off.gather(B[g], spans[g], consumed[g]) for g in G}
                    for first, g, si in order:
                        s, e = spans[g][si]
                        hooks.window(li, gidx[g][s:e])
                        if off is not None:
                            a, kw = off._span_args(pending[g][2], pending[g][3], B[g], spans[g], si, maps[g])
                            out = orig(layer, *a, **kw)
                            del a, kw
                            gathers[g].add(si, out)
                            if si == len(spans[g]) - 1:
                                outs[g] = off.finish(gathers[g], consumed[g], g)
                                gathers[g] = None
                            del out
                            continue
                        out = _run_span(orig, layer, pending[g][2], pending[g][3], B[g], micro, maps[g], spans[g], si)
                        if micro >= B[g]:
                            outs[g] = out
                        else:
                            gathers[g].add(si, out)
                            if si == len(spans[g]) - 1:
                                outs[g] = gathers[g].result()
                                gathers[g] = None
                        del out
                    dev_sync()
                    t_comp = time.time() - t2
                    t3 = time.time()
                    hooks.layer_end(li, layer, site)
                    t_end = time.time() - t3
                finally:
                    self.unbind(saved)
                    del saved
                hooks.layer_unbound(li)
                per_layer.append({"layer": li, "read_s": round(rsec, 3), "wait_s": round(t_wait, 3),
                                  "bind_s": round(t_bind, 3), "compute_s": round(t_comp, 3), "end_s": round(t_end, 3),
                                  "read_bytes": rbytes})
                for g in G:  # hand each group its output; it runs alone until its next layer call (or the end)
                    pending[g] = None
                    to_worker[g].put(outs.pop(g))
                    pending[g] = recv(g, threads[g])
            for g in G:
                if pending[g][0] != "done":
                    raise RuntimeError(f"group {lens[g]}: expected the end of the base forward, got {pending[g][:2]}")
                hidden[g] = pending[g][1]
                if off is not None:
                    h = off.host_of(hidden[g])
                    if h is None:
                        raise RuntimeError(f"group {lens[g]}: the base forward's last hidden state is not an offloaded "
                                           "one (the base forward transformed it)")
                    hidden[g] = h
        except BaseException:
            for g, t in threads.items():
                if t.is_alive():
                    to_worker[g].put(_Abort("lockstep pass aborted"))
            for t in threads.values():
                t.join(timeout=120)
            raise
        finally:
            reader.close()
            if n < n_all:
                self.base.layers = layers_all
            for m in installed:
                m.__dict__.pop("forward", None)
        for t in threads.values():
            t.join(timeout=120)
        self.read_stats = (reader.read_bytes, reader.read_seconds)
        out_h = None
        if keep_hidden:
            out_h = [None] * len(rows)
            for g in G:
                for j, i in enumerate(gidx[g]):
                    out_h[i] = hidden[g][j]
        dev_sync()
        if off is not None:
            off.cur = {}
            off.prune()
        return {"hidden": out_h, "per_layer": per_layer, "seconds": time.time() - t_start,
                "groups": {int(lens[g]): B[g] for g in G}, "verify": dict(self.verify_info),
                "read_gib": round(reader.read_bytes / 2**30, 2), "read_s": round(reader.read_seconds, 1),
                **({"offload": {k: round(v, 3) if isinstance(v, float) else v for k, v in off.stats.items()}}
                   if off is not None else {})}


# =====================================================================================================================
# per-layer statistics


class MoETap:
    """forward pre-hook on a site's experts module: fn(x [T, H], idx [T, k], wts [T, k] or None) for every call (the
    module's first three arguments, positional or by name)."""

    def __init__(self, site: arch.MoESite, fn):
        m = site.ref
        self.names = list(inspect.signature(type(m).forward).parameters)[1:4]
        self.fn, self.site = fn, site
        self.handle = m.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _arg(self, args, kwargs, i):
        if len(args) > i:
            return args[i]
        return kwargs.get(self.names[i]) if i < len(self.names) else None

    def _pre(self, _mod, args, kwargs):
        x, idx, w = self._arg(args, kwargs, 0), self._arg(args, kwargs, 1), self._arg(args, kwargs, 2)
        x = x.reshape(-1, x.shape[-1])
        idx = idx.reshape(-1, idx.shape[-1])
        if idx.shape[0] != x.shape[0]:
            raise RuntimeError(f"{self.site.name}: {x.shape[0]} tokens, {idx.shape[0]} index rows")
        if w is not None:
            w = w.reshape(idx.shape) if w.numel() == idx.numel() else None
        self.fn(x, idx, w)
        return None

    def remove(self):
        self.handle.remove()


class _Acc:
    """tools/dense_stats.py's Acc: sum x^2 [k], count, sum x x^T [k, k] (float32 per window, float64 sums)."""

    def __init__(self, k: int, dev, full: bool):
        self.k = k
        self.sumsq = torch.zeros(k, dtype=torch.float64, device=dev)
        self.xtx = torch.zeros(k, k, dtype=torch.float64, device=dev) if full else None
        self.count = 0
        self.calls = 0

    def add(self, x: torch.Tensor) -> None:
        x2 = x.reshape(-1, self.k).float()
        self.sumsq += x2.square().sum(0).double()
        if self.xtx is not None:
            self.xtx += (x2.t() @ x2).double()
        self.count += x2.shape[0]
        self.calls += 1

    def entry(self) -> dict:
        out = {"sumsq": tq.to_host_checked(self.sumsq.float()), "count": torch.tensor(self.count, dtype=torch.int64)}
        if self.xtx is not None:
            out["xtx"] = tq.to_host_checked(self.xtx.float())
        return out


class DenseStats:
    """Input statistics of the packed dense weights of one layer: a forward pre-hook per module (its first argument,
    reshaped to [-1, in_features]). Modules that receive the same live tensor object in one call (q / k / v) share
    the first one's accumulator (decided on the first call; a later disagreement raises)."""

    def __init__(self, modules: dict, device, xtx: bool):
        self.dev, self.full = device, xtx
        self.acc: dict[str, _Acc] = {}
        self.owner: dict[str, str] = {}
        self.unusable: dict[str, str] = {}
        self._seen: dict[int, tuple] = {}
        self.handles = []
        for name, mod in modules.items():
            self.handles.append(mod.register_forward_pre_hook(self._make(name, mod), with_kwargs=True))

    def new_call(self) -> None:
        self._seen.clear()

    def _make(self, name, mod):
        k = int(mod.weight.shape[-1])

        def pre(_m, args, kwargs):
            x = args[0] if args else next((v for v in kwargs.values() if isinstance(v, torch.Tensor)), None)
            if x is None or x.shape[-1] != k:
                self.unusable[name] = f"input {None if x is None else list(x.shape)} vs in_features {k}"
                return None
            hit = self._seen.get(id(x))
            if hit is not None and hit[0]() is x:
                own = self.owner.setdefault(name, hit[1])
                if own != hit[1]:
                    raise RuntimeError(f"{name}: input shared with {hit[1]} in this call, with {own} before")
                return None
            own = self.owner.setdefault(name, name)
            if own != name:
                raise RuntimeError(f"{name}: shared {own}'s input before, a tensor of its own now")
            self._seen[id(x)] = (weakref.ref(x), name)
            acc = self.acc.get(name)
            if acc is None:
                acc = self.acc[name] = _Acc(k, self.dev, self.full)
            acc.add(x)
            return None
        return pre

    def entry(self, name: str) -> dict | None:
        own = self.owner.get(name)
        return None if own is None else self.acc[own].entry()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class SumsqStats(stream.ExpertStats):
    """ramog.stream.ExpertStats (sum of squares of the routed inputs, float64) for the H4 importance without Hessians;
    result()'s device -> host copies are checked (design-quality amendment 10)."""

    def result(self) -> dict:
        if self.gu is None or self.dn is None:
            raise RuntimeError(f"{self.site.name}: no routed tokens were seen")
        if int(self.dn_rows) != int(self.cnt.sum()):
            raise RuntimeError(f"{self.site.name}: down rows {int(self.dn_rows)} != routed pairs {int(self.cnt.sum())}")
        cnt = tq.to_host_checked(self.cnt)
        return {"gate_up": {"sumsq": tq.to_host_checked(self.gu.float()), "count": cnt.clone()},
                "down": {"sumsq": tq.to_host_checked(self.dn.float()), "count": cnt.clone()}}


# =====================================================================================================================
# multi-rung layer job


def encode_rungs(Wg: torch.Tensor, Wd: torch.Tensor, h_gu: tq.LayerHessians, h_dn: tq.LayerHessians, rungs: dict,
                 layer: int, users: dict, device, lut_dir, timings: dict, group: int, votes: int) -> dict:
    """tcq_encode.encode_layer for several plan entries from the same Hessians: per input kind, the factors of each
    distinct (damp, min_tokens) once, then every rung's matrices (the same calls as encode_layer, hence the same codes
    as one-rung conversions). Returns {rung: {"gate"|"up"|"down": record}}."""
    out = {r: {} for r in rungs}
    if not rungs:
        return out
    I = Wd.shape[2]  # noqa: E741
    for hs, kind in ((h_gu, "gu"), (h_dn, "dn")):
        n = hs.H.shape[-1]
        rot = tq.checked_rotation(n, layer * tq.ROT_SEED, device)
        factors: dict = {}
        for r, ent in rungs.items():
            fk = (float(ent.get("damp", 0.3)), int(ent.get("min_tokens", 8)))
            if fk not in factors:
                t0 = time.time()
                factors[fk] = tq.hessian_factors(hs, rot, *fk)
                tq._sync(device)
                timings["factor_s"] = timings.get("factor_s", 0.0) + time.time() - t0
            U = factors[fk]
            mats = ((("gate", Wg[:, :I], ent["gu"]), ("up", Wg[:, I:], ent["gu"])) if kind == "gu"
                    else (("down", Wd, ent["dn"]),))
            for m, W, spec in mats:
                t0 = time.time()
                sp = tq.parse_spec(spec)
                u = (users.get(r) or {}).get(sp["lut_key"], 1)
                out[r][m] = tq.encode_matrix(W, U, spec, layer, rot, device, lut_dir, u, m, hs.ids, group=group,
                                             votes=votes)
                timings[f"encode_{r}_{m}_s"] = timings.get(f"encode_{r}_{m}_s", 0.0) + time.time() - t0
        del factors
    return out


def h4_importance_from_hessians(hs: tq.LayerHessians) -> torch.Tensor:
    """sumsq [n_e, n] = diag(H_e * scale_e) * count_e, element for element hot_rot.sumsq_from_hess (rot=None), then
    kquant.importance_from_stats: WP11's H4 importance (agents/qint/qd/h4_stats.py)."""
    diag = hs.H.diagonal(dim1=1, dim2=2).float() * hs.scale.float()[:, None]
    sumsq = diag * hs.count.to(diag.device).float()[:, None]
    return kquant.importance_from_stats(sumsq, hs.count)


class MultiLayerJob(tq.LayerJob):
    """tcq_encode.LayerJob that finishes into several rungs and the H4 blobs from the same per-chunk Hessians.
    finish_multi(W_gu, W_dn) -> {"rungs": {rung: {matrix: record}}, "h4": uint8 [E, nbytes] (CPU) or None}."""

    def __init__(self, layer, n_experts, hidden, inter, rungs: dict, users: dict, device, n_windows, *,
                 h4_layout: BlobLayout | None = None, **kw):
        first = next(iter(rungs.values())) if rungs else {"gu": None, "dn": None, "damp": 0.3, "min_tokens": 8}
        super().__init__(layer, n_experts, hidden, inter, first, device, n_windows, **kw)
        self.rungs, self.rung_users, self.h4_layout = rungs, users, h4_layout

    @torch.no_grad()
    def finish_multi(self, W_gu: torch.Tensor, W_dn: torch.Tensor) -> dict:
        if self.dead:
            raise tq.LayerFailed(f"layer {self.layer}: job aborted")
        if self.acc.seen != self.n_windows:
            raise RuntimeError(f"layer {self.layer}: {self.acc.seen} forwards seen, {self.n_windows} expected")
        try:
            return self._finish_multi(W_gu, W_dn)
        except tq.LayerFailed:
            self.abort()
            raise
        finally:
            if self.store is not None:
                self.store.close()
            self.captured = None

    def _finish_multi(self, W_gu, W_dn) -> dict:
        parts = {r: {m: [] for m in tq.MATRICES} for r in self.rungs}
        h4_parts = []
        counts = None
        for ci, (e0, e1) in enumerate(self.chunks):  # LayerJob._finish's Hessian part, unchanged
            t0 = time.time()
            Wg = tq.to_device_verified(W_gu[e0:e1], self.dev)
            if ci == 0:
                acc = self.acc
                self.acc = None
            else:
                acc = self._replay(ci, e0, e1, Wg)
                if counts is not None and not torch.equal(acc.cnt, counts):
                    raise tq.LayerFailed(f"layer {self.layer}: replayed routing differs from the live pass")
            try:
                h_gu, h_dn = acc.result()
            except tq.AccumulationFlake as err:
                if ci or self.captured is None:
                    raise
                self._count("replay_retries", f"layer {self.layer} chunk 0 (live): {err}; replayed from the captures")
                acc = self._replay(0, e0, e1, Wg)
                h_gu, h_dn = acc.result()
            counts = acc.cnt
            del acc
            tq._sync(self.dev)
            self.timings["hessian_finish_s"] = self.timings.get("hessian_finish_s", 0.0) + time.time() - t0
            Wd = tq.to_device_verified(W_dn[e0:e1], self.dev)
            for attempt in range(len(tq.FINISH_PAUSES_S) + 1):  # the chunk's Hessians stay: retry in place first
                try:
                    recs = encode_rungs(Wg, Wd, h_gu, h_dn, self.rungs, self.layer, self.rung_users, self.dev,
                                        self.lut_dir, self.timings, self.group, self.votes)
                    break
                except tq.LayerFailed as err:
                    if attempt == len(tq.FINISH_PAUSES_S):
                        raise
                    self._count("finish_retries", f"layer {self.layer} chunk {ci} encode attempt {attempt}: {err}; "
                                f"again in {tq.FINISH_PAUSES_S[attempt]} s")
                    time.sleep(tq.FINISH_PAUSES_S[attempt])
            for r in self.rungs:
                for m in tq.MATRICES:
                    parts[r][m].append(recs[r][m])
            if self.h4_layout is not None:
                t0 = time.time()
                imp_gu, imp_dn = h4_importance_from_hessians(h_gu), h4_importance_from_hessians(h_dn)
                for a in range(0, e1 - e0, H4_CHUNK):
                    b = min(e1 - e0, a + H4_CHUNK)
                    blob = self.h4_layout.quantize(Wg[a:b], Wd[a:b], (imp_gu[a:b], imp_dn[a:b]), device=self.dev,
                                                   info=self.timings.setdefault("h4_info", {}))
                    h4_parts.append(tq.to_host_checked(blob))
                    del blob
                tq._sync(self.dev)
                self.timings["h4_s"] = self.timings.get("h4_s", 0.0) + time.time() - t0
            del Wg, Wd, h_gu, h_dn, recs
        self.refs = torch.zeros(0, 6, dtype=torch.int64)
        self.counts = tq.to_host_checked(counts)
        out = {r: {m: tq.concat_records(parts[r][m]) for m in tq.MATRICES} for r in self.rungs}
        for r, recs in out.items():
            tq.check_records(recs, f"layer {self.layer} {r}")
        return {"rungs": out, "h4": torch.cat(h4_parts) if h4_parts else None}


# =====================================================================================================================
# the conversion


def _sha_of(path: Path) -> str:
    return file_sha(path, 64)


def code_shas() -> dict:
    return {f: file_sha(ROOT / f, 16) for f in CODE_FILES}


def git_rev() -> str | None:
    try:
        import subprocess

        r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def packed_modules(model: nn.Module, prefix: str, roles) -> dict[str, tuple[nn.Module, str]]:
    """{module name: (module, role)} of the modules under `prefix` with a 2-D weight that a dense recipe packs (roles
    of ramog.dense_recipe.roles_from_arch: q/k/v/o, other selected linears = "default", lm_head, router, embed) or
    keeps in bf16 on purpose (role "bf16": linears the dense plan excludes, e.g. shared_expert_gate)."""
    out = {}
    for name, m in model.named_modules():
        if not name.startswith(prefix):
            continue
        w = m._parameters.get("weight")
        if w is None or w.dim() != 2:
            continue
        r = roles(name)
        if r in PACKED_ROLES or r == "bf16":
            out[name] = (m, r)
    return out


class Conversion(PassHooks):
    """State of one conversion run (see convert())."""

    def __init__(self, S: LockstepStreamer, out: Path, cfg: dict):
        self.S, self.out, self.cfg = S, out, cfg
        self.dev = S.dev
        self.sites = S.sites
        self.lay = None
        if self.sites:
            from .convert_any import uniform_experts

            self.lay = uniform_experts([self.sites[k] for k in sorted(self.sites)])
        self.rungs = cfg["rungs"]  # name -> {"plan": plan dict, "rep": validate report}
        self.users = {r: v["rep"]["users"] for r, v in self.rungs.items()}
        self.recipes = cfg["dense"]
        self.specs = {r: dr.recipe_spec(r, S.entry.model_type) for r in self.recipes}
        self.n_windows = cfg["n_windows"]
        self.n_tok = cfg["n_tokens"]
        self.roles = dr.roles_from_arch(S.model, [self.sites[k] for k in sorted(self.sites)], S.entry)
        self.want: set = set()
        self.done: dict = {}
        self.failed: dict = {}
        self.jobs: dict = {}
        self.taps: list = []
        self.counts: dict = {}
        self.dstats: DenseStats | None = None
        self.sumsq: SumsqStats | None = None
        self._mods: dict = {}
        self.t_layer: dict = {}
        self.need_xtx = any(f.gptq for r in self.recipes for f in self.specs[r].values())
        self.h4_layout = (BlobLayout(H4_BITS, H4_GROUP, self.lay.gu_shape, self.lay.dn_shape, "k4row")
                          if cfg["h4"] and self.lay is not None else None)
        self.h4_from_hess = bool(self.rungs) and cfg.get("h4_importance", "auto") in ("auto", "hess")
        if cfg.get("h4_importance") == "hess" and not self.rungs:
            self.h4_from_hess = True
        for d in ("layers", "codes", "h4", "bias"):
            (out / d).mkdir(parents=True, exist_ok=True)
        for r in self.rungs:
            (out / "codes" / r).mkdir(parents=True, exist_ok=True)
        for r in self.recipes:
            (out / r).mkdir(parents=True, exist_ok=True)

    # -- hooks --------------------------------------------------------------------------------------------------------

    def layer_begin(self, li, layer, site):
        self.t_layer[li] = time.time()
        if li not in self.want:
            return
        if self.recipes:
            mods = packed_modules(self.S.model, f"{self.S.layers_prefix}.{li}.", self.roles)
            self.dstats = DenseStats({n: m for n, (m, r) in mods.items() if r != "bf16"}, self.dev, self.need_xtx)
            self._mods = mods
        if site is None:
            return
        lay = site.layout
        self.counts[li] = torch.zeros(lay.E, dtype=torch.int64, device=self.dev)
        m = site.ref
        W_gu = m.gate_up_proj.transpose(1, 2) if lay.transposed else m.gate_up_proj
        gu_bias = m.gate_up_proj_bias if lay.bias else None
        job = None
        if self.rungs or (self.h4_layout is not None and self.h4_from_hess):
            cfg = self.cfg
            chunk = cfg["expert_chunk"]
            if chunk == "auto":
                free = (torch.cuda.mem_get_info(self.dev)[0] + torch.cuda.memory_reserved(self.dev)
                        - torch.cuda.memory_allocated(self.dev))
                chunk = tq.pick_expert_chunk(lay.E, lay.H, lay.I, int(0.85 * free), cfg["expert_group"])
            else:
                chunk = int(chunk) or lay.E
            entries = {r: tcq_plan.entry(v["plan"], li) for r, v in self.rungs.items()}
            job = MultiLayerJob(li, lay.E, lay.H, lay.I, entries, self.users, self.dev, self.n_windows,
                                h4_layout=self.h4_layout if self.h4_from_hess else None, flush=cfg["flush"],
                                weighted=cfg["weighted"], expert_chunk=chunk, glu=m._apply_gate,
                                lut_dir=cfg["luts"], group=cfg["expert_group"], votes=cfg["encode_votes"],
                                gu_bias=gu_bias)
            self.jobs[li] = job
        if self.h4_layout is not None and not self.h4_from_hess:
            self.sumsq = SumsqStats(site, self.dev)
        cnt = self.counts[li]

        def tap(x, idx, w):
            cnt.add_(torch.bincount(idx.reshape(-1), minlength=cnt.numel()))
            if job is not None and not job.dead:
                try:
                    job.add(x, idx, w, W_gu)
                except tq.LayerFailed as err:
                    self._fail(li, err)
        self.taps.append(MoETap(site, tap))

    def window(self, li, ids):
        if self.dstats is not None:
            self.dstats.new_call()

    def _fail(self, li, err):
        job = self.jobs.get(li)
        if job is not None:
            job.abort()
        self.failed[li] = f"{type(err).__name__}: {err}"[:2000]
        log(f"layer {li}: FAILED ({self.failed[li][:400]}); it will be converted again in another pass")

    def layer_end(self, li, layer, site):
        for t in self.taps:
            t.remove()
        self.taps = []
        if self.dstats is not None:
            self.dstats.remove()
        if li not in self.want:
            return
        if li in self.failed:
            self._cleanup(li)
            return
        rec = {"layer": li, "files": {}, "row": {}}
        try:
            if self.recipes:
                self._dense_layer(li, rec)
            if site is not None:
                self._experts_layer(li, site, rec)
        except tq.LayerFailed as err:
            self._fail(li, err)
            self._cleanup(li)
            return
        rec["row"]["layer_wall_s"] = round(time.time() - self.t_layer[li], 2)
        rec["row"].update(host_mem())
        rec["key_sha"] = self.cfg["key_sha"]
        write_json(self.out / "layers" / f"layer{li:03d}.done.json", rec)
        self.done[li] = rec
        self._cleanup(li)
        r = rec["row"]
        extra = ""
        if "rel_w_err" in r:
            extra = f"; rel w-err {r['rel_w_err']}"
        log(f"layer {li}: done in {r['layer_wall_s']}s (finish {r.get('finish_s', 0)}s, dense {r.get('dense_s', 0)}s)"
            f"{extra}; GPU peak {r.get('gpu_peak_alloc_gib')} GiB; host {r.get('rss_gib')} GiB")

    def _cleanup(self, li):
        self.jobs.pop(li, None)
        self.counts.pop(li, None)
        self.dstats = None
        if self.sumsq is not None:
            self.sumsq.remove()
            self.sumsq = None
        self._mods = {}

    # -- dense parts --------------------------------------------------------------------------------------------------

    def _pack(self, name, W, bias, role, entry, recipe, prep_cache, rows: dict, tensors: dict):
        fmt = dr.BF16 if role == "bf16" else dr.recipe_format(self.specs[recipe], role)
        if fmt.kind == "bf16":
            rows.setdefault("bf16", []).append(name)
            return
        info: dict = {}
        warn = None
        if fmt.gptq and (entry is None or entry.get("xtx") is None):  # never called in calibration: plain grid
            warn = "no input statistics for GPTQ: packed without GPTQ and importance"
            fmt = dr.Fmt("k4row", fmt.bits, fmt.group, gptq=False, imp=False)
        t0 = time.time()
        kp = dr.quantize_tensor(W, fmt, entry, device=self.dev, gptq_path=self.cfg["gptq_path"],
                                prep_cache=prep_cache, info=info)
        tq._sync(self.dev)
        dt = time.time() - t0
        for f in ("qweight", "sb", "d", "dmin"):
            tensors[f"{name}.{f}"] = tq.to_host_checked(getattr(kp, f))
        if bias is not None:
            tensors[f"{name}.bias"] = tq.to_host_checked(bias.detach())
        n, k = W.shape
        rows["tensors"][name] = {"role": role, "shape": [n, k], "bits": fmt.bits, "group": fmt.group,
                                 "gptq": fmt.gptq, "imp": fmt.imp, "bits_total": fmt.bits_total(n, k),
                                 "fallback_groups": kp.fallback, "bias": bias is not None, "seconds": round(dt, 3),
                                 "info": {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in info.items()},
                                 **({"warning": warn} if warn else {})}

    def _dense_layer(self, li, rec):
        t0 = time.time()
        by_owner: dict = {}  # one entry object per accumulator: modules sharing an input share the GPTQ factors
        entries: dict = {}
        for name, (_m, role) in self._mods.items():
            own = None if (self.dstats is None or role == "bf16") else self.dstats.owner.get(name)
            if own is not None and own not in by_owner:
                by_owner[own] = self.dstats.entry(own)
            entries[name] = None if own is None else by_owner[own]
        for recipe in self.recipes:
            tensors: dict = {}
            rows: dict = {"tensors": {}}
            prep_cache: dict = {}
            for name, (m, role) in sorted(self._mods.items()):
                ent = entries[name]
                bias = m._parameters.get("bias") if isinstance(m, nn.Linear) else None
                self._pack(name, m.weight.detach(), bias, role, ent, recipe, prep_cache, rows, tensors)
                if name in rows["tensors"]:
                    rows["tensors"][name]["stats_owner"] = None if self.dstats is None else self.dstats.owner.get(name)
                    if self.dstats is not None and name in self.dstats.unusable:
                        rows["tensors"][name]["stats_unusable"] = self.dstats.unusable[name]
            fname = f"layer{li:03d}.safetensors"
            path = self.out / recipe / fname
            save_safetensors(tensors, path, {"format": dr.FORMAT, "version": str(dr.VERSION), "recipe": recipe})
            rec["files"][f"{recipe}/{fname}"] = {"sha256": _sha_of(path)}
            rec.setdefault("dense", {})[recipe] = rows
            del tensors
        rec["row"]["dense_s"] = round(time.time() - t0, 2)

    # -- experts ------------------------------------------------------------------------------------------------------

    def _experts_layer(self, li, site, rec):
        lay = site.layout
        m = site.ref
        W_gu = m.gate_up_proj.transpose(1, 2) if lay.transposed else m.gate_up_proj
        W_dn = m.down_proj.transpose(1, 2) if lay.transposed else m.down_proj
        cnt = tq.to_host_checked(self.counts[li])
        top_k = site.top_k
        if int(cnt.sum()) != self.n_tok * top_k:
            raise tq.LayerFailed(f"layer {li}: {int(cnt.sum())} routed pairs != {self.n_tok} x {top_k}")
        row = rec["row"]
        job = self.jobs.get(li)
        t0 = time.time()
        chol0, enc0, fac0 = dict(tq.CHOL_STATS), dict(tq.ENCODE_STATS), dict(tq.FACTOR_STATS)
        torch.cuda.reset_peak_memory_stats(self.dev) if self.dev.type == "cuda" else None
        res = {"rungs": {}, "h4": None}
        if job is not None:
            if job.dead:
                raise tq.LayerFailed(f"layer {li}: job aborted")
            res = job.finish_multi(W_gu, W_dn)
            if not torch.equal(job.counts, cnt):
                raise tq.LayerFailed(f"layer {li}: the Hessian accumulator's routed counts differ from the tap's")
            row.update({"accumulate_s": round(job.timings.get("accumulate_s", 0.0), 2),
                        "factor_s": round(job.timings.get("factor_s", 0.0), 2),
                        "hessian_finish_s": round(job.timings.get("hessian_finish_s", 0.0), 2),
                        "h4_s": round(job.timings.get("h4_s", 0.0), 2),
                        "encode_s": {k[7:-2]: round(v, 2) for k, v in job.timings.items()
                                     if k.startswith("encode_") and k.endswith("_s")},
                        "replay": dict(job.stats), "replay_errors": job.replay_errors[:20],
                        "expert_chunk": job.chunks[0][1] - job.chunks[0][0], "chunks": len(job.chunks)})
        if self.h4_layout is not None and res["h4"] is None:  # H4 from directly accumulated sums of squares
            t1 = time.time()
            st = self.sumsq.result()
            if not torch.equal(st["gate_up"]["count"], cnt):
                raise tq.LayerFailed(f"layer {li}: ExpertStats counts differ from the tap's")
            imp = (kquant.importance_from_stats(tq.to_device_verified(st["gate_up"]["sumsq"], self.dev).to(torch.float32),
                                                st["gate_up"]["count"]),
                   kquant.importance_from_stats(tq.to_device_verified(st["down"]["sumsq"], self.dev).to(torch.float32),
                                                st["down"]["count"]))
            parts = []
            for a in range(0, lay.E, H4_CHUNK):
                b = min(lay.E, a + H4_CHUNK)
                blob = self.h4_layout.quantize(W_gu[a:b], W_dn[a:b], (imp[0][a:b], imp[1][a:b]), device=self.dev)
                parts.append(tq.to_host_checked(blob))
            res["h4"] = torch.cat(parts)
            row["h4_s"] = round(time.time() - t1, 2)
        row["finish_s"] = round(time.time() - t0, 2)
        for r, recs in res["rungs"].items():
            files = {}
            for mname in tq.MATRICES:
                p = tq.save_record(recs[mname], str(self.out / "codes" / r))
                files[mname] = os.path.basename(p)
                rec["files"][f"codes/{r}/{os.path.basename(p)}"] = {"sha256": _sha_of(Path(p))}
            row.setdefault("rel_w_err", {})[r] = {mn: round(tq.relative_error(recs[mn]), 5) for mn in tq.MATRICES}
            row.setdefault("rel_w_err_expert_max", {})[r] = {
                mn: round(float(tq.expert_relative_errors(recs[mn]).max()), 5) for mn in tq.MATRICES}
            row.setdefault("bits_per_expert", {})[r] = {mn: recs[mn]["bits_per_expert"] for mn in tq.MATRICES}
        if res["h4"] is not None:
            h4 = res["h4"]
            if tuple(h4.shape) != (lay.E, self.h4_layout.nbytes):
                raise RuntimeError(f"layer {li}: H4 blobs {tuple(h4.shape)}, expected {(lay.E, self.h4_layout.nbytes)}")
            p = self.out / "h4" / f"layer{li:03d}.cold.bin"
            write_bytes(p, h4)
            rec["files"][f"h4/{p.name}"] = {"sha256": _sha_of(p)}
            row["h4_fallback_groups"] = (job.timings.get("h4_info", {}).get("fallback", 0) if job is not None else None)
        if lay.bias:
            from .convert_any import BIAS_KEYS

            tensors = {BIAS_KEYS[0]: tq.to_host_checked(m.gate_up_proj_bias.detach()).contiguous(),
                       BIAS_KEYS[1]: tq.to_host_checked(m.down_proj_bias.detach()).contiguous()}
            p = self.out / "bias" / f"layer{li:03d}.bias.safetensors"
            save_safetensors(tensors, p, {"format": "pt"})
            rec["files"][f"bias/{p.name}"] = {"sha256": _sha_of(p)}
        rec["counts"] = cnt.tolist()
        row.update({"count_min": int(cnt.min()), "count_median": int(cnt.sort().values[len(cnt) // 2]),
                    "cholesky_matrices_fixed": tq.CHOL_STATS["matrices_fixed"] - chol0["matrices_fixed"],
                    "factor_votes": {k: tq.FACTOR_STATS[k] - fac0.get(k, 0) for k in tq.FACTOR_STATS},
                    "encode_votes": {k: tq.ENCODE_STATS[k] - enc0.get(k, 0) for k in tq.ENCODE_STATS}})
        if self.dev.type == "cuda":
            row["gpu_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(self.dev) / 2**30, 3)
            row["gpu_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(self.dev) / 2**30, 3)

    # -- model level --------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def model_level(self, hidden: list) -> dict:
        """The embedding and the head of every dense recipe: lm_head statistics from the base model's last hidden
        states (the head's input) of every calibration window, in file order; the embedding by plain MSE."""
        t0 = time.time()
        rec = {"layer": "model", "files": {}, "row": {}}
        model = self.S.model
        head = self.S.head
        emb = model.get_input_embeddings()
        emb_name = next(n for n, mm in model.named_modules() if mm is emb)
        head_name = next(n for n, mm in model.named_modules() if mm is head)
        tied = head.weight is emb.weight
        need = self.need_xtx
        acc = _Acc(int(head.weight.shape[1]), self.dev, need)
        off = self.S.offload
        for h in hidden:
            acc.add(off.h2d(h) if off is not None and h.device.type == "cpu" else h)
        lm_entry = acc.entry()
        for recipe in self.recipes:
            tensors: dict = {}
            rows: dict = {"tensors": {}}
            self._pack(emb_name, emb.weight.detach(), None, "embed", None, recipe, {}, rows, tensors)
            self._pack(head_name, head.weight.detach(), getattr(head, "bias", None), "lm_head", lm_entry, recipe, {},
                       rows, tensors)
            if tied:
                rows["tensors"][head_name]["tied_to"] = emb_name
            p = self.out / recipe / "model.safetensors"
            save_safetensors(tensors, p, {"format": dr.FORMAT, "version": str(dr.VERSION), "recipe": recipe})
            rec["files"][f"{recipe}/model.safetensors"] = {"sha256": _sha_of(p)}
            rec.setdefault("dense", {})[recipe] = rows
        rec["row"]["seconds"] = round(time.time() - t0, 2)
        rec["row"]["lm_head_tokens"] = acc.count
        rec["key_sha"] = self.cfg["key_sha"]
        rec["tie_word_embeddings"] = tied
        write_json(self.out / "layers" / "model.done.json", rec)
        self.done["model"] = rec
        return rec


def run_key(model_dir, tokens_sha, n_windows, rungs, cfg) -> dict:
    luts = sorted({str(x.get("sha256")) for v in rungs.values() for x in v["rep"]["luts"]})
    return {"format": FORMAT, "model": os.path.abspath(model_dir), "tokens_sha256": tokens_sha, "windows": n_windows,
            "rungs": {r: v["plan"] for r, v in sorted(rungs.items())}, "h4": cfg["h4"],
            "h4_importance": cfg.get("h4_importance", "auto"), "dense": list(cfg["dense"]), "flush": cfg["flush"],
            "weighted": cfg["weighted"], "expert_group": cfg["expert_group"], "encode_votes": cfg["encode_votes"],
            "factor_votes": cfg["factor_votes"], "gptq_path": cfg["gptq_path"], "micro": 1, "code": code_shas(),
            "lut_sha256": luts, "transformers": hfcompat.TF_VERSION, "torch": torch.__version__}


def layer_done(out: Path, li, key_sha: str):
    """The layer's done.json when it parses and every file it names exists with its sha256; else None."""
    name = "model.done.json" if li == "model" else f"layer{li:03d}.done.json"
    d = read_json_or_none(out / "layers" / name)
    if d is None or d.get("key_sha") != key_sha or not isinstance(d.get("files"), dict):
        return None
    try:
        for rel, f in d["files"].items():
            p = out / rel
            if not p.exists() or _sha_of(p) != f["sha256"]:
                return None
    except (KeyError, TypeError):
        return None
    return d


def load_rungs(specs: list[str], dims: dict, n_layers: int, lut_dir) -> dict:
    """--rung NAME=PLAN.json | NAME=depth16 -> {name: {"plan": normalized plan, "rep": validate report}}."""
    from . import tcq_rate

    out = {}
    for s in specs:
        name, _, src = s.partition("=")
        if not name or not src:
            raise SystemExit(f"--rung {s!r}: NAME=PLAN.json or NAME=depth16")
        if src == "depth16":
            if name not in tcq_rate.LADDER:
                raise SystemExit(f"--rung {name}=depth16: {name} is not a ladder rung {list(tcq_rate.LADDER)}")
            plan = tcq_rate.make_plan(dims, tcq_rate.LADDER[name], signal="depth16", n_layers=n_layers,
                                      lut_dir=lut_dir, rung=name)
        else:
            plan = src
        plan = tcq_plan.load_plan(plan)
        rep = tcq_plan.validate(plan, dims, lut_dir)
        out[name] = {"plan": plan, "rep": rep}
    return out


def convert(model_dir: str, tokens: str, out_dir: str, *, rungs: list[str] = (), h4: bool = True,
            dense: tuple = ("dense-hi", "dense-c"), device: str = "cuda:0", expert_group: int = tq.EXPERT_GROUP,
            expert_chunk="0", encode_votes: int = tq.ENCODE_VOTES, factor_votes: int = tq.FACTOR_VOTES,
            flush: int = 32, weighted: bool = False, luts: str | None = None, gptq_path: str = "cpu",
            max_layers: int = 0, layer_retries: int = 2, resume: bool = True, stores: bool = True,
            verify: bool = True, h4_importance: str = "auto", threads: int = 32, depth: int = 2,
            corpus_name: str | None = None, offload_hidden: bool = False) -> dict:
    t_start = time.time()
    tq.FACTOR_VOTES = factor_votes
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = torch.load(tokens)
    rows = list(rows) if not isinstance(rows, list) else rows
    if isinstance(rows[0], dict):  # a t2-calib-windows/1 file is not flat
        raise SystemExit(f"{tokens}: pass a flat list of token windows (tools/calib_corpus.py flat)")
    rows = [r.reshape(-1).long() for r in rows]
    n_tok = sum(int(r.numel()) for r in rows)
    tokens_sha = file_sha(tokens)
    for r in dense:
        if r not in RECIPES:
            raise SystemExit(f"unknown dense recipe {r!r}: {list(RECIPES)}")
    log(f"{len(rows)} windows, {n_tok} tokens from {tokens}")
    S = LockstepStreamer(model_dir, device, verify=verify, threads=threads, depth=depth, log=log,
                         offload_hidden=offload_hidden)
    if offload_hidden:
        log("hidden states in host memory (--offload-hidden, design-quality amendment 10)")
    sites = [S.sites[k] for k in sorted(S.sites)]
    if not sites:
        raise SystemExit(f"{model_dir}: no MoE site")
    from .convert_any import uniform_experts

    lay = uniform_experts(sites)
    n_layers_model = len(S.layers)
    dims = {"moe_layers": [s.layer_idx for s in sites], "n_experts": lay.E, "hidden": lay.H, "inter": lay.I}
    rung_d = load_rungs(list(rungs), dims, n_layers_model, luts)
    for r, v in rung_d.items():
        rep = v["rep"]
        log(f"rung {r}: {rep['bpw']:.6f} bpw (gate {rep['by_matrix']['gate']:.4f}, up {rep['by_matrix']['up']:.4f}, "
            f"down {rep['by_matrix']['down']:.4f}), damp {v['plan']['damp']}, {len(rep['luts'])} LUTs"
            + (f"; WARNINGS {rep['warnings']}" if rep["warnings"] else ""))
    if expert_group == 0 and (expert_chunk == "auto" or 0 < int(expert_chunk) < lay.E):
        raise SystemExit("--expert-group 0 (T4's whole-stack batches) cannot be combined with expert chunking")
    if not lay.gated:
        raise SystemExit("non-gated experts: no _apply_gate for the down-input recompute")
    n_layers = max_layers or n_layers_model
    cfg = {"rungs": rung_d, "h4": bool(h4), "dense": list(dense), "n_windows": len(rows), "n_tokens": n_tok,
           "flush": flush, "weighted": weighted, "expert_group": expert_group, "expert_chunk": expert_chunk,
           "encode_votes": encode_votes, "factor_votes": factor_votes, "luts": luts, "gptq_path": gptq_path,
           "h4_importance": h4_importance}
    key = run_key(model_dir, tokens_sha, len(rows), rung_d, cfg)
    key_sha = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    cfg["key_sha"] = key_sha
    run_file = out / "run.json"
    old = read_json_or_none(run_file) if run_file.exists() else None
    if resume and run_file.exists():
        if old is None:
            raise SystemExit(f"{run_file} is unreadable; use another --out, or --no-resume")
        if old.get("key_sha") != key_sha:
            diff = sorted(k for k in set(key) | set(old.get("key", {})) if key.get(k) != old.get("key", {}).get(k))
            raise SystemExit(f"{out} holds a conversion with other settings ({diff}); use another --out, or "
                             "--no-resume to convert every layer again into it")
    write_json(run_file, {"key_sha": key_sha, "key": key, "argv": sys.argv, "started": time.time(),
                          "git": git_rev()})
    conv = Conversion(S, out, cfg)
    for r, v in rung_d.items():
        write_json(out / "codes" / r / "plan.json", v["plan"])
    all_layers = list(range(n_layers))
    if resume:
        for li in all_layers + ["model"]:
            d = layer_done(out, li, key_sha)
            if d is not None:
                conv.done[li] = d
            elif (out / "layers" / ("model.done.json" if li == "model" else f"layer{li:03d}.done.json")).exists():
                log(f"layer {li}: done.json corrupt or not matching its files / key; converting it again")
        if conv.done:
            log(f"resume: {sorted(k for k in conv.done if k != 'model')} done"
                + (", model level done" if "model" in conv.done else ""))
    passes = []
    todo = [li for li in all_layers if li not in conv.done]
    need_model = "model" not in conv.done and bool(dense) and n_layers == n_layers_model
    if dense and n_layers != n_layers_model:
        log("--max-layers: the embedding / head of the dense parts are not converted (no final hidden states)")
    while (todo or need_model) and len(passes) <= layer_retries:
        conv.want = set(todo)
        conv.failed = {}
        full = need_model
        n_pass = n_layers if full else max(todo) + 1
        if passes:
            log(f"pass {len(passes)}: converting layers {tcq_plan.format_layers(todo)} again (streaming layers "
                f"0-{n_pass - 1})")
        info = S.run_ordered(rows, micro=1, n_layers=n_pass, hooks=conv, keep_hidden=full)
        if full and not conv.failed:
            conv.model_level(info["hidden"])
            need_model = False
        info.pop("hidden", None)
        passes.append({"layers": tcq_plan.format_layers(todo) if todo else "", "failed": dict(conv.failed),
                       "seconds": round(info["seconds"], 1), "read_gib": info["read_gib"], "read_s": info["read_s"],
                       "groups": info["groups"], "verify": info["verify"],
                       **({"offload": info["offload"]} if "offload" in info else {})})
        todo = sorted(conv.failed)
    tq.PINNED.release()
    missing = [li for li in all_layers if li not in conv.done]
    summary = {"out": str(out), "key_sha": key_sha, "passes": passes, "missing": missing,
               "model_level_done": "model" in conv.done, "seconds": round(time.time() - t_start, 1),
               "replay_checks": dict(tq.REPLAY_STATS), "cholesky_voted": dict(tq.CHOL_STATS),
               "factor_votes": dict(tq.FACTOR_STATS), "encode_votes": dict(tq.ENCODE_STATS), **host_mem()}
    if missing or (dense and n_layers == n_layers_model and "model" not in conv.done):
        write_json(out / "summary.json", summary)
        log(f"MISSING layers {tcq_plan.format_layers(missing) if missing else '-'} after {len(passes)} passes")
        sys.exit(EXIT_LAYERS_MISSING)
    # ---- finalize: counts, dense meta, stores
    moe = [s.layer_idx for s in sites if s.layer_idx < n_layers]
    counts = torch.tensor([conv.done[li]["counts"] for li in moe], dtype=torch.float32)
    cmeta = {"format": "moe-tiering/counts/1", "corpus": corpus_name or os.path.abspath(tokens),
             "tokens_sha256": tokens_sha, "tokens": str(n_tok), "windows": str(len(rows)),
             "layers": json.dumps(moe), "note": "float32 [n_sites, E]: tokens routed to each expert"}
    save_safetensors({"counts": counts.contiguous(), "layers": torch.tensor(moe, dtype=torch.int32)},
                     out / "counts.safetensors", cmeta)
    for recipe in dense:
        _dense_meta(conv, recipe, model_dir, out, n_layers, tokens, n_tok, len(rows), corpus_name)
    meta = {"format": FORMAT, "model": os.path.abspath(model_dir), "arch": S.entry.model_type,
            "transformers": hfcompat.TF_VERSION, "torch": torch.__version__, "git": git_rev(), "code_sha256": key["code"],
            "tokens": os.path.abspath(tokens), "tokens_sha256": tokens_sha, "windows": len(rows), "n_tokens": n_tok,
            "corpus": corpus_name, "rungs": {r: {"bpw": v["rep"]["bpw"], "bpw_exact": v["rep"]["bpw_exact"],
                                                 "by_matrix": v["rep"]["by_matrix"]} for r, v in rung_d.items()},
            "h4": bool(h4), "h4_importance": "hessian diagonal (hot_rot rot=0 imp=hess)" if conv.h4_from_hess
            else "routed-input sum of squares (ramog.stream.ExpertStats)", "dense": list(dense),
            "settings": {k: cfg[k] for k in ("flush", "weighted", "expert_group", "expert_chunk", "encode_votes",
                                             "factor_votes", "gptq_path")},
            "encoder": {"ramog/tcq_encode.py": tq.source_sha(), "sha256": file_sha(ROOT / "ramog/tcq_encode.py"),
                        "expert_group": expert_group},
            "moe_layers": moe, "layers": n_layers, "summary": summary}
    write_json(out / "meta.json", meta)
    if stores and n_layers == n_layers_model and (h4 or rung_d):
        sys.path.insert(0, str(ROOT))
        from ramog import build_store_v2 as bsv

        dense_bf16 = bsv.write_dense_bf16(S.keymap, out / "common", log=log)
        dense_dirs = {DENSE_DIRS[r]: out / r for r in dense}
        calib = {"corpus": corpus_name or os.path.abspath(tokens), "tokens": n_tok, "windows": len(rows),
                 "tokens_sha256": tokens_sha, "hessians": "per expert, per layer in the converter (never stored)",
                 "converter": CONVERTER, "run_key_sha": key_sha}
        enc = meta["encoder"]
        bias_dir = out / "bias" if lay.bias else None
        if h4:
            bsv.build_store_v2(None, dense_dirs, dense_bf16, model_dir, out / "counts.safetensors",
                               out / "stores" / "H4", "H4", None, cold_dir=out / "h4", bias_dir=bias_dir,
                               encoder={"ramog/kquant.py": file_sha(ROOT / "ramog/kquant.py", 16),
                                        "importance": meta["h4_importance"], **enc},
                               calibration={**calib, "hessians": "diagonals only" if conv.h4_from_hess else "none"},
                               keymap=S.keymap, log=log)
        for r, v in rung_d.items():
            bsv.build_store_v2(out / "codes" / r, dense_dirs, dense_bf16, model_dir, out / "counts.safetensors",
                               out / "stores" / r, r, v["plan"], bias_dir=bias_dir, encoder=enc,
                               calibration={**calib, "damp": v["plan"]["damp"]}, keymap=S.keymap, log=log)
    log(f"done in {time.time() - t_start:.0f}s ({len(passes)} pass(es)); verify {S.verify_info}; host {host_mem()}")
    return meta


def _dense_meta(conv: Conversion, recipe, model_dir, out: Path, n_layers, tokens, n_tok, n_win, corpus):
    tensors, bf16 = {}, []
    timing: dict = {}
    for li in list(range(n_layers)) + ["model"]:
        d = conv.done.get(li)
        if d is None:
            continue
        rows = (d.get("dense") or {}).get(recipe)
        if rows is None:
            continue
        fname = "model.safetensors" if li == "model" else f"layer{li:03d}.safetensors"
        for name, e in rows["tensors"].items():
            tensors[name] = {**e, "file": fname}
            timing[e["role"]] = timing.get(e["role"], 0.0) + e["seconds"]
        bf16 += rows.get("bf16", [])
    spec = conv.specs[recipe]
    bits = sum(e["bits_total"] for e in tensors.values())
    weights = sum(e["shape"][0] * e["shape"][1] for e in tensors.values())
    tied = (conv.done.get("model") or {}).get("tie_word_embeddings")
    meta = {"format": dr.FORMAT, "version": dr.VERSION, "recipe": recipe,
            "recipe_spec": {k: asdict(v) for k, v in spec.items()}, "source": os.path.abspath(model_dir),
            "names": "the transformers text model's module names (ramog.arch.build_meta)",
            "stats": {"kind": "accumulated in the conversion pass (ramog.convert_tcq, tools/dense_stats.py's Acc)",
                      "corpus": corpus or os.path.abspath(tokens), "tokens": n_tok, "windows": n_win},
            "gptq_path": conv.cfg["gptq_path"], "device": str(conv.dev), "layers": list(range(n_layers)),
            "tensors": tensors, "bf16": sorted(set(bf16)), "skipped_expert_tensors": 0, "tie_word_embeddings": tied,
            "bits": bits, "weights": weights, "gib": bits / 8 / 2**30, "converter": CONVERTER,
            "engine_sha256": {f: file_sha(ROOT / "engine" / f, 16) for f in ("gptq.py", "kquant.py", "dense_recipe.py")},
            "timing": {"quant_s_per_role": {k: round(v, 2) for k, v in sorted(timing.items())}}}
    write_json(out / recipe / "meta.json", meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True, help="torch file: flat list of 1-D token windows (any lengths)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rung", action="append", default=[], help="NAME=PLAN.json or NAME=depth16 (repeatable)")
    ap.add_argument("--h4", action="store_true", help="also the H4 variant (k4row 4-bit g16, Hessian-diagonal "
                    "importance)")
    ap.add_argument("--h4-importance", default="auto", choices=("auto", "hess", "sumsq"),
                    help="auto: Hessian diagonals when a rung is converted, else directly accumulated sums of squares")
    ap.add_argument("--dense", default="dense-hi,dense-c", help="dense recipes (comma list; '' for none)")
    ap.add_argument("--corpus-name", default=None, help="name recorded for the calibration corpus (e.g. t2v1)")
    ap.add_argument("--expert-group", type=int, default=tq.EXPERT_GROUP)
    ap.add_argument("--expert-chunk", default="0")
    ap.add_argument("--encode-votes", type=int, default=tq.ENCODE_VOTES, choices=(1, 2))
    ap.add_argument("--factor-votes", type=int, default=tq.FACTOR_VOTES, choices=(1, 2))
    ap.add_argument("--flush", type=int, default=32)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--luts", default=None)
    ap.add_argument("--gptq-path", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--max-layers", type=int, default=0, help="debug: only the first N decoder layers")
    ap.add_argument("--layer-retries", type=int, default=2)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-stores", action="store_true", help="do not build the store v2 directories")
    ap.add_argument("--counts-only", action="store_true", help="only the routed counts (no rung, H4 or dense part)")
    ap.add_argument("--no-verify", action="store_true", help="skip the fingerprint check of every bind")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--offload-hidden", action="store_true", help="keep the calibration hidden states in host memory, "
                    "one window on the device at a time (design-quality amendment 10; same output bits)")
    a = ap.parse_args(argv)
    if a.counts_only:
        a.rung, a.h4, a.dense = [], False, ""
    elif not a.rung and not a.h4 and not a.dense:
        ap.error("nothing to convert (--rung, --h4, --dense; --counts-only for the counts alone)")
    convert(a.model, a.tokens, a.out, rungs=a.rung, h4=a.h4, dense=tuple(x for x in a.dense.split(",") if x),
            device=a.device, expert_group=a.expert_group, expert_chunk=a.expert_chunk, encode_votes=a.encode_votes,
            factor_votes=a.factor_votes, flush=a.flush, weighted=a.weighted, luts=a.luts, gptq_path=a.gptq_path,
            max_layers=a.max_layers, layer_retries=a.layer_retries, resume=not a.no_resume, stores=not a.no_stores,
            verify=not a.no_verify, h4_importance=a.h4_importance, threads=a.threads, depth=a.depth,
            corpus_name=a.corpus_name, offload_hidden=a.offload_hidden)


if __name__ == "__main__":
    main()
