"""Generic layer-by-layer streamer for reference distributions and calibration statistics (design-core B7; WP6).

The model is transformers' own text model (ramog.arch.build_meta) on the meta device. Its base forward
(`model.model(input_ids=[B, S])`) runs over a whole batch of windows, unchanged, so every architecture's glue between
the decoder layers stays transformers code: masks and rotary tables, GLM-5.3's IndexShare carry (`prev_topk_indices`
returned by one layer and passed to the next), GLM-5.3-Flash's mHC expand / collapse and hc_head, Gemma-4's per-layer-
type masks and embed_scale, Qwen3.5's mrope positions. Only these pieces are ours:

- Resident stage 0: every tensor outside the decoder layers (embedding, final norm, lm_head, hc_head, ...) is loaded
  once on the device, tied heads re-tied; the non-persistent buffers (rotary inv_freq, Gemma-4's embed_scale, ...) are
  rebuilt by from_pretrained's own sequence (design-core A6): allocated with empty_like (NaN-filled, so a buffer the
  init misses is caught), every other tensor flagged `_is_hf_initialized`, then transformers' initialize_weights().
- dtypes and routing are the engine's (they come with the model and the key map, not from this file): KeyMap's
  target dtype is hfcompat.DtypePlan's, i.e. from_pretrained(dtype=bf16)'s rule (the strict-fp32 plan, else the meta
  dtype) plus the architecture's fp32_keys (ramog.arch: MiniMax-M2's F32 router weight and correction bias,
  GLM-5.3-Flash's F32 mHC base / scale, kept F32 as the release stores them and as transformers' FP8 load holds
  them); and build_meta runs the routers of ArchEntry.sorted_topk architectures with torch.topk as a stable sort
  (ramog.arch.topk_stable: the selected set and the slot order the experts sum in are a function of the scores,
  ties by expert index; torch.topk over GLM-5.3-Flash's 288 experts is not run-to-run deterministic on MI250).
  ramog.loader and the converter share both, so references and engine agree.
- StreamedLayer: each decoder layer's forward is replaced (on the instance; the module tree, hooks and attributes stay
  as they are) by: (1) bind the layer's parameters and persistent buffers, decoded through the converter's own
  ramog.ckpt.KeyMap (ramog.dequant, the dtype rule above), from raw bytes read ahead into
  pinned host buffers by a background reader (`depth` layers ahead, parallel positional reads); (2) run the layer's
  own forward on micro-batches of `micro` windows, slicing every tensor argument (also inside tuples) whose dim 0 is
  the batch size B; (3) concatenate the outputs along dim 0, tuple outputs per element (None kept); (4) unbind
  (the original meta tensors are put back, so the device memory is freed). The default micro=1 runs every layer on
  one window at a time, exactly as calib_stream.py / make_refs.py do.
- Host tensors (ramog.ckpt host targets, ArchEntry.host_keys: DeepSeek-V4.1's engram n-gram tables) are bound once,
  at setup, as zero-copy memory maps of the checkpoint in their release dtype (KeyMap.host_view): never read ahead,
  decoded, put on the device or unbound; the modules that own them gather the rows they need.
- The base model's other children that see the hidden states after the layers (final norm, GLM-5.3-Flash's hc_head,
  Gemma-4's per-layer projection when present) get the same micro-batching (no weights to bind): per window, as a
  full forward of one window computes them. Embeddings (a gather) and rotary modules (they take position ids) are
  called once.
- Mapping arguments (Gemma-4's shared_kv_states: the last layer of each attention type stores its K/V there for
  later KV-sharing layers) get one copy per micro-batch span that lasts the whole pass, so every span reads what its
  own windows stored. A cache argument is refused when a call is split (it would mix windows).

Statistics (ExpertStats) are collected on the passes named in collect_stats, over every routed (token, expert) pair:
gate_up = the hidden state entering the experts module (its forward's first argument), down = the output of the
module's own `_apply_gate` (the default chunk(2) gate, gpt-oss's interleaved clamp/alpha gate, GLM-5.3-Flash's clamp
gate, ...), which needs experts_implementation "grouped_mm" (rows sorted by expert id); count = tokens routed to the
expert. Accumulated in float64 on the device, saved as float32, sizes taken from the tensors; the arithmetic is
calib_stream.ExpertStats's, so Qwen3-30B's stats are bit-identical to stats-30b.

Two passes over the weights (design-core B7): calibration windows (1024 tokens) and eval windows (512) are separate
base forwards. The read-ahead runs across the pass boundary.

    s = Streamer(model_dir, "cuda:0")
    res = s.run({"calib": calib_rows, "eval": eval_rows}, collect_stats=("calib",), micro=1)
    logits = s.logits(res["eval"].hidden[i:i + 1])   # lm_head (+ Gemma's final softcap), per window
"""

from __future__ import annotations

import copy
import inspect
import math
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import arch, ckpt, hfcompat

ALIGN = 256  # byte alignment of every raw tensor inside a read-ahead buffer (any dtype view is aligned)
HAVE_PREADV = hasattr(os, "preadv")


# ---------------------------------------------------------------------------------------------------------------------
# raw read-ahead


def _align(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class LayerReader:
    """Background reader: raw checkpoint tensors of each plan, in plan order, into `depth` reused host buffers (pinned
    when `pin`). plans = [(key, [checkpoint keys])]; get() returns (key, {checkpoint key: tensor view of the buffer},
    buffer, read seconds, bytes); release(buffer) hands the buffer back. Reads are positional (os.preadv where it
    exists, else one unbuffered file object per thread), split in `chunk_mb` pieces over `threads` threads; nothing is
    decoded here (KeyMap does that on the device)."""

    def __init__(self, index: ckpt.CheckpointIndex, plans: Sequence[tuple[object, Sequence[str]]], depth: int = 2,
                 threads: int = 32, chunk_mb: int = 16, drop_cache: bool = False, pin: bool = True):
        self.dir = index.dir
        self.layouts = []
        for key, ckeys in plans:
            entries, pos = [], 0
            for ck in dict.fromkeys(ckeys):  # unique, in order
                ti = index[ck]
                ti.torch_dtype  # noqa: B018 - raises on an unsupported dtype
                off = _align(pos)
                entries.append((ck, ti, off))
                pos = off + ti.nbytes
            self.layouts.append((key, entries, _align(pos)))
        self.size = max((n for _k, _e, n in self.layouts), default=0)
        self.chunk = max(1, chunk_mb) << 20
        self.drop = drop_cache
        self.free: queue.Queue = queue.Queue()
        for _ in range(max(1, min(depth, len(self.layouts)))):
            self.free.put(torch.empty(max(self.size, 1), dtype=torch.uint8, pin_memory=pin))
        self.ready: queue.Queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max(1, threads))
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._stop = False
        self.read_bytes = 0
        self.read_seconds = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _fd(self, path: str) -> int:
        fd = self._fds.get(path)
        if fd is None:
            with self._lock:
                fd = self._fds.get(path)
                if fd is None:
                    fd = self._fds[path] = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        return fd

    def _read(self, mv: memoryview, path: str, off: int, dst: int, n: int) -> int:
        got = 0
        if HAVE_PREADV:
            fd = self._fd(path)
            while got < n:
                r = os.preadv(fd, [mv[dst + got: dst + n]], off + got)
                if r <= 0:
                    raise OSError(f"{path}: short read at {off + got}")
                got += r
            return n
        files = getattr(self._tls, "files", None)
        if files is None:
            files = self._tls.files = {}
        f = files.get(path)
        if f is None:
            f = files[path] = open(path, "rb", buffering=0)
        f.seek(off)
        while got < n:
            r = f.readinto(mv[dst + got: dst + n])
            if not r:
                raise OSError(f"{path}: short read at {off + got}")
            got += r
        return n

    def _run(self):
        try:
            for key, entries, nbytes in self.layouts:
                buf = self.free.get()
                if buf is None or self._stop:
                    return
                t0 = time.time()
                mv = memoryview(buf.numpy()) if buf.numel() else None
                pieces = []
                for _ck, ti, off in entries:
                    path = os.path.join(self.dir, ti.file)
                    for s in range(0, ti.nbytes, self.chunk):
                        pieces.append((path, ti.start + s, off + s, min(self.chunk, ti.nbytes - s)))
                total = sum(self.pool.map(lambda j: self._read(mv, *j), pieces))
                if self.drop and hasattr(os, "posix_fadvise"):
                    for _ck, ti, _off in entries:
                        os.posix_fadvise(self._fd(os.path.join(self.dir, ti.file)), ti.start, ti.nbytes,
                                         os.POSIX_FADV_DONTNEED)
                raw = {}
                for ck, ti, off in entries:
                    raw[ck] = buf[off: off + ti.nbytes].view(ti.torch_dtype).reshape(ti.shape)
                dt = time.time() - t0
                self.read_bytes += total
                self.read_seconds += dt
                self.ready.put((key, raw, buf, dt, total))
        except BaseException as e:  # noqa: BLE001 - surfaced in get()
            self.ready.put(e)

    def get(self):
        item = self.ready.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def release(self, buf: torch.Tensor) -> None:
        self.free.put(buf)

    def close(self) -> None:
        self._stop = True
        self.free.put(None)  # wakes the reader if it waits for a buffer; it stops after an in-flight layer
        self.thread.join(timeout=300)
        self.pool.shutdown(wait=True)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()


class RawIndex:
    """Stands in for KeyMap.index: serves read_many() from raw tensors read ahead by LayerReader, copied to the
    requested device (always a copy: ramog.dequant.plain returns its input when the dtype already matches, and the
    read-ahead buffer is reused for the next layer)."""

    def __init__(self, raw: Mapping[str, torch.Tensor]):
        self.raw = raw

    def read_many(self, requests, device="cpu", pin=None) -> list[torch.Tensor]:
        out = []
        for key, rows in requests:
            try:
                t = self.raw[key]
            except KeyError:
                raise KeyError(f"{key} was not read ahead for this layer") from None
            if rows is not None:
                t = t[rows[0]:rows[1]]
            out.append(t.to(device=device, copy=True))
        return out


# ---------------------------------------------------------------------------------------------------------------------
# micro-batching


def _is_cache(x) -> bool:
    return x is not None and type(x).__name__.endswith("Cache") and hasattr(x, "get_seq_length")


def slice_batch(x, s: int, e: int, B: int):
    """x with every tensor whose dim 0 is B cut to [s, e); tuples / lists element-wise; anything else unchanged."""
    if isinstance(x, torch.Tensor):
        return x[s:e] if x.dim() >= 1 and x.shape[0] == B else x
    if isinstance(x, tuple) and not hasattr(x, "_fields"):
        return tuple(slice_batch(v, s, e, B) for v in x)
    if isinstance(x, list):
        return [slice_batch(v, s, e, B) for v in x]
    return x


def concat_batch(outs: list, B: int):
    """Inverse of slice_batch over the outputs of consecutive micro-batches: tensors concatenated along dim 0 (their
    dim 0 must add up to B), tuples element-wise (an element None in every micro-batch stays None)."""
    first = outs[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(o, torch.Tensor) for o in outs):
            raise TypeError("micro-batch outputs mix tensors and other values")
        if sum(o.shape[0] for o in outs) != B:
            raise RuntimeError(f"micro-batch outputs have dim 0 {[o.shape[0] for o in outs]}, expected a total of {B}")
        return torch.cat(outs, dim=0)
    if isinstance(first, tuple):
        if not all(isinstance(o, tuple) and len(o) == len(first) for o in outs):
            raise TypeError("micro-batch outputs are tuples of different lengths")
        parts = [concat_batch([o[i] for o in outs], B) for i in range(len(first))]
        return type(first)(*parts) if hasattr(first, "_fields") else tuple(parts)
    if first is None:
        if any(o is not None for o in outs):
            raise TypeError("micro-batch outputs mix None and values")
        return None
    if all(o == first for o in outs):
        return first
    raise TypeError(f"cannot concatenate micro-batch outputs of type {type(first).__name__}")


def describe_args(args, kwargs, B: int) -> dict:
    """{argument: "sliced [shape]" | "whole [shape]" | type name} of a call: what micro-batching does with each."""
    def one(v):
        if isinstance(v, torch.Tensor):
            return f"{'sliced' if v.dim() >= 1 and v.shape[0] == B else 'whole'} {list(v.shape)}"
        if isinstance(v, (tuple, list)):
            return [one(x) for x in v]
        return type(v).__name__
    out = {f"arg{i}": one(v) for i, v in enumerate(args)}
    out.update({k: one(v) for k, v in kwargs.items()})
    return out


def _span_mapping(m: Mapping, spans: list[tuple[int, int]], B: int, maps: dict | None, si: int):
    """The micro-batch span's own copy of a mapping argument (e.g. Gemma-4's shared_kv_states, which one layer writes
    and later KV-sharing layers read): made once per pass from the original (batch tensors sliced), then handed to
    every call of that span, so what a layer stores for its windows is what later layers read for the same windows,
    exactly as in a forward of those windows alone."""
    if maps is None:
        raise RuntimeError("a mapping argument needs the pass-level micro-batch state")
    rec = maps.get(id(m))
    if rec is None or rec[0] is not m:
        try:
            copies = [type(m)({k: slice_batch(v, s, e, B) for k, v in m.items()}) for s, e in spans]
        except TypeError as exc:
            raise TypeError(f"cannot copy a {type(m).__name__} mapping argument per micro-batch") from exc
        rec = maps[id(m)] = (m, copies)
    return rec[1][si]


def micro_call(fn: Callable, module: nn.Module, args: tuple, kwargs: dict, B: int, micro: int,
               maps: dict | None = None):
    """fn(module, *args, **kwargs) on consecutive micro-batches of `micro` windows of a batch of B; one call when
    micro >= B. Mapping arguments get a per-span copy that lasts the pass (`maps`, see _span_mapping); a cache
    argument is refused (it would hold several micro-batches' keys)."""
    if micro >= B:
        return fn(module, *args, **kwargs)
    spans = [(s, min(B, s + micro)) for s in range(0, B, micro)]
    vals = list(args) + list(kwargs.values())
    for v in vals:
        if _is_cache(v):
            raise RuntimeError(f"{type(module).__name__}: a cache cannot be shared by {len(spans)} micro-batches")
    outs = []
    for si, (s, e) in enumerate(spans):
        def one(x):
            return _span_mapping(x, spans, B, maps, si) if isinstance(x, Mapping) else slice_batch(x, s, e, B)

        a = tuple(one(x) for x in args)
        kw = {k: one(v) for k, v in kwargs.items()}
        outs.append(fn(module, *a, **kw))
    return concat_batch(outs, B)


# ---------------------------------------------------------------------------------------------------------------------
# expert statistics


class ExpertStats:
    """Sum of squares of the expert inputs over every routed (token, expert) pair of one MoE site (see the module
    docstring). result() -> {"gate_up": {"sumsq": float32 [E, H], "count": int64 [E]}, "down": {"sumsq": float32
    [E, I], "count"}}."""

    def __init__(self, site: arch.MoESite, device):
        m = site.ref
        impl = getattr(getattr(m, "config", None), "_experts_implementation", None)
        if impl != "grouped_mm":
            raise RuntimeError(f"{site.name}: expert statistics need experts_implementation 'grouped_mm' (rows sorted "
                               f"by expert id at _apply_gate), this model resolved {impl!r}")
        if not site.layout.gated:
            raise NotImplementedError(f"{site.name}: non-gated experts (act_fn without _apply_gate)")
        self.site, self.mod, self.dev = site, m, torch.device(device)
        self.E = site.layout.E
        params = list(inspect.signature(type(m).forward).parameters)[1:]
        self.names = params[:2]  # hidden states, top-k indices (gpt-oss: router_indices)
        self.gu = self.dn = None
        self.cnt = torch.zeros(self.E, dtype=torch.int64, device=self.dev)
        self.dn_rows = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.calls = 0
        self._ids = None
        self.handle = m.register_forward_pre_hook(self._pre, with_kwargs=True)
        orig = type(m)._apply_gate
        m._apply_gate = lambda gate_up_out: self._gate(orig(m, gate_up_out))

    def _arg(self, args, kwargs, i):
        return args[i] if len(args) > i else kwargs[self.names[i]]

    def _pre(self, mod, args, kwargs):
        x, idx = self._arg(args, kwargs, 0), self._arg(args, kwargs, 1)
        x = x.reshape(-1, x.shape[-1])
        idx = idx.reshape(-1, idx.shape[-1])
        if idx.shape[0] != x.shape[0]:
            raise RuntimeError(f"{self.site.name}: {x.shape[0]} tokens, {idx.shape[0]} index rows")
        if self.gu is None:
            self.gu = torch.zeros(self.E, x.shape[-1], dtype=torch.float64, device=self.dev)
        x2 = x.double().square()
        for k in range(idx.shape[1]):
            self.gu.index_add_(0, idx[:, k], x2)
        flat = idx.reshape(-1)
        self.cnt += torch.bincount(flat, minlength=self.E)
        self._ids = torch.sort(flat).values  # the row order of grouped_mm_experts_forward
        self.calls += 1
        return None

    def _gate(self, act):
        if self._ids is None:
            raise RuntimeError(f"{self.site.name}: _apply_gate called without a routed input")
        if act.shape[0] != self._ids.numel():
            raise RuntimeError(f"{self.site.name}: activation rows {act.shape[0]} != routed pairs {self._ids.numel()}")
        if self.dn is None:
            self.dn = torch.zeros(self.E, act.shape[-1], dtype=torch.float64, device=self.dev)
        self.dn.index_add_(0, self._ids, act.double().square())
        self.dn_rows += act.shape[0]
        self._ids = None
        return act

    def remove(self):
        self.handle.remove()
        if "_apply_gate" in self.mod.__dict__:
            del self.mod.__dict__["_apply_gate"]

    def result(self) -> dict:
        if self.gu is None or self.dn is None:
            raise RuntimeError(f"{self.site.name}: no routed tokens were seen")
        if int(self.dn_rows) != int(self.cnt.sum()):
            raise RuntimeError(f"{self.site.name}: down rows {int(self.dn_rows)} != routed pairs {int(self.cnt.sum())}")
        cnt = self.cnt.cpu()
        return {"gate_up": {"sumsq": self.gu.float().cpu(), "count": cnt.clone()},
                "down": {"sumsq": self.dn.float().cpu(), "count": cnt.clone()}}


# ---------------------------------------------------------------------------------------------------------------------
# streamer


@dataclass
class PassResult:
    """One base forward over a group of windows. hidden: the base model's last_hidden_state [B, S, H] (after its final
    norm / hc_head), in the caller's window order; stats: {decoder layer: ExpertStats.result()} when collected."""

    group: str
    hidden: torch.Tensor
    stats: dict[int, dict] | None
    per_layer: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    order: list[int] | None = None
    args: dict = field(default_factory=dict)  # describe_args of the first decoder layer call


def _split(name: str) -> tuple[str, str]:
    return name.rsplit(".", 1) if "." in name else ("", name)


class Streamer:
    """See the module docstring. Attributes: config (as config.json gives it), text_config, entry (ramog.arch),
    model (text model: meta decoder layers, resident rest), base (model.model), keymap, index, sites {layer: MoESite},
    layer_keys {layer: model keys}, resident_keys, info (load facts)."""

    def __init__(self, model_dir: str | os.PathLike, device: str | torch.device = "cuda:0", *, threads: int = 32,
                 depth: int = 2, drop_cache: bool = False, chunk: int = 16, log: Callable[[str], None] = print):
        from transformers import AutoConfig

        t0 = time.time()
        self.model_dir = str(model_dir)
        self.dev = torch.device(device)
        self.log = log
        self.threads, self.depth, self.drop, self.chunk = threads, depth, drop_cache, max(1, int(chunk))
        self.config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=False)
        self.entry = arch.arch_for(self.config)
        self.model = arch.build_meta(self.config)
        self.index = ckpt.CheckpointIndex(self.model_dir)
        self.keymap = ckpt.KeyMap.build(self.model, self.entry, self.index)  # strict: raises on any problem
        self.text_config = arch.text_config(self.config)
        self.layers = arch.decoder_layers(self.model)
        self.base = self.model.model
        if self.base.layers is not self.layers:
            raise TypeError(f"{type(self.model).__name__}: decoder layers are not model.model.layers")
        self.layers_prefix = next(n for n, m in self.model.named_modules() if m is self.layers)
        self.sites = {s.layer_idx: s for s in self.keymap.sites}
        n = len(self.layers)
        by_layer = self.keymap.keys_by_layer()
        bad = sorted(k for k in by_layer if k is not None and not 0 <= k < n)
        if bad:
            raise RuntimeError(f"key map has tensors of decoder layers {bad} outside 0..{n - 1}")
        self.layer_keys = {li: list(by_layer.get(li, [])) for li in range(n)}
        host = set(self.keymap.host_keys())
        self.host_keys = [k for k in self.keymap.targets if k in host]
        self.layer_bind_keys = {li: [k for k in ks if k not in host] for li, ks in self.layer_keys.items()}
        self.resident_keys = [k for k in by_layer.get(None, []) if k not in host]
        state = self.model.state_dict(keep_vars=True)
        for li in range(n):
            want = {k for k in state if k.startswith(f"{self.layers_prefix}.{li}.")}
            if want != set(self.layer_keys[li]):
                raise RuntimeError(f"layer {li}: model keys and key-map targets differ: "
                                   f"{sorted(want ^ set(self.layer_keys[li]))[:5]}")
        self._owner = {}
        for k in self.keymap.targets:
            parent, leaf = _split(k)
            mod = self.model.get_submodule(parent)
            if leaf not in mod._parameters and leaf not in mod._buffers:
                raise RuntimeError(f"{k}: neither a parameter nor a buffer of {type(mod).__name__}")
            self._owner[k] = (mod, leaf)
        self.head = self.model.get_output_embeddings()
        self.softcap = getattr(self.text_config, "final_logit_softcapping", None)
        routers = arch.router_modules(self.model)
        n_sorted = sum(bool(getattr(r, "_mt_sorted_topk", False)) for r in routers)
        if self.entry.sorted_topk and n_sorted != len(routers):
            raise RuntimeError(f"{self.entry.model_type}: {len(routers) - n_sorted} of {len(routers)} routers do not "
                               "run the stable top-k (ramog.arch.build_meta / sort_router_topk)")
        extra = hfcompat.DtypePlan(self.model).extra
        ex_rx = [re.compile(p.replace("*", ".*")) for p in extra]
        self.info = {"model_dir": self.model_dir, "model_type": self.text_config.model_type,
                     "transformers": hfcompat.TF_VERSION, "torch": torch.__version__,
                     "attn": getattr(self.text_config, "_attn_implementation", None),
                     "experts": getattr(self.text_config, "_experts_implementation", None),
                     "layers": n, "moe_layers": sorted(self.sites), "keymap_seconds": round(self.keymap.seconds, 2),
                     "source_formats": self.keymap.source_formats(), "skipped": dict(self.keymap.skipped),
                     "softcap": self.softcap,
                     "router_topk": "sorted" if self.entry.sorted_topk else "as transformers",
                     "routers_sorted": n_sorted,
                     "fp32_keys": sorted(extra),
                     "fp32_keys_tensors": sum(1 for k in self.keymap.targets if any(rx.search(k) for rx in ex_rx))}
        self._bind_host()
        self._load_resident()
        self.info["setup_seconds"] = round(time.time() - t0, 1)
        log(f"{self.model_dir}: {type(self.model).__name__} transformers {hfcompat.TF_VERSION} "
            f"attn={self.info['attn']} experts={self.info['experts']} layers={n} moe={len(self.sites)} "
            f"checkpoint {self.index.nbytes() / 2**30:.1f} GiB; resident {self.info['resident_tensors']} tensors "
            f"({self.info['resident_bytes'] / 2**30:.2f} GiB), {self.info['nonpersistent']} non-persistent buffers "
            f"rebuilt; setup {self.info['setup_seconds']}s")

    # -- binding ----------------------------------------------------------------------------------------------------

    def _value(self, km: ckpt.KeyMap, key: str) -> torch.Tensor:
        """Model tensor `key` on the device in its KeyMap target dtype (hfcompat.DtypePlan). Expert tensors are decoded
        `chunk` experts at a time into the full tensor (bitwise the same as one load; bounds the decode's transient
        memory, KeyMap.load stacks and concatenates)."""
        t = km.targets[key]
        if t.is_expert and len(t.shape) >= 1 and t.shape[0] > self.chunk:
            out = torch.empty(t.shape, dtype=t.dtype, device=self.dev)
            for e0 in range(0, t.shape[0], self.chunk):
                e1 = min(t.shape[0], e0 + self.chunk)
                part = km.load(key, self.dev, rows=(e0, e1))
                if tuple(part.shape) != (e1 - e0, *t.shape[1:]) or part.dtype != t.dtype:
                    raise RuntimeError(f"{key}[{e0}:{e1}]: decoded {part.dtype} {list(part.shape)}, expected "
                                       f"{t.dtype} {[e1 - e0, *t.shape[1:]]}")
                out[e0:e1].copy_(part)
                del part
            return out
        v = km.load(key, self.dev)
        if tuple(v.shape) != tuple(t.shape) or v.dtype != t.dtype:
            raise RuntimeError(f"{key}: decoded {v.dtype} {list(v.shape)}, expected {t.dtype} {list(t.shape)}")
        return v

    def _set(self, key: str, value: torch.Tensor):
        mod, leaf = self._owner[key]
        if leaf in mod._parameters:
            orig = mod._parameters[leaf]
            mod._parameters[leaf] = nn.Parameter(value, requires_grad=False)
            return (mod, leaf, "p", orig)
        orig = mod._buffers[leaf]
        mod._buffers[leaf] = value
        return (mod, leaf, "b", orig)

    def bind(self, keys: Iterable[str], raw: Mapping[str, torch.Tensor] | None = None) -> list:
        """Decode `keys` (from read-ahead raw tensors, else from the checkpoint) onto the device and put them into the
        model; returns what unbind() needs."""
        km = self.keymap
        if raw is not None:
            km = copy.copy(self.keymap)
            km.index = RawIndex(raw)
        saved = []
        try:
            for k in keys:
                saved.append(self._set(k, self._value(km, k)))
        except BaseException:
            self.unbind(saved)
            raise
        return saved

    @staticmethod
    def unbind(saved: list) -> None:
        for mod, leaf, kind, orig in reversed(saved):
            (mod._parameters if kind == "p" else mod._buffers)[leaf] = orig

    def checkpoint_keys(self, keys: Iterable[str]) -> list[str]:
        """Every checkpoint key (stored tensors and their side tensors) the model keys are made from."""
        out = []
        for k in keys:
            for p in self.keymap.targets[k].all_parts():
                out.append(p.key)
                out.extend(p.sides)
        return list(dict.fromkeys(out))

    def _bind_host(self) -> None:
        """Host tensors: zero-copy memory maps of the checkpoint, bound for the streamer's lifetime."""
        nbytes = 0
        for k in self.host_keys:
            v = self.keymap.host_view(k)
            nbytes += v.numel() * v.element_size()
            self._set(k, v)
        self.info.update(host_tensors=len(self.host_keys), host_bytes=nbytes)

    @torch.no_grad()
    def _load_resident(self) -> None:
        nbytes = 0
        for k in self.resident_keys:
            v = self._value(self.keymap, k)
            nbytes += v.numel() * v.element_size()
            self._set(k, v)
        for head_key, emb_key in self.keymap.tied.items():  # a tied head shares the embedding's Parameter
            hmod, hleaf = _split(head_key)
            emod, eleaf = self._owner[emb_key]
            self.model.get_submodule(hmod)._parameters[hleaf] = emod._parameters[eleaf]
        self.index.close()  # drops the staging buffer of the resident reads
        nonpers = list(hfcompat.named_non_persistent_buffers(self.model))
        names = {n for n, _ in nonpers}
        for name, b in nonpers:
            parent, leaf = _split(name)
            buf = torch.empty_like(b, device=self.dev)
            if buf.is_floating_point():
                buf.fill_(float("nan"))
            self.model.get_submodule(parent)._buffers[leaf] = buf
        for n, t in list(self.model.named_parameters(remove_duplicate=False)) + \
                list(self.model.named_buffers(remove_duplicate=False)):
            if n not in names:
                t._is_hf_initialized = True
        hfcompat.initialize_weights(self.model)
        bad = [n for n, b in hfcompat.named_non_persistent_buffers(self.model)
               if b.device.type == "meta" or (b.is_floating_point() and torch.isnan(b).any())]
        if bad:
            raise RuntimeError(f"non-persistent buffers not rebuilt by initialize_weights: {bad[:5]}")
        left = [n for n, t in self.model.named_parameters() if t.device.type == "meta"
                and not n.startswith(f"{self.layers_prefix}.")]
        if left:
            raise RuntimeError(f"tensors outside the decoder layers left on meta: {left[:5]}")
        self.info.update(resident_tensors=len(self.resident_keys), resident_bytes=nbytes, nonpersistent=len(names),
                         tied=dict(self.keymap.tied))

    # -- head -------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """The CausalLM head on base-model hidden states: lm_head, then Gemma's final softcap where the CausalLM applies
        it (logits / cap, tanh, * cap: Gemma4ForCausalLM's own ops). Call it per window, as make_refs.py does."""
        out = self.head(h)
        if self.softcap is not None:
            out = out / self.softcap
            out = torch.tanh(out)
            out = out * self.softcap
        return out

    # -- passes -----------------------------------------------------------------------------------------------------

    def _tail_modules(self) -> list[nn.Module]:
        """Children of the base model that are micro-batched without weights to bind (see the module docstring)."""
        out = []
        for name, m in self.base.named_children():
            if m is self.layers or isinstance(m, nn.Embedding) or "Rotary" in type(m).__name__:
                continue
            out.append(m)
        return out

    @torch.no_grad()
    def run(self, windows_by_group: Mapping[str, Sequence[torch.Tensor] | torch.Tensor],
            collect_stats: Iterable[str] | bool = ("calib",), micro: int = 1,
            order: Mapping[str, Sequence[int]] | None = None, n_layers: int | None = None,
            on_layer: Callable[[str, int, dict | None, dict], None] | None = None) -> dict[str, PassResult]:
        """One base forward per group (each group's windows share one length). collect_stats: the groups whose expert
        statistics are collected (True: all). order: per group, the order in which the windows are batched (a
        permutation of range(B)); results come back in the given window order. n_layers: debug, only the first n
        decoder layers (then the final norm). on_layer(group, layer, stats or None, info) after each layer."""
        groups = list(windows_by_group)
        if collect_stats is True:
            collect = set(groups)
        elif collect_stats is False or collect_stats is None:
            collect = set()
        else:
            collect = set(collect_stats)
        if not collect <= set(groups):
            raise ValueError(f"collect_stats {sorted(collect - set(groups))} are not groups {groups}")
        if micro < 1:
            raise ValueError("micro must be >= 1")
        n_all = len(self.layers)
        n = n_all if n_layers is None else int(n_layers)
        if not 1 <= n <= n_all:
            raise ValueError(f"n_layers {n} outside 1..{n_all}")
        plans = [((g, li), self.checkpoint_keys(self.layer_bind_keys[li])) for g in groups for li in range(n)]
        reader = LayerReader(self.index, plans, depth=self.depth, threads=self.threads, drop_cache=self.drop,
                             pin=self.dev.type == "cuda")
        layers_all = self.layers
        results: dict[str, PassResult] = {}
        try:
            if n < n_all:
                self.base.layers = nn.ModuleList(list(layers_all)[:n])
            for g in groups:
                rows = windows_by_group[g]
                ids = torch.stack(list(rows)) if not isinstance(rows, torch.Tensor) else rows
                if ids.dim() != 2:
                    raise ValueError(f"group {g}: windows must be 1-D token rows of one length, got {list(ids.shape)}")
                B = ids.shape[0]
                perm = list(order[g]) if order and g in order else None
                if perm is not None:
                    if sorted(perm) != list(range(B)):
                        raise ValueError(f"group {g}: order is not a permutation of range({B})")
                    ids = ids[torch.tensor(perm)]
                results[g] = self._pass(g, ids, B, micro, n, g in collect, reader, on_layer, perm)
        finally:
            reader.close()
            if n < n_all:
                self.base.layers = layers_all
        self.read_stats = (reader.read_bytes, reader.read_seconds)
        return results

    def _pass(self, group, ids, B, micro, n, stats_on, reader, on_layer, perm) -> PassResult:
        t0 = time.time()
        state = {"next": 0, "args": None}
        maps: dict = {}  # per-span copies of mapping arguments, for the whole pass (micro_call)
        stats: dict[int, dict] = {}
        per_layer: list[dict] = []
        installed: list[nn.Module] = []
        dev_sync = (lambda: torch.cuda.synchronize(self.dev)) if self.dev.type == "cuda" else (lambda: None)

        def make_layer_forward(li: int, layer: nn.Module):
            orig = type(layer).forward
            keys = self.layer_bind_keys[li]
            site = self.sites.get(li)

            def forward(*args, **kwargs):
                if state["next"] != li:
                    raise RuntimeError(f"{group}: decoder layer {li} called, expected layer {state['next']}")
                tw = time.time()
                key, raw, buf, rsec, rbytes = reader.get()
                t_wait = time.time() - tw
                if key != (group, li):
                    raise RuntimeError(f"read-ahead delivered {key}, the forward is at {(group, li)}")
                t1 = time.time()
                try:
                    saved = self.bind(keys, raw)
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
                if state["args"] is None:
                    state["args"] = describe_args(args, kwargs, B)
                t2 = time.time()
                st = None
                try:
                    st = ExpertStats(site, self.dev) if (stats_on and site is not None) else None
                    out = micro_call(orig, layer, args, kwargs, B, micro, maps)
                    dev_sync()
                finally:
                    if st is not None:
                        st.remove()
                    self.unbind(saved)
                    del saved
                t_comp = time.time() - t2
                res = st.result() if st is not None else None
                if res is not None:
                    stats[li] = res
                info = {"read_s": rsec, "wait_s": t_wait, "bind_s": t_bind, "compute_s": t_comp,
                        "read_bytes": rbytes, "micro_batches": math.ceil(B / micro)}
                if st is not None:
                    info["stats_calls"] = st.calls
                per_layer.append({"layer": li, **{k: (round(v, 3) if isinstance(v, float) else v)
                                                   for k, v in info.items()}})
                state["next"] = li + 1
                if on_layer is not None:
                    on_layer(group, li, res, info)
                return out

            return forward

        def make_tail_forward(m: nn.Module):
            orig = type(m).forward

            def forward(*args, **kwargs):
                return micro_call(orig, m, args, kwargs, B, micro, maps)

            return forward

        try:
            for li in range(n):
                layer = self.layers[li]
                layer.forward = make_layer_forward(li, layer)
                installed.append(layer)
            for m in self._tail_modules():
                m.forward = make_tail_forward(m)
                installed.append(m)
            out = self.base(input_ids=ids.to(self.dev), use_cache=False)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        finally:
            for m in installed:
                if "forward" in m.__dict__:
                    del m.__dict__["forward"]
        if state["next"] != n:
            raise RuntimeError(f"{group}: the base forward ran {state['next']} of {n} decoder layers")
        if hidden.shape[0] != B:
            raise RuntimeError(f"{group}: last hidden state {list(hidden.shape)}, batch {B}")
        if perm is not None:
            inv = torch.empty(B, dtype=torch.long)
            inv[torch.tensor(perm)] = torch.arange(B)
            hidden = hidden[inv.to(hidden.device)]
        dev_sync()
        return PassResult(group, hidden, stats if stats_on else None, per_layer, time.time() - t0, perm,
                          state["args"] or {})


def stream(model_dir: str | os.PathLike, windows_by_group: Mapping[str, Sequence[torch.Tensor]],
           collect_stats: Iterable[str] | bool = ("calib",), micro: int = 1, *, device: str = "cuda:0",
           **kwargs) -> tuple[Streamer, dict[str, PassResult]]:
    """Streamer(model_dir, device, **kwargs).run(windows_by_group, collect_stats, micro): design-core B7's entry
    point. Returns the streamer (for its logits() head) and the per-group results."""
    run_kw = {k: kwargs.pop(k) for k in ("order", "n_layers", "on_layer") if k in kwargs}
    s = Streamer(model_dir, device, **kwargs)
    return s, s.run(windows_by_group, collect_stats, micro, **run_kw)
