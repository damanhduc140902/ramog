"""Expert weights as packed blobs (host side) and the GPU-driven expert cache.

Every expert of a layer is stored, per precision, as one contiguous blob holding the packed gate_up and down
matrices, in one of two formats (BlobLayout.format):
  rtn    (ramog.quant)   per matrix: qweight, fp16 scale and fp16 minimum per group;
  k4row  (ramog.kquant)  per matrix: qweight, one byte of 4-bit (sc, mc) codes per group, fp16 (d, dmin) per row.
Pools of blobs are exposed to the kernels as strided [slots, rows, cols] views, so no repacking is needed after a
copy; the hot cache, the cold blobs and the prefetch staging may each use a different layout.

Precision follows the cache. A per-layer table `slot_of[expert]` lives on the GPU: the decode kernels read 4-bit
("hot") weights from the cache slot when slot_of >= 0, and otherwise read the expert's 2-bit ("cold") blob
straight from pinned host memory (zero-copy). Nothing on the decode path waits for the host.

The cache is maintained asynchronously by `update()`, called every few decode steps:
  * usage counts accumulate on the GPU and are snapshotted host-side one interval late (no stall);
  * promotions copy a hot blob into a free slot (on a side stream, overlapping with decode; on ROCm on the compute
    stream, see fill_stream_kind) and are published in slot_of only after the copy is known complete;
  * evictions set slot_of = -1 first; the slot is reused only after the compute stream has passed that point,
    so no in-flight kernel ever reads a slot that is being overwritten.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import torch

from . import kquant
from .quant import Packed, dequantize, quantize

FORMATS = ("rtn", "k4row")
ALIGN = 16  # k4row arrays start on 16-byte boundaries (int32 views, aligned vector loads); rtn keeps its layout


def _round_up(n: int, a: int) -> int:
    return -(-n // a) * a


def fill_stream_kind() -> str:
    """Where ExpertCache.update() copies promoted blobs into the cache: "side" (a side stream, overlapping with
    decode) or "compute" (the compute stream, in order with decode). ROCm uses "compute": on MI250 (ROCm 7.2.3,
    torch 2.11) a side-stream fill enqueued while 2-3 graph-captured decode steps were still queued stopped the GPU
    in the next replay, every time on the Qwen3-235B at the default host lead of 2; with the fills on the compute
    stream no run hung, even with the host lead unbounded (results/polish/fix/). Decode is about 3% slower on the
    Qwen3-30B (the copies no longer overlap with compute). ENGINE_FILL_STREAM=side|compute overrides."""
    kind = os.environ.get("ENGINE_FILL_STREAM") or ("compute" if torch.version.hip else "side")
    if kind not in ("side", "compute"):
        raise ValueError(f"ENGINE_FILL_STREAM={kind!r}: expected 'side' or 'compute'")
    return kind


@dataclass(frozen=True)
class BlobLayout:
    """Byte layout of one expert blob (gate_up, then down) for a given precision and format.

    rtn   per matrix [n, k]: qweight uint8 [n, k*bits/8], scale fp16 [n, k/group], minimum fp16 [n, k/group]
          (contiguous, no padding: the layout of every existing store);
    k4row per matrix [n, k]: qweight uint8 [n, k*bits/8], sb uint8 [n, k/group], d fp16 [n], dmin fp16 [n],
          each array padded to a multiple of 16 bytes (so every offset and nbytes are multiples of 4).
    """

    bits: int
    group: int
    gu_shape: tuple[int, int]  # gate_up: (2I, H)
    dn_shape: tuple[int, int]  # down: (H, I)
    format: str = "rtn"

    def __post_init__(self):
        if self.format not in FORMATS:
            raise ValueError(f"unknown blob format {self.format!r} (expected one of {FORMATS})")
        for n, k in (self.gu_shape, self.dn_shape):
            # whole int32 words per row and per group: the kernel reads 32 // bits codes per word
            if 8 % self.bits or k % self.group or (k * self.bits) % 32 or self.group % (32 // self.bits):
                raise ValueError(f"{self.format}: bits={self.bits}, group={self.group} do not fit [{n}, {k}]")

    def _arrays(self):
        """(name, shape per expert, dtype, bytes, padded bytes) of every array of a blob, in storage order."""
        out = []
        for m, (n, k) in (("gu", self.gu_shape), ("dn", self.dn_shape)):
            q = (f"{m}_q", (n, k * self.bits // 8), torch.uint8, n * k * self.bits // 8)
            if self.format == "rtn":
                g = n * (k // self.group) * 2
                arrs = [q, (f"{m}_s", (n, k // self.group), torch.float16, g),
                        (f"{m}_m", (n, k // self.group), torch.float16, g)]
                out += [(*a, a[3]) for a in arrs]
            else:
                arrs = [q, (f"{m}_sb", (n, k // self.group), torch.uint8, n * (k // self.group)),
                        (f"{m}_d", (n,), torch.float16, 2 * n), (f"{m}_dm", (n,), torch.float16, 2 * n)]
                out += [(*a, _round_up(a[3], ALIGN)) for a in arrs]
        return out

    def _sizes(self):
        return [a[4] for a in self._arrays()]

    @property
    def nbytes(self) -> int:
        return sum(self._sizes())

    def offsets(self):
        offs, o = [], 0
        for s in self._sizes():
            offs.append(o)
            o += s
        return offs

    def describe(self) -> dict:
        return {"format": self.format, "bits": self.bits, "group": self.group, "nbytes": self.nbytes}

    def views(self, pool: torch.Tensor):
        """pool: uint8 [S, nbytes] (GPU or pinned host) -> dict of strided views for the kernel.
        rtn: {gu,dn}_q, _s, _m; k4row: {gu,dn}_q, _sb, _d, _dm (each [S, *shape])."""
        s = pool.shape[0]
        out = {}
        for (name, shape, dtype, size, _), off in zip(self._arrays(), self.offsets(), strict=True):
            raw = pool[:, off:off + size]
            out[name] = raw.view(s, *shape) if dtype == torch.uint8 else raw.view(dtype).view(s, *shape)
        return out

    def gemv_args(self, views: dict, which: str) -> dict:
        """Keyword arguments of kernels.grouped_gemv for matrix `which` ("gu" or "dn") of a pool's views."""
        if self.format == "rtn":
            return {"qweight": views[f"{which}_q"], "scale": views[f"{which}_s"], "minimum": views[f"{which}_m"],
                    "bits": self.bits, "group": self.group}
        return {"qweight": views[f"{which}_q"], "scale": views[f"{which}_sb"], "minimum": None, "bits": self.bits,
                "group": self.group, "fmt": 1, "d": views[f"{which}_d"], "dmin": views[f"{which}_dm"]}

    def dequant(self, views: dict, idx: int, which: str, dtype=torch.float32) -> torch.Tensor:
        """Dequantize matrix `which` of slot `idx` of a pool's views -> [out, in] in `dtype`."""
        if self.format == "rtn":
            shape = self.gu_shape if which == "gu" else self.dn_shape
            p = Packed(views[f"{which}_q"][idx], views[f"{which}_s"][idx], views[f"{which}_m"][idx], self.bits,
                       self.group, shape)
            return dequantize(p, dtype)
        return kquant.dequantize(views[f"{which}_q"][idx], views[f"{which}_sb"][idx], views[f"{which}_d"][idx],
                                 views[f"{which}_dm"][idx], self.bits, self.group, dtype)

    def assemble(self, arrays: dict, e: int) -> torch.Tensor:
        """Per-array tensors with a leading expert dim -> uint8 blobs [e, nbytes] (zero padding)."""
        parts = []
        for name, _shape, _dtype, size, padded in self._arrays():
            t = arrays[name].reshape(e, -1)
            t = t if t.dtype == torch.uint8 else t.view(torch.uint8)
            assert t.shape[1] == size, (name, t.shape, size)
            parts.append(t)
            if padded > size:
                parts.append(t.new_zeros(e, padded - size))
        blob = torch.cat(parts, dim=1)
        assert blob.shape[1] == self.nbytes, (blob.shape, self.nbytes)
        return blob

    def quantize(self, gate_up: torch.Tensor, down: torch.Tensor, importance=None, device=None,
                 info: dict | None = None) -> torch.Tensor:
        """Quantize E experts (gate_up [E, 2I, H], down [E, H, I], nn.Linear layout) into uint8 blobs [E, nbytes],
        on `device` (default: the weights' device), one matrix at a time.
        importance: optional per-expert input-channel weights (gate_up [E, H], down [E, I]) for the k4row search
        (see kquant.importance_from_stats); rtn ignores it. info: optional dict; k4row adds the number of groups
        that needed fallback codes to info["fallback"]."""
        e = gate_up.shape[0]
        arrays = {}
        for j, (m, w) in enumerate((("gu", gate_up), ("dn", down))):
            n, k = w.shape[1:]
            w = w.to(device if device is not None else w.device, torch.float32)
            if self.format == "rtn":
                p: Packed = quantize(w.reshape(e * n, k), self.bits, self.group)
                arrays.update({f"{m}_q": p.qweight, f"{m}_s": p.scale, f"{m}_m": p.minimum})
            else:
                a = None if importance is None else importance[j]
                kp = kquant.quantize(w, self.bits, self.group, a)
                arrays.update({f"{m}_q": kp.qweight, f"{m}_sb": kp.sb, f"{m}_d": kp.d, f"{m}_dm": kp.dmin})
                if info is not None:
                    info["fallback"] = info.get("fallback", 0) + kp.fallback
            del w
        return self.assemble(arrays, e)

    def pack(self, gate_up: torch.Tensor, down: torch.Tensor, importance=None) -> torch.Tensor:
        """Quantize one expert (weights in nn.Linear layout [out, in]) into a uint8 blob.
        importance: optional (gate_up [H], down [I]) input-channel weights (k4row only)."""
        imp = None if importance is None else tuple(a.reshape(1, -1) for a in importance)
        blob = self.quantize(gate_up[None], down[None], imp)[0]
        assert blob.numel() == self.nbytes, (blob.numel(), self.nbytes)
        return blob


class LayerExperts:
    """Host-side packed experts of one MoE layer: hot (4-bit) and cold (2-bit) blobs for every expert.

    host_row (design-quality WP12, the split): None = row e of hot / cold is expert e (every expert has a host copy,
    today's layout). Otherwise int32 [E] on the CPU: the row of expert e in the PARTIAL host tensors hot / cold
    ([E - F, nbytes], the experts that are not frozen in VRAM), -1 for a frozen expert (VRAM only, no host copy).
    Every host read (zero-copy GEMV, prefill, cache fills, prefetch copies) goes through it."""

    def __init__(self, hot: torch.Tensor, cold: torch.Tensor, hot_layout: BlobLayout, cold_layout: BlobLayout,
                 host_row: torch.Tensor | None = None):
        self.hot = hot  # uint8 [E, hot_layout.nbytes], pinned ([E - F, ...] with host_row)
        self.cold = cold  # uint8 [E, cold_layout.nbytes], pinned; read by the GPU zero-copy
        self.hot_layout = hot_layout
        self.cold_layout = cold_layout
        self.cold_views = cold_layout.views(cold)
        self.host_row = None
        self.host_rows: list[int] | None = None  # host_row as a list (host-side fills)
        if host_row is not None:
            hr = host_row.to("cpu", torch.int32).reshape(-1)
            have = sorted(int(r) for r in hr.tolist() if r >= 0)
            if have != list(range(cold.shape[0])) or hot.shape[0] != cold.shape[0]:
                raise ValueError(f"host_row must map the non-frozen experts one to one onto the {cold.shape[0]} host "
                                 f"rows (hot {hot.shape[0]} rows), got rows {have[:8]}...")
            self.host_row = hr
            self.host_rows = hr.tolist()

    @property
    def num_experts(self) -> int:
        return self.hot.shape[0] if self.host_row is None else self.host_row.numel()

    def row(self, e: int) -> int:
        """Row of expert e in hot / cold (-1: frozen, no host copy)."""
        return e if self.host_rows is None else self.host_rows[e]


class ExpertCache:
    def __init__(self, layer: LayerExperts, slots: int, device, stream: torch.cuda.Stream | None,
                 decay: float = 0.8, frozen=None):
        """stream: the side stream for cache fills, or None to fill on the compute stream (the current stream when
        update() runs), see fill_stream_kind().
        frozen (design-quality WP12, the split): expert ids that sit in slots 0..F-1 from construction, are published
        in slot_of at once and are never evicted (excluded from the eviction ranking and from the candidates); the
        dynamic LFU runs on the other slots - F slots over the other experts. The caller puts their bytes into
        pool[:F] (ramog.loader streams them from the store file) before anything runs. With a host_row map in
        `layer` the frozen experts must be exactly the ones without a host row."""
        self.layer = layer
        self.slots = slots
        self.device = device
        self.stream = stream
        self.decay = decay
        e = layer.num_experts
        self.pool = torch.empty(max(1, slots), layer.hot_layout.nbytes, dtype=torch.uint8, device=device)
        self.views = layer.hot_layout.views(self.pool)
        self.slot_of = torch.full((e,), -1, dtype=torch.int32, device=device)  # published table (GPU)
        self.cached: dict[int, int] = {}  # expert -> slot, as published
        self.frozen = [int(x) for x in frozen] if frozen is not None else []
        n_f = len(self.frozen)
        if n_f or layer.host_row is not None:
            no_host = {i for i in range(e) if layer.row(i) < 0}
            if len(set(self.frozen)) != n_f or not all(0 <= x < e for x in self.frozen) or n_f > slots:
                raise ValueError(f"frozen experts must be {n_f} distinct ids in 0..{e - 1}, at most the {slots} slots")
            if layer.host_row is not None and set(self.frozen) != no_host:
                raise ValueError(f"frozen experts {sorted(self.frozen)[:8]}... differ from the experts without a host "
                                 f"row {sorted(no_host)[:8]}...")
        self._frozen_set = set(self.frozen)
        self.free = list(range(n_f, slots))
        if n_f:
            self.cached = {x: s for s, x in enumerate(self.frozen)}
            self.slot_of[torch.tensor(self.frozen, dtype=torch.long, device=device)] = torch.arange(
                n_f, dtype=torch.int32, device=device)
            self._frozen_idx = torch.tensor(self.frozen, dtype=torch.long)
        self.reclaim: deque[tuple[int, torch.cuda.Event]] = deque()  # evicted slots waiting for the compute stream
        self.inflight: list[tuple[int, int, torch.cuda.Event]] = []  # (expert, slot, copy done) not yet published
        self.counts = torch.zeros(e, dtype=torch.float32, device=device)  # usage, accumulated on the GPU
        self.snap = torch.zeros(e, dtype=torch.float32).pin_memory()
        self.snap_event: torch.cuda.Event | None = None
        self.prev = torch.zeros(e, dtype=torch.float32)
        self.score = torch.zeros(e, dtype=torch.float32)
        self.fill_bytes = 0

    # -- decode-path helpers (GPU only, no host sync) ------------------------------------------------
    def record(self, idx: torch.Tensor) -> None:
        self.counts.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=idx.device))

    # -- maintenance (host side, called every few steps) ---------------------------------------------
    def _to_dev(self, values: list[int], dtype) -> torch.Tensor:
        # pinned + non_blocking: a plain host->device copy would block until the GPU drains its queue
        return torch.tensor(values, dtype=dtype).pin_memory().to(self.device, non_blocking=True)

    def _publish_and_reclaim(self) -> None:
        flags = [ev.query() for _, _, ev in self.inflight]
        done = [(e, s) for (e, s, _), f in zip(self.inflight, flags, strict=True) if f]
        if done:
            self.inflight = [t for t, f in zip(self.inflight, flags, strict=True) if not f]
            self.slot_of[self._to_dev([e for e, _ in done], torch.long)] = self._to_dev([s for _, s in done], torch.int32)
            for e, s in done:
                self.cached[e] = s
        while self.reclaim and self.reclaim[0][1].query():
            self.free.append(self.reclaim.popleft()[0])

    def update(self, max_moves: int = 2) -> None:
        if self.slots - len(self.frozen) <= 0:  # no dynamic slot (no slot at all without frozen experts)
            return
        self._publish_and_reclaim()
        # usage since the last snapshot (one interval late, so this never waits for the GPU)
        if self.snap_event is not None and self.snap_event.query():
            cur = self.snap.clone()
            self.score = self.score * self.decay + (cur - self.prev)
            self.prev = cur
        self.snap.copy_(self.counts, non_blocking=True)
        self.snap_event = torch.cuda.Event()
        self.snap_event.record(torch.cuda.current_stream(self.device))

        busy = {e for e, _, _ in self.inflight}
        if self.frozen:  # the dynamic LFU: the other slots over the other experts
            sc = self.score.clone()
            sc[self._frozen_idx] = float("-inf")
            dyn = min(self.slots, sc.numel()) - len(self.frozen)
            want = torch.topk(sc, dyn).indices.tolist()
        else:
            want = torch.topk(self.score, min(self.slots, self.score.numel())).indices.tolist()
        candidates = [e for e in want if e not in self.cached and e not in busy and self.score[e] > 0][:max_moves]
        if not candidates:
            return
        evict = []
        need = max(0, len(candidates) - len(self.free))
        if need:
            # lowest-scoring cached experts, only if a candidate beats them
            ranked = sorted((e for e in self.cached if e not in self._frozen_set), key=lambda e: float(self.score[e]))
            for e in ranked[:need]:
                if float(self.score[e]) < float(self.score[candidates[-1]]):
                    evict.append(e)
        if evict:
            self.slot_of[self._to_dev(evict, torch.long)] = -1  # compute stream: later kernels read the cold blob
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            for e in evict:
                self.reclaim.append((self.cached.pop(e), ev))
        self._publish_and_reclaim()
        fill = self.stream if self.stream is not None else torch.cuda.current_stream(self.device)
        with torch.cuda.stream(fill):
            for e in candidates:
                if not self.free:
                    break
                s = self.free.pop()
                self.pool[s].copy_(self.layer.hot[self.layer.row(e)], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(fill)
                self.inflight.append((e, s, ev))
                self.fill_bytes += self.layer.hot_layout.nbytes

    def warm(self, experts: list[int]) -> None:
        """Synchronously fill free slots with `experts` (used after a prefill)."""
        todo = [e for e in experts if e not in self.cached][: len(self.free)]
        if not todo:
            return
        slots = [self.free.pop() for _ in todo]
        for e, s in zip(todo, slots, strict=True):
            self.pool[s].copy_(self.layer.hot[self.layer.row(e)], non_blocking=True)
            self.cached[e] = s
            self.fill_bytes += self.layer.hot_layout.nbytes
        self.slot_of[torch.tensor(todo, device=self.device)] = torch.tensor(slots, dtype=torch.int32, device=self.device)
        self.score[todo] += 1.0
