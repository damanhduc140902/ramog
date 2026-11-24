"""Drop-in MoE block for Qwen3-MoE style models that runs experts from the tiered store.

Decode (few tokens) runs entirely on the GPU: router -> slot = slot_of[expert] -> for each matrix one launch
over the hot pool (4-bit, VRAM) and one over the cold blobs (2-bit, pinned host memory read zero-copy), both
adding into the same output -> activation -> weighted sum. No host synchronization.
Each weight source (hot pool, cold blobs, prefetch staging) is launched with its own BlobLayout, so any mix of
formats works (e.g. hot rtn 4-bit + cold k4row 2-bit; with hot_from_cold the cache holds cold-format blobs; the
staging buffers always use the cold layout).
Prefill (more tokens): the (token, expert) pairs are grouped by expert on the GPU and two grouped dequant-GEMM
launches per matrix (hot pool; cold blobs zero-copy, in bigger blocks since each block re-reads its weights over
the host link) read every used expert's weights once per block of its pairs, again without a host
synchronization; afterwards the busiest experts fill any free cache slots. Prompts are processed PREFILL_CHUNK
tokens at a time (bounded scratch). ENGINE_PREFILL=dequant selects the previous path (each used expert
dequantized to bf16 and multiplied as a regular matmul, one host synchronization per expert), kept as the
reference.
A shared Manager runs cache maintenance every few decode steps.

Expert math (design-core B1; WP8). By default an expert is Qwen3-MoE's: a = act(h[:, :I]) * h[:, I:] on the float32
gate_up output h, then down. Three optional hooks generalize it, each a no-op when unset (the default code path is
then unchanged, byte for byte):
- glu: a = glu(h) replaces the default gate (the replaced transformers module's own `_apply_gate`, e.g. gpt-oss's
  interleaved clamp / alpha / (up + 1) gate or GLM-5.3-Flash's clamp-10 SwiGLU), run on the float32 h;
- gu_bias / dn_bias: float32 [E, 2I] / [E, H] tables in VRAM, gathered per (token, expert) pair: h += gu_bias[e]
  before the gate, y += dn_bias[e] before the routing weight (gpt-oss's order, modeling_gpt_oss 5.19 :107-110);
- predict: x [T, H] -> expert ids [T, k], the next-layer prediction for prefetch (moe_tiering.arch.make_probe: the
  layer's own router), instead of topk(F.linear(x, router.weight)).

Trellis blobs (design-quality WP15; layout moe_tiering.tcq.TcqLayout, kernels moe_tiering.tcq_kernels). With rot = (signs_gu,
b_gu, signs_dn, b_dn) (moe_tiering.experts.tcq_rotation: the layer's input rotation, seed layer * 1000) the same sources
are launched with the rotated-domain kernels, one tier, hot from cold (the cache holds the layer's own trellis blobs):
- decode: x_rot = rotate(x) in float32 once per layer and step, tcq_gemv over the hot / host zero-copy / staged
  sources (mode 1, host reads through WP12's host_row), a = act(h) cast to x.dtype exactly as above, a_rot =
  rotate(a) per pair (float32), down the same way. tcq_gemv has no split-K, so a pair gives the same bits from VRAM,
  pinned host or the staging buffer;
- prefill: x_rot = rotate(x) once, tcq_moe_gemm with FLOAT32 operands (design-quality amendment 8: with bf16
  operands the rotation spreads an attention-sink token's massive channel over every entry and bf16 rounding of
  x_rot swamps the rest; up to 31% error in the MoE output), the same row map for host experts; one block_m for the
  hot and the cold launch (the host rule), so a pair's arithmetic does not depend on where its expert sits;
- the shared Staging / Prefetcher buffers are sized by the largest layer blob; each layer reads them through its
  own layout's views (views_for).
rot=None (every k4row / rtn store) keeps the code path above unchanged.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tcq
from .kernels import block_m_for, copy_blobs, grouped_gemv, moe_blocks, moe_gemm
from .store import ExpertCache, LayerExperts
from .tcq_kernels import rotate, tcq_gemv, tcq_moe_gemm

DECODE_MAX_TOKENS = 8
PREFILL_IMPL = os.environ.get("ENGINE_PREFILL", "gemm")  # "gemm" or "dequant" (the previous path)
PREFILL_CHUNK = 1024  # tokens per grouped-GEMM pass: scratch <= 1024 * top_k * (2I + H) * 4 bytes


class Staging:
    """GPU buffer for one cold blob during prefill, shared by all layers (they run in stream order)."""

    def __init__(self, layout, device):
        self.layout = layout
        self.pool = torch.empty(1, layout.nbytes, dtype=torch.uint8, device=device)
        self.views = layout.views(self.pool)
        self._views: dict = {}

    def views_for(self, layout):
        """The buffer's views for one layer's layout: `views` for the layout it was sized by, else that layout's
        views of its first layout.nbytes bytes (per-layer trellis layouts; design-quality WP15)."""
        if layout == self.layout:
            return self.views
        if layout not in self._views:
            if layout.nbytes > self.pool.shape[1]:
                raise ValueError(f"{layout} needs {layout.nbytes} B, the staging buffer has {self.pool.shape[1]}")
            self._views[layout] = layout.views(self.pool[:, :layout.nbytes])
        return self._views[layout]


class Manager:
    """Calls ExpertCache.update() on every layer once every `interval` decode steps.

    The host never runs more than `max_lead` decode steps ahead of the GPU: before enqueuing a step it waits until
    the GPU has finished the step `max_lead` steps back (blocking-sync events: the waiting thread sleeps). The GPU
    still always has `max_lead` steps queued, so it never starves. ExpertCache.update() reads usage counts one
    interval late (it only consumes a snapshot whose copy has completed by the next update); graph replays cost the
    host almost nothing and some drivers queue very deep (Windows/WDDM: the host was 150 decode steps ahead after
    160 replays), so every update found its snapshot still pending and the cache never changed after the prefill.
    max_lead must be >= 1: there is no unbounded setting.
    A queue of graph replays also hung ROCm decode. On MI250 (ROCm 7.2.3, torch 2.11) a cache fill copied on a side
    stream while 2-3 decode steps were still queued stopped the GPU in the next replay, for good (the host then
    blocks in hipGraphLaunch or in the cache update): Qwen3-30B at max_lead 3 and Qwen3-235B at the default 2 hung
    in every run. How deep a queue it takes grows with the model, so no bound is safe in general; the fills on
    ROCm therefore go on the compute stream (store.fill_stream_kind), and then no run hung, not even with the host
    lead unbounded (tests/test_decode_watchdog.py; measurements in results/polish/fix/)."""

    def __init__(self, interval: int = 4, max_moves: int = 2, max_lead: int = 2):
        self.interval = interval
        self.max_moves = max_moves
        self.max_lead = max_lead  # decode steps the host may run ahead of the GPU (>= 1)
        self.caches: list[ExpertCache] = []
        self.steps = 0
        self.enabled = True
        self.external = False  # True when a graph runner calls step() between replays
        self._lead: list[torch.cuda.Event] = []

    @property
    def max_lead(self) -> int:
        return self._max_lead

    @max_lead.setter
    def max_lead(self, value: int) -> None:
        if int(value) < 1:
            raise ValueError(f"max_lead must be >= 1 (got {value}): an unbounded host lead freezes the expert cache "
                             "on deep-queueing drivers (see Manager)")
        self._max_lead = int(value)

    def step(self) -> None:
        self.steps += 1
        ev = torch.cuda.Event(blocking=True)
        ev.record()
        self._lead.append(ev)
        while len(self._lead) > self._max_lead:
            self._lead.pop(0).synchronize()
        if self.enabled and self.steps % self.interval == 0:
            for c in self.caches:
                c.update(self.max_moves)


class Prefetcher:
    """Shared state for next-layer prefetch: a side stream and two staging buffers (layer parity), so the copy
    for layer L+1 never overwrites the buffer layer L is still reading."""

    def __init__(self, layout, pred_k: int, device):
        self.pred_k = pred_k
        self.layout = layout
        self.stream = torch.cuda.Stream(device)
        self.bufs = [torch.empty(pred_k, layout.nbytes, dtype=torch.uint8, device=device) for _ in range(2)]
        self.views = [layout.views(b) for b in self.bufs]
        self._views: dict = {}
        self.positions = torch.arange(pred_k, device=device, dtype=torch.int32)
        self.enabled = True

    def views_for(self, layout) -> list:
        """[views of bufs[0], views of bufs[1]] for one layer's layout (Staging.views_for). copy_blobs writes a
        layer's blob into the first nbytes of a row, which these views read."""
        if layout == self.layout:
            return self.views
        if layout not in self._views:
            if layout.nbytes > self.bufs[0].shape[1]:
                raise ValueError(f"{layout} needs {layout.nbytes} B, the prefetch buffers have "
                                 f"{self.bufs[0].shape[1]}")
            self._views[layout] = [layout.views(b[:, :layout.nbytes]) for b in self.bufs]
        return self._views[layout]


def _combine_eager(y, weights, t: int, k: int, dtype):
    """sum over a token's k pairs of weight x expert output, float32, cast to the activation dtype."""
    return (y.view(t, k, -1) * weights.view(t, k, 1).float()).sum(1).to(dtype)


_combine_compiled = None


def _combine(y, weights, t, k, dtype):
    """_combine_eager as one fused kernel at decode (torch.compile; three kernels otherwise), eager when compiling
    is off (MOE_TIERING_COMPILE=0) or fails."""
    global _combine_compiled
    if os.environ.get("MOE_TIERING_COMPILE", "1") == "0" or not y.is_cuda:
        return _combine_eager(y, weights, t, k, dtype)
    if _combine_compiled is None:
        try:
            _combine_compiled = torch.compile(_combine_eager, dynamic=False)
        except Exception:  # noqa: BLE001
            _combine_compiled = _combine_eager
    return _combine_compiled(y, weights, t, k, dtype)


class TieredMoE(nn.Module):
    def __init__(self, router: nn.Module, experts: LayerExperts, cache: ExpertCache, top_k: int, staging: Staging,
                 manager: Manager, first: bool, act=F.silu, *, glu=None, gu_bias: torch.Tensor | None = None,
                 dn_bias: torch.Tensor | None = None, predict=None, rot=None):
        super().__init__()
        self.router = router
        self.experts = experts
        self.cache = cache
        self.top_k = top_k
        self.staging = staging
        self.manager = manager
        self.first = first
        self.act = act
        dev = cache.device
        # generalized expert math (module docstring); plain attributes: no state-dict keys
        if glu is None and act is None:
            raise ValueError("TieredMoE needs act (default gate) or glu")
        self.glu = glu
        n_exp, n_gu, n_dn = experts.num_experts, experts.hot_layout.gu_shape[0], experts.hot_layout.dn_shape[0]
        for name, t, n in (("gu_bias", gu_bias, n_gu), ("dn_bias", dn_bias, n_dn)):
            if t is not None and (tuple(t.shape) != (n_exp, n) or t.dtype != torch.float32
                                  or t.device.type != torch.device(dev).type):
                raise ValueError(f"{name} must be float32 [{n_exp}, {n}] on {dev}, got {t.dtype} "
                                 f"{tuple(t.shape)} on {t.device}")
        self.gu_bias = gu_bias
        self.dn_bias = dn_bias
        self.predict = predict
        self.rot = None  # trellis blobs (module docstring): (signs_gu, b_gu, signs_dn, b_dn)
        if rot is not None or isinstance(experts.cold_layout, tcq.TcqLayout) \
                or isinstance(experts.hot_layout, tcq.TcqLayout):
            self._init_tcq(rot, dev)
        self._tok = {t: torch.arange(t, device=dev, dtype=torch.int32).repeat_interleave(top_k)
                     for t in range(1, DECODE_MAX_TOKENS + 1)}
        self._pairs = torch.arange(top_k * DECODE_MAX_TOKENS, device=dev, dtype=torch.int32)
        self.hit_count = torch.zeros((), dtype=torch.float64, device=dev)  # read lazily, no sync on the hot path
        self.staged_count = torch.zeros((), dtype=torch.float64, device=dev)
        self.copied_count = torch.zeros((), dtype=torch.float64, device=dev)
        self.pair_count = 0
        # next-layer prefetch (wired up by build(): .prefetcher, .index, .next, .prev)
        self.prefetcher: Prefetcher | None = None
        self.index = 0
        self.next: TieredMoE | None = None
        self.prev: TieredMoE | None = None
        self.staged_of = torch.full((experts.num_experts,), -1, dtype=torch.int32, device=dev)
        self.copy_done = torch.cuda.Event()
        self.neg = torch.full((top_k * DECODE_MAX_TOKENS,), -1, dtype=torch.int32, device=dev)
        # Persistent inputs of the side-stream copy. They must outlive the call: a temporary allocated on the
        # main stream would be recycled by the caching allocator before the side stream has read it.
        self.pf_pred = torch.zeros(top_k * 2, dtype=torch.int32, device=dev)
        self.pf_want = torch.zeros(top_k * 2, dtype=torch.int32, device=dev)
        # The split (design-quality WP12): experts.host_row maps expert -> row of the partial host tensors (-1 for
        # a frozen expert, which always holds a cache slot). Every host read goes through it: the zero-copy GEMV
        # (one GPU gather, graph-safe), prefill (the cold launch runs on row-renumbered blocks), the prefetch copies
        # (without the map a staged pair would read another expert's bytes) and the cache fills (store.py).
        self.host_row = None
        if experts.host_row is not None:
            hr = experts.host_row.to(torch.long)
            n_rows = int((hr >= 0).sum())
            self.host_row = hr.to(dev, torch.int32)
            self.n_rows = n_rows
            row_expert = torch.empty(n_rows, dtype=torch.long)
            row_expert[hr[hr >= 0]] = torch.nonzero(hr >= 0).flatten()
            self.row_expert = row_expert.to(dev)  # expert of each host row
            self.cold_id = torch.where(hr >= 0, hr, n_rows).to(dev)  # prefill: row id, n_rows for frozen experts
            self.pf_rows = torch.zeros(top_k * 2, dtype=torch.int32, device=dev)  # persistent (side-stream input)

    def _init_tcq(self, rot, dev) -> None:
        ex = self.experts
        lay = ex.cold_layout
        if rot is None:
            raise ValueError("trellis (tcq) expert blobs need rot=(signs_gu, b_gu, signs_dn, b_dn) "
                             "(moe_tiering.experts.tcq_rotation)")
        if not isinstance(lay, tcq.TcqLayout) or ex.hot_layout != lay:
            raise NotImplementedError(f"rot= runs trellis blobs one tier, hot from cold (hot layout == cold layout); "
                                      f"got hot {ex.hot_layout}, cold {lay} (two tiers: design-quality WP18)")
        sg, bg, sd, bd = rot
        for name, s, b, n in (("gu", sg, bg, lay.gu_shape[1]), ("dn", sd, bd, lay.dn_shape[1])):
            if (s.dtype != torch.float32 or tuple(s.shape) != (n,) or s.device.type != torch.device(dev).type
                    or n % int(b)):
                raise ValueError(f"rot {name}: signs must be float32 [{n}] on {dev} and the block divide {n}, got "
                                 f"{s.dtype} {tuple(s.shape)} on {s.device}, block {b}")
        self.rot = (sg, int(bg), sd, int(bd))
        # the patterns' packed LUTs, on the GPU before any graph capture; the upload is checked (amendment 6)
        self.tcq_lut = {}
        for p in (lay.gu_pattern, lay.dn_pattern):
            p = tcq.parse_pattern(p)
            if p not in self.tcq_lut:
                lut = tcq.lut_packed(p, dev)
                ref = tcq.lut(p, "cpu").contiguous().view(torch.int32).view(-1)
                if not torch.equal(lut.cpu(), ref):
                    raise RuntimeError(f"LUT {tcq.pattern_name(*p)}: the copy on {dev} differs from the checked file")
                self.tcq_lut[p] = lut

    def _tcq_gemv(self, x, token, slot, *, qweight, scale, pattern, mode, out):
        return tcq_gemv(x, token, slot, qweight, scale, pattern, mode=mode, out=out,
                        lut32=self.tcq_lut[tcq.parse_pattern(pattern)])

    def _tcq_gemm(self, x, order, blocks, slot, *, qweight, scale, pattern, num_experts, out, block_m, mode, in_div):
        return tcq_moe_gemm(x, order, blocks, slot, qweight, scale, pattern, num_experts=num_experts, out=out,
                            block_m=block_m, mode=mode, in_div=in_div, f32=True,
                            lut32=self.tcq_lut[tcq.parse_pattern(pattern)])

    def _tracking(self) -> bool:
        """Whether the decode step must record expert use: only a cache with dynamic slots and a running manager
        reads the counts. A static cache (every slot frozen, or the manager off: every expert resident) skips the
        record and the hit counter, two to four tiny kernels per layer and token."""
        return bool(self.manager.enabled) and len(self.cache.frozen) < self.cache.slots

    @property
    def stats(self) -> dict:
        hit = float(self.hit_count) if self._tracking() else float(self.pair_count)  # static: every pair is a hit
        return {"pairs": self.pair_count, "hit": hit, "staged": float(self.staged_count),
                "copied": float(self.copied_count)}

    def reset_stats(self) -> None:
        for t in (self.hit_count, self.staged_count, self.copied_count):
            t.zero_()
        self.pair_count = 0

    def _prefetch_active(self, t: int) -> bool:
        return self.prefetcher is not None and self.prefetcher.enabled and t == 1

    def prefetch(self, x: torch.Tensor) -> None:
        """Called by the previous layer: predict this layer's experts from its hidden state and copy the cold
        blobs of predicted misses into VRAM on the side stream."""
        pf = self.prefetcher
        k = pf.pred_k
        pred, want = self.pf_pred[:k], self.pf_want[:k]
        if self.predict is not None:  # this layer's own router (moe_tiering.arch.make_probe), no host sync
            pred.copy_(self.predict(x).reshape(-1))
        else:
            b = getattr(self.router, "bias", None)
            logits = F.linear(x, self.router.weight, b)
            pred.copy_(torch.topk(logits.float(), k, dim=-1).indices.reshape(-1))
        want.copy_(self.cache.slot_of[pred] < 0)
        self.staged_of.fill_(-1)
        self.staged_of[pred.long()] = torch.where(want > 0, pf.positions, self.neg[:k])
        rows = pred
        if self.host_row is not None:  # host rows of the predicted experts (a frozen one is never wanted)
            rows = self.pf_rows[:k]
            rows.copy_(self.host_row[pred.long()])
        main = torch.cuda.current_stream(x.device)
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(pf.stream):
            pf.stream.wait_event(ready)
            copy_blobs(self.experts.cold, pf.bufs[self.index % 2], rows, want)
            self.copy_done.record(pf.stream)
        self.copied_count += want.sum()

    @staticmethod
    def _dequant(views, layout, idx: int, which: str, dtype):
        return layout.dequant(views, idx, which, dtype)

    def _gate(self, h: torch.Tensor, expert: torch.Tensor, dtype) -> torch.Tensor:
        """float32 gate_up outputs h [pairs, 2I] (scratch, changed in place) of experts `expert` [pairs] -> the down
        projection's input [pairs, I] in the activations' dtype. glu=None and no gu_bias: the default gate, as
        before WP8."""
        if self.gu_bias is not None:
            h += self.gu_bias[expert]
        if self.glu is None:
            inter = h.shape[1] // 2
            return (self.act(h[:, :inter]) * h[:, inter:]).to(dtype)
        return self.glu(h).to(dtype).contiguous()

    def _down_bias(self, y: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        """y [pairs, H] float32 (scratch) += dn_bias[expert], before the routing weights (gpt-oss :107-110)."""
        if self.dn_bias is not None:
            y += self.dn_bias[expert]
        return y

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        _, weights, idx = self.router(x)
        if x.shape[0] <= DECODE_MAX_TOKENS:
            if self.first and not self.manager.external:
                self.manager.step()
            out = self._decode(x, weights, idx)
        else:
            out = self._prefill(x, weights, idx)
        return out.reshape(shape)

    def _decode(self, x, weights, idx):
        t = x.shape[0]
        k = self.top_k
        n_pairs = t * k
        cache, ex = self.cache, self.experts
        hot_l, cold_l = ex.hot_layout, ex.cold_layout
        flat = idx.reshape(-1)
        tok = self._tok[t]
        pairs = self._pairs[:n_pairs]
        prefetch = self._prefetch_active(t)
        if prefetch and self.next is not None:
            self.next.prefetch(x)  # overlaps this layer's compute with the next layer's cold-blob copies

        # three weight sources, selected per pair; -1 means "not from this source"
        hot_idx = cache.slot_of[flat]
        hv, cv = cache.views, ex.cold_views
        stg_idx = None
        if self.host_row is not None and self.n_rows == 0:
            # every expert frozen in VRAM: no host rows, nothing to read from them (and no miss to resolve)
            miss = None
            srcs = [(hot_idx, hv, hot_l)]
        else:
            miss = hot_idx < 0
            neg = self.neg[:n_pairs]
            rows = flat.to(torch.int32) if self.host_row is None else self.host_row[flat]  # host row of each pair
            if prefetch and self.prev is not None:
                torch.cuda.current_stream(x.device).wait_event(self.copy_done)
                stg = self.staged_of[flat]
                stg_idx = torch.where(miss, stg, neg)
                host_idx = torch.where(miss & (stg < 0), rows, neg)
                sv = self.prefetcher.views_for(cold_l)[self.index % 2]
            else:
                host_idx = torch.where(miss, rows, neg)
            srcs = [(hot_idx, hv, hot_l), (host_idx, cv, cold_l)]
            if stg_idx is not None:
                srcs.append((stg_idx, sv, cold_l))

        gemv, xin = grouped_gemv, x
        if self.rot is not None:  # trellis: rotated-domain GEMV on float32 x_rot (module docstring)
            gemv, xin = self._tcq_gemv, rotate(x, self.rot[0], self.rot[1])
        h = torch.zeros(n_pairs, hot_l.gu_shape[0], device=x.device, dtype=torch.float32)
        for sel, v, lay in srcs:
            gemv(xin, tok, sel, **lay.gemv_args(v, "gu"), mode=1, out=h)
        a = self._gate(h, flat, x.dtype)
        if self.rot is not None:
            a = rotate(a, self.rot[2], self.rot[3])  # a_rot per pair, float32
        y = torch.zeros(n_pairs, hot_l.dn_shape[0], device=x.device, dtype=torch.float32)
        for sel, v, lay in srcs:
            gemv(a, pairs, sel, **lay.gemv_args(v, "dn"), mode=1, out=y)
        y = self._down_bias(y, flat)
        out = _combine(y, weights, t, k, x.dtype)

        if self._tracking() and miss is not None:
            cache.record(flat)
            self.hit_count += (~miss).sum()
            if stg_idx is not None:
                self.staged_count += (stg_idx >= 0).sum()
        self.pair_count += n_pairs
        return out

    def _prefill(self, x, weights, idx):
        if PREFILL_IMPL == "dequant":
            return self._prefill_dequant(x, weights, idx)
        cache, ex = self.cache, self.experts
        counts = torch.zeros(ex.num_experts, dtype=torch.long, device=x.device)
        out = torch.empty(x.shape[0], ex.hot_layout.dn_shape[0], device=x.device, dtype=x.dtype)
        for t0 in range(0, x.shape[0], PREFILL_CHUNK):  # bounds the [pairs, 2I] / [pairs, H] float32 scratch
            t1 = min(x.shape[0], t0 + PREFILL_CHUNK)
            out[t0:t1] = self._prefill_gemm(x[t0:t1], weights[t0:t1], idx[t0:t1], counts)
        cache.counts += counts.float()
        if cache.free:  # free slots (a cold start): fill them with this prompt's busiest experts, as before
            c = counts.tolist()
            cache.warm(sorted((e for e, n in enumerate(c) if n), key=lambda e: -c[e]))
        return out

    def _prefill_gemm(self, x, weights, idx, counts):
        """Grouped dequant-GEMM over the (token, expert) pairs of a chunk, no host synchronization: every expert's
        weights are read once per block of up to block_m of its pairs, from the VRAM cache (hot) or zero-copy from
        its cold blob, exactly the sources and precisions the decode path uses."""
        t, k = idx.shape
        cache, ex = self.cache, self.experts
        hot_l, cold_l = ex.hot_layout, ex.cold_layout
        n_exp = ex.num_experts
        flat = idx.reshape(-1)
        on_host = ex.cold.device.type == "cpu"
        tq = self.rot is not None
        srcs = []  # (mode, views, layout, block_m, order, blocks, slot table, experts): hot cache, then cold blobs
        for mode, v, lay, host in ((1, cache.views, hot_l, False), (2, ex.cold_views, cold_l, on_host)):
            # trellis: one block_m (the host rule) for both launches, so a pair's bits do not depend on its source
            bm = block_m_for(t * k, n_exp, on_host if tq else host)
            if mode == 2 and self.host_row is not None:
                # split: the cold launch runs on blocks of host ROWS (a frozen expert -> n_rows, past the last row:
                # skipped); per expert the same pairs in the same order and block_m as with expert ids, so every
                # pair's arithmetic is unchanged. Its slot table is indexed by row (rows whose expert is cached skip).
                if self.n_rows:
                    order, blocks, _ = moe_blocks(self.cold_id[flat], self.n_rows + 1, bm)
                    srcs.append((mode, v, lay, bm, order, blocks, cache.slot_of[self.row_expert], self.n_rows))
                continue
            if srcs and srcs[0][3] == bm:
                order, blocks = srcs[0][4], srcs[0][5]
            else:
                order, blocks, cnt = moe_blocks(flat, n_exp, bm)
                if not srcs:
                    counts += cnt
            srcs.append((mode, v, lay, bm, order, blocks, cache.slot_of, n_exp))
        gemm, xin = moe_gemm, x
        if tq:  # trellis: x_rot once, float32 GEMM operands (amendment 8)
            gemm, xin = self._tcq_gemm, rotate(x, self.rot[0], self.rot[1])
        h = torch.empty(t * k, hot_l.gu_shape[0], device=x.device, dtype=torch.float32)
        for mode, v, lay, bm, order, blocks, sl, ne in srcs:
            gemm(xin, order, blocks, sl, **lay.gemv_args(v, "gu"), num_experts=ne, out=h,
                 block_m=bm, mode=mode, in_div=k)
        a = self._gate(h, flat, x.dtype)  # h rows are pairs in `flat` order (moe_gemm writes row = pair)
        del h, xin
        if tq:
            a = rotate(a, self.rot[2], self.rot[3])  # a_rot per pair, float32
        y = torch.empty(t * k, hot_l.dn_shape[0], device=x.device, dtype=torch.float32)
        for mode, v, lay, bm, order, blocks, sl, ne in srcs:
            gemm(a, order, blocks, sl, **lay.gemv_args(v, "dn"), num_experts=ne, out=y,
                 block_m=bm, mode=mode, in_div=1)
        y = self._down_bias(y, flat)
        return (y.view(t, k, -1) * weights.view(t, k, 1).float()).sum(1).to(x.dtype)

    def _prefill_dequant(self, x, weights, idx):
        """The previous prefill (PREFILL_IMPL = "dequant"): every used expert dequantized to bf16, one matmul each
        (one host synchronization per expert). Kept as the reference for the GEMM path."""
        out = torch.zeros(x.shape[0], self.experts.hot_layout.dn_shape[0], device=x.device, dtype=torch.float32)
        cache, ex = self.cache, self.experts
        counts = torch.bincount(idx.reshape(-1), minlength=ex.num_experts).tolist()
        order = sorted((e for e, c in enumerate(counts) if c), key=lambda e: -counts[e])
        inter = ex.hot_layout.gu_shape[0] // 2
        for e in order:
            tok, j = torch.where(idx == e)
            s = cache.cached.get(e, -1)
            if s >= 0:
                views, lay, i = cache.views, ex.hot_layout, s
            else:  # a per-layer layout fills the first nbytes of the shared (largest-blob) buffer
                self.staging.pool[0, :ex.cold_layout.nbytes].copy_(ex.cold[ex.row(e)], non_blocking=True)
                views, lay, i = self.staging.views_for(ex.cold_layout), ex.cold_layout, 0
            wg = self._dequant(views, lay, i, "gu", x.dtype)
            wd = self._dequant(views, lay, i, "dn", x.dtype)
            hgu = x[tok] @ wg.t()
            if self.glu is None and self.gu_bias is None:
                a = self.act(hgu[:, :inter]) * hgu[:, inter:]
            else:  # generalized math: on the float32 h, as the GEMV / GEMM paths
                a = self._gate(hgu.float(), torch.full_like(tok, e), x.dtype)
            yd = self._down_bias((a @ wd.t()).float(), torch.full_like(tok, e))
            out.index_add_(0, tok, yd * weights[tok, j, None].float())
        cache.warm(order)
        cache.counts += torch.bincount(idx.reshape(-1), minlength=ex.num_experts).float()
        return out.to(x.dtype)
