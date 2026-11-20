"""Expert blobs straight from the store files into VRAM, through one small reusable pinned staging buffer
(design-quality amendment 1; WP11).

Why: moe_tiering.loader.build_engine reads EVERY layer file into exact-size page-locked host memory before the cache is
filled (hostmem.read_file), also with hot_from_cold and also when every expert ends up resident. A resident store
then needs its whole expert size in pinned RAM (Qwen3-30B H4: 15.27 GiB), which a 16 GB machine (pinned budget
11.5 GiB) does not have, although the rule (agents/qint/qd/autoconfig.py) counts "resident" as 0 GiB pinned. Here the
VRAM-only experts go file -> pinned staging (bounded chunks, one buffer reused) -> their VRAM rows, and nothing else
is pinned; every upload is checked against an exact fingerprint of the host bytes (amendment 6: the MI250 node has
corrupted host -> device copies under concurrent load), re-copied on a mismatch, refused after `tries` attempts.

Interfaces
    stream_rows(path, nbytes, rows, dst, staging, verify=True, tries=3) -> dict
        dst[i] <- row rows[i] of the uint8 [E, nbytes] file (expert e = row e, core's exact-size layer file);
        dst: a CUDA uint8 tensor [len(rows), nbytes] (e.g. the frozen slots of a split: WP12 reads its frozen rows
        with this). Runs of consecutive rows are read with one seek each and copied in pieces of at most the
        staging size (a row bigger than the staging buffer is split).
    PinnedStaging(nbytes)                  one exact-size page-locked buffer (moe_tiering.hostmem), reused for every chunk
    build_engine_resident(store, device, vram_gb, dense_bits, ..., tier="cold") -> (model, info)
        the RESIDENT placement without a host copy: moe_tiering.loader's own steps (load_dense, attach: the engine's own
        cache sizing, amendment 5) with placeholder (meta-device) host blobs, then every layer file streamed into its
        cache pool (slot e = expert e), the caches published as full (cache.cached, slot_of, no free slot), the
        manager and the prefetcher off (nothing can miss). LayerExperts.hot / .cold then alias the VRAM pool, so
        every read the decode, prefill and prefetch code could make reads the right expert's bytes. Refused when the
        engine's sizing at vram_gb gives fewer slots than experts (that is the pinned or split placement).
        tier="cold": the cold blobs (hot from cold; H4 / trellis-style one-tier stores); tier="hot": the hot blobs
        of a two-tier store (regression gate G1's setting: every expert resident at 4-bit).

Values: identical to build_engine + warm every expert (the G1 / --resident setting): the same blobs in VRAM slots,
every pair served from the slot, the same kernels; the cold-source launches select no pair in either case.

moe_tiering.loader uses stream_rows since design-quality WP12: build_engine(frozen=...) streams the frozen experts of a
split into their VRAM slots (moe_tiering.loader.load_frozen), and with every expert of every layer frozen it is this
module's resident placement inside the loader (no host copy, manager and prefetch off; tests/test_split.py checks it
bitwise against build_engine_resident). build_engine_resident stays for tier="hot" and the WP11 tools.
"""

from __future__ import annotations

import gc
import os
import time

import torch

GiB = 1024**3
DEFAULT_STAGING = 64 << 20  # bytes of the one pinned staging buffer


class UploadFailed(RuntimeError):
    """A host -> device copy kept differing from the host bytes (transient platform fault, or a bad device)."""


class PinnedStaging:
    """One page-locked uint8 host buffer of exactly `nbytes` (moe_tiering.hostmem, exact-size registration), reused for
    every chunk of every file. close() unregisters it."""

    def __init__(self, nbytes: int = DEFAULT_STAGING):
        from . import hostmem

        self.nbytes = int(nbytes)
        self.buf = hostmem.pin_copy(torch.empty(self.nbytes, dtype=torch.uint8))
        self.np = self.buf.numpy()

    def close(self) -> None:
        self.buf = None
        self.np = None
        gc.collect()  # hostmem unregisters the buffer when it is collected


def _runs(rows: list[int]) -> list[tuple[int, int, int]]:
    """[(first file row, first dst row, count)] for maximal runs of consecutive file rows."""
    out = []
    for i, r in enumerate(rows):
        if out and r == out[-1][0] + out[-1][2] and i == out[-1][1] + out[-1][2]:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + 1)
        else:
            out.append((r, i, 1))
    return out


def _read_exact(f, offset: int, mv: memoryview) -> None:
    f.seek(offset)
    got = 0
    while got < len(mv):
        n = f.readinto(mv[got:])
        if not n:
            raise EOFError(f"{f.name}: read {got} of {len(mv)} bytes at offset {offset}")
        got += n


@torch.no_grad()
def stream_rows(path: str, nbytes: int, rows, dst: torch.Tensor, staging: PinnedStaging, *, verify: bool = True,
                tries: int = 3, stats: dict | None = None, fault=None) -> dict:
    """Copy rows `rows` of the layer file `path` (uint8 [E, nbytes], row e = expert e) into dst [len(rows), nbytes]
    (CUDA uint8), through `staging`. With verify, each piece's device copy is checked against the fingerprint of the
    host bytes (moe_tiering.tcq_encode.fingerprint: exact, [sum, position-weighted sum] of the 4-byte words; the device
    one is computed on the device), re-copied on a mismatch, UploadFailed after `tries` attempts.
    fault(out, attempt): test hook called after each device copy lands (tests/test_h4_store.py injects corruption)."""
    from .tcq_encode import fingerprint

    rows = [int(r) for r in rows]
    if dst.dtype != torch.uint8 or dst.dim() != 2 or tuple(dst.shape) != (len(rows), nbytes):
        raise ValueError(f"dst must be uint8 [{len(rows)}, {nbytes}], got {dst.dtype} {tuple(dst.shape)}")
    if dst.device.type != "cuda" or not dst.is_contiguous():
        raise ValueError("dst must be a contiguous CUDA tensor")
    size = os.path.getsize(path)
    if size % nbytes:
        raise ValueError(f"{path}: {size} bytes is not a multiple of {nbytes}")
    n_file = size // nbytes
    bad = [r for r in rows if not 0 <= r < n_file]
    if bad:
        raise IndexError(f"{path}: rows {bad[:4]} outside 0..{n_file - 1}")
    st = stats if stats is not None else {}
    st.setdefault("bytes", 0)
    st.setdefault("pieces", 0)
    st.setdefault("retries", 0)
    st.setdefault("seconds_read", 0.0)
    st.setdefault("seconds_copy", 0.0)
    flat = dst.view(-1)
    cap = staging.nbytes - staging.nbytes % 4 if staging.nbytes >= 4 else staging.nbytes
    dev = dst.device
    with open(path, "rb", buffering=0) as f:
        for r0, d0, cnt in _runs(rows):
            off, total = r0 * nbytes, cnt * nbytes
            base = d0 * nbytes
            done = 0
            while done < total:
                n = min(cap, total - done)
                t0 = time.perf_counter()
                host = staging.buf[:n]
                _read_exact(f, off + done, memoryview(staging.np)[:n])
                want = fingerprint(host) if verify else None
                t1 = time.perf_counter()
                out = flat[base + done:base + done + n]
                for attempt in range(tries):
                    torch.cuda.synchronize(dev)  # nothing queued while the copy lands, and the staging is free again
                    out.copy_(host, non_blocking=True)
                    torch.cuda.synchronize(dev)  # the staging buffer is reused by the next piece
                    if fault is not None:
                        fault(out, attempt)
                    if not verify or torch.equal(fingerprint(out).cpu(), want):
                        break
                    st["retries"] += 1
                else:
                    raise UploadFailed(f"{path}: bytes [{off + done}, {off + done + n}) kept differing from the host "
                                       f"copy after {tries} uploads")
                st["seconds_read"] += t1 - t0
                st["seconds_copy"] += time.perf_counter() - t1
                st["bytes"] += n
                st["pieces"] += 1
                done += n
    return st


# ---------------------------------------------------------------------------------------------------------------------
# the resident placement without a host copy


def _placeholder(n_exp: int, nbytes: int) -> torch.Tensor:
    """A [E, nbytes] uint8 tensor on the meta device: LayerExperts / ExpertCache / attach only read its shape (and the
    cold views, built here on meta), so nothing is allocated for the host blobs."""
    return torch.empty((n_exp, nbytes), dtype=torch.uint8, device="meta")


def _proc_status() -> dict:
    """VmHWM / VmRSS / RssAnon / RssFile / VmLck / VmPin of this process in bytes (Linux; {} elsewhere)."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "VmLck", "VmPin"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def _bias_bytes(sites, biases: dict) -> int:
    """VRAM attach allocates for the expert-bias tables (core_kwargs: float32 [E, 2I] + [E, H]) before it measures."""
    n = 0
    for s in sites:
        if s.layer_idx in biases:
            gub, dnb = biases[s.layer_idx]
            n += 4 * (gub.numel() + dnb.numel())
    return n


@torch.no_grad()
def build_engine_resident(store, device="cuda:0", vram_gb: float = 24.0, dense_bits: int = 0, *,
                          tier: str = "cold", embed_host: bool = False, head_host: bool = False,
                          dense_format: str = "rtn", staging_bytes: int = DEFAULT_STAGING, verify: bool = True,
                          dense_q: str | None = None, log=print):
    """Resident engine with no host copy of the experts (module docstring). Returns (model, info) like
    moe_tiering.loader.build_engine, plus info["placement"] = "resident-vram", info["vram_load"] (bytes, retries,
    seconds, the peak page-locked bytes during the load) and info["proc_status"]."""
    from . import formats, hostmem, loader
    from .experts import unsupported_reason
    from .store import LayerExperts

    t0 = time.time()
    sm = loader.read_store_meta(store)
    if tier not in ("cold", "hot"):
        raise ValueError(f"tier {tier!r}: 'cold' (hot from cold) or 'hot'")
    entry = sm.cold if tier == "cold" else sm.hot
    if entry is None:
        raise ValueError(f"{sm.root} has no {tier} blobs")
    dev = torch.device(device)
    total = torch.cuda.get_device_properties(dev).total_memory
    pinned_before = hostmem.registered_bytes()
    dm = loader.load_dense(sm, dev, dense_bits, dense_format=dense_format, embed_host=embed_host,
                           head_host=head_host, dense_q=dense_q, log=log)
    model, sites, arch_entry = dm.model, dm.sites, dm.entry
    bad = {s.layer_idx: unsupported_reason(s.ref) for s in sites if unsupported_reason(s.ref)}
    if bad:
        (layer, why), = list(bad.items())[:1]
        raise NotImplementedError(f"{arch_entry.model_type}: {len(bad)} MoE layers cannot run on TieredMoE "
                                  f"(layer {layer}: {why})")
    gu, dn = sites[0].layout.gu_shape, sites[0].layout.dn_shape
    lay = formats.expert_layout(entry, gu, dn)
    if not formats.servable(lay):
        raise NotImplementedError(f"expert format {lay.format!r} has no decode kernel yet (moe_tiering.formats)")
    n_exp, nb = sm.num_experts, lay.nbytes
    files = {s.layer_idx: sm.layer_file(s.layer_idx, tier) for s in sites}
    for li, p in files.items():
        if os.path.getsize(p) != n_exp * nb:
            raise ValueError(f"{p}: {os.path.getsize(p)} bytes, expected {n_exp} x {nb}")
    layers = {}
    for s in sites:
        ph = _placeholder(n_exp, nb)
        layers[s.layer_idx] = LayerExperts(ph, ph, lay, lay)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * GiB / total), dev)
    biases = {s.layer_idx: loader.read_biases(sm, s) for s in sites if s.layout.bias}

    # the engine's own sizing (moe_tiering.loader.attach = build._attach: dense + one staging blob + reserve) at vram_gb
    torch.cuda.synchronize(dev)
    dense = torch.cuda.memory_allocated(dev) + _bias_bytes(sites, biases)
    per_slot = nb * len(layers)
    reserve = max(0.75, 0.08 * vram_gb) * GiB
    slots_at_budget = max(0, int((vram_gb * GiB - dense - nb - reserve) // per_slot))
    if slots_at_budget < n_exp:
        raise RuntimeError(f"not resident at {vram_gb:g} GiB: the engine's sizing gives {slots_at_budget} cache slots "
                           f"per layer for {n_exp} experts (dense {dense / GiB:.2f} GiB, {per_slot / GiB:.3f} GiB per "
                           "slot); use the pinned or split placement")
    # attach sized for exactly E slots (the budget handed to it is the one that gives E; reserve form as attach's)
    need = dense + nb + n_exp * per_slot + per_slot // 2
    v = (need + 0.75 * GiB) / GiB
    if 0.08 * v > 0.75:
        v = need / GiB / 0.92
    model, info = loader.attach(model, sites, layers, dev, v, lay, lay, arch_entry, log, biases=biases)
    if info["slots"] != n_exp:
        raise RuntimeError(f"attach sized {info['slots']} slots, expected {n_exp} (sizing formula changed?)")

    # stream every layer file into its pool; publish the caches as full and frozen
    staging_bytes = min(int(staging_bytes), n_exp * nb)  # never more than one layer file
    staging = PinnedStaging(staging_bytes)
    peak_pinned = hostmem.registered_bytes()
    st: dict = {}
    t1 = time.time()
    mgr = info["manager"]
    ids = torch.arange(n_exp, device=dev, dtype=torch.int32)
    try:
        for n, (s, cache) in enumerate(zip(sites, mgr.caches, strict=True)):
            le = layers[s.layer_idx]
            if cache.layer is not le or cache.pool.shape != (n_exp, nb):
                raise RuntimeError(f"layer {s.layer_idx}: unexpected cache wiring {tuple(cache.pool.shape)}")
            stream_rows(files[s.layer_idx], nb, range(n_exp), cache.pool, staging, verify=verify, stats=st)
            cache.cached = {e: e for e in range(n_exp)}
            cache.free = []
            cache.slot_of.copy_(ids)
            cache.score += 1.0  # as ExpertCache.warm
            cache.fill_bytes += n_exp * nb
            le.hot = cache.pool  # the host-blob aliases now point at the VRAM pool (row e = expert e)
            le.cold = cache.pool
            le.cold_views = lay.views(cache.pool)
            if n % 16 == 0:
                log(f"  experts of layer {s.layer_idx} streamed into VRAM ({time.time() - t1:.0f}s)")
            peak_pinned = max(peak_pinned, hostmem.registered_bytes())
    finally:
        staging.close()
    mgr.enabled = False  # nothing to move: every expert holds a slot, nothing is ever evicted
    pf = info.get("prefetcher")
    if pf is not None:
        pf.enabled = False  # no pair can miss
    torch.cuda.synchronize(dev)
    gib = st.get("bytes", 0) / GiB
    log(f"expert blobs: {gib:.2f} GiB streamed into VRAM through a {staging_bytes / 2**20:.0f} MiB pinned staging "
        f"buffer in {time.time() - t1:.0f}s ({st.get('pieces', 0)} pieces, {st.get('retries', 0)} re-copies, "
        f"verify {'on' if verify else 'off'}); page-locked now {hostmem.registered_bytes() / GiB:.3f} GiB")
    info.update(dm.info)
    info.update({"store": sm.root, "store_version": sm.version, "seconds": round(time.time() - t0, 1),
                 "placement": "resident-vram", "tier": tier, "slots_at_budget": slots_at_budget,
                 "vram_load": {**st, "staging_bytes": int(staging_bytes), "verify": bool(verify),
                               "peak_pinned_bytes_during_load": int(peak_pinned - pinned_before),
                               "pinned_bytes_after_load": int(hostmem.registered_bytes() - pinned_before),
                               "seconds": round(time.time() - t1, 1)},
                 "proc_status": _proc_status()})
    return model, info
