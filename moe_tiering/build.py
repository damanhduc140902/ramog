"""Build a tiered engine under a GPU memory budget: from a converted store (build_converted) or by quantizing a
Hugging Face Qwen3-MoE checkpoint in this process (build).

Since the WP9 switch-over (design-core C9) both builders put the engine into transformers' own model through the
experts-only hook (design-core A1): every MoE layer's `experts` module becomes a TieredExperts around a TieredMoE
core, and routers, shared experts and attention stay transformers code.
- build_converted(store, ...) delegates to moe_tiering.loader.build_engine: any architecture of moe_tiering.arch.ARCHES, store
  meta.json v1 (moe_tiering.convert before WP9) or v2 (moe_tiering.convert_any).
- build(model_path, ...) loads the checkpoint on the CPU, quantizes every expert into a 4-bit (hot) and a 2-bit
  (cold) blob on the GPU, keeps the blobs in pinned host memory and attaches them with moe_tiering.loader.attach (the same
  cache sizing and Manager / Staging / Prefetcher wiring). Qwen3-MoE only; other models are converted first
  (moe_tiering.convert_any) and built with build_converted. Packed experts can be saved and reloaded (`packed_path`).
Both return (model, info) with info["device"] and info["entry"] (moe_tiering.arch.ArchEntry) next to the v0.1 keys.

ENGINE_HOOK=block (kept for one cycle, design-core WP9) selects the legacy block path instead: TieredMoE replaces the
whole Qwen3-MoE block at layer.mlp (Qwen3-MoE only). build_converted then still loads the dense part through
moe_tiering.loader.load_dense, whose buffer sequence (transformers' initialize_weights) replaces the old rotary re-init
`type(rotary)(config=config, device=dev)`, which transformers >= 5.18 refuses (design-core WP9: build.py:196 removed).
"""

from __future__ import annotations

import os
import time

import torch

from . import hostmem, tunecache
from .moe import Manager, Prefetcher, Staging, TieredMoE
from .store import BlobLayout, ExpertCache, LayerExperts, fill_stream_kind

GiB = 1024**3
HOOKS = ("experts", "block")  # ENGINE_HOOK values; "experts" is the default since WP9


def engine_hook() -> str:
    """The integration ENGINE_HOOK selects (read at every build): "experts" (default) or "block" (legacy)."""
    hook = os.environ.get("ENGINE_HOOK", "").strip().lower() or "experts"
    if hook not in HOOKS:
        raise ValueError(f"ENGINE_HOOK={hook!r}: expected one of {HOOKS}")
    return hook


@torch.no_grad()
def pack_layer(gate_up: torch.Tensor, down: torch.Tensor, layout: BlobLayout, device,
               importance: tuple[torch.Tensor, torch.Tensor] | None = None, info: dict | None = None) -> torch.Tensor:
    """gate_up [E, 2I, H], down [E, H, I] (nn.Linear layout) -> uint8 blobs [E, layout.nbytes] on the CPU.
    importance: optional per-expert input-channel importance (gate_up [E, H], down [E, I]) for the k4row search
    (moe_tiering.kquant.importance_from_stats); the rtn format ignores it. info: see BlobLayout.quantize."""
    imp = None if importance is None else tuple(a.to(device) for a in importance)
    blob = layout.quantize(gate_up, down, imp, device=device, info=info).cpu()
    assert blob.shape[1] == layout.nbytes
    return blob


def layout_from_meta(entry: dict, gu_shape, dn_shape) -> BlobLayout:
    """meta.json "hot"/"cold" entry -> BlobLayout. No "format" (or the old "rtn-g64") means rtn."""
    fmt = entry.get("format", "rtn")
    if fmt.startswith("rtn"):
        fmt = "rtn"
    lay = BlobLayout(entry["bits"], entry["group"], tuple(gu_shape), tuple(dn_shape), fmt)
    if "nbytes" in entry and entry["nbytes"] != lay.nbytes:
        raise ValueError(f"meta says {entry['nbytes']} bytes per expert, layout {lay} has {lay.nbytes}")
    return lay


def _moe_blocks(model):
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "experts") and hasattr(mlp, "gate"):
            yield i, layer


def build(model_path: str, device="cuda:0", vram_gb: float = 8.0, hot_bits: int = 4, cold_bits: int = 2,
          group: int = 64, packed_path: str | None = None, log=print):
    from transformers import AutoModelForCausalLM

    from . import arch, loader

    hook = engine_hook()
    t0 = time.time()
    dev = torch.device(device)
    total = torch.cuda.get_device_properties(dev).total_memory

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    cfg = model.config
    entry = arch.arch_for(cfg)
    if entry.model_type != "qwen3_moe":
        raise ValueError(f"build() quantizes Qwen3-MoE checkpoints in-process; {entry.model_type} needs a store: "
                         "python -m moe_tiering.convert_any, then build_converted (eval_engine.py / bench_graph.py "
                         "--converted)")
    hidden, inter, top_k = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok
    hot_l = BlobLayout(hot_bits, group, (2 * inter, hidden), (hidden, inter))
    cold_l = BlobLayout(cold_bits, group, (2 * inter, hidden), (hidden, inter))
    log(f"loaded {model_path} on CPU in {time.time() - t0:.0f}s; expert blob {hot_l.nbytes / 2**20:.2f} MiB hot, "
        f"{cold_l.nbytes / 2**20:.2f} MiB cold")
    sites = {s.layer_idx: s for s in arch.find_moe_sites(model, cfg)} if hook == "experts" else None

    saved = torch.load(packed_path) if packed_path and os.path.exists(packed_path) else None
    layers = {}
    t1 = time.time()
    for i, layer in _moe_blocks(model):
        ex = layer.mlp.experts
        if saved is not None:
            hot, cold = saved[i]
        else:
            hot = pack_layer(ex.gate_up_proj.data, ex.down_proj.data, hot_l, dev)
            cold = pack_layer(ex.gate_up_proj.data, ex.down_proj.data, cold_l, dev)
        layers[i] = LayerExperts(hostmem.pin_copy(hot), hostmem.pin_copy(cold), hot_l, cold_l)  # exact size
        if sites is None:
            layer.mlp.experts = None  # drop the bf16 expert weights
        else:  # the site keeps the module (its class carries the expert math); its bf16 weights go
            ex.to("meta")
            layer.mlp.experts = loader.Removed("experts", sites[i].name)
    if packed_path and saved is None:
        torch.save({i: (le.hot, le.cold) for i, le in layers.items()}, packed_path)
    log(f"packed {len(layers)} layers in {time.time() - t1:.0f}s"
        f"{' (from ' + packed_path + ')' if saved is not None else ''}")

    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()  # quantization scratch is gone; from here on the budget is enforced
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * GiB / total), dev)
    model.to(dev)
    if sites is None:
        model, info = _attach(model, layers, dev, vram_gb, top_k, hot_l, cold_l, log)
        info["entry"], info["model_type"] = entry, entry.model_type
        return model, info
    return loader.attach(model, list(sites.values()), layers, dev, vram_gb, hot_l, cold_l, entry, log)


def _attach(model, layers: dict, dev, vram_gb: float, top_k: int, hot_l: BlobLayout, cold_l: BlobLayout, log):
    """Legacy block path (ENGINE_HOOK=block): size the expert caches to the VRAM left after the dense part and swap
    in TieredMoE blocks."""
    torch.cuda.synchronize(dev)
    dense = torch.cuda.memory_allocated(dev)
    staging = cold_l.nbytes
    # activations, KV cache, prefill dequant scratch, allocator slack; grows with the model/budget
    reserve = int(max(0.75, 0.08 * vram_gb) * GiB)
    free = vram_gb * GiB - dense - staging - reserve
    slots = max(0, int(free // (hot_l.nbytes * len(layers))))
    stream = torch.cuda.Stream(dev) if fill_stream_kind() == "side" else None  # None: fills on the compute stream
    shared = Staging(cold_l, dev)
    manager = Manager()
    prefetcher = Prefetcher(cold_l, top_k, dev)
    moes = []
    for n, (i, layer) in enumerate(_moe_blocks(model)):
        cache = ExpertCache(layers[i], slots, dev, stream)
        manager.caches.append(cache)
        layer.mlp = TieredMoE(layer.mlp.gate, layers[i], cache, top_k, shared, manager, first=n == 0)
        layer.mlp.prefetcher, layer.mlp.index = prefetcher, n
        moes.append(layer.mlp)
    for a, b in zip(moes, moes[1:]):
        a.next, b.prev = b, a
    tuned = tunecache.load(dev)  # saved kernel choices for this GPU and versions: no re-benchmarking
    if tunecache.path():
        log(f"autotune cache: {tuned} saved choices loaded from {tunecache.path()}")
    n_exp = next(iter(layers.values())).num_experts
    host = sum(le.cold.numel() + (le.hot.numel() if le.hot is not le.cold else 0) for le in layers.values())
    log(f"VRAM budget {vram_gb:.1f} GiB: dense {dense / GiB:.2f} GiB, staging {staging / GiB:.2f} GiB, "
        f"cache {slots}/{n_exp} experts per layer ({slots * hot_l.nbytes * len(layers) / GiB:.2f} GiB); "
        f"host blobs {host / GiB:.1f} GiB; expert blob hot {hot_l.format} {hot_l.bits}-bit g{hot_l.group} "
        f"{hot_l.nbytes} B, cold {cold_l.format} {cold_l.bits}-bit g{cold_l.group} {cold_l.nbytes} B; cache fills "
        f"on the {'side' if stream is not None else 'compute'} stream; legacy block path (ENGINE_HOOK=block)")
    return model, {"slots": slots, "num_experts": n_exp, "layers": len(layers), "manager": manager,
                   "cold_nbytes": cold_l.nbytes, "hot_nbytes": hot_l.nbytes, "prefetcher": prefetcher,
                   "device": dev}


def build_converted(packed_dir: str, device="cuda:0", vram_gb: float = 24.0, dense_bits: int | None = 4,
                    hot_from_cold: bool = False, log=print, *, dense_q: str | None = None, frozen=None):
    """Build an engine from a converted store (moe_tiering.convert / moe_tiering.convert_any output) without touching the
    original checkpoint: moe_tiering.loader.build_engine (dense_bits None or 0 keeps the dense part in bf16; dense_q runs
    a packed dense part the store ships, design-quality WP10; frozen: the split's frozen experts, design-quality
    WP12, see moe_tiering.loader.build_engine), or the legacy block path under ENGINE_HOOK=block."""
    from . import loader

    if engine_hook() == "experts":
        model, info = loader.build_engine(packed_dir, device=device, vram_gb=vram_gb, dense_bits=dense_bits or 0,
                                          hot_from_cold=hot_from_cold, dense_q=dense_q, frozen=frozen, log=log)
        _log_dense(info, log)
        return model, info

    t0 = time.time()
    if frozen:
        raise ValueError("frozen experts (the split) run on the experts hook only (unset ENGINE_HOOK)")
    sm = loader.read_store_meta(packed_dir)
    if sm.model_type != "qwen3_moe":
        raise ValueError(f"ENGINE_HOOK=block runs the legacy Qwen3-MoE block only; {sm.root} holds {sm.model_type} "
                         "(unset ENGINE_HOOK)")
    if sm.hot is None and not hot_from_cold:
        raise ValueError(f"{sm.root} is a cold-only store (converted with --cold-only, no hot blobs): build it with "
                         "hot_from_cold=True (--hot-from-cold)")
    dev = torch.device(device)
    total = torch.cuda.get_device_properties(dev).total_memory
    # dense part (bf16, or attention + lm_head quantized from the file in row blocks): the generic loader's, so the
    # buffers are rebuilt by transformers' initialize_weights on every version
    dm = loader.load_dense(sm, dev, dense_bits or 0, dense_q=dense_q, log=log)
    model = dm.model
    log(f"dense loaded in {time.time() - t0:.0f}s (legacy block path)")
    _log_dense(dm.info, log)

    top_k = model.config.num_experts_per_tok
    gu_shape, dn_shape = sm.gu_shape, sm.dn_shape
    cold_l = layout_from_meta(sm.cold, gu_shape, dn_shape)
    if hot_from_cold:  # cache the cold blobs themselves: no second copy in RAM, every expert at cold precision
        hot_l = cold_l
    else:
        hot_l = layout_from_meta(sm.hot, gu_shape, dn_shape)
    n_exp = sm.num_experts
    layers = {}
    t1 = time.time()
    for i, _layer in _moe_blocks(model):
        blobs = {}
        for kind, lay in (("cold", cold_l),) if hot_from_cold else (("hot", hot_l), ("cold", cold_l)):
            # read into a page-aligned buffer that is locked in place at exact size (no power-of-two rounding)
            blobs[kind] = hostmem.read_file(sm.layer_file(i, kind), n_exp * lay.nbytes).view(n_exp, lay.nbytes)
        if hot_from_cold:
            blobs["hot"] = blobs["cold"]
        layers[i] = LayerExperts(blobs["hot"], blobs["cold"], hot_l, cold_l)
        if i % 16 == 0:
            log(f"  experts of layer {i} pinned ({time.time() - t1:.0f}s)")
    fb = hostmem.fallback_bytes()
    log(f"expert blobs: {hostmem.registered_bytes() / GiB:.2f} GiB page-locked at exact size"
        + (f", {fb / GiB:.2f} GiB through pin_memory() (rounded up)" if fb else "")
        + f" ({time.time() - t1:.0f}s)")
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * GiB / total), dev)
    model, info = _attach(model, layers, dev, vram_gb, top_k, hot_l, cold_l, log)
    info.update({"entry": dm.entry, "model_type": dm.entry.model_type, "store": sm.root})
    return model, info


def _log_dense(info: dict, log) -> None:
    """v0.1 build_converted's dense summary line (bench_readme.py's bench parser reads it), from load_dense's facts."""
    if info.get("dense_bits") and info.get("dense_quantized"):
        log(f"dense linears -> {info['dense_bits']}-bit ({len(info['dense_quantized'])} quantized from the file in row "
            f"blocks), saved {info['saved_bytes'] / GiB:.2f} GiB; dense part on the GPU "
            f"{info.get('dense_vram', 0) / GiB:.2f} GiB")


def stats(model) -> dict:
    """Decode-path totals over the TieredMoE cores of any engine model (moe_tiering.arch.tiered: hook or block path):
    expert uses, cache hits, bytes read zero-copy, bytes copied into the cache."""
    from .loader import stats as _stats

    return _stats(model)


def reset_stats(model) -> None:
    from .loader import reset_stats as _reset

    _reset(model)
