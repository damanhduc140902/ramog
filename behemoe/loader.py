"""Generic engine builder: a converted store -> transformers text model with TieredExperts (design-core B3; WP4).

    model, info = build_engine(store, device, vram_gb, dense_bits=4, hot_from_cold=False, embed_host=False,
                               head_host=False, dense_q=None)

Since the WP9 switch-over this is the default build path: behemoe/build.py's build_converted delegates here (and
build() attaches through attach()), so eval_engine.py, bench_graph.py and the tools run on it; ENGINE_HOOK=block keeps
the legacy Qwen3-MoE block for one cycle (behemoe/build.py).

Steps (load_dense, then attach):
1. Store metadata (read_store_meta): meta.json v1 (behemoe.convert, Qwen3-MoE: loads as qwen3_moe with every layer
   MoE) or v2 (behemoe.convert_any, WP5). The config is rebuilt from the dict stored in meta.json (a composite VLM
   config for VLM checkpoints); transformers builds the text model only (behemoe.arch.build_meta).
2. Meta build and MoE sites (behemoe.arch.find_moe_sites, three checks); the store's MoE layers, expert count and
   matrix shapes must match the sites.
3. Every site's experts module, every dense nn.Linear the 4-bit plan selects (behemoe.arch.dense_quant_plan) and,
   with embed_host, the input embedding are replaced by placeholders, so nothing is allocated for them.
4. dtype rule (design-core A7): a tensor matching transformers' dtype plan for a bf16 load (the strict-fp32 list)
   gets that dtype, every other tensor keeps the meta parameter's dtype (core_model_loading 5.19 :1698-1703). The
   meta tensors get their final dtype, then model.to_empty(device) allocates everything at once; ties (lm_head =
   embedding) are restored by name.
5. dense.safetensors is streamed one tensor at a time (peak host RAM: one tensor), cast to its load dtype on the
   CPU, copied into place. Unexpected keys and shape mismatches raise.
6. Buffers, from_pretrained's own sequence (design-core A6): every parameter or buffer that was not loaded must be a
   non-persistent buffer or a tied alias, else RuntimeError; non-persistent buffers were allocated by to_empty
   (empty_like); every loaded tensor is flagged `_is_hf_initialized`; then transformers' initialize_weights()
   rebuilds the rest (rotary inv_freq, Gemma-4's embed_scale and per-layer-type inv_freq, ...).
7. Dense 4-bit modules are quantized from the file in row blocks on the GPU (behemoe.formats dense format, default
   QuantLinear rtn 4-bit g64, as build_converted does); a tied head is quantized from the embedding table, before the
   table moves to pinned host memory (HostEmbedding, behemoe/hostemb.py). head_host keeps the quantized head's
   packed arrays in pinned host memory, read zero-copy.
8. attach: expert blobs are read into exact-size pinned memory (behemoe.hostmem), layouts come from the format
   registry (behemoe.formats), the VRAM cache is sized exactly as build._attach does, and each site's experts module
   becomes a TieredExperts around a TieredMoE core whose expert math is the replaced module's (behemoe.experts
   core_kwargs, WP8: act_fn or the module's own _apply_gate as glu; expert biases from layer{L:03d}.bias.safetensors
   as float32 VRAM tables). Prefetch is on for every architecture (WP8): Qwen3-MoE keeps TieredMoE's built-in
   predictor, the others predict through the next layer's own router (arch.make_probe as TieredMoE.predict);
   prev/next links run over MoE sites only.
   Per-layer layouts (design-quality WP14): expert_layouts gives every site its own layouts (one shared object for
   k4row / rtn, as before; one behemoe.store_v2.TcqStoreLayout per layer for a trellis store, checked first by
   behemoe.store_v2.check), read_experts pins each layer file at its own size, and attach sizes the cache with
   cache_slots (slot cost = sum of the layers' hot blob sizes, one staging blob of the largest cold layout) and
   allocates the shared Staging / Prefetcher buffers for the largest blob. Trellis stores (design-quality WP15) run
   one tier, hot from cold: each core gets its layer's rotation (behemoe.experts.rotation_kwargs) and reads the shared
   buffers through its own layout's views (behemoe.moe Staging / Prefetcher views_for).
   Host tail guard (design-core B3; WP7 fixed the kernels, rtn and k4row register host_words=1): a cold layout of a
   format that still declares host_words > 1 whose rows are not a multiple of it is refused unless every expert fits
   in the VRAM cache; then every expert is made resident.

The split (design-quality WP12; frozen=None | F | {layer: [expert ids]}, hot from cold only): F experts per layer are
   FROZEN in VRAM (slots 0..F-1 of the layer's cache, published at construction, never evicted; by default the F most
   routed by counts.safetensors, behemoe.store_v2.frozen_sets). PARTIAL HOST LOAD: their rows of layer{L:03d}.cold.bin
   go file -> one reusable pinned staging buffer -> their VRAM slots (behemoe.vram_load.stream_rows, every piece
   fingerprint-checked, amendments 1 and 6), the other E - F rows into one exact-size pinned [E - F, nbytes_L] tensor
   (behemoe.hostmem.read_rows), with LayerExperts.host_row int32 [E] (row of expert e there, -1 for a frozen expert);
   every host read of the engine goes through host_row (behemoe/moe.py, behemoe/store.py). The dynamic LFU runs on the
   other slots over the other experts. F = E for every layer is the resident placement with no host copy of any
   expert (the manager and the prefetcher are then off). frozen=None (or F = 0) is today's full load, byte for byte.

Packed dense part (design-quality WP10; dense_q="dense-hi" | "dense-c" | a directory name, instead of dense_bits):
   a store v2 lists its packed dense parts in meta.json "dense_q" (amendment 2: "dense" keeps core's meaning, the
   dense.safetensors writer's entry): [{dir, recipe, format "moe-tiering/dense-q", version 1, gib, meta_sha256?}].
   open_dense_q checks format, version, recipe and the directory meta.json's sha256 (when recorded), then reads it
   with behemoe.dense_recipe.PackedDense. Every packed linear (attention, lm_head, dense MLPs, shared experts, ...)
   becomes the registered dense-q kind's module (behemoe.formats: k4row -> behemoe.klinear.KLinear), the input
   embedding a QEmbedding (in VRAM, or in pinned host memory with embed_host); their weights are never read from
   dense.safetensors (the bf16 embedding table is read on the CPU for initialize_weights, then dropped). Routers and
   embeddings other than the input embedding stay as dense.safetensors holds them (design-core B3). Every packed
   array goes up through a fingerprint-checked copy (amendment 6). head_host with a packed head is refused (KLinear
   has no host variant).

Host tensors (ArchEntry.host_keys: DeepSeek-V4.1's engram n-gram tables, 189 GiB of FP8 + E8M0): meta.json "host"
   {file: host.safetensors, keys} (behemoe.convert_any). They are taken out of the meta model before step 4 (never
   cast or allocated on the device), and bound after step 5 as zero-copy views of a private memory map of the file
   (behemoe.ckpt.mmap_file) in their release dtype: pages are read on first touch and live in the page cache, not in
   pinned memory; the model's own forward gathers rows on the CPU (behemoe/models/deepseek_v41.py). A model with host
   keys and a store without them (or with other keys) is refused.

meta.json v2 fields read here (the converter, WP5, writes them; v1 has none of the starred ones):
    format* = 2, arch* (text model_type), transformers* (version used to convert), config (dict as in config.json),
    source, num_experts, moe_layers* (decoder layer indices), gu_shape, dn_shape, hot ({format, bits, group,
    nbytes, ...} or null), cold, experts* {E, gu_shape, dn_shape, gated, bias, transposed, concatenated}.
Files: dense.safetensors (module key names, source dtype), layer{L:03d}.cold.bin / .hot.bin (uint8 [E, nbytes]),
layer{L:03d}.bias.safetensors (gpt-oss: gate_up_proj_bias [E, 2I], down_proj_bias [E, H]; read by build_engine).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from . import arch, formats, hfcompat

GiB = 1024**3


# ---------------------------------------------------------------------------------------------------------------------
# store metadata


@dataclass
class StoreMeta:
    root: str
    version: int
    raw: dict
    config: object  # transformers PretrainedConfig (composite for VLM checkpoints)
    model_type: str  # text model type
    num_experts: int
    gu_shape: tuple[int, int]
    dn_shape: tuple[int, int]
    hot: dict | None
    cold: dict
    moe_layers: list[int]
    transformers: str | None = None

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def layer_file(self, layer: int, kind: str) -> str:
        return self.path(f"layer{layer:03d}.{kind}.bin")


def config_from_dict(d: dict):
    """The config transformers would build from this config.json dict (AutoConfig.from_pretrained does
    config_class.from_dict(config_dict))."""
    from transformers import CONFIG_MAPPING

    mt = d.get("model_type")
    if mt not in CONFIG_MAPPING:
        raise ValueError(f"model_type {mt!r} unknown to transformers {hfcompat.TF_VERSION}")
    return CONFIG_MAPPING[mt].from_dict(dict(d))


def read_store_meta(store: str) -> StoreMeta:
    root = os.path.abspath(store)
    with open(os.path.join(root, "meta.json")) as f:
        raw = json.load(f)
    version = int(raw.get("format", 1))
    if version not in (1, 2):
        raise ValueError(f"{root}: meta.json format {version} is not supported (1 or 2)")
    cfg = config_from_dict(raw["config"])
    text = arch.text_config(cfg)
    if version == 1:
        if text.model_type != "qwen3_moe":
            raise ValueError(f"{root}: a v1 store (behemoe.convert) holds Qwen3-MoE, its config says {text.model_type!r}")
        n = text.num_hidden_layers
        moe_layers = list(range(n))  # behemoe.convert wrote a blob pair for every layer
        if arch.config_sparse_layers(cfg) != moe_layers:
            raise ValueError(f"{root}: v1 stores have every layer MoE; the config's sparse layers differ")
    else:
        if raw.get("arch") not in (None, text.model_type):
            raise ValueError(f"{root}: meta arch {raw.get('arch')!r} != config text model_type {text.model_type!r}")
        moe_layers = [int(x) for x in raw["moe_layers"]]
    ex = raw.get("experts", {})
    gu = tuple(raw.get("gu_shape") or ex["gu_shape"])
    dn = tuple(raw.get("dn_shape") or ex["dn_shape"])
    n_exp = int(raw.get("num_experts") or ex["E"])
    return StoreMeta(root, version, raw, cfg, text.model_type, n_exp, gu, dn, raw.get("hot"), raw["cold"],
                     moe_layers, raw.get("transformers"))


def _check_sites(sm: StoreMeta, sites: list) -> None:
    got = [s.layer_idx for s in sites]
    want = sm.moe_layers
    if got != want:
        raise ValueError(f"{sm.root}: store MoE layers {want} != the model's MoE sites {got}")
    for s in sites:
        lay = s.layout
        if (lay.E, lay.gu_shape, lay.dn_shape) != (sm.num_experts, sm.gu_shape, sm.dn_shape):
            raise ValueError(f"{sm.root}: layer {s.layer_idx} experts E={lay.E} gu={lay.gu_shape} dn={lay.dn_shape}, "
                             f"store E={sm.num_experts} gu={sm.gu_shape} dn={sm.dn_shape}")


# ---------------------------------------------------------------------------------------------------------------------
# module surgery helpers


class Removed(nn.Module):
    """Placeholder for a module the loader took out (nothing is allocated for it); calling it is an error."""

    def __init__(self, what: str, name: str):
        super().__init__()
        self.what, self.name = what, name

    def forward(self, *args, **kwargs):
        raise RuntimeError(f"{self.name} was removed by behemoe.loader ({self.what}) and not replaced")

    def extra_repr(self) -> str:
        return f"{self.what}: {self.name}"


def _split(name: str) -> tuple[str, str]:
    return name.rsplit(".", 1) if "." in name else ("", name)


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, attr = _split(name)
    setattr(model.get_submodule(parent), attr, module)


def _module_name(model: nn.Module, target: nn.Module) -> str:
    for n, m in model.named_modules(remove_duplicate=False):
        if m is target:
            return n
    raise LookupError(f"{type(target).__name__} not found")


def _tie_groups(model: nn.Module) -> list[list[str]]:
    """Names that share one Parameter object (e.g. lm_head.weight and model.embed_tokens.weight), canonical first."""
    by_id: dict[int, list[str]] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(p), []).append(n)
    return [names for names in by_id.values() if len(names) > 1]


def _get_tensor(model: nn.Module, name: str):
    parent, attr = _split(name)
    mod = model.get_submodule(parent)
    if attr in mod._parameters:
        return mod._parameters[attr]
    if attr in mod._buffers:
        return mod._buffers[attr]
    return None


def _set_param(model: nn.Module, name: str, p: nn.Parameter) -> None:
    parent, attr = _split(name)
    model.get_submodule(parent)._parameters[attr] = p


def _take_host_params(model: nn.Module, entry, sm: StoreMeta) -> dict[str, nn.Parameter]:
    """Remove the host tensors (ArchEntry.host_keys) from the meta model; returns {name: meta parameter}. The store
    must list exactly these keys in meta.json "host"."""
    import re

    rx = [re.compile(p) for p in getattr(entry, "host_keys", ())]
    names = [n for n, _ in model.named_parameters(remove_duplicate=False) if any(r.search(n) for r in rx)]
    want = (sm.raw.get("host") or {}).get("keys")
    if names or want:
        if not want:
            raise ValueError(f"{sm.root}: {entry.model_type} keeps {len(names)} tensors in host memory ({names[:2]}), "
                             "the store has none (meta.json \"host\"): reconvert with this engine")
        if sorted(want) != sorted(names):
            raise ValueError(f"{sm.root}: meta.json host keys {sorted(want)[:4]} differ from the model's {names[:4]}")
    out = {}
    for n in names:
        parent, attr = _split(n)
        out[n] = model.get_submodule(parent)._parameters.pop(attr)
    return out


def _bind_host_params(model: nn.Module, sm: StoreMeta, params: dict[str, nn.Parameter]) -> int:
    """Bind each host tensor as a zero-copy view of host.safetensors (behemoe.ckpt.mmap_file), in the meta
    parameter's dtype and shape; returns the bytes mapped."""
    if not params:
        return 0
    from . import ckpt

    maps = ckpt.mmap_file(sm.path(sm.raw["host"]["file"]))
    nbytes = 0
    for n, meta_p in params.items():
        v = maps.get(n)
        want = meta_p.numel() * meta_p.element_size()
        if v is None or v.numel() * v.element_size() != want or v.element_size() != meta_p.element_size():
            got = None if v is None else (str(v.dtype), list(v.shape))
            raise RuntimeError(f"{sm.root}: host tensor {n}: file has {got}, model {meta_p.dtype} {list(meta_p.shape)}")
        _set_param(model, n, nn.Parameter(v.view(meta_p.dtype).view(meta_p.shape), requires_grad=False))
        nbytes += want
    return nbytes


def _apply_dtype_rule(model: nn.Module, plan: hfcompat.DtypePlan, prefix: str = "") -> dict[str, torch.dtype]:
    """Give every persistent tensor (parameters and persistent buffers, on meta) its load dtype; returns the changes."""
    changed = {}
    for mname, mod in model.named_modules():
        base = f"{prefix}{mname}." if mname else prefix
        for pname, p in list(mod._parameters.items()):
            if p is None:
                continue
            want = hfcompat.load_dtype(plan, base + pname, p.dtype)
            if want != p.dtype:
                mod._parameters[pname] = nn.Parameter(torch.empty_like(p, dtype=want), requires_grad=p.requires_grad)
                changed[base + pname] = want
        for bname, b in list(mod._buffers.items()):
            if b is None or bname in mod._non_persistent_buffers_set:
                continue
            want = hfcompat.load_dtype(plan, base + bname, b.dtype)
            if want != b.dtype:
                mod._buffers[bname] = torch.empty_like(b, dtype=want)
                changed[base + bname] = want
    return changed


@dataclass
class _Deferred:
    name: str
    has_bias: bool
    dtype: torch.dtype
    in_features: int
    out_features: int


@dataclass
class DenseModel:
    """What load_dense returns: the model with every dense tensor loaded and the experts still placeholders."""

    model: nn.Module
    entry: arch.ArchEntry
    sites: list
    store: StoreMeta
    device: torch.device
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------------
# dense part


@torch.no_grad()
def load_dense(store, device="cuda:0", dense_bits: int = 0, *, dense_format: str = "rtn", embed_host: bool = False,
               head_host: bool = False, dense_q: str | None = None, log=print) -> DenseModel:
    """Steps 1-7 of the module docstring. The experts stay placeholders (site.parent.<attr> is a Removed module;
    site.ref keeps the original experts module on meta). Runs on any device for dense_bits=0 without host
    placement (tests compare it with from_pretrained on the CPU); 4-bit modules and host placement need the GPU.
    dense_q: a packed dense part the store ships (meta.json "dense_q", by recipe or directory name, e.g. "dense-hi"):
    see the module docstring's "Packed dense part". None keeps today's path."""
    from safetensors import safe_open

    t0 = time.time()
    sm = store if isinstance(store, StoreMeta) else read_store_meta(store)
    dev = torch.device(device)
    if dense_q and dense_bits:
        raise ValueError(f"dense_q={dense_q!r} and dense_bits={dense_bits} both quantize the dense part: pick one")
    if (dense_bits or embed_host or head_host or dense_q) and dev.type != "cuda":
        raise ValueError("dense 4-bit modules and host placement run on the GPU kernels: use a cuda device")
    if sm.transformers and sm.transformers != hfcompat.TF_VERSION:
        log(f"warning: {sm.root} was converted with transformers {sm.transformers}, this is {hfcompat.TF_VERSION} "
            "(engine and reference must use the same version, design-core A5)")
    cfg = sm.config
    entry = arch.arch_for(cfg)
    model = arch.build_meta(cfg)
    sites = arch.find_moe_sites(model, cfg)
    _check_sites(sm, sites)
    plan = hfcompat.DtypePlan(model, torch.bfloat16)
    host_params = _take_host_params(model, entry, sm)  # out of the model until step 5b: never cast or allocated

    selected, excluded = arch.dense_quant_plan(model, sites, entry) if dense_bits else ([], {})
    ties = _tie_groups(model)
    tie_of = {alias: g[0] for g in ties for alias in g[1:]}
    emb = model.get_input_embeddings()
    emb_name = _module_name(model, emb)
    head = model.get_output_embeddings()
    head_name = _module_name(model, head) if head is not None else None
    pk, dq_entry, qmods, kept = None, None, {}, []
    if dense_q:
        pk, dq_entry = open_dense_q(sm, dense_q)
        qmods, kept = _dense_q_modules(model, pk, emb, emb_name)
        if head_host and head_name in qmods:
            raise NotImplementedError(f"head_host with the packed head of {dense_q!r}: KLinear has no host variant")
    q_embed = emb_name in qmods
    cpu_emb = embed_host or q_embed  # the bf16 table goes through the CPU (initialize_weights), then is replaced
    if head_host and head_name not in selected:
        raise ValueError(f"head_host needs a quantized lm_head ({head_name} is not in the dense 4-bit plan; "
                         f"dense_bits={dense_bits})")

    # 3. placeholders
    for s in sites:
        setattr(s.parent, s.attr, Removed("experts", s.name))
    deferred = []
    for name in selected:
        lin = model.get_submodule(name)
        deferred.append(_Deferred(name, lin.bias is not None, lin.weight.dtype, lin.in_features, lin.out_features))
        _set_module(model, name, Removed("dense 4-bit", name))
    packed = []
    for name in qmods:
        if name == emb_name:
            continue
        lin = model.get_submodule(name)
        packed.append(_Deferred(name, lin.bias is not None, lin.weight.dtype, lin.in_features, lin.out_features))
        _set_module(model, name, Removed("dense-q", name))
    if cpu_emb:
        _set_module(model, emb_name, Removed("host embedding", emb_name))

    # 4. dtypes, allocation, ties
    changed = _apply_dtype_rule(model, plan)
    if cpu_emb:
        changed.update(_apply_dtype_rule(emb, plan, prefix=f"{emb_name}."))
    model.to_empty(device=dev)
    if cpu_emb:
        emb.to_empty(device="cpu")  # the table is loaded on the CPU, initialized with the model, then pinned
        _set_module(model, emb_name, emb)
    copy_after_load = []  # tied aliases on another device than their canonical tensor (embed_host + bf16 head)
    for group in ties:
        canon = _get_tensor(model, group[0])
        for alias in group[1:]:
            cur = _get_tensor(model, alias)
            if canon is None or cur is None:
                continue  # one side was replaced (quantized head): nothing to tie
            if cur.device == canon.device:
                _set_param(model, alias, canon)
            else:
                copy_after_load.append((alias, group[0]))

    # 5. stream the dense tensors
    state = dict(model.state_dict(keep_vars=True))
    skip = {f"{d.name}.{t}" for d in deferred + packed for t in ("weight", "bias")}
    loaded: set[str] = set()
    f = safe_open(sm.path("dense.safetensors"), "pt")
    keys = list(f.keys())
    unexpected = [k for k in keys if k not in state and k not in skip]
    if unexpected:
        raise RuntimeError(f"{sm.root}: {len(unexpected)} dense tensors match no model key: {unexpected[:8]}")
    n_bytes = 0
    for k in sorted(keys):
        if k in skip:
            continue
        canon = tie_of.get(k)
        if canon is not None and canon in keys and canon not in skip:
            continue  # a tied alias stored next to its canonical tensor: the canonical one is loaded
        dst = state[k]
        t = f.get_tensor(k)
        if tuple(t.shape) != tuple(dst.shape):
            raise RuntimeError(f"{k}: file {tuple(t.shape)}, model {tuple(dst.shape)}")
        dst.copy_(t.to(dst.dtype))
        loaded.add(k)
        n_bytes += t.numel() * t.element_size()
        del t
    for group in ties:  # a tensor shared by several names is loaded under every one of them
        if any(n in loaded for n in group):
            first = next(_get_tensor(model, n) for n in group if n in loaded)
            loaded.update(n for n in group if _get_tensor(model, n) is first)
    for alias, canon in copy_after_load:  # the same values on another device
        src, dst = (canon, alias) if canon in loaded else (alias, canon)
        if src in loaded:
            _get_tensor(model, dst).copy_(_get_tensor(model, src))
            loaded.add(dst)
    # 5b. host tensors: memory-mapped from host.safetensors, release dtype, on the CPU
    host_bytes = _bind_host_params(model, sm, host_params)
    loaded.update(host_params)

    # 6. buffers: hard check, flags, initialize_weights
    nonpers = {n for n, _ in hfcompat.named_non_persistent_buffers(model)}
    missing = []
    for n, _t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if n not in loaded and n not in nonpers:
            missing.append(n)
    if missing:
        raise RuntimeError(f"{sm.root}: {len(missing)} model tensors were not loaded and are not non-persistent "
                           f"buffers or tied aliases: {missing[:8]}")
    for n, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        if n not in nonpers:
            t._is_hf_initialized = True
    hfcompat.initialize_weights(model)

    # 7. dense 4-bit modules (after initialize_weights: it never sees them), host embedding
    saved = 0
    for d in deferred:
        wkey, bkey = f"{d.name}.weight", f"{d.name}.bias"
        if wkey in keys:
            w = f.get_tensor(wkey)
        elif tie_of.get(wkey) is not None and _get_tensor(model, tie_of[wkey]) is not None:
            w = _get_tensor(model, tie_of[wkey]).detach()  # tied head: quantized from the table (wherever it is)
        else:
            raise RuntimeError(f"{sm.root}: dense tensor {wkey} missing")
        w = w.to(hfcompat.load_dtype(plan, wkey, d.dtype))
        b = None
        if d.has_bias:
            if bkey not in keys:
                raise RuntimeError(f"{sm.root}: dense tensor {bkey} missing")
            b = f.get_tensor(bkey)
            b = b.to(hfcompat.load_dtype(plan, bkey, d.dtype))
        q = formats.dense_module(dense_format, w, b, dense_bits, dev)
        saved += w.numel() * w.element_size() - (q.nbytes() if hasattr(q, "nbytes") else 0)
        if head_host and d.name == head_name:
            _to_host(q)
        _set_module(model, d.name, q)
        loaded.update({wkey} | ({bkey} if d.has_bias else set()))
        del w, b
    del f, state
    dq_info = None
    if pk is not None:  # packed dense part: KLinear for the linears (and the head), QEmbedding below
        dq_info = _attach_dense_q(model, pk, dq_entry, packed, plan, dev)
        dq_info["kept_bf16"] = kept
        for d in packed:
            loaded.update({f"{d.name}.weight"} | ({f"{d.name}.bias"} if d.has_bias else set()))
    host_emb = None
    if q_embed:
        scale = emb._buffers.get("embed_scale") if "embed_scale" in emb._non_persistent_buffers_set else None
        host_emb = _dense_q_embedding(pk, emb_name, emb, scale, dev, embed_host)
        dq_info["modules"] += 1
        dq_info["bytes"] += host_emb.nbytes()
        dq_info["embed_host"] = bool(embed_host)
        _set_module(model, emb_name, host_emb)
        del emb
    elif embed_host:
        from .hostemb import HostEmbedding

        scale = emb._buffers.get("embed_scale") if "embed_scale" in emb._non_persistent_buffers_set else None
        host_emb = HostEmbedding(emb.weight.detach(), scale, device=dev, padding_idx=getattr(emb, "padding_idx", None))
        _set_module(model, emb_name, host_emb)
        del emb
    if pk is not None:  # the verified uploads' fingerprint tables (128 MiB) are no part of the dense part (WP13)
        from .tcq_encode import release_fp_buffers

        release_fp_buffers()
    info = {"dense_seconds": round(time.time() - t0, 1), "dense_loaded": len(loaded), "dense_bytes_read": n_bytes,
            "dense_quantized": [d.name for d in deferred], "dense_excluded": excluded, "dtype_forced": changed,
            "dense_format": dense_format if deferred else None, "dense_bits": dense_bits, "saved_bytes": saved,
            "embed_host": bool(embed_host), "head_host": bool(head_host), "tied": ties,
            "host_tensors": sorted(host_params), "host_bytes": host_bytes,
            "nonpersistent": sorted(nonpers), "dense_q": dq_info}
    if dev.type == "cuda":
        info["dense_vram"] = torch.cuda.memory_allocated(dev)
    log(f"dense: {len(loaded)} tensors loaded ({n_bytes / GiB:.2f} GiB read) in {info['dense_seconds']:.0f}s; "
        f"{len(deferred)} linears -> {dense_bits}-bit {dense_format if deferred else ''}"
        f"{f', saved {saved / GiB:.2f} GiB' if deferred else ''}; {len(changed)} tensors forced to another dtype; "
        f"{len(nonpers)} non-persistent buffers rebuilt"
        + (f"; embedding {host_emb.nbytes() / GiB:.2f} GiB on pinned host" if embed_host else "")
        + (f"; {len(host_params)} host tensors memory-mapped ({host_bytes / GiB:.2f} GiB)" if host_params else "")
        + (f"; dense-q {dq_info['recipe']}: {dq_info['modules']} packed modules, {dq_info['bytes'] / GiB:.3f} GiB "
           f"({dq_info['kinds']}), {len(dq_info['kept_bf16'])} kept bf16" if dq_info else ""))
    return DenseModel(model, entry, sites, sm, dev, info)


# ---------------------------------------------------------------------------------------------------------------------
# packed dense part (design-quality WP10)


def dense_q_entries(sm: StoreMeta) -> list[dict]:
    """meta.json "dense_q" (store v2, design-quality amendment 2): the packed dense parts the store ships."""
    return list(sm.raw.get("dense_q") or [])


def open_dense_q(sm: StoreMeta, name: str):
    """(behemoe.dense_recipe.PackedDense, its meta.json "dense_q" entry) for a recipe or directory name. Refuses an
    unknown name, another format or version, a recipe that differs from the entry's, and a directory meta.json whose
    sha256 differs from the one the entry records."""
    import hashlib

    from . import dense_recipe as dr

    entries = dense_q_entries(sm)
    e = next((x for x in entries if name in (x.get("recipe"), x.get("dir"))), None)
    if e is None:
        raise ValueError(f"{sm.root}: no packed dense part {name!r} (meta.json dense_q lists "
                         f"{[x.get('recipe') for x in entries]})")
    if e.get("format") != dr.FORMAT or int(e.get("version", 0)) != dr.VERSION:
        raise ValueError(f"{sm.root}: dense_q {name!r} is {e.get('format')} v{e.get('version')}, this engine reads "
                         f"{dr.FORMAT} v{dr.VERSION}")
    path = sm.path(e["dir"])
    if e.get("meta_sha256"):
        with open(os.path.join(path, "meta.json"), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != e["meta_sha256"]:
            raise ValueError(f"{path}/meta.json: sha256 {got} differs from the store's dense_q entry")
    pk = dr.PackedDense(path)
    if pk.meta.get("recipe") != e.get("recipe"):
        raise ValueError(f"{path}: holds recipe {pk.meta.get('recipe')!r}, the store says {e.get('recipe')!r}")
    return pk, e


def _dense_q_modules(model: nn.Module, pk, emb: nn.Module, emb_name: str) -> tuple[dict, list]:
    """({module name: role} the packed part replaces, [packed modules kept as dense.safetensors holds them]).
    Routers stay bf16 (design-core B3: the engine's router and prefetch probes read them; dense-hi's 8-bit router
    is qbench's only). Embeddings other than the input embedding (Gemma-4's per-layer tables) stay bf16 too."""
    out, kept = {}, []
    for name, e in pk.meta["tensors"].items():
        role = e["role"]
        try:
            m = model.get_submodule(name)
        except AttributeError:
            raise ValueError(f"{pk.path}: packed tensor {name!r} is no module of the model") from None
        if role == "router" or (role == "embed" and m is not emb):
            kept.append(name)
            continue
        if role == "embed" and name != emb_name:
            raise ValueError(f"{pk.path}: {name} is the input embedding under another name ({emb_name})")
        if role != "embed" and not isinstance(m, nn.Linear):
            raise ValueError(f"{pk.path}: packed tensor {name!r} ({role}) is a {type(m).__name__}, not an nn.Linear")
        if role != "embed" and list(e["shape"]) != [m.out_features, m.in_features]:
            raise ValueError(f"{pk.path}: {name} packed {e['shape']}, module [{m.out_features}, {m.in_features}]")
        out[name] = role
    return out, kept


def _kind(pk, role: str) -> str:
    spec = pk.meta.get("recipe_spec") or {}
    f = spec.get(role) or spec.get("default") or {}
    return f.get("kind", "k4row")


def _verified(kp, dev):
    """The packed arrays on the device, every upload checked against the host bytes' fingerprint (amendment 6)."""
    from . import kquant
    from .tcq_encode import to_device_verified

    up = [to_device_verified(t.contiguous(), dev) for t in (kp.qweight, kp.sb, kp.d, kp.dmin)]
    return kquant.KPacked(*up, kp.bits, kp.group, kp.fallback)


def _attach_dense_q(model: nn.Module, pk, entry: dict, packed: list, plan, dev) -> dict:
    from .tcq_encode import to_device_verified

    kinds: dict[str, int] = {}
    nbytes = 0
    for d in packed:
        role = pk.meta["tensors"][d.name]["role"]
        kp, bias = pk.get(d.name, "cpu")
        if (bias is not None) != d.has_bias:
            raise ValueError(f"{pk.path}: {d.name} bias {'present' if bias is not None else 'missing'}, the module "
                             f"{'has' if d.has_bias else 'has no'} bias")
        b = None
        if bias is not None:
            b = to_device_verified(bias.to(hfcompat.load_dtype(plan, f"{d.name}.bias", d.dtype)).contiguous(), dev)
        kind = _kind(pk, role)
        q = formats.dense_q_module(kind, role, _verified(kp, dev), b, dev)
        _set_module(model, d.name, q)
        kinds[kind] = kinds.get(kind, 0) + 1
        nbytes += q.nbytes()
    return {"recipe": entry["recipe"], "dir": entry["dir"], "modules": len(packed), "bytes": nbytes, "kinds": kinds,
            "embed_host": False}


def _dense_q_embedding(pk, emb_name: str, emb: nn.Module, scale, dev, host: bool) -> nn.Module:
    kp, _bias = pk.get(emb_name, "cpu")
    if tuple(pk.meta["tensors"][emb_name]["shape"]) != (emb.num_embeddings, emb.embedding_dim):
        raise ValueError(f"{pk.path}: {emb_name} packed {pk.meta['tensors'][emb_name]['shape']}, embedding "
                         f"[{emb.num_embeddings}, {emb.embedding_dim}]")
    kp = kp if host else _verified(kp, dev)  # host: packed into pinned memory on the CPU, no upload
    return formats.dense_q_module(_kind(pk, "embed"), "embed", kp, None, dev, host=host, embed_scale=scale,
                                  padding_idx=getattr(emb, "padding_idx", None))


def _to_host(q: nn.Module) -> None:
    """Move a QuantLinear's packed arrays (qweight, scale, minimum) to exact-size pinned host memory: the GEMV and
    GEMM kernels then read them zero-copy (their host variants); bias and index buffers stay on the GPU."""
    from . import hostmem

    for name in ("qweight", "scale", "minimum"):
        t = getattr(q, name)
        q._buffers[name] = hostmem.pin_copy(t.cpu())


# ---------------------------------------------------------------------------------------------------------------------
# experts


def build_engine(store, device="cuda:0", vram_gb: float = 24.0, dense_bits: int = 4, hot_from_cold: bool = False,
                 embed_host: bool = False, head_host: bool = False, *, dense_format: str = "rtn",
                 dense_q: str | None = None, frozen=None, slots: int | None = None, log=print):
    """Build the engine from a converted store (see the module docstring). Returns (model, info); info carries
    slots, num_experts, layers, manager, prefetcher (shared by every site since WP8), cold/hot nbytes, device, entry,
    sites (layer, name, router, top_k) and the dense part's facts (load_dense). dense_q: the store's packed dense
    part to run (meta.json "dense_q"; dense_bits must then be 0); None = today's path. A trellis store (cold format
    "tcq", design-quality WP14) is checked (behemoe.store_v2.check) before anything is loaded and its per-layer
    layouts built; it runs hot from cold only (WP15: rotated-domain kernels, behemoe.moe).
    frozen (design-quality WP12, the split; module docstring): None (today's full host load), an int F (the F most
    routed experts per layer by counts.safetensors) or {layer: [expert ids]}; needs hot_from_cold. info then also
    carries "placement" ("split" or "resident-vram"), "frozen" and "vram_load". slots: at most this many cache slots
    per layer (the auto-config rule's count; its dense estimate counts what the speedups add after the build, e.g.
    the bf16 kv_b_proj, so the budget's own sizing can give one more slot, which would make a static split dynamic)."""
    from . import hostmem

    t0 = time.time()
    sm = read_store_meta(store)
    if sm.hot is None and not hot_from_cold:
        raise ValueError(f"{sm.root} is a cold-only store (converted with --cold-only, no hot blobs): build it with "
                         "hot_from_cold=True (--hot-from-cold)")
    fz = frozen_arg(sm, frozen)
    if fz and not hot_from_cold:
        raise ValueError("frozen experts (the split) are for one-tier stores run hot from cold: the VRAM slot holds the "
                         "cold blob itself (hot_from_cold=True)")
    if formats.canonical_name(sm.cold) == "tcq":  # before anything is loaded or pinned
        from . import store_v2

        if not hot_from_cold:
            raise ValueError(f"{sm.root} is a trellis store: it runs one tier, hot from cold (hot_from_cold=True); "
                             "a separate hot tier over it is design-quality WP18 (not built)")
        store_v2.check(sm)
    dev = torch.device(device)
    total = torch.cuda.get_device_properties(dev).total_memory
    dm = load_dense(sm, dev, dense_bits, dense_format=dense_format, embed_host=embed_host, head_host=head_host,
                    dense_q=dense_q, log=log)
    model, sites, entry = dm.model, dm.sites, dm.entry
    from .experts import unsupported_reason

    bad = {s.layer_idx: unsupported_reason(s.ref) for s in sites if unsupported_reason(s.ref)}
    if bad:
        (layer, why), = list(bad.items())[:1]
        raise NotImplementedError(f"{entry.model_type}: {len(bad)} MoE layers cannot run on TieredMoE "
                                  f"(layer {layer}: {why})")
    hot_ls, cold_ls = expert_layouts(sm, sites, hot_from_cold, check=False)
    for lay in {id(x): x for x in (*cold_ls.values(), *hot_ls.values())}.values():
        if not formats.servable(lay):
            raise NotImplementedError(f"expert format {lay.format!r} has no decode kernel yet (behemoe.formats)")
    cold_l, hot_l = cold_ls[sites[0].layer_idx], hot_ls[sites[0].layer_idx]
    t1 = time.time()
    layers = read_experts(sm, sites, hot_ls, cold_ls, hot_from_cold, log, frozen=fz)
    fb = hostmem.fallback_bytes()
    log(f"expert blobs: {hostmem.registered_bytes() / GiB:.2f} GiB page-locked at exact size"
        + (f", {fb / GiB:.2f} GiB through pin_memory() (rounded up)" if fb else "")
        + (f" (partial host load: {sum(len(v) for v in fz.values())} frozen experts have no host copy)" if fz else "")
        + f" ({time.time() - t1:.0f}s)")
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * GiB / total), dev)
    biases = {s.layer_idx: read_biases(sm, s) for s in sites if s.layout.bias}
    model, info = attach(model, sites, layers, dev, vram_gb, hot_l, cold_l, entry, log, biases=biases, frozen=fz,
                         max_slots=slots)
    if fz:
        info.update(load_frozen(sm, sites, layers, info, fz, log))
    info.update(dm.info)
    info.update({"store": sm.root, "store_version": sm.version, "seconds": round(time.time() - t0, 1)})
    return model, info


def expert_layouts(sm: StoreMeta, sites: list, hot_from_cold: bool, check: bool = True) -> tuple[dict, dict]:
    """({layer: hot layout}, {layer: cold layout}) of every MoE site (behemoe.formats.layer_layouts: one shared layout
    for k4row / rtn, one per layer for a trellis store, design-quality WP14). A trellis cold entry is checked first
    (behemoe.store_v2.check: codec, LUT sha256s, rotation sign hashes, the encoder record, file sizes; SHA256SUMS are
    the publishing tools' check, not the loader's), so a store the engine must not decode fails before anything is
    pinned (build_engine runs the check before load_dense and passes check=False)."""
    gu, dn = sites[0].layout.gu_shape, sites[0].layout.dn_shape
    idx = [s.layer_idx for s in sites]
    if check and formats.canonical_name(sm.cold) == "tcq":
        from . import store_v2

        store_v2.check(sm)
    cold_ls = formats.layer_layouts(sm.cold, gu, dn, idx)
    hot_ls = cold_ls if hot_from_cold else formats.layer_layouts(sm.hot, gu, dn, idx)
    return hot_ls, cold_ls


def frozen_arg(sm: StoreMeta, frozen) -> dict | None:
    """build_engine's `frozen` argument -> {layer: sorted expert ids} for every MoE layer, or None (no frozen expert:
    today's path). An int F picks the F most routed experts per layer (behemoe.store_v2.frozen_sets)."""
    if frozen is None:
        return None
    if isinstance(frozen, bool):
        raise TypeError("frozen: None, an int F or {layer: [expert ids]}")
    if isinstance(frozen, int):
        from . import store_v2

        fz = store_v2.frozen_sets(sm, frozen)
    else:
        extra = sorted(set(int(k) for k in frozen) - set(sm.moe_layers))
        if extra:
            raise ValueError(f"frozen experts for layers {extra[:4]} that are no MoE layers of {sm.root}")
        fz = {li: [int(e) for e in frozen.get(li, frozen.get(str(li), []))] for li in sm.moe_layers}
    E = sm.num_experts
    for li, ids in fz.items():
        if len(set(ids)) != len(ids) or not all(0 <= e < E for e in ids):
            raise ValueError(f"layer {li}: frozen experts must be distinct ids in 0..{E - 1}")
        fz[li] = sorted(ids)
    return fz if any(fz.values()) else None


def read_experts(sm: StoreMeta, sites: list, hot_ls: dict, cold_ls: dict, hot_from_cold: bool, log=print, *,
                 frozen: dict | None = None) -> dict:
    """{layer: LayerExperts}: every site's layer{L:03d}.cold.bin (and .hot.bin) read into exact-size pinned memory
    (behemoe.hostmem) as uint8 [E, nbytes_L], with the layer's own layouts.
    frozen {layer: [ids]} (the split, hot from cold): only the other E - F rows of a layer's cold file are read, into
    an exact-size pinned [E - F, nbytes_L] tensor (behemoe.hostmem.read_rows), with host_row int32 [E] (-1 = frozen);
    the frozen rows go straight into VRAM later (load_frozen). A layer with every expert frozen gets a placeholder."""
    from . import hostmem
    from .store import LayerExperts

    n_exp = sm.num_experts
    layers = {}
    t1 = time.time()
    for n, s in enumerate(sites):
        hot_l, cold_l = hot_ls[s.layer_idx], cold_ls[s.layer_idx]
        ids = (frozen or {}).get(s.layer_idx)
        if ids:
            if not hot_from_cold:
                raise ValueError("frozen experts need hot_from_cold")
            path, nb = sm.layer_file(s.layer_idx, "cold"), cold_l.nbytes
            if os.path.getsize(path) != n_exp * nb:
                raise ValueError(f"{path}: {os.path.getsize(path)} bytes, expected {n_exp} x {nb}")
            fset = set(ids)
            keep = [e for e in range(n_exp) if e not in fset]
            host_row = torch.full((n_exp,), -1, dtype=torch.int32)
            host_row[keep] = torch.arange(len(keep), dtype=torch.int32)
            part = (hostmem.read_rows(path, nb, keep) if keep
                    else torch.empty((0, nb), dtype=torch.uint8, device="meta"))  # aliased to VRAM by load_frozen
            layers[s.layer_idx] = LayerExperts(part, part, hot_l, cold_l, host_row=host_row)
            if n % 16 == 0:
                log(f"  experts of layer {s.layer_idx}: {len(keep)} of {n_exp} pinned ({time.time() - t1:.0f}s)")
            continue
        blobs = {}
        for kind, lay in (("cold", cold_l),) if hot_from_cold else (("hot", hot_l), ("cold", cold_l)):
            blobs[kind] = hostmem.read_file(sm.layer_file(s.layer_idx, kind), n_exp * lay.nbytes).view(n_exp, lay.nbytes)
        if hot_from_cold:
            blobs["hot"] = blobs["cold"]
        layers[s.layer_idx] = LayerExperts(blobs["hot"], blobs["cold"], hot_l, cold_l)
        if n % 16 == 0:
            log(f"  experts of layer {s.layer_idx} pinned ({time.time() - t1:.0f}s)")
    return layers


def cache_slots(vram_gb: float, dense_bytes: int, hot_nbytes, cold_nbytes) -> dict:
    """The engine's VRAM cache sizing (build._attach's formula, design-quality amendment 5): the budget less the dense
    part, ONE staging blob (the largest cold blob of any layer) and the reserve max(0.75 GiB, 8% of the budget),
    divided by the cost of one slot in every layer, i.e. the sum over layers of the hot blob size (nbytes x layers
    when one layout serves every layer: today's arithmetic, to the byte). hot_nbytes / cold_nbytes: one entry per MoE
    layer."""
    hot_nbytes, cold_nbytes = [int(x) for x in hot_nbytes], [int(x) for x in cold_nbytes]
    staging = max(cold_nbytes)
    reserve = int(max(0.75, 0.08 * vram_gb) * GiB)
    free = vram_gb * GiB - dense_bytes - staging - reserve
    slot_cost = sum(hot_nbytes)
    return {"slots": max(0, int(free // slot_cost)), "staging": staging, "reserve": reserve, "free": free,
            "slot_cost": slot_cost}


def read_biases(sm: StoreMeta, site) -> tuple[torch.Tensor, torch.Tensor]:
    """(gate_up_proj_bias [E, 2I], down_proj_bias [E, H]) of a site's experts from layer{L:03d}.bias.safetensors
    (behemoe.convert_any), in transformers' layout and row order; shapes checked against the site."""
    from safetensors.torch import load_file

    path = sm.path(f"layer{site.layer_idx:03d}.bias.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: the experts of layer {site.layer_idx} have biases, the store has no bias file")
    t = load_file(path)
    lay = site.layout
    want = {"gate_up_proj_bias": (lay.E, lay.gu_shape[0]), "down_proj_bias": (lay.E, lay.H)}
    for k, shape in want.items():
        if k not in t or tuple(t[k].shape) != shape:
            raise ValueError(f"{path}: {k} {tuple(t[k].shape) if k in t else 'missing'}, expected {shape}")
    return t["gate_up_proj_bias"], t["down_proj_bias"]


def attach(model: nn.Module, sites: list, layers: dict, dev, vram_gb: float, hot_l, cold_l, entry, log=print, *,
           biases: dict | None = None, frozen: dict | None = None, max_slots: int | None = None):
    """build._attach over the site list: size the VRAM caches to what the budget leaves after the dense part, then
    put a TieredExperts on every site (same formula, same Manager / Staging / Prefetcher wiring). biases: {layer:
    (gate_up_proj_bias, down_proj_bias)} for experts with biases (read_biases).
    Per-layer layouts (design-quality WP14): the sizing reads every layer's own layouts from `layers` (cache_slots:
    slot cost = sum over layers of the hot blob size, staging = the largest cold blob), and the shared Staging /
    Prefetcher buffers are allocated for the largest cold layout. hot_l / cold_l are the layouts the callers built
    the layers with (one per store before WP14); with one layout for every layer all of this is today's arithmetic.
    Layers whose layouts differ (the trellis rungs) read the shared buffers through their own layout's views
    (behemoe.moe Staging / Prefetcher views_for, design-quality WP15); a trellis layer's core gets its rotation
    (behemoe.experts.rotation_kwargs).
    frozen {layer: [ids]} (the split, design-quality WP12): each layer's ExpertCache gets its frozen experts in slots
    0..F-1 (behemoe.store.ExpertCache frozen=); the slot count is capped at E and must hold every layer's F. The caller
    puts the frozen blobs into the pools (load_frozen). None: today's caches."""
    from . import tunecache
    from .experts import TieredExperts, core_kwargs, rotation_kwargs
    from .moe import Manager, Prefetcher, Staging, TieredMoE
    from .store import ExpertCache, fill_stream_kind

    biases = biases or {}
    # the cores' expert math (act / glu; expert-bias tables, allocated here so the budget below counts them)
    kws = {s.layer_idx: core_kwargs(s.ref, biases.get(s.layer_idx), dev) for s in sites}
    torch.cuda.synchronize(dev)
    dense = torch.cuda.memory_allocated(dev)
    lays = list(layers.values())
    cold_ls, hot_ls = [le.cold_layout for le in lays], [le.hot_layout for le in lays]
    sz = cache_slots(vram_gb, dense, [la.nbytes for la in hot_ls], [la.nbytes for la in cold_ls])
    staging, slots = sz["staging"], sz["slots"]
    if max_slots is not None and slots > max_slots:
        slots = max_slots
    big = max(cold_ls, key=lambda la: la.nbytes)  # the first largest: the shared layout when there is one
    mixed = len({repr(la) for la in cold_ls}) > 1 or len({repr(la) for la in hot_ls}) > 1
    n_exp = lays[0].num_experts
    if frozen:
        slots = min(slots, n_exp)  # more slots than experts would only hold nothing
        need = max(len(v) for v in frozen.values())
        if need > slots:
            raise RuntimeError(f"the split freezes {need} experts per layer in VRAM, the budget of {vram_gb:g} GiB gives "
                               f"{slots} cache slots per layer (slot cost {sz['slot_cost'] / GiB:.3f} GiB): freeze fewer "
                               "experts or raise vram_gb")
    resident = False
    for cl in {repr(la): la for la in cold_ls}.values():
        if formats.host_tail_ok(cl):
            continue
        words = formats.host_row_words(cl)
        if slots < n_exp:
            raise RuntimeError(f"{cl.format} {cl.bits}-bit rows of {words} int32 words: the format declares "
                               f"host_words={formats.expert_format(cl.format).host_words} (behemoe.formats), so "
                               f"the host GEMV is not known to read these rows right and every expert must be "
                               f"resident: {slots} cache slots per layer for {n_exp} experts (raise vram_gb)")
        resident = True
    stream = torch.cuda.Stream(dev) if fill_stream_kind() == "side" else None
    shared = Staging(big, dev)
    manager = Manager()
    top_k = sites[0].top_k
    prefetcher = Prefetcher(big, top_k, dev)
    cores = []
    for n, s in enumerate(sites):
        cache = ExpertCache(layers[s.layer_idx], slots, dev, stream,
                            **({"frozen": frozen.get(s.layer_idx) or None} if frozen else {}))
        manager.caches.append(cache)
        # Qwen3-MoE ("linear" probe) keeps TieredMoE's own predictor; the others ask the layer's own router
        predict = None if entry.probe.call == "linear" else arch.make_probe(s, entry)
        core = TieredMoE(s.router, layers[s.layer_idx], cache, s.top_k, shared, manager, first=n == 0,
                         **kws[s.layer_idx], predict=predict, **rotation_kwargs(layers[s.layer_idx].cold_layout, dev))
        mod = TieredExperts(core, s.ref, layer_idx=s.layer_idx)
        if prefetcher is not None:
            core.prefetcher, core.index = prefetcher, n
        setattr(s.parent, s.attr, mod)
        cores.append(core)
    for a, b in zip(cores, cores[1:]):
        # plain attributes: registered submodules would make every core reachable from every other one (a cycle
        # that state_dict(), named_modules() and .to() recurse into); build._attach registers them, never calls those
        object.__setattr__(a, "next", b)
        object.__setattr__(b, "prev", a)
    if resident:
        for c in manager.caches:
            c.warm(list(range(n_exp)))
    tuned = tunecache.load(dev)
    if tunecache.path():
        log(f"autotune cache: {tuned} saved choices loaded from {tunecache.path()}")
    host = sum(le.cold.numel() + (le.hot.numel() if le.hot is not le.cold else 0) for le in layers.values())

    def blob(la, nb):  # one layout, or the range of a per-layer one
        if mixed:
            return f"{la.format} per layer {min(nb)}..{max(nb)} B"
        return (f"{la.format} {la.bits}-bit g{la.group} {la.nbytes} B" if la.bits is not None else
                f"{la.format} {la.describe().get('gu_pattern')}/{la.describe().get('dn_pattern')} {la.nbytes} B")

    hot_l, cold_l = (hot_ls[0], cold_ls[0]) if not mixed else (max(hot_ls, key=lambda la: la.nbytes), big)
    log(f"VRAM budget {vram_gb:.1f} GiB: dense {dense / GiB:.2f} GiB, staging {staging / GiB:.2f} GiB, "
        f"cache {slots}/{n_exp} experts per layer ({slots * sz['slot_cost'] / GiB:.2f} GiB); "
        f"host blobs {host / GiB:.1f} GiB; expert blob hot {blob(hot_l, [la.nbytes for la in hot_ls])}, "
        f"cold {blob(cold_l, [la.nbytes for la in cold_ls])}; cache fills "
        f"on the {'side' if stream is not None else 'compute'} stream; {len(sites)} MoE sites ({entry.model_type}), "
        f"prefetch {'on' if prefetcher is not None else 'off'}" + ("; every expert resident (host tail guard)"
                                                                    if resident else ""))
    info = {"slots": slots, "num_experts": n_exp, "layers": len(layers), "manager": manager,
            "cold_nbytes": cold_l.nbytes, "hot_nbytes": hot_l.nbytes, "prefetcher": prefetcher, "device": dev,
            "entry": entry, "model_type": entry.model_type, "resident_forced": resident,
            "hot": hot_l.describe(), "cold": cold_l.describe(), "dense_vram_total": dense,
            "sites": [{"layer": s.layer_idx, "name": s.name, "router": s.router_name, "top_k": s.top_k}
                      for s in sites]}
    if mixed:
        info["slot_cost"] = sz["slot_cost"]
        info["cold_layers"] = {s.layer_idx: layers[s.layer_idx].cold_layout.nbytes for s in sites}
    if frozen:
        fl = [len(frozen.get(s.layer_idx) or []) for s in sites]
        info["frozen"] = {"F": fl[0] if len(set(fl)) == 1 else fl, "experts": sum(fl),
                          "dynamic_slots": [slots - f for f in fl] if len(set(fl)) > 1 else slots - fl[0],
                          "host_bytes": host}
    return model, info


@torch.no_grad()
def load_frozen(sm: StoreMeta, sites: list, layers: dict, info: dict, frozen: dict, log=print, *,
                staging_bytes: int | None = None, verify: bool = True) -> dict:
    """The VRAM half of the partial host load (design-quality WP12, amendment 1): every layer's frozen rows of
    layer{L:03d}.cold.bin go file -> one reusable pinned staging buffer (behemoe.vram_load.PinnedStaging, at most 64 MiB)
    -> their slots 0..F-1 of the layer's cache pool (behemoe.vram_load.stream_rows: every piece fingerprint-checked,
    re-copied on a mismatch, refused after 3 tries). A layer with every expert frozen has no host rows: its hot / cold
    tensors then alias the VRAM pool (slot e = expert e, as behemoe.vram_load's resident placement), and with every
    layer so the manager and the prefetcher are off (nothing can miss or move). Returns the info keys it adds."""
    from . import hostmem, vram_load

    t0 = time.time()
    caches = info["manager"].caches
    E = sm.num_experts
    big = max(len(frozen.get(s.layer_idx) or []) * layers[s.layer_idx].cold_layout.nbytes for s in sites)
    staging = vram_load.PinnedStaging(min(int(staging_bytes or vram_load.DEFAULT_STAGING), big))
    peak = hostmem.registered_bytes()
    st: dict = {}
    try:
        for n, (s, cache) in enumerate(zip(sites, caches, strict=True)):
            ids = frozen.get(s.layer_idx) or []
            le = layers[s.layer_idx]
            if cache.layer is not le or cache.frozen != ids:
                raise RuntimeError(f"layer {s.layer_idx}: cache wiring differs from the frozen sets")
            if not ids:
                continue
            nb = le.cold_layout.nbytes
            vram_load.stream_rows(sm.layer_file(s.layer_idx, "cold"), nb, ids, cache.pool[:len(ids)], staging,
                                  verify=verify, stats=st)
            if len(ids) == E:  # no host row: the tensors alias the pool (slot e = expert e: ids are sorted)
                le.hot = le.cold = cache.pool
                le.cold_views = le.cold_layout.views(cache.pool)
            peak = max(peak, hostmem.registered_bytes())
            if n % 16 == 0:
                log(f"  {len(ids)} frozen experts of layer {s.layer_idx} streamed into VRAM ({time.time() - t0:.0f}s)")
    finally:
        staging.close()
        if verify:  # stream_rows' device fingerprints built a 128 MiB position table: not kept past the load
            from .tcq_encode import release_fp_buffers

            release_fp_buffers()
    every = all(len(frozen.get(s.layer_idx) or []) == E for s in sites)
    static = all(len(frozen.get(s.layer_idx) or []) >= c.slots for s, c in zip(sites, caches, strict=True))
    if every or static:  # no dynamic slot anywhere: the manager and the prefetcher have nothing to do
        info["manager"].enabled = False
        if info.get("prefetcher") is not None:
            info["prefetcher"].enabled = False
    torch.cuda.synchronize()
    log(f"split: {st.get('bytes', 0) / GiB:.2f} GiB of frozen experts streamed into VRAM through a "
        f"{staging.nbytes / 2**20:.0f} MiB pinned staging buffer in {time.time() - t0:.0f}s ({st.get('pieces', 0)} "
        f"pieces, {st.get('retries', 0)} re-copies, verify {'on' if verify else 'off'}); host rows "
        f"{info['frozen']['host_bytes'] / GiB:.2f} GiB pinned" + ("; every expert resident, manager and prefetch off"
                                                                   if every else ""))
    return {"placement": "resident-vram" if every else "split",
            "vram_load": {**st, "staging_bytes": staging.nbytes, "verify": bool(verify),
                          "peak_pinned_bytes_during_load": int(peak), "seconds": round(time.time() - t0, 1)}}


def stats(model) -> dict:
    """build.stats over the TieredMoE cores of any engine model (arch.tiered)."""
    tot = {"pairs": 0, "hit": 0.0, "staged": 0.0, "zero_copy_bytes": 0, "prefetch_bytes": 0, "fill_bytes": 0,
           "layers": 0}
    for m in arch.tiered(model):
        s = m.stats
        nb = m.experts.cold_layout.nbytes
        tot["layers"] += 1
        tot["pairs"] += s["pairs"]
        tot["hit"] += s["hit"]
        tot["staged"] += s["staged"]
        tot["zero_copy_bytes"] += (s["pairs"] - s["hit"] - s["staged"]) * nb
        tot["prefetch_bytes"] += s["copied"] * nb
        tot["fill_bytes"] += m.cache.fill_bytes
    return tot


def reset_stats(model) -> None:
    for m in arch.tiered(model):
        m.reset_stats()
        m.cache.fill_bytes = 0
