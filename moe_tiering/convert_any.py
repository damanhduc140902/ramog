"""Generic streaming converter: any approved checkpoint -> tiered engine store, meta.json v2 (design-core B4; WP5).

What moe_tiering.convert does for Qwen3-MoE, for every architecture in moe_tiering.arch.ARCHES, one MoE layer at a time:
- names, decoding and dtypes come from moe_tiering.ckpt (WP3): CheckpointIndex reads the shards, KeyMap maps every
  checkpoint key onto the transformers text model (renames, converters, FP8 / MXFP4 side tensors, named skips for
  MTP layers and vision / audio towers; anything unmapped raises), ExpertSource gives each MoE layer's experts in
  chunks as gate_up [n, 2I, H] / down [n, H, I] ([out, in], rows in transformers' output order: gpt-oss stays
  interleaved) in the dtype transformers loads (bf16), whatever the source layout (per-expert, fused, MXFP4) and
  format (BF16, FP8 block-scaled with F32 / BF16 / E8M0 scales, MXFP4) is;
- each tier's blobs come from an ExpertQuantizer looked up by store format name (QUANTIZERS; "rtn" and "k4row" are
  registered here on top of moe_tiering.formats / moe_tiering.store.BlobLayout). A new store format (the quality sprint's
  trellis-coded experts, design-quality WP16) registers its own quantizer with register_expert_quantizer() and its
  store layout with moe_tiering.formats.register_expert_format(); nothing here names k4row except the defaults of the CLI;
- the dense part comes from a dense writer looked up by name (DENSE_WRITERS), streamed into dense.safetensors one
  tensor at a time (moe_tiering.ckpt.StreamingWriter) under the model's key names, which is what moe_tiering.loader reads.
  "plain" (default): every non-expert tensor unquantized, in the checkpoint's own dtype when it is stored unquantized
  (lossless; MiniMax-M2's F32 router stays F32), else decoded to the load dtype (FP8 / MXFP4 -> bf16, strict-fp32
  tensors float32); "bf16": every tensor in the dtype from_pretrained(dtype=bfloat16) gives it. moe_tiering.loader casts
  to the load dtype (design-core A7), so both load the same model. A packed dense format (the quality sprint's
  dense-q) registers its own writer with register_dense_writer();
- expert biases (gpt-oss) go to layer{L:03d}.bias.safetensors (gate_up_proj_bias [E, 2I], down_proj_bias [E, H]).

For Qwen3-MoE with the same options it writes the same bytes as moe_tiering.convert: the same chunks, the same
BlobLayout.quantize calls on the same bf16 values, the same importance arithmetic (tests/test_convert_equiv_qwen3.py
compares layer blobs and dense tensors with a store moe_tiering.convert wrote).

Output layout (moe_tiering.convert's, plus the v2 fields; moe_tiering.loader.read_store_meta reads it):
    <out>/meta.json                 format 2: arch (text model_type), model_type (the checkpoint config's), transformers,
                                    torch, source, config (config.json as stored), num_layers, num_experts, moe_layers,
                                    gu_shape, dn_shape, hot, cold, experts {E, gu_shape, dn_shape, gated, bias, transposed,
                                    concatenated, gate, act, sources}, bias, dense, source_formats, skipped, keymap
    <out>/dense.safetensors         non-expert tensors under model key names (dense writer "plain" or "bf16")
    <out>/host.safetensors          host tensors (ArchEntry.host_keys: DeepSeek-V4.1's engram tables), the stored bytes
                                    in their release dtype under model key names (moe_tiering.ckpt.copy_host); meta.json
                                    "host" lists them; moe_tiering.loader memory-maps the file (architectures with host
                                    keys only)
    <out>/layer{L:03d}.hot.bin      uint8 [E, hot nbytes]   (not written with --cold-only)
    <out>/layer{L:03d}.cold.bin     uint8 [E, cold nbytes]
    <out>/layer{L:03d}.bias.safetensors   expert biases (architectures with biases only)
    <out>/layer{L:03d}.json         how the layer's blobs were made; checked on resume
Layer files exist for MoE layers only (L = decoder layer index; dense layers such as GLM's first ones have none).

One process (resumable):
    python -m moe_tiering.convert_any --model models/GLM-4.7-Flash --out stores/glm47f-k4row \
        --cold-format k4row --cold-group 16 [--stats DIR]
Split across processes / GPUs (as moe_tiering.convert: nothing shared is written until --finalize):
    HIP_VISIBLE_DEVICES=0 python -m moe_tiering.convert_any <options> --layers 0-23 &
    HIP_VISIBLE_DEVICES=1 python -m moe_tiering.convert_any <options> --layers 24-47 &
    wait; python -m moe_tiering.convert_any <options> --finalize
--layers takes decoder layer indices; indices of dense (non-MoE) layers are ignored. --arch NAME asserts the
detected architecture (default: auto, from config.json). --plugin MODULE imports a module that registers more
quantizers / dense writers before the arguments are checked. --stats applies to every tier whose quantizer takes
per-expert importance (k4row); stats files are <stats>/layer{L:03d}.pt with gate_up/down sumsq [E, width] and
count [E] (calib_stream.py / stream_ref.py format).

Runs on a CPU too (--device cpu): no Triton kernel is imported.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch

from . import arch, ckpt, formats, hfcompat

BIAS_KEYS = ("gate_up_proj_bias", "down_proj_bias")


# ---------------------------------------------------------------------------------------------------------------------
# expert quantizers (the converter side of a store expert format)


class ExpertQuantizer:
    """Makes one tier's blobs (hot or cold) for one store expert format.

    Constructed once per conversion with the format's parameters; then, per MoE layer: begin_layer(layer), quantize()
    for consecutive expert chunks, end_layer(layer). The store side of the format (layout: nbytes, views, dequant,
    kernels) is moe_tiering.formats'; layout(layer) must give the layout the loader will build from describe().
    Subclass for a new format and register it with register_expert_quantizer(name, cls).

    importance: the quantizer takes per-expert input-channel importance (from --stats).
    """

    importance = False

    def __init__(self, fmt: str, bits: int, group: int, experts: arch.ExpertsLayout, device, *,
                 stats_dir: str | None = None, **options):
        self.format, self.bits, self.group = fmt, bits, group
        self.experts = experts
        self.device = torch.device(device)
        self.stats_dir = os.path.abspath(stats_dir) if stats_dir else None
        self.options = options

    def layout(self, layer: int):
        raise NotImplementedError

    def describe(self) -> dict:
        """The meta.json "hot"/"cold" entry (moe_tiering.formats.expert_layout builds the loader's layout from it)."""
        raise NotImplementedError

    def begin_layer(self, layer: int) -> None:
        pass

    def quantize(self, layer: int, e0: int, e1: int, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        """Experts [e0, e1) of `layer` (gate_up [n, 2I, H], down [n, H, I], nn.Linear layout) -> uint8 blobs
        [n, layout(layer).nbytes] on the CPU."""
        raise NotImplementedError

    def end_layer(self, layer: int) -> dict:
        """Per-layer facts for the layer sidecar."""
        return {}


class BlobQuantizer(ExpertQuantizer):
    """moe_tiering.store.BlobLayout formats (rtn, k4row): one layout for every layer; layout.quantize does the work, as
    moe_tiering.build.pack_layer does in moe_tiering.convert (same call, same arguments, so the same bytes)."""

    def __init__(self, fmt, bits, group, experts, device, *, stats_dir=None, **options):
        super().__init__(fmt, bits, group, experts, device, stats_dir=stats_dir, **options)
        self.importance = fmt == "k4row"  # rtn has no importance-weighted search
        self._layout = formats.new_expert_layout(fmt, bits, group, experts.gu_shape, experts.dn_shape)
        self._imp = None
        self._info: dict = {}

    def layout(self, layer):
        return self._layout

    def describe(self):
        d = self._layout.describe()
        if self.importance and self.stats_dir:
            d["stats"] = self.stats_dir
        return d

    def begin_layer(self, layer):
        self._info = {}
        self._imp = None
        if self.importance and self.stats_dir:
            lay = self.experts
            self._imp = load_importance(self.stats_dir, layer, lay.E, lay.H, lay.I, self.device)

    def quantize(self, layer, e0, e1, gate_up, down):
        imp = None if self._imp is None else (self._imp[0][e0:e1], self._imp[1][e0:e1])
        blob = self._layout.quantize(gate_up, down, imp, device=self.device, info=self._info).cpu()
        if tuple(blob.shape) != (e1 - e0, self._layout.nbytes):
            raise AssertionError(f"{self.format}: blob {tuple(blob.shape)}, expected {(e1 - e0, self._layout.nbytes)}")
        return blob

    def end_layer(self, layer):
        out = {}
        if self.format == "k4row":
            out["fallback_groups"] = self._info.get("fallback", 0)
        self._imp = None
        return out


QUANTIZERS: dict[str, type[ExpertQuantizer]] = {}


def register_expert_quantizer(name: str, cls: type[ExpertQuantizer], *, replace: bool = False) -> None:
    """Make `name` a converter tier format. The store layout must be registered in moe_tiering.formats as well (the loader
    reads stores through it); a quantizer of a format without kernels (formats.servable False) still converts."""
    if name in QUANTIZERS and not replace:
        raise ValueError(f"expert quantizer {name!r} is already registered")
    QUANTIZERS[name] = cls


def expert_quantizer(name: str, bits: int, group: int, experts: arch.ExpertsLayout, device, **kw) -> ExpertQuantizer:
    if name not in QUANTIZERS:
        raise ValueError(f"unknown expert format {name!r} for the converter; registered: {sorted(QUANTIZERS)}")
    return QUANTIZERS[name](name, bits, group, experts, device, **kw)


register_expert_quantizer("rtn", BlobQuantizer)
register_expert_quantizer("k4row", BlobQuantizer)


def load_importance(stats_dir: str, layer: int, n_exp: int, hidden: int, inter: int, device):
    """<stats>/layer{L:03d}.pt (calib_stream.py / stream_ref.py format) -> per-expert importance (gate_up [E, H],
    down [E, I]) on `device`. The arithmetic of moe_tiering.convert.load_importance, operation for operation (that module
    imports the Triton kernels through moe_tiering.build, so it is not imported here)."""
    from .kquant import importance_from_stats

    st = torch.load(os.path.join(stats_dir, f"layer{layer:03d}.pt"), map_location="cpu")
    out = []
    for key, width in (("gate_up", hidden), ("down", inter)):
        s, c = st[key]["sumsq"], st[key]["count"]
        if tuple(s.shape) != (n_exp, width) or tuple(c.shape) != (n_exp,):
            raise ValueError(f"{stats_dir} layer {layer} {key}: sumsq {tuple(s.shape)}, count {tuple(c.shape)}; "
                             f"expected ({n_exp}, {width}) and ({n_exp},)")
        out.append(importance_from_stats(s.to(device, torch.float32), c))
    return tuple(out)


# ---------------------------------------------------------------------------------------------------------------------
# dense writers


DENSE_WRITERS: dict[str, Callable] = {}


def register_dense_writer(name: str, fn: Callable, *, replace: bool = False) -> None:
    """fn(keymap, out_dir: Path, *, device, log) -> the meta.json "dense" entry. Must leave every dense model tensor
    (keymap.dense_keys()) where moe_tiering.loader finds it."""
    if name in DENSE_WRITERS and not replace:
        raise ValueError(f"dense writer {name!r} is already registered")
    DENSE_WRITERS[name] = fn


def dense_file_dtype(keymap: ckpt.KeyMap, key: str) -> torch.dtype:
    """The dtype the "plain" writer stores a dense tensor in: the checkpoint's own dtype when every source part is
    stored unquantized (lossless: MiniMax-M2's F32 router, DeepSeek-V3.2's F32 norms stay F32; GLM-5.3-Flash's BF16
    q/k/v_conv1d stay BF16), else the load dtype (FP8 / MXFP4 decoded to bf16, strict-fp32 tensors float32: what
    transformers' dequantize gives). moe_tiering.loader casts every tensor to its load dtype (design-core A7), so the
    loaded model is the same either way."""
    t = keymap.targets[key]
    parts = t.all_parts()
    if parts and all(p.fmt in ("plain", "raw") for p in parts):
        dts = {keymap.index[p.key].torch_dtype for p in parts}
        if len(dts) == 1:
            return dts.pop()
    return t.dtype


def _reuse_dense(path: Path, specs: dict) -> dict | None:
    """An existing dense file (an earlier run) is kept when its header has exactly these names, dtypes and shapes."""
    if not path.exists():
        return None
    hdr, _meta, _start = ckpt.parse_safetensors_header(str(path))
    want = {k: (ckpt.st_dtype_name(dt), tuple(shape)) for k, (dt, shape) in specs.items()}
    have = {k: (v[0], tuple(v[1])) for k, v in hdr.items()}
    if have != want:
        raise RuntimeError(f"{path} exists with {len(have)} tensors that differ from this checkpoint's {len(want)} "
                           "dense tensors (different model or dense format? use another --out)")
    return {"tensors": len(have), "bytes": sum(v[3] for v in hdr.values()), "reused": True}


def _write_dense(keymap: ckpt.KeyMap, out: Path, name: str, dtype_of, *, device="cpu", log=print) -> dict:
    path = out / "dense.safetensors"
    keys = keymap.dense_keys()
    specs = {k: (dtype_of(keymap, k), keymap.targets[k].shape) for k in keys}
    entry = {"format": name, "file": path.name,
             "dtypes": dict(sorted(Counter(str(dt).replace("torch.", "") for dt, _ in specs.values()).items()))}
    reused = _reuse_dense(path, specs)
    if reused is not None:
        return {**entry, **reused}
    t0 = time.time()
    nbytes = 0
    meta = {"format": "pt", "converter": "moe_tiering.convert_any", "dense_format": name,
            "transformers": hfcompat.TF_VERSION}
    with ckpt.StreamingWriter(path, specs, meta) as w:  # header first, then one tensor at a time
        for i, k in enumerate(keys):
            t = keymap.load(k, device, dtype=specs[k][0])
            w.write(k, t)
            nbytes += t.numel() * t.element_size()
            del t
            if (i + 1) % 2000 == 0:
                log(f"dense {i + 1}/{len(keys)} ({nbytes / 2**30:.2f} GiB)")
    log(f"dense ({name}): {len(keys)} tensors, {nbytes / 2**30:.2f} GiB in {time.time() - t0:.0f}s, {entry['dtypes']}")
    return {**entry, "tensors": len(keys), "bytes": nbytes}


def _dense_plain(keymap: ckpt.KeyMap, out: Path, *, device="cpu", log=print) -> dict:
    """Every dense model tensor unquantized, under its model key name, in dense.safetensors: stored dtype for tensors
    the checkpoint stores unquantized, load dtype for decoded FP8 / MXFP4 ones (dense_file_dtype)."""
    return _write_dense(keymap, out, "plain", dense_file_dtype, device=device, log=log)


def _dense_bf16(keymap: ckpt.KeyMap, out: Path, *, device="cpu", log=print) -> dict:
    """Every dense model tensor exactly as from_pretrained(dtype=bfloat16) holds it (moe_tiering.ckpt.write_dense's dtypes:
    strict-fp32 tensors float32, the rest the meta dtype)."""
    return _write_dense(keymap, out, "bf16", lambda km, k: km.targets[k].dtype, device=device, log=log)


register_dense_writer("plain", _dense_plain)
register_dense_writer("bf16", _dense_bf16)

HOST_FILE = "host.safetensors"


def _write_host(keymap: ckpt.KeyMap, out: Path, *, log=print) -> dict | None:
    """Host tensors (ArchEntry.host_keys) into host.safetensors, byte for byte as stored (moe_tiering.ckpt.copy_host);
    an existing file with exactly these names, dtypes and shapes is kept. None when the architecture has none."""
    specs = ckpt.host_specs(keymap)
    if not specs:
        return None
    path = out / HOST_FILE
    entry = {"file": HOST_FILE, "keys": list(specs), "placement": "mmap"}
    if path.exists():
        hdr, _meta, _start = ckpt.parse_safetensors_header(str(path))
        have = {k: (v[0], tuple(v[1])) for k, v in hdr.items()}
        if have != specs:
            raise RuntimeError(f"{path} exists with {len(have)} tensors that differ from this checkpoint's host "
                               f"tensors {sorted(specs)} (different model? use another --out)")
        return {**entry, "tensors": len(have), "bytes": sum(v[3] for v in hdr.values()),
                "dtypes": dict(sorted(Counter(v[0] for v in hdr.values()).items())), "reused": True}
    info = ckpt.copy_host(keymap, path, {"converter": "moe_tiering.convert_any", "content": "host tensors, release bytes"},
                          log=log)
    log(f"host: {info['tensors']} tensors, {info['bytes'] / 2**30:.2f} GiB in {info['seconds']:.0f}s {info['dtypes']}")
    return {**entry, **info}


# ---------------------------------------------------------------------------------------------------------------------
# helpers


def parse_layers(spec: str | None, n_layers: int) -> list[int] | None:
    """'0-23', '0,5,9' or '0-11,24-35' -> sorted layer list (None for None); moe_tiering.convert's syntax."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        out.update(range(lo, hi + 1))
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"layers {bad} out of range 0..{n_layers - 1}")
    return sorted(out)


def _atomic_write(path: Path, write) -> None:
    """write(tmp_path) then rename: readers never see a partial file; concurrent writers use distinct temps."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def uniform_experts(sites: list) -> arch.ExpertsLayout:
    """The experts layout shared by every MoE site (the store format has one E and one pair of shapes)."""
    lay = sites[0].layout
    for s in sites[1:]:
        if s.layout != lay:
            raise NotImplementedError(f"MoE layer {s.layer_idx} has experts {s.layout}, layer {sites[0].layer_idx} "
                                      f"{lay}: stores with per-layer expert shapes are not supported")
    return lay


def bias_file(out: Path, layer: int) -> Path:
    return out / f"layer{layer:03d}.bias.safetensors"


def _layer_done(out: Path, li: int, want: dict, tiers: list, n_exp: int, bias: bool) -> bool:
    """Every blob file present with the expected size (and the bias file when the experts have biases); a layer
    sidecar, when present, must match the requested tiers (refuses to mix settings in one store)."""
    for kind, q in tiers:
        p = out / f"layer{li:03d}.{kind}.bin"
        if not p.exists():
            return False
        nb = q.layout(li).nbytes
        if p.stat().st_size != n_exp * nb:
            raise RuntimeError(f"{p}: {p.stat().st_size} bytes, expected {n_exp} x {nb} for {want[kind]} "
                               "(different settings? use another --out)")
    if bias and not bias_file(out, li).exists():
        return False
    side = out / f"layer{li:03d}.json"
    if side.exists():
        got = json.loads(side.read_text())
        mine = {"hot": want["hot"], "cold": want["cold"]}
        theirs = {"hot": got.get("hot"), "cold": got.get("cold")}
        if want["hot"] is None:  # a cold-only request accepts a layer that also has hot blobs
            theirs["hot"] = None
        if theirs != mine:
            raise RuntimeError(f"{side}: layer was converted with {theirs}, this run asks for {mine}")
    return True


def _formats_by_layer(es: ckpt.ExpertSource, layers: list[int]) -> list[dict]:
    """The experts' source formats per role, grouped over layers: [{formats, layers: [first, ...], count}]."""
    groups: dict[str, list[int]] = {}
    for li in layers:
        groups.setdefault(json.dumps(es.formats(li), sort_keys=True), []).append(li)
    return [{"formats": json.loads(k), "count": len(v), "layers": v[:4]} for k, v in groups.items()]


def _keymap_summary(km: ckpt.KeyMap) -> dict:
    r = km.report()
    keep = ("checkpoint_keys", "mapped_direct", "mapped_converter", "side", "skipped", "tied", "targets",
            "expert_targets", "dense_targets", "fp8_block", "fp8_block_inferred", "transforms", "balance_ok", "complete")
    return {k: r[k] for k in keep if k in r}


# ---------------------------------------------------------------------------------------------------------------------
# conversion


def convert(model_dir: str, out_dir: str, device="cuda:0", hot_bits=4, cold_bits=2, group=64, chunk=16,
            layers: list[int] | None = None, log=print, *, cold_format: str = "rtn", cold_group: int | None = None,
            stats_dir: str | None = None, cold_only: bool = False, finalize: bool | None = None,
            hot_format: str = "rtn", dense_format: str = "plain", arch_name: str | None = None) -> dict:
    """Convert the MoE layers in `layers` (decoder layer indices; default: all) of the checkpoint (moe_tiering.convert's
    arguments; group is the hot group size and the rtn cold one unless cold_group is given; k4row defaults to group
    16). finalize: write dense.safetensors and meta.json after checking that every MoE layer is present (default:
    True for a full run, False when `layers` is given). hot_format: the hot tier's store format (moe_tiering.convert's hot
    tier is always rtn). Returns a summary (layers converted, skipped, seconds, the meta written)."""
    t0 = time.time()
    if cold_format not in QUANTIZERS or hot_format not in QUANTIZERS:
        bad = cold_format if cold_format not in QUANTIZERS else hot_format
        raise ValueError(f"expert format {bad!r} not in {sorted(QUANTIZERS)}")
    if dense_format not in DENSE_WRITERS:
        raise ValueError(f"dense format {dense_format!r} not in {sorted(DENSE_WRITERS)}")
    model_dir = str(model_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        raw_cfg = json.load(f)
    km, meta_model, index = ckpt.build_keymap(model_dir)  # strict: any unmapped key or uncovered tensor raises
    cfg = meta_model.config
    entry = km.entry
    if arch_name not in (None, "auto", entry.model_type):
        raise ValueError(f"{model_dir}: architecture {entry.model_type!r}, --arch says {arch_name!r}")
    text = arch.text_config(cfg)
    n_layers = text.num_hidden_layers
    sites = km.sites
    moe_layers = [s.layer_idx for s in sites]
    lay = uniform_experts(sites)
    es = ckpt.ExpertSource(km)
    dev = torch.device(device)
    cg = cold_group if cold_group is not None else (16 if cold_format == "k4row" else group)
    qkw = {"stats_dir": stats_dir}
    cold_q = expert_quantizer(cold_format, cold_bits, cg, lay, dev, **qkw)
    hot_q = None if cold_only else expert_quantizer(hot_format, hot_bits, group, lay, dev, **qkw)
    tiers = [("cold", cold_q)] + ([("hot", hot_q)] if hot_q is not None else [])
    if stats_dir and not any(q.importance for _, q in tiers):
        raise ValueError(f"--stats only applies to formats with an importance-weighted search (k4row); tiers are "
                         f"{[(k, q.format) for k, q in tiers]}")
    # moe_tiering.convert's entries: the cold one always names its statistics (None without), the hot one only when used
    want = {"hot": hot_q.describe() if hot_q else None, "cold": cold_q.describe()}
    want["cold"].setdefault("stats", None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    finalize = layers is None if finalize is None else finalize
    asked = list(range(n_layers)) if layers is None else list(layers)
    todo = [li for li in asked if li in moe_layers]  # [] with finalize: check + dense + meta only
    not_moe = [li for li in asked if li not in moe_layers] if layers is not None else []
    span = "all" if layers is None else (f"{len(todo)} ({todo[0]}..{todo[-1]})" if todo else "none")
    log(f"convert {model_dir} -> {out}: {entry.model_type} ({len(sites)} MoE layers of {n_layers}, E={lay.E}, "
        f"gate_up {list(lay.gu_shape)}, down {list(lay.dn_shape)}, sources {es.kinds()}, formats "
        f"{km.source_formats()}); hot {want['hot']}, cold {want['cold']}, layers {span}, finalize {finalize}"
        + (f"; ignoring dense layers {not_moe}" if not_moe else ""))
    dense_info = host_info = None
    if finalize and layers is None:
        dense_info = DENSE_WRITERS[dense_format](km, out, device=dev, log=log)  # first: a crash leaves the dense part
        host_info = _write_host(km, out, log=log)

    done_s, converted = 0.0, []
    for li in todo:
        if _layer_done(out, li, want, tiers, lay.E, lay.bias):
            continue  # resumable
        t1 = time.time()
        for _, q in tiers:
            q.begin_layer(li)
        tmp = {kind: out / f"layer{li:03d}.{kind}.bin.tmp{os.getpid()}" for kind, _ in tiers}
        fh = {kind: open(p, "wb") for kind, p in tmp.items()}
        secs = {kind: 0.0 for kind, _ in tiers}
        t_read = 0.0
        biases = ([], [])
        try:
            for e0 in range(0, lay.E, chunk):
                e1 = min(lay.E, e0 + chunk)
                ta = time.time()
                gu, dn, gub, dnb = es.read(li, e0, e1, dev)
                t_read += time.time() - ta
                for kind, q in tiers:  # each tier's blob chunk is appended as it is made: peak RAM one chunk
                    tb = time.time()
                    blob = q.quantize(li, e0, e1, gu, dn)
                    fh[kind].write(memoryview(blob.numpy()))
                    secs[kind] += time.time() - tb
                    del blob
                if lay.bias:
                    biases[0].append(gub.cpu())
                    biases[1].append(dnb.cpu())
                del gu, dn, gub, dnb
            for kind, f in fh.items():
                f.flush()
                os.fsync(f.fileno())
                f.close()
            for kind, _ in tiers:
                os.replace(tmp[kind], out / f"layer{li:03d}.{kind}.bin")
        finally:
            for f in fh.values():
                if not f.closed:
                    f.close()
            for p in tmp.values():
                if p.exists():
                    p.unlink()
        side_info = {}
        for kind, q in tiers:
            side_info.update({f"{kind}_{k}" if kind == "hot" else k: v for k, v in q.end_layer(li).items()})
        if lay.bias:
            from safetensors.torch import save_file

            tensors = {BIAS_KEYS[0]: torch.cat(biases[0]).contiguous(), BIAS_KEYS[1]: torch.cat(biases[1]).contiguous()}
            _atomic_write(bias_file(out, li), lambda t, tensors=tensors: save_file(tensors, str(t),
                                                                                  metadata={"format": "pt"}))
        dt = time.time() - t1
        done_s += dt
        side = {"layer": li, **want, **side_info, "seconds": round(dt, 2), "seconds_read": round(t_read, 2),
                **{f"seconds_{k}": round(v, 2) for k, v in secs.items()}, "device": str(dev), "chunk": chunk,
                "source": es.kind(li), "source_formats": es.formats(li), "bias": lay.bias,
                "converter": "moe_tiering.convert_any"}
        _atomic_write(out / f"layer{li:03d}.json", lambda t, side=side: t.write_text(json.dumps(side, indent=1)))
        converted.append(li)
        extra = f", fallback groups {side_info['fallback_groups']}" if "fallback_groups" in side_info else ""
        log(f"layer {li}/{n_layers}: {dt:.1f}s (read {t_read:.1f}s, "
            + ", ".join(f"{k} {v:.1f}s" for k, v in secs.items()) + f"{extra}; total {time.time() - t0:.0f}s)")

    meta = None
    if finalize:
        missing = [li for li in moe_layers if not _layer_done(out, li, want, tiers, lay.E, lay.bias)]
        if missing:
            raise RuntimeError(f"cannot finalize {out}: MoE layers {missing[:10]}{'...' if len(missing) > 10 else ''} "
                               "missing or incomplete (run the --layers parts first)")
        if dense_info is None:
            dense_info = DENSE_WRITERS[dense_format](km, out, device=dev, log=log)
            host_info = _write_host(km, out, log=log)
        secs_total = 0.0
        for li in moe_layers:
            p = out / f"layer{li:03d}.json"
            if p.exists():
                secs_total += json.loads(p.read_text()).get("seconds", 0.0)
        exp = {"E": lay.E, "gu_shape": list(lay.gu_shape), "dn_shape": list(lay.dn_shape), "gated": lay.gated,
               "bias": lay.bias, "transposed": lay.transposed, "concatenated": lay.concatenated, "gate": lay.gate,
               "act": lay.act, "sources": es.kinds(), "source_formats": _formats_by_layer(es, moe_layers)}
        meta = {"format": 2, "arch": entry.model_type, "model_type": raw_cfg.get("model_type"),
                "transformers": hfcompat.TF_VERSION, "torch": torch.__version__, "converter": "moe_tiering.convert_any",
                "source": model_dir, "config": raw_cfg, "num_layers": n_layers, "num_experts": lay.E,
                "moe_layers": moe_layers, "gu_shape": list(lay.gu_shape), "dn_shape": list(lay.dn_shape),
                "hot": want["hot"], "cold": want["cold"], "experts": exp,
                "bias": {"file": "layer{L:03d}.bias.safetensors", "keys": list(BIAS_KEYS),
                         "shapes": [[lay.E, lay.gu_shape[0]], [lay.E, lay.H]]} if lay.bias else None,
                "dense": dense_info, **({"host": host_info} if host_info else {}),
                "source_formats": km.source_formats(), "skipped": dict(km.skipped),
                "keymap": _keymap_summary(km), "layer_seconds_total": round(secs_total, 1)}
        _atomic_write(out / "meta.json", lambda t: t.write_text(json.dumps(meta, indent=2, default=str)))
    index.close()
    log(f"done in {time.time() - t0:.0f}s ({done_s:.0f}s converting {len(converted)} layers) -> {out}")
    return {"out": str(out), "arch": entry.model_type, "converted": converted, "moe_layers": moe_layers,
            "ignored_dense_layers": not_moe, "seconds": round(time.time() - t0, 1), "meta": meta}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="auto", help="assert the architecture (default: auto, from config.json)")
    ap.add_argument("--layers", help="subset of decoder layers, e.g. 0-23 or 0,5,9 (default: all; dense layers are "
                                     "ignored). A subset writes only its layer files; run --finalize once every part "
                                     "is done")
    ap.add_argument("--finalize", action="store_true",
                    help="check that every MoE layer is present, then write the dense part and meta.json")
    ap.add_argument("--chunk", type=int, default=16, help="experts quantized per GPU batch")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hot-format", default="rtn", help="hot tier store format (default rtn, as moe_tiering.convert)")
    ap.add_argument("--hot-bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64, help="hot group size (and rtn cold group by default)")
    ap.add_argument("--cold-format", default="rtn")
    ap.add_argument("--cold-bits", type=int, default=2)
    ap.add_argument("--cold-group", type=int, help="default: --group for rtn, 16 for k4row")
    ap.add_argument("--stats", help="importance statistics dir (layer{L:03d}.pt with sumsq/count), k4row tiers")
    ap.add_argument("--cold-only", action="store_true", help="no hot blobs (the store needs --hot-from-cold)")
    ap.add_argument("--dense-format", default="plain",
                    help="dense writer: plain (default: unquantized, stored dtypes kept) or bf16 (from_pretrained's "
                         "dtypes)")
    ap.add_argument("--plugin", action="append", default=[],
                    help="import this module first (it registers more quantizers / dense writers)")
    args = ap.parse_args(argv)
    for mod in args.plugin:
        importlib.import_module(mod)
    for name in (args.hot_format, args.cold_format):
        if name not in QUANTIZERS:
            ap.error(f"expert format {name!r}: registered {sorted(QUANTIZERS)}")
    if args.dense_format not in DENSE_WRITERS:
        ap.error(f"dense format {args.dense_format!r}: registered {sorted(DENSE_WRITERS)}")
    from transformers import AutoConfig

    n_layers = arch.text_config(AutoConfig.from_pretrained(args.model, trust_remote_code=False)).num_hidden_layers
    layers = parse_layers(args.layers, n_layers)
    if args.finalize and layers is not None:
        ap.error("--finalize checks and finishes the whole store: do not combine it with --layers")
    convert(args.model, args.out, device=args.device, hot_bits=args.hot_bits, cold_bits=args.cold_bits,
            group=args.group, chunk=args.chunk, layers=[] if args.finalize else layers,
            log=lambda *a: print(*a, flush=True), cold_format=args.cold_format, cold_group=args.cold_group,
            stats_dir=args.stats, cold_only=args.cold_only, finalize=args.finalize or layers is None,
            hot_format=args.hot_format, dense_format=args.dense_format, arch_name=args.arch)


if __name__ == "__main__":
    main()
