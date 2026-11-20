"""Store v2 (design-quality section 2; WP14): a published variant's meta.json "cold" entry <-> its per-layer expert
layouts, and the checks the loader and the publishing tools run on a store.

    cold_layouts(meta) -> {layer: layout}         every MoE layer's cold layout: a TcqStoreLayout per layer for a
                                                   trellis rung (its pattern from cold["layers"][str(L)]), one shared
                                                   BlobLayout for k4row / rtn (H4, core stores)
    layer_entry(layout) -> {"gu", "dn", "nbytes"}  a trellis layer's meta entry (the inverse of the factory)
    check(meta, verify_sums=False, *, root=None, strict=True) -> report
                                                   raises StoreError listing every problem (strict=False: returns it)
    signs_sha256(n, layer)                         the rotation sign hash recorded per layer and input size
    encoder_matches(meta)                          amendment 7: the store's codes were made by this tree's encoder
    frozen_sets(meta, F) -> {layer: [expert ids]}  WP12, the split: the F most routed experts per layer by
                                                   counts.safetensors (the frozen VRAM-only experts)

meta: the parsed meta.json dict, an moe_tiering.loader.StoreMeta, or a store directory (root is then known).

What check() verifies (section 2.4: the loader refuses an unknown codec, a LUT whose sha256 differs and a sign hash that
differs; SHA256SUMS catches corruption inside a blob):
  - every MoE layer has a layout (formats registry); trellis: codec 1, domain "rotated", each layer's patterns are
    engine patterns and its nbytes equals the TcqLayout's; no layer entry outside moe_layers;
  - trellis rotation: kind "signs+block_hadamard", seed "layer*1000", block {gu, dn} = moe_tiering.quantizers.hadamard_block
    of the input sizes, and every layer's sign hashes recomputed from moe_tiering.quantizers.rotation (a change in torch's
    RNG fails here instead of decoding garbage);
  - trellis LUTs: every pattern a layer uses is named in cold["luts"] with the engine manifest's sha256, and the
    engine's LUT file itself matches the manifest (moe_tiering.tcq.lut);
  - trellis encoder record (amendment 7): meta["encoder"] = {"sha256": 64 hex digits of the encoder source that made
    the codes (moe_tiering/tcq_encode.py), "expert_group": int >= 0, "moe_tiering/tcq_encode.py": a prefix of it (optional)}.
    Decoding does not depend on it; re-encoding does (HEAD's encoder does not reproduce codes of an older one; T4 /
    QW3-era codes need --expert-group 0). The report says whether it is this tree's encoder;
  - with root: every layer{L:03d}.cold.bin is E x nbytes_L bytes; every dense_q directory is a moe-tiering/dense-q
    directory of the recipe named; counts.safetensors has shape [len(moe_layers), E]; plan.json exists when "plan"
    names it;
  - verify_sums (with root): every line of SHA256SUMS matches its file, and every file of the store is listed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from . import formats, tcq
from .quantizers import hadamard_block, rotation

CODECS = (1,)  # the trellis bitstreams this engine decodes (moe_tiering.tcq: L 12, V 2, 16x16 tail-biting tiles)
ROTATION_KIND = "signs+block_hadamard"
ROTATION_SEED = "layer*1000"
ENCODER_FILE = Path(__file__).resolve().parent / "tcq_encode.py"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class StoreError(ValueError):
    """A store v2 directory or meta entry the engine must not load."""


@dataclass(frozen=True)
class TcqStoreLayout(tcq.TcqLayout):
    """moe_tiering.tcq.TcqLayout of one MoE layer of a store, with what the loader and the format registry read: the layer
    (its rotation seed is layer * 1000), format "tcq" (no bits / group: the rate is the layer's pattern pair) and
    dequant() through the reference decoder."""

    layer: int = 0
    format: ClassVar[str] = "tcq"
    bits: ClassVar[None] = None
    group: ClassVar[None] = None

    def pattern(self, which: str) -> tuple[int, int]:
        return self.gu_pattern if which == "gu" else self.dn_pattern

    def dequant(self, views: dict, idx: int, which: str, dtype=torch.float32) -> torch.Tensor:
        """One matrix of one slot in the ORIGINAL input basis (moe_tiering.tcq.decode_unrotated, float32 arithmetic)."""
        w = tcq.decode_unrotated(views[f"{which}_w"][idx], views[f"{which}_s"][idx], self.pattern(which), self.layer)
        return w.to(dtype)

    def describe(self) -> dict:
        return {**super().describe(), "layer": self.layer}


def tcq_layer_layout(entry: dict, gu_shape, dn_shape, layer: int) -> TcqStoreLayout:
    """The "tcq" format factory (moe_tiering.formats): cold entry + layer -> that layer's layout, codec and nbytes checked."""
    codec = entry.get("codec")
    if codec not in CODECS:
        raise StoreError(f"unknown trellis codec {codec!r}: this engine decodes codec(s) {list(CODECS)} "
                         "(moe_tiering.tcq: L12, V2, 16x16 tail-biting tiles, fp16 row scales)")
    if entry.get("domain") != "rotated":
        raise StoreError(f"trellis domain {entry.get('domain')!r}: the engine decodes rotated-domain codes only")
    ent = (entry.get("layers") or {}).get(str(layer))
    if ent is None:
        raise StoreError(f"layer {layer}: no entry in cold.layers")
    try:
        lay = TcqStoreLayout(tuple(gu_shape), tuple(dn_shape), tcq.parse_pattern(ent["gu"]),
                             tcq.parse_pattern(ent["dn"]), layer=int(layer))
    except (KeyError, ValueError) as e:
        raise StoreError(f"layer {layer}: {e}") from e
    if ent.get("nbytes") != lay.nbytes:
        raise StoreError(f"layer {layer}: meta says {ent.get('nbytes')} bytes per expert, the {ent['gu']}/{ent['dn']} "
                         f"layout has {lay.nbytes}")
    return lay


def layer_entry(layout: tcq.TcqLayout) -> dict:
    return {"gu": tcq.pattern_name(*layout.gu_pattern), "dn": tcq.pattern_name(*layout.dn_pattern),
            "nbytes": layout.nbytes}


def signs_sha256(n: int, layer: int) -> str:
    """sha256 of the int8 (+1 / -1) signs per input channel of moe_tiering.quantizers.rotation(n, layer * 1000, "cpu")."""
    signs = rotation(n, tcq.rotation_seed(layer), "cpu")[0]
    return hashlib.sha256(signs.to(torch.int8).contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _resolve(meta, root):
    if isinstance(meta, (str, os.PathLike)):
        root = root or meta
        with open(os.path.join(meta, "meta.json")) as f:
            meta = json.load(f)
    elif hasattr(meta, "raw"):  # moe_tiering.loader.StoreMeta
        root = root or meta.root
        meta = meta.raw
    return meta, (Path(root) if root is not None else None)


def cold_layouts(meta) -> dict:
    """{layer: layout} of every MoE layer (formats.layer_layouts on the cold entry)."""
    meta, _ = _resolve(meta, None)
    return formats.layer_layouts(meta["cold"], meta["gu_shape"], meta["dn_shape"], meta["moe_layers"])


def frozen_sets(meta, F: int, *, root=None) -> dict[int, list[int]]:
    """{layer: [expert ids]} for a split (design-quality WP12, section 4): the F most routed experts of every MoE layer
    by counts.safetensors (tokens routed to each expert on the calibration corpus; ties go to the lower id), each
    list sorted by id. The choice changes speed, never values (hot from cold). F = 0: empty lists, no file read."""
    meta, root = _resolve(meta, root)
    moe = [int(x) for x in meta["moe_layers"]]
    E = int(meta.get("num_experts") or meta["experts"]["E"])
    F = int(F)
    if not 0 <= F <= E:
        raise ValueError(f"F={F} frozen experts per layer: expected 0..{E}")
    if F == 0:
        return {li: [] for li in moe}
    c = meta.get("counts")
    if not c or root is None:
        raise StoreError("a split needs the store's per-expert calibration counts (meta.json \"counts\", "
                         "counts.safetensors) and the store directory")
    from safetensors import safe_open

    with safe_open(str(root / c["file"]), "pt") as f:
        counts = f.get_tensor(c.get("key", "counts")).float()
        layers = f.get_tensor("layers").tolist() if "layers" in f.keys() else moe
    if list(counts.shape) != [len(moe), E] or [int(x) for x in layers] != moe:
        raise StoreError(f"counts {list(counts.shape)} for layers {layers[:4]}..., expected {[len(moe), E]} for the "
                         f"MoE layers {moe[:4]}...")
    out = {}
    for i, li in enumerate(moe):
        row = counts[i].tolist()
        out[li] = sorted(sorted(range(E), key=lambda e: (-row[e], e))[:F])
    return out


def encoder_record(meta) -> dict:
    """The store's encoder record (amendment 7) checked for form; StoreError when missing or malformed."""
    meta, _ = _resolve(meta, None)
    enc = meta.get("encoder")
    if not isinstance(enc, dict) or not _HEX64.match(str(enc.get("sha256", ""))):
        raise StoreError(f"no encoder sha256 recorded (meta encoder = {enc!r}): a trellis store must name the encoder "
                         "that made its codes (design-quality amendment 7)")
    g = enc.get("expert_group")
    if not isinstance(g, int) or isinstance(g, bool) or g < 0:
        raise StoreError(f"encoder record without a valid expert_group ({g!r}; amendment 7)")
    short = enc.get("moe_tiering/tcq_encode.py")
    if short is not None and not enc["sha256"].startswith(str(short)):
        raise StoreError(f"encoder record: moe_tiering/tcq_encode.py {short} is not a prefix of sha256 {enc['sha256']}")
    return enc


def encoder_matches(meta) -> bool:
    """True when the store's codes were made by this tree's moe_tiering/tcq_encode.py (byte for byte)."""
    return encoder_record(meta)["sha256"] == file_sha256(ENCODER_FILE)


def check(meta, verify_sums: bool = False, *, root=None, strict: bool = True) -> dict:
    """Check a store v2 (module docstring). Returns the report; raises StoreError on any problem unless strict=False."""
    meta, root = _resolve(meta, root)
    probs: list[str] = []
    cold = meta["cold"]
    name = formats.canonical_name(cold)
    moe = [int(x) for x in meta["moe_layers"]]
    E = int(meta.get("num_experts") or meta["experts"]["E"])
    gu, dn = tuple(meta["gu_shape"]), tuple(meta["dn_shape"])
    rep = {"store": str(root) if root else None, "variant": meta.get("variant"), "format": name, "layers": len(moe),
           "num_experts": E}
    lays = {}
    try:
        fmt = formats.expert_format(cold)
        for li in moe:
            try:
                lays[li] = formats.expert_layout(cold, gu, dn, li) if fmt.per_layer else \
                    lays.get(moe[0]) or formats.expert_layout(cold, gu, dn)
            except (ValueError, KeyError) as e:
                probs.append(str(e) if str(e).startswith(f"layer {li}:") else f"layer {li}: {e}")
    except ValueError as e:
        probs.append(str(e))
    if lays:
        nb = [lays[li].nbytes for li in moe if li in lays]
        rep["nbytes"] = {"min": min(nb), "max": max(nb), "distinct": len(set(nb))}
        rep["experts_bytes"] = E * sum(nb)
    if name == "tcq":
        extra = sorted(set(cold.get("layers") or {}) - {str(li) for li in moe})
        if extra:
            probs.append(f"cold.layers has entries for layers {extra[:8]} that are not MoE layers")
        rot = cold.get("rotation") or {}
        if rot.get("kind") != ROTATION_KIND or rot.get("seed") != ROTATION_SEED:
            probs.append(f"rotation kind {rot.get('kind')!r} seed {rot.get('seed')!r}: the engine rotates "
                         f"{ROTATION_KIND!r} with seed {ROTATION_SEED!r}")
        want_block = {"gu": hadamard_block(gu[1]), "dn": hadamard_block(dn[1])}
        if rot.get("block") != want_block:
            probs.append(f"rotation block {rot.get('block')} != {want_block} (moe_tiering.quantizers.hadamard_block)")
        hashes = rot.get("signs_sha256") or {}
        for li in moe:
            want = {"gu": signs_sha256(gu[1], li), "dn": signs_sha256(dn[1], li)}
            got = hashes.get(str(li))
            if got != want:
                probs.append(f"layer {li}: rotation sign hash {got} != {want} (moe_tiering.quantizers.rotation)")
        used = sorted({tcq.pattern_name(*p) for la in lays.values() for p in (la.gu_pattern, la.dn_pattern)})
        luts = cold.get("luts") or {}
        for p in used:
            if p not in luts:
                probs.append(f"LUT {p}: used by a layer, not named in cold.luts")
                continue
            try:
                ref = tcq.lut_sha256(p)
                tcq.lut(p, "cpu")  # the engine's own LUT file against its manifest
            except (RuntimeError, FileNotFoundError, KeyError) as e:
                probs.append(f"LUT {p}: engine file: {e}")
                continue
            if luts[p] != ref:
                probs.append(f"LUT {p}: store names sha256 {str(luts[p])[:16]}.., the engine's is {ref[:16]}..")
        rep["luts"] = used
        try:
            enc = encoder_record(meta)
            rep["encoder"] = {"sha256": enc["sha256"], "expert_group": enc["expert_group"],
                              "this_tree": enc["sha256"] == file_sha256(ENCODER_FILE)}
        except StoreError as e:
            probs.append(str(e))
    if root is not None:
        for li in moe:
            if li not in lays:
                continue
            p = root / f"layer{li:03d}.cold.bin"
            size = p.stat().st_size if p.exists() else None
            if size != E * lays[li].nbytes:
                probs.append(f"{p.name}: {size} bytes, expected {E} x {lays[li].nbytes}")
        for d in meta.get("dense_q") or []:
            mp = root / d["dir"] / "meta.json"
            try:
                dm = json.load(open(mp))
            except (OSError, ValueError) as e:
                probs.append(f"dense_q {d['dir']}: {e}")
                continue
            if dm.get("format") != "moe-tiering/dense-q" or dm.get("recipe") != d.get("recipe"):
                probs.append(f"dense_q {d['dir']}: format {dm.get('format')!r} recipe {dm.get('recipe')!r}, meta "
                             f"names {d.get('recipe')!r}")
        c = meta.get("counts")
        if c:
            try:
                from safetensors import safe_open

                with safe_open(str(root / c["file"]), "pt") as f:
                    shape = list(f.get_slice("counts").get_shape())
                if shape != [len(moe), E]:
                    probs.append(f"counts {shape}, expected {[len(moe), E]}")
            except Exception as e:  # noqa: BLE001  (a missing or unreadable file is a problem to report)
                probs.append(f"counts: {e}")
        pl = meta.get("plan")
        if pl and not (root / pl["file"]).exists():
            probs.append(f"plan file {pl['file']} missing")
    rep["sums_checked"] = None
    if verify_sums:
        if root is None:
            raise ValueError("verify_sums needs the store directory (root)")
        sums = root / "SHA256SUMS"
        if not sums.exists():
            probs.append("SHA256SUMS missing")
        else:
            listed = set()
            lines = sums.read_text().splitlines()
            for line in lines:
                sha, rel = line.split("  ", 1)
                listed.add(rel)
                p = root / rel
                if not p.is_file():
                    probs.append(f"{rel}: listed in SHA256SUMS, missing")
                elif file_sha256(p) != sha:
                    probs.append(f"{rel}: sha256 differs from SHA256SUMS")
            files = {p.relative_to(root).as_posix() for p in root.rglob("*")
                     if p.is_file() and p.name != "SHA256SUMS" and ".tmp" not in p.name}
            for rel in sorted(files - listed)[:8]:
                probs.append(f"{rel}: not in SHA256SUMS")
            rep["sums_checked"] = len(lines)
    rep["problems"] = probs
    rep["ok"] = not probs
    if strict and probs:
        raise StoreError(f"store {rep['store'] or rep['variant']}: {len(probs)} problem(s): " + "; ".join(probs[:6]))
    return rep
