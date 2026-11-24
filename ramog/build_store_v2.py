"""Store v2 directories (design-quality section 2): one published variant = one directory.

    build_store_v2(codes, dense_dirs, bf16_dense, config, counts, out, variant, plan)    (WP14's interface)

writes
    meta.json                core's format 2 (ramog.convert_any's fields: arch, config, moe_layers, experts, ...)
                             + variant, cold (trellis "tcq" entry with per-layer patterns, LUT sha256s and rotation
                             sign hashes; or H4's k4row 4/16 entry), hot: null, dense (core's meaning: the
                             dense.safetensors writer entry), dense_q (the packed dense recipes present; amendment 2),
                             counts, plan, calibration, encoder (amendment 7: the encoder's sha256), converter
    layer{L:03d}.cold.bin    uint8 [E, nbytes_L]: trellis blobs assembled from T4-format codes (ramog.tcq.from_t4 /
                             fuse_gate_up / TcqLayout.assemble), or H4's k4row blobs copied from cold_dir
    layer{L:03d}.bias.safetensors   expert biases (gpt-oss), from bias_dir
    dense/, dense-c/         the packed dense parts (moe-tiering/dense-q v1 directories), hard-linked when possible
    dense.safetensors        every dense tensor unquantized (ramog.convert_any's "plain" writer): today's loader
                             reads the store with it; a WP10 loader reads the packed parts from dense/ and dense-c/
    counts.safetensors       float32 [n_sites, E] routed tokens per expert on the calibration corpus
    plan.json                trellis rungs: the tcq-plan/1 plan
    SHA256SUMS               every other file, relative paths
Every trellis layer file is checked after it is written: the blob views of every expert equal the source code words
and row scales (torch.equal), and 3 experts decode to the same rotated-domain weights (ramog.tcq.decode).
A trellis store must record the encoder that made its codes (design-quality amendment 7): `encoder` = {"sha256": the
encoder source's sha256, "expert_group": N, ...}; ramog.convert_tcq passes its own, the CLI takes --encoder and
--expert-group (the codes files do not say which encoder made them).
verify_store(dir) re-checks a store with ramog.store_v2.check, the loader's check (sizes, codec, LUT sha256s against
the engine's manifest, sign hashes, the encoder record, dense-q directories, counts; SHA256SUMS with sums=True).

python ramog/build_store_v2.py --codes CODES --model MODEL --dense dense=DIR [--dense dense-c=DIR] \
    [--bf16-dense FILE|auto] --counts FILE --variant TQ2.57 [--plan PLAN] --encoder FILE|SHA256 --expert-group N \
    --out DIR
python ramog/build_store_v2.py --verify STORE [--sums]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ramog import arch, ckpt, hfcompat, store_v2, tcq  # noqa: E402

MATRICES = ("gate", "up", "down")
SIGNS_HASH = "sha256 of the int8 (+1/-1) signs per input channel of ramog.quantizers.rotation(n, layer * 1000, 'cpu')"


def file_sha(path, n: int = 64) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:n]


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _link(src: Path, dst: Path) -> str:
    """Hard link src -> dst (same file system), else copy; returns how."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


signs_sha256 = store_v2.signs_sha256  # one definition for the writer and the loader's check


def write_dense_bf16(keymap: ckpt.KeyMap, out_dir: Path, log=print) -> dict:
    """Every dense tensor of the checkpoint, unquantized, into out_dir/dense.safetensors (ramog.convert_any's "plain"
    writer, reused when a matching file exists). Returns {"path", "entry"}."""
    from ramog import convert_any

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entry = convert_any.DENSE_WRITERS["plain"](keymap, out_dir, device="cpu", log=log)
    entry.pop("reused", None)  # the same entry whether the file was written now or by an earlier run
    return {"path": str(out_dir / "dense.safetensors"), "entry": entry}


def core_meta(model_dir: str, keymap: ckpt.KeyMap | None = None) -> tuple[dict, arch.ExpertsLayout, list]:
    """ramog.convert_any's format-2 fields (no tiers, no dense entry) for the checkpoint in model_dir."""
    from ramog import convert_any

    index = None
    if keymap is None:
        keymap, _meta, index = ckpt.build_keymap(str(model_dir))
    raw_cfg = json.load(open(os.path.join(model_dir, "config.json")))
    sites = keymap.sites
    lay = convert_any.uniform_experts(sites)
    es = ckpt.ExpertSource(keymap)
    moe = [s.layer_idx for s in sites]
    from transformers import AutoConfig

    n_layers = arch.text_config(AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False)).num_hidden_layers
    exp = {"E": lay.E, "gu_shape": list(lay.gu_shape), "dn_shape": list(lay.dn_shape), "gated": lay.gated,
           "bias": lay.bias, "transposed": lay.transposed, "concatenated": lay.concatenated, "gate": lay.gate,
           "act": lay.act, "sources": es.kinds(), "source_formats": convert_any._formats_by_layer(es, moe)}
    meta = {"format": 2, "arch": keymap.entry.model_type, "model_type": raw_cfg.get("model_type"),
            "transformers": hfcompat.TF_VERSION, "torch": torch.__version__, "source": str(model_dir),
            "config": raw_cfg, "num_layers": n_layers, "num_experts": lay.E, "moe_layers": moe,
            "gu_shape": list(lay.gu_shape), "dn_shape": list(lay.dn_shape), "experts": exp,
            "bias": {"file": "layer{L:03d}.bias.safetensors", "keys": list(convert_any.BIAS_KEYS),
                     "shapes": [[lay.E, lay.gu_shape[0]], [lay.E, lay.H]]} if lay.bias else None,
            "source_formats": keymap.source_formats(), "skipped": dict(keymap.skipped),
            "keymap": convert_any._keymap_summary(keymap)}
    if index is not None:
        index.close()
    return meta, lay, moe


def tcq_layer_blobs(codes: Path, layer: int, E: int, gu_shape, dn_shape) -> tuple[torch.Tensor, tcq.TcqLayout, dict]:
    """One layer's trellis blobs from T4-format codes, checked: (uint8 [E, nbytes], layout, facts)."""
    recs = {m: torch.load(codes / f"layer{layer:03d}.{m}.pt", map_location="cpu") for m in MATRICES}
    for m, r in recs.items():
        if r["layer"] != layer or r["experts"] != list(range(E)) or r.get("rot", "layer") != "layer":
            raise ValueError(f"{codes} layer {layer} {m}: layer {r['layer']}, {len(r['experts'])} experts, rot "
                             f"{r.get('rot')}")
    gate, up, down = (tcq.from_t4(recs[m]) for m in MATRICES)
    gu_w, gu_s, gu_p = tcq.fuse_gate_up(gate, up)
    dn_w, dn_s, dn_p = down
    lay = tcq.TcqLayout(tuple(gu_shape), tuple(dn_shape), gu_p, dn_p)
    arrays = {"gu_w": gu_w, "gu_s": gu_s, "dn_w": dn_w, "dn_s": dn_s}
    blobs = lay.assemble(arrays, E)
    if tuple(blobs.shape) != (E, lay.nbytes):
        raise RuntimeError(f"layer {layer}: blobs {tuple(blobs.shape)}, layout {lay.nbytes} B")
    v = lay.views(blobs)
    for k, a in arrays.items():  # every expert's arrays, bit for bit
        if not torch.equal(v[k].contiguous().view(torch.uint8), a.contiguous().view(torch.uint8)):
            raise RuntimeError(f"layer {layer}: {k} of the assembled blobs differs from the codes")
    for e in sorted({0, E // 2, E - 1}):  # decode spot check through the store views
        for which, w, s, p in (("gu", gu_w, gu_s, gu_p), ("dn", dn_w, dn_s, dn_p)):
            a = tcq.decode(v[f"{which}_w"][e], v[f"{which}_s"][e], p)
            b = tcq.decode(w[e], s[e], p)
            if not torch.equal(a, b):
                raise RuntimeError(f"layer {layer} expert {e} {which}: decode of the blob differs from the codes")
    facts = {"gu": tcq.pattern_name(*gu_p), "dn": tcq.pattern_name(*dn_p), "nbytes": lay.nbytes,
             "bits_per_expert": {m: recs[m]["bits_per_expert"] for m in MATRICES},
             "lut_sha256": {tcq.pattern_name(*gu_p): recs["gate"]["lut_sha256"],
                            tcq.pattern_name(*dn_p): recs["down"]["lut_sha256"]}}
    return blobs, lay, facts


def sha256sums(out: Path) -> int:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and ".tmp" not in p.name)
    lines = [f"{file_sha(p)}  {p.relative_to(out).as_posix()}\n" for p in files]
    tmp = out / f"SHA256SUMS.tmp{os.getpid()}"
    tmp.write_text("".join(lines))
    os.replace(tmp, out / "SHA256SUMS")
    return len(lines)


def build_store_v2(codes, dense_dirs: dict, bf16_dense, config, counts, out, variant: str, plan=None, *,
                   cold_dir=None, cold_entry: dict | None = None, bias_dir=None, encoder: dict | None = None,
                   calibration: dict | None = None, quality: dict | None = None,
                   keymap: ckpt.KeyMap | None = None, log=print) -> dict:
    """A store v2 directory (module docstring).
    codes: T4-format code directory of a trellis rung (None for H4); cold_dir: a directory of layer{L:03d}.cold.bin
    k4row 4/16 blobs (H4); dense_dirs: {name in the store ("dense", "dense-c"): packed dense directory};
    bf16_dense: write_dense_bf16()'s result, or a path to a dense.safetensors; config: the model directory;
    counts: counts.safetensors; plan: tcq-plan/1 dict or path (rungs). Returns meta."""
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = str(config)
    meta, lay, moe = core_meta(model_dir, keymap)
    E = lay.E
    gib = 0.0
    weights = len(moe) * E * (lay.gu_shape[0] * lay.gu_shape[1] + lay.dn_shape[0] * lay.dn_shape[1])
    if codes is not None:
        codes = Path(codes)
        try:
            store_v2.encoder_record({"encoder": encoder})
        except store_v2.StoreError as e:
            raise ValueError(f"{codes}: {e}") from e
        if plan is None and (codes / "plan.json").exists():
            plan = json.load(open(codes / "plan.json"))
        if isinstance(plan, (str, os.PathLike)):
            plan = json.load(open(plan))
        layers, luts, signs, nbytes_total = {}, {}, {}, 0
        b_gu, _ = tcq.block_structure(lay.H)
        b_dn, _ = tcq.block_structure(lay.I)
        for li in moe:
            blobs, tl, facts = tcq_layer_blobs(codes, li, E, lay.gu_shape, lay.dn_shape)
            p = out / f"layer{li:03d}.cold.bin"
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            with open(tmp, "wb") as f:
                f.write(memoryview(blobs.numpy()))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
            if p.stat().st_size != E * tl.nbytes:
                raise RuntimeError(f"{p}: {p.stat().st_size} B, expected {E} x {tl.nbytes}")
            back = torch.from_numpy(np.fromfile(p, dtype=np.uint8)).view(E, tl.nbytes)
            if not torch.equal(back, blobs):
                raise RuntimeError(f"{p}: file differs from the assembled blobs")
            layers[str(li)] = {"gu": facts["gu"], "dn": facts["dn"], "nbytes": tl.nbytes}
            for pat, sha in facts["lut_sha256"].items():
                if luts.setdefault(pat, sha) != sha:
                    raise RuntimeError(f"pattern {pat}: two LUT sha256s in one store")
                if sha != tcq.lut_sha256(pat):
                    raise RuntimeError(f"pattern {pat}: codes made with LUT {sha[:16]}, engine has "
                                       f"{tcq.lut_sha256(pat)[:16]}")
            signs[str(li)] = {"gu": signs_sha256(lay.H, li), "dn": signs_sha256(lay.I, li)}
            nbytes_total += E * tl.nbytes
            del blobs, back
        gib = nbytes_total / 2**30
        meta["cold"] = {"format": "tcq", "codec": 1, "domain": "rotated",
                        "rotation": {"kind": "signs+block_hadamard", "seed": "layer*1000",
                                     "block": {"gu": b_gu, "dn": b_dn}, "signs_hash": SIGNS_HASH,
                                     "signs_sha256": signs},
                        "luts": dict(sorted(luts.items())), "layers": layers,
                        "bpw": plan.get("bpw") if plan else None, "bpw_exact": plan.get("bpw_exact") if plan else None,
                        "blob_bpw": 8 * nbytes_total / weights, "gib": gib,
                        "codes": {"format": "t4-codes (quant_methods/t4_quant.py quantize_from_codes)",
                                  "source": str(codes)}}
        _write_json(out / "plan.json", plan)
        meta["plan"] = {"file": "plan.json", "method": (plan.get("signal") or {}).get("method")
                        or plan.get("method") or ("t4" if "t4" in plan else None),
                        "budget": plan.get("budget") or (plan.get("t4") or {}).get("budget"),
                        "bpw": plan.get("bpw"), "source": plan.get("source")}
    else:
        if cold_dir is None:
            raise ValueError("codes (trellis) or cold_dir (H4 blobs) is required")
        from ramog.store import BlobLayout

        bl = BlobLayout(4, 16, tuple(lay.gu_shape), tuple(lay.dn_shape), "k4row")
        for li in moe:
            src = Path(cold_dir) / f"layer{li:03d}.cold.bin"
            if src.stat().st_size != E * bl.nbytes:
                raise RuntimeError(f"{src}: {src.stat().st_size} B, expected {E} x {bl.nbytes}")
            _link(src, out / f"layer{li:03d}.cold.bin")
        gib = len(moe) * E * bl.nbytes / 2**30
        meta["cold"] = {**bl.describe(), "stats": None,
                        "importance": (cold_entry or {}).get("importance")
                        or {"kind": "routed-input second moments (Hessian diagonal)",
                            "corpus": (calibration or {}).get("corpus"), "tokens": (calibration or {}).get("tokens")},
                        "bpw": 8 * bl.nbytes / (lay.gu_shape[0] * lay.gu_shape[1] + lay.dn_shape[0] * lay.dn_shape[1]),
                        "gib": gib}
        meta["plan"] = None
    meta["variant"] = variant
    meta["hot"] = None
    meta["converter"] = "ramog.convert_tcq" if calibration else "ramog/build_store_v2.py"
    if bias_dir is not None:
        for li in moe:
            _link(Path(bias_dir) / f"layer{li:03d}.bias.safetensors", out / f"layer{li:03d}.bias.safetensors")
    if isinstance(bf16_dense, (str, os.PathLike)):
        bf16_dense = {"path": str(bf16_dense), "entry": {"format": "plain", "file": "dense.safetensors"}}
    _link(Path(bf16_dense["path"]), out / "dense.safetensors")
    meta["dense"] = bf16_dense["entry"]
    dq = []
    for name, d in dense_dirs.items():
        d = Path(d)
        dm = json.load(open(d / "meta.json"))
        if dm.get("format") != "moe-tiering/dense-q":
            raise ValueError(f"{d}: not a moe-tiering/dense-q directory")
        (out / name).mkdir(exist_ok=True)
        for f in sorted(d.iterdir()):
            if f.is_file() and ".tmp" not in f.name:
                _link(f, out / name / f.name)
        bits = sum(e["bits_total"] for e in dm["tensors"].values())
        dq.append({"dir": name, "recipe": dm["recipe"], "format": dm["format"], "version": dm["version"],
                   "gib": bits / 8 / 2**30, "tensors": len(dm["tensors"]), "bf16_modules": len(dm.get("bf16", []))})
    meta["dense_q"] = dq
    if counts is not None:
        _link(Path(counts), out / "counts.safetensors")
        from safetensors import safe_open

        with safe_open(str(out / "counts.safetensors"), "pt") as f:
            cm = f.metadata() or {}
            shape = list(f.get_slice("counts").get_shape())
        if shape != [len(moe), E]:
            raise ValueError(f"{counts}: counts {shape}, expected {[len(moe), E]}")
        meta["counts"] = {"file": "counts.safetensors", "corpus": cm.get("corpus"), "tokens": int(cm.get("tokens", 0)),
                          "windows": int(cm.get("windows", 0)), "shape": shape}
    else:
        meta["counts"] = None
    meta["calibration"] = calibration
    meta["encoder"] = encoder
    meta["quality"] = quality
    meta["experts_gib"] = gib
    _write_json(out / "meta.json", meta)
    n = sha256sums(out)
    log(f"store {variant} -> {out}: {len(moe)} layer files, {gib:.2f} GiB of experts, {n} files in SHA256SUMS "
        f"({time.time() - t0:.0f}s)")
    return meta


def verify_store(store, sums: bool = False) -> dict:
    """Writer-side check of a store v2 directory: ramog.store_v2.check (the loader's own check: every layer file is
    E x nbytes_L; trellis: codec, every LUT sha256 against the engine's manifest, every sign hash recomputed, the
    encoder record; dense directories are dense-q; counts; SHA256SUMS and their coverage with sums)."""
    store = Path(store)
    r = store_v2.check(store, verify_sums=sums, strict=False)
    return {"store": str(store), "variant": r["variant"], "format": r["format"], "layers": r["layers"],
            "sums_checked": r["sums_checked"], "encoder": r.get("encoder"), "bad": r["problems"], "ok": r["ok"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", default=None, help="check this store instead of building one")
    ap.add_argument("--sums", action="store_true", help="with --verify: also check SHA256SUMS")
    ap.add_argument("--codes", default=None, help="T4-format trellis codes (a rung)")
    ap.add_argument("--cold-dir", default=None, help="k4row 4/16 blobs layer{L:03d}.cold.bin (H4)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dense", action="append", default=[], help="NAME=DIR (dense=..., dense-c=...)")
    ap.add_argument("--bf16-dense", default="auto", help="dense.safetensors, or auto (written into OUT/../common)")
    ap.add_argument("--counts", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--bias-dir", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--encoder", default=None, help="trellis codes: the encoder source file that made them (its "
                    "sha256 is recorded), or its sha256 (64 hex digits); amendment 7")
    ap.add_argument("--expert-group", type=int, default=None, help="trellis codes: the encoder's expert group "
                    "(0 = T4's batches, T4 / QW3-era codes)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.verify:
        r = verify_store(a.verify, a.sums)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    if not (a.model and a.out and a.variant):
        ap.error("--model, --out and --variant are required to build")
    bf16 = a.bf16_dense
    if bf16 == "auto":
        km, _m, idx = ckpt.build_keymap(a.model)
        bf16 = write_dense_bf16(km, Path(a.out).parent / "common")
        idx.close()
    dense = dict(x.split("=", 1) for x in a.dense)
    enc = None
    if a.encoder is not None:
        if os.path.isfile(a.encoder):
            enc = {os.path.basename(a.encoder): file_sha(a.encoder, 16), "sha256": file_sha(a.encoder),
                   "source": os.path.abspath(a.encoder), "expert_group": a.expert_group}
        else:
            enc = {"sha256": a.encoder.lower(), "expert_group": a.expert_group}
    build_store_v2(a.codes, dense, bf16, a.model, a.counts, a.out, a.variant, a.plan, cold_dir=a.cold_dir,
                   bias_dir=a.bias_dir, encoder=enc)


if __name__ == "__main__":
    main()
