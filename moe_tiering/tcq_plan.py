"""Rate plans for trellis-coded (TCQ) experts: which code every MoE layer's gate/up and down stacks use, the
encoder settings, and the exact bits that costs.

Plan file (JSON), format "tcq-plan/1":
    {"format": "tcq-plan/1",
     "base":   {"gu": SPEC, "dn": SPEC},                   optional: layers no range names
     "layers": {"0-2": {"gu": SPEC, "dn": SPEC}, "3": {...}, ...},   decoder-layer indices of MoE layers
     "damp": 0.3, "min_tokens": 8, "rot": "layer",          BlockLDLQ settings (moe_tiering.tcq_encode)
     ...free metadata ("source", "budget", "predicted_kl", ...)}
SPEC = moe_tiering.tcq_encode.parse_spec's "tcq:L12:V2:b5" (2.5 bits per weight), "tcq:L12:V2:b4,5" (2.25), ...
`gu` codes the gate and the up stack (same input, same Hessian), `dn` the down stack. Ranges must not overlap and
must name MoE layers only; every MoE layer needs a spec (from a range or the base).
T4's allocator output (results/quality/t4/plans/alloc-*.json: {"layers", "bpw_codes_rows", "predicted_kl",
"share_gu", "budget"}) loads as-is, with T4's final-run settings: base gu = dn = tcq:L12:V2:b5, damp 0.3.

Bits (validate): per expert matrix [R, n] coded with spec b: R * n * bpw_codes code bits + 16 * R (fp16 row scales)
+ an even share of its LUT (2^L * V * 16 bits, stored once per model) over every expert matrix decoding with it.
This is t4_quant.matrix_bits / table_users exactly, so the result equals qbench's bit account of the same codes.

python3 -m moe_tiering.tcq_plan PLAN.json [--config MODEL_DIR | --dims E,H,I,LAYERS] [--luts DIR] [--out report.json]
"""

from __future__ import annotations

import json
import os
from fractions import Fraction

from moe_tiering.tcq_encode import load_lut, lut_file, parse_spec, table_bits

FORMAT = "tcq-plan/1"
T4_BASE = "tcq:L12:V2:b5"  # the base spec of T4's final runs (every layer is overridden by the allocator's ranges)
T4_DAMP = 0.3


def parse_layers(key: str) -> list[int]:
    """'0-2' -> [0, 1, 2]; '5' -> [5]; '0-2,7' -> [0, 1, 2, 7]."""
    out: list[int] = []
    for part in str(key).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def format_layers(ls) -> str:
    """[0, 1, 2, 7] -> '0-2,7' (parse_layers' inverse)."""
    out: list[list[int]] = []
    for li in sorted(set(int(x) for x in ls)):
        if out and li == out[-1][1] + 1:
            out[-1][1] = li
        else:
            out.append([li, li])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in out)


def load_plan(src, damp: float | None = None, base: str | None = None) -> dict:
    """A plan from a path or a dict, normalised to format tcq-plan/1 (T4 alloc files are converted)."""
    if isinstance(src, (str, os.PathLike)):
        with open(src) as f:
            obj = json.load(f)
        obj.setdefault("source", str(src))
    else:
        obj = dict(src)
    if obj.get("format") not in (None, FORMAT):
        raise ValueError(f"unknown plan format {obj.get('format')!r}")
    if obj.get("format") is None:  # T4 allocator output
        meta = {k: v for k, v in obj.items() if k != "layers"}
        b = base or T4_BASE
        obj = {"format": FORMAT, "base": {"gu": b, "dn": b}, "layers": obj["layers"],
               "damp": T4_DAMP if damp is None else damp, "min_tokens": 8, "rot": "layer", "t4": meta,
               "source": meta.get("source")}
    elif damp is not None:
        obj["damp"] = damp
    obj.setdefault("damp", T4_DAMP)
    obj.setdefault("min_tokens", 8)
    obj.setdefault("rot", "layer")
    if obj["rot"] != "layer":
        raise ValueError("only rot='layer' (seed = layer * 1000, shared by the layer's experts) is supported")
    return obj


def entry(plan: dict, layer: int) -> dict:
    """{"gu", "dn", "damp", "min_tokens", "rot"} of one MoE layer (the range naming it, else the base)."""
    hit = [o for k, o in plan["layers"].items() if layer in parse_layers(k)]
    if len(hit) > 1:
        raise ValueError(f"layer {layer} is named by {len(hit)} ranges")
    base = plan.get("base") or {}
    o = hit[0] if hit else {}
    gu = o.get("gu", base.get("gu"))
    dn = o.get("dn", base.get("dn", gu))
    if gu is None or dn is None:
        raise ValueError(f"layer {layer}: no spec (no range names it and the plan has no base)")
    return {"gu": gu, "dn": dn, "damp": float(plan["damp"]), "min_tokens": int(plan["min_tokens"]),
            "rot": plan["rot"]}


def dims_from_config(cfg) -> dict:
    """{"moe_layers", "n_experts", "hidden", "inter"} from a transformers config (Qwen3-MoE, DeepSeek/GLM style
    first_k_dense_replace, mlp_layer_types). moe_tiering.arch.find_moe_sites is the authority for other layouts."""
    g = lambda k, d=None: getattr(cfg, k, d)  # noqa: E731
    n_layers = g("num_hidden_layers")
    E = g("num_experts") or g("n_routed_experts") or g("num_local_experts")
    inter = g("moe_intermediate_size") or g("intermediate_size")
    if g("mlp_layer_types"):
        moe = [i for i, t in enumerate(g("mlp_layer_types")) if t == "sparse"]
    elif g("first_k_dense_replace") is not None:
        moe = list(range(g("first_k_dense_replace"), n_layers))
    else:
        only = set(g("mlp_only_layers") or [])
        step = g("decoder_sparse_step") or 1
        moe = [i for i in range(n_layers) if i not in only and (i + 1) % step == 0]
    return {"moe_layers": moe, "n_experts": int(E), "hidden": int(g("hidden_size")), "inter": int(inter)}


def validate(plan: dict, dims: dict, lut_dir: str | None = None, check_luts: bool = True) -> dict:
    """Check a plan against a model's dims and count its bits exactly. Raises ValueError listing every problem.
    Returns {"bpw" (float), "bpw_exact" ("p/q"), "bits", "weights", "by_matrix": {gate|up|down: bpw}, "users":
    {lut_key: expert matrices}, "luts": [...], "layers": {L: {"gu", "dn", "bits_per_expert": {m: bits}}},
    "warnings": [...]}"""
    errors, warnings = [], []
    moe = list(dims["moe_layers"])
    E, H, I = int(dims["n_experts"]), int(dims["hidden"]), int(dims["inter"])
    named: dict[int, str] = {}
    for k in plan["layers"]:
        try:
            ls = parse_layers(k)
        except ValueError:
            errors.append(f"range {k!r} does not parse")
            continue
        for li in ls:
            if li not in moe:
                errors.append(f"range {k!r} names layer {li}, which is not a MoE layer")
            if li in named:
                errors.append(f"layer {li} is named by ranges {named[li]!r} and {k!r}")
            named[li] = k
    if not (plan.get("damp", 0) > 0):
        errors.append(f"damp must be > 0 (got {plan.get('damp')})")
    entries, specs = {}, {}
    for li in moe:
        try:
            entries[li] = entry(plan, li)
        except ValueError as e:
            errors.append(str(e))
            continue
        for key in ("gu", "dn"):
            s = entries[li][key]
            if s not in specs:
                try:
                    specs[s] = parse_spec(s)
                except (ValueError, KeyError, IndexError) as e:
                    errors.append(f"layer {li} {key}: bad spec {s!r} ({e})")
    if errors:
        raise ValueError("invalid TCQ plan:\n  " + "\n  ".join(errors))
    unnamed = [li for li in moe if li not in named]
    if unnamed and plan["layers"]:  # a plan of ranges that misses layers: say so (they silently took the base)
        bs = plan.get("base") or {}
        warnings.append(f"{len(unnamed)} MoE layers named by no range take the base spec (layers "
                        f"{format_layers(unnamed)}: gu {bs.get('gu')}, dn {bs.get('dn', bs.get('gu'))}"
                        + (", T4's default base for its allocator files)" if "t4" in plan else ")"))
    shapes = {"gate": (I, H), "up": (I, H), "down": (H, I)}
    for s, sp in specs.items():
        for m, (R, n) in shapes.items():
            if any((entries[li]["dn" if m == "down" else "gu"] == s) for li in moe) and (R % sp["tr"] or n % sp["tc"]):
                errors.append(f"{s}: {m} [{R}, {n}] does not tile into {sp['tr']} x {sp['tc']}")
        if sp["bits_per_tile"] % 32:
            errors.append(f"{s}: {sp['bits_per_tile']} bits per tile is not a whole number of 32-bit words")
        if len(sp["b"]) > 2 or sp["N"] % 2 or sp["V"] not in (1, 2):
            warnings.append(f"{s}: the Triton Viterbi kernel needs a pattern of 1-2 values, N even, V 1 or 2; the "
                            "torch reference (about 50x slower) would run")
    users: dict = {}
    for li in moe:
        for m in ("gate", "up", "down"):
            k = specs[entries[li]["dn" if m == "down" else "gu"]]["lut_key"]
            users[k] = users.get(k, 0) + E
    luts = []
    for s, sp in specs.items():
        if any(x["lut_key"] == sp["lut_key"] for x in luts):
            continue
        row = {"lut_key": sp["lut_key"], "spec": s, "bits": table_bits(sp), "users": users[sp["lut_key"]]}
        if check_luts:
            try:
                row["file"] = lut_file(sp, lut_dir)
                row["sha256"] = load_lut(sp, "cpu", lut_dir)[1]
            except (FileNotFoundError, RuntimeError) as e:
                errors.append(f"{s}: {e}")
        luts.append(row)
    if errors:
        raise ValueError("invalid TCQ plan:\n  " + "\n  ".join(errors))
    total = Fraction(0)
    by_m = {m: Fraction(0) for m in shapes}
    layers = {}
    for li in moe:
        row = {"gu": entries[li]["gu"], "dn": entries[li]["dn"], "bits_per_expert": {}}
        for m, (R, n) in shapes.items():
            sp = specs[entries[li]["dn" if m == "down" else "gu"]]
            codes = Fraction(sp["bits_per_tile"]) * (R // sp["tr"]) * (n // sp["tc"])
            per = codes + 16 * R + Fraction(table_bits(sp), users[sp["lut_key"]])
            row["bits_per_expert"][m] = float(per)
            by_m[m] += per * E
            total += per * E
        layers[li] = row
    w_m = len(moe) * E * H * I  # weights per matrix type
    bpw = total / (3 * w_m)
    return {"bpw": float(bpw), "bpw_exact": f"{bpw.numerator}/{bpw.denominator}", "bits": float(total),
            "bytes": float(total / 8), "weights": 3 * w_m,
            "by_matrix": {m: float(v / w_m) for m, v in by_m.items()},
            "users": users, "luts": luts, "layers": layers, "warnings": warnings,
            "dims": {"moe_layers": moe, "n_experts": E, "hidden": H, "inter": I},
            "settings": {"damp": plan["damp"], "min_tokens": plan["min_tokens"], "rot": plan["rot"]}}


def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--config", help="model directory: dims from its config.json (transformers AutoConfig)")
    ap.add_argument("--dims", help="E,H,I,LAYERS (every layer MoE) or E,H,I,FIRST-LAST")
    ap.add_argument("--luts", default=None, help="LUT directory (default: moe_tiering.tcq_encode.lut_dirs())")
    ap.add_argument("--no-luts", action="store_true", help="do not look for the LUT files")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    plan = load_plan(a.plan)
    if a.config:
        from transformers import AutoConfig

        dims = dims_from_config(AutoConfig.from_pretrained(a.config))
    elif a.dims:
        E, H, I, ls = a.dims.split(",")
        dims = {"n_experts": int(E), "hidden": int(H), "inter": int(I),
                "moe_layers": parse_layers(ls) if "-" in ls else list(range(int(ls)))}
    else:
        raise SystemExit("--config or --dims")
    rep = validate(plan, dims, a.luts, not a.no_luts)
    print(f"{a.plan}: {rep['bpw']:.6f} bpw ({rep['bpw_exact']}), gate {rep['by_matrix']['gate']:.4f} up "
          f"{rep['by_matrix']['up']:.4f} down {rep['by_matrix']['down']:.4f}; {rep['bytes'] / 2**30:.3f} GiB of experts; "
          f"{len(rep['luts'])} LUTs; warnings {rep['warnings']}")
    if a.out:
        rep = dict(rep, users={str(k): v for k, v in rep["users"].items()},
                   luts=[dict(x, lut_key=str(x["lut_key"])) for x in rep["luts"]])
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    _main()
