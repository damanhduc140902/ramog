"""Trellis rate plans for any MoE model (design-quality section 6): per-layer code patterns at a fixed expert budget.

T4's allocator (agents/quality/t4/alloc.py, as QD's agents/qint/qd/plans.py reproduces it; ported here with the
model's dims as arguments instead of module globals, operation for operation, so a plan made here equals QD's / the
q235 lane's for the same inputs):
  units u = (allocation group g of 3 consecutive MoE layers, gate/up or down); KL ~= sum_u c_u M(R_u) with
  c_u = s_g * share_m / M(base_m) (base 2.25 for gate/up, 3.0 for down), M = the trellis code's Gaussian MSE per rate
  (the LUT files' "gaussian_mse"); greedy marginal allocation over the rates 1.75 .. 4.0 at a fixed budget of code +
  fp16 row-scale bits per expert weight. Only the RELATIVE group sensitivities s_g and the gate/up : down share steer
  the plan.
Signals (section 6's rule for a new model):
  s4fit    (default) 4 grouped sensitivity runs at depth fractions [0, .5), [.5, .75), [.75, .875), [.875, 1] (KL of
           DEV and OOD-DEV above the BF16 floor, only those layers at tcq gu 2.25 / dn 3.0) and the share from two
           whole-model runs: the depth curve s(f) = c (1 + a exp((f - 1) / tau)) fitted to the runs (fit_depth), then
           evaluated per layer at f = layer / (layers - 1) and summed per allocation group
  depth16  no eval possible: the Qwen3-30B curve (a 5.2, tau 0.11) with share 0.82 (untested on other architectures)
  runs     per-layer sensitivities from N grouped runs taken as steps (s4 / T4's 16 runs)
The exact bits of a plan (moe_tiering.tcq_plan.validate: codes + row scales + each LUT's share) include the LUT bits the
allocator does not see; make_plan lowers the allocator budget until the exact bpw is at or below the target (the q235
lane's s4fit.py procedure).

    from moe_tiering import tcq_rate
    plan = tcq_rate.make_plan(dims, 2.5747, signal="depth16")            # tcq-plan/1 dict, validated
    plan = tcq_rate.make_plan(dims, 3.9913, signal="s4fit", runs=[(0, 23, 0.0242), ...], share=0.817)
python -m moe_tiering.tcq_rate --config MODEL_DIR --rung TQ2.57 [--signal depth16|s4fit --runs a-b:kl,... --share S] --out P
"""

from __future__ import annotations

import json
import math

import torch

from . import tcq_plan
from .tcq_encode import lut_file, parse_spec

RATES = {1.75: "3,4", 2.0: "4", 2.25: "4,5", 2.5: "5", 2.75: "5,6", 3.0: "6", 3.25: "6,7", 3.5: "7", 3.75: "7,8",
         4.0: "8"}
# the ladder (design-quality sections 1.2 / 5): rung -> exact expert bpw cap (the Qwen3-30B rung's exact bpw)
LADDER = {"TQ2.0": 1.9966, "TQ2.26": 2.2570, "TQ2.57": 2.5747, "TQ3.0": 2.9966, "TQ3.5": 3.4966, "TQ4.0": 3.9913}
DEPTH16 = {"c": 1.0, "a": 5.2, "tau": 0.11, "share": 0.82}  # section 6, rule 2 (Qwen3-30B's curve)
S4_FRACTIONS = ((0.0, 0.5), (0.5, 0.75), (0.75, 0.875), (0.875, 1.0))
BASE_GU, BASE_DN = 2.25, 3.0


def mse_table(lut_dir: str | None = None) -> dict:
    """{rate: Gaussian MSE of the trellis code at that rate} from the LUT files (T4's "gaussian_mse")."""
    out = {}
    for r, p in RATES.items():
        o = torch.load(lut_file(parse_spec(f"tcq:L12:V2:b{p}"), lut_dir), map_location="cpu")
        out[r] = float(o["gaussian_mse"])
    return out


def alloc_groups(moe_layers: list[int], size: int = 3) -> list[list[int]]:
    """T4's allocation units: consecutive MoE layers in groups of `size` (the last one shorter)."""
    ls = list(moe_layers)
    return [ls[a:a + size] for a in range(0, len(ls), size)]


def s4_groups(n_layers: int, moe_layers: list[int]) -> list[tuple[int, int]]:
    """The 4 grouped sensitivity runs of section 6: MoE layers at depth fractions [0, .5), [.5, .75), [.75, .875),
    [.875, 1] of the decoder (f = layer / n_layers), as (first, last) layer. Qwen3-30B: 0-23, 24-35, 36-41, 42-47;
    Qwen3-235B: 0-46, 47-70, 71-82, 83-93."""
    out = []
    for lo, hi in S4_FRACTIONS:
        ls = [li for li in moe_layers if lo <= li / n_layers < hi or (hi == 1.0 and li / n_layers >= lo)]
        if not ls:
            raise ValueError(f"no MoE layer at depth fractions [{lo}, {hi})")
        out.append((ls[0], ls[-1]))
    return out


def depth_curve(params, f: float) -> float:
    c, a, tau = params
    return c * (1.0 + a * math.exp((f - 1.0) / tau))


def fit_depth(runs: list[tuple[list[int], float]], n_layers: int) -> tuple[float, float, float]:
    """QD's fit (plans.fit_depth): relative least squares of the group means of c (1 + a exp((f - 1) / tau)) against
    the runs' per-layer KL, tau on the grid 0.01 .. 1.00, closed form for c and c a. runs = [(layers, KL above the
    floor)]. Returns (c, a, tau)."""
    best = None
    for tau in [0.01 * k for k in range(1, 101)]:
        rows, ys = [], []
        for layers, v in runs:
            fs = [li / (n_layers - 1) for li in layers]
            e = sum(math.exp((f - 1) / tau) for f in fs) / len(fs)
            rows.append((1.0, e))
            ys.append(v / len(layers))
        w2 = [1 / (y * y) for y in ys]
        s00 = sum(w2)
        s01 = sum(w * r[1] for w, r in zip(w2, rows))
        s11 = sum(w * r[1] * r[1] for w, r in zip(w2, rows))
        t0 = sum(w * y for w, y in zip(w2, ys))
        t1 = sum(w * r[1] * y for w, r, y in zip(w2, rows, ys))
        det = s00 * s11 - s01 * s01
        if abs(det) < 1e-300:
            continue
        c, b = (t0 * s11 - t1 * s01) / det, (s00 * t1 - s01 * t0) / det
        if c <= 0 or b < 0:
            continue
        err = sum(w * (c + b * r[1] - y) ** 2 for w, r, y in zip(w2, rows, ys))
        if best is None or err < best[0]:
            best = (err, c, b / c, tau)
    if best is None:
        raise ValueError("depth fit failed (no tau with c > 0 and a >= 0)")
    return best[1], best[2], best[3]


def per_layer_from_runs(runs: list[tuple[list[int], float]], moe_layers: list[int]) -> dict[int, float]:
    s: dict[int, float] = {}
    for layers, v in runs:
        for li in layers:
            s[li] = v / len(layers)
    missing = [li for li in moe_layers if li not in s]
    if missing:
        raise ValueError(f"sensitivity runs do not cover MoE layers {missing[:8]}")
    return s


def allocate(sg: list[float], sd: list[float], budget: float, M: dict, groups: list[list[int]], E: int, H: int,
             I: int, base_gu: float = BASE_GU, base_dn: float = BASE_DN, min_rate: float = 1.75,  # noqa: E741
             max_rate: float = 4.0) -> tuple[dict, float]:
    """T4's greedy (QD plans.allocate with the dims as arguments): sg / sd per allocation group. Returns ({"a-b":
    {"gu": spec, "dn": spec}}, bpw of codes + row scales)."""
    units = []
    for gi, g in enumerate(groups):
        n = len(g) * E
        units.append({"g": gi, "m": "gu", "c": sg[gi] / M[base_gu], "w": n * 2 * I * H, "rows": n * 2 * I})
        units.append({"g": gi, "m": "dn", "c": sd[gi] / M[base_dn], "w": n * I * H, "rows": n * H})
    rates = sorted(r for r in M if min_rate <= r <= max_rate)
    W = sum(u["w"] for u in units)
    row_bits = 16 * sum(u["rows"] for u in units)
    budget_bits = budget * W - row_bits
    for u in units:
        u["r"] = rates[0]
    spent = sum(u["r"] * u["w"] for u in units)
    while True:
        best = None
        for u in units:
            i = rates.index(u["r"])
            if i + 1 == len(rates):
                continue
            dr = rates[i + 1] - u["r"]
            cost = dr * u["w"]
            if spent + cost > budget_bits + 1e-6:
                continue
            gain = u["c"] * (M[u["r"]] - M[rates[i + 1]]) / cost
            if best is None or gain > best[0]:
                best = (gain, u, rates[i + 1], cost)
        if best is None:
            break
        _, u, r, cost = best
        u["r"] = r
        spent += cost
    layers = {}
    for gi, g in enumerate(groups):
        rg = {u["m"]: u["r"] for u in units if u["g"] == gi}
        layers[f"{g[0]}-{g[-1]}"] = {"gu": f"tcq:L12:V2:b{RATES[rg['gu']]}", "dn": f"tcq:L12:V2:b{RATES[rg['dn']]}"}
    return layers, (spent + row_bits) / W


def group_sensitivities(per_layer: dict[int, float], groups: list[list[int]], share: float):
    s = [sum(per_layer[li] for li in g) for g in groups]
    tot = sum(s)
    return [x / tot * share for x in s], [x / tot * (1 - share) for x in s], [x / tot for x in s]


def make_plan(dims: dict, target: float, *, signal: str = "depth16", runs=None, share: float | None = None,
              fit: tuple | None = None, n_layers: int | None = None, M: dict | None = None,
              lut_dir: str | None = None, damp: float = 0.3, min_tokens: int = 8, rung: str | None = None) -> dict:
    """A validated tcq-plan/1 plan whose exact bpw is at or below `target`.
    dims: moe_tiering.tcq_plan dims ({"moe_layers", "n_experts", "hidden", "inter"}); n_layers: decoder layers (default:
    last MoE layer + 1). signal "depth16" (section 6 rule 2), "s4fit" (runs = [(first, last, KL above floor)] of the
    4 grouped runs, share required; or fit = (c, a, tau) given directly) or "runs" (runs as steps)."""
    moe = list(dims["moe_layers"])
    n_layers = n_layers or (max(moe) + 1)
    E, H, I = int(dims["n_experts"]), int(dims["hidden"]), int(dims["inter"])  # noqa: E741
    groups = alloc_groups(moe)
    M = M or mse_table(lut_dir)
    info: dict = {"method": signal, "alloc_groups": [[g[0], g[-1]] for g in groups], "n_layers": n_layers}
    if signal == "depth16":
        fit = (DEPTH16["c"], DEPTH16["a"], DEPTH16["tau"])
        share = DEPTH16["share"] if share is None else share
        per_layer = {li: depth_curve(fit, li / (n_layers - 1)) for li in moe}
        info["fit"] = {"c": fit[0], "a": fit[1], "tau": fit[2], "source": "design-quality section 6 rule 2 (Qwen3-30B)"}
    elif signal == "s4fit":
        if share is None:
            raise ValueError("s4fit needs the gate/up : down share")
        if fit is None:
            rr = [(list(range(a, b + 1)) if not isinstance(a, list) else a, v) for a, b, v in runs]
            rr = [([li for li in ls if li in moe], v) for ls, v in rr]
            fit = fit_depth(rr, n_layers)
            info["runs"] = [[ls[0], ls[-1], v] for ls, v in rr]
        per_layer = {li: depth_curve(fit, li / (n_layers - 1)) for li in moe}
        info["fit"] = {"c": fit[0], "a": fit[1], "tau": fit[2]}
    elif signal == "runs":
        if share is None:
            raise ValueError("runs needs the gate/up : down share")
        rr = [([li for li in range(a, b + 1) if li in moe], v) for a, b, v in runs]
        per_layer = per_layer_from_runs(rr, moe)
        info["runs"] = [[a, b, v] for a, b, v in runs]
    else:
        raise ValueError(f"unknown signal {signal!r}")
    sg, sd, gsens = group_sensitivities(per_layer, groups, share)
    info.update(share_gu=share, group_sens=gsens)
    budget, tries = target, []
    for _ in range(6):
        layers, bpw_cr = allocate(sg, sd, budget, M, groups, E, H, I)
        plan = {"format": tcq_plan.FORMAT, "layers": layers, "damp": damp, "min_tokens": min_tokens, "rot": "layer"}
        rep = tcq_plan.validate(tcq_plan.load_plan(dict(plan)), dims, lut_dir)
        tries.append({"budget": budget, "bpw_codes_rows": bpw_cr, "bpw_exact": rep["bpw"]})
        if rep["bpw"] <= target + 1e-12:
            break
        budget -= rep["bpw"] - target + 1e-9
    else:
        raise ValueError(f"no plan at or below {target} bpw (tries {tries})")
    plan.update({"rung": rung, "budget": target, "bpw": rep["bpw"], "bpw_exact": rep["bpw_exact"],
                 "bpw_codes_rows": bpw_cr, "by_matrix": rep["by_matrix"], "experts_gib": rep["bytes"] / 2**30,
                 "allocator_tries": tries, "signal": info, "tool": "moe_tiering.tcq_rate (T4's allocator, QD's port)"})
    return plan


def _main():
    import argparse

    ap = argparse.ArgumentParser(description="trellis rate plan for a model (design-quality section 6)")
    ap.add_argument("--config", required=True, help="model directory (its config.json gives the dims)")
    ap.add_argument("--rung", default=None, help=f"one of {list(LADDER)} (its exact bpw cap)")
    ap.add_argument("--target", type=float, default=None, help="exact expert bpw cap (instead of --rung)")
    ap.add_argument("--signal", default="depth16", choices=("depth16", "s4fit", "runs"))
    ap.add_argument("--runs", default=None, help="a-b:kl,... (KL of DEV and OOD-DEV above the BF16 floor)")
    ap.add_argument("--share", type=float, default=None)
    ap.add_argument("--luts", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoConfig

    from . import arch

    cfg = AutoConfig.from_pretrained(a.config)
    text = arch.text_config(cfg)
    dims = tcq_plan.dims_from_config(text)
    dims["moe_layers"] = arch.config_sparse_layers(cfg)
    target = a.target if a.target is not None else LADDER[a.rung]
    runs = None
    if a.runs:
        runs = []
        for part in a.runs.split(","):
            rng, _, v = part.partition(":")
            lo, _, hi = rng.partition("-")
            runs.append((int(lo), int(hi or lo), float(v)))
    plan = make_plan(dims, target, signal=a.signal, runs=runs, share=a.share, n_layers=text.num_hidden_layers,
                     lut_dir=a.luts, rung=a.rung)
    plan["model"] = a.config
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"{a.out}: {plan['bpw']:.6f} bpw exact (target {target}); "
          + " ".join(f"{k}:{v['gu'].rsplit('b', 1)[1]}/{v['dn'].rsplit('b', 1)[1]}" for k, v in plan["layers"].items()))


if __name__ == "__main__":
    _main()
