"""WP7 (design-core): the grouped dequant-GEMV on rows that no primary launch config divides (the kernel tail fix).

The GEMV loop reads whole steps of WORDS int32 words per row, so a launch is right only if WORDS divides the row's
K * BITS / 32 words. Before WP7 the host-memory list offered WORDS {8, 16} only and the prune fell back to its first
config (WORDS 8) when neither divided: 2-bit gpt-oss rows (K 2880: 180 words) and Gemma-4 down rows (K 704: 44
words) read from pinned host memory lost their last 4 words (64 inputs), silently. WP7 appends tail configs (WORDS
4 / 2 / 1, SPLIT_K 1) to both lists; the prune offers them only when no primary config divides, and raises when
nothing does.

Part 1, prune property (CPU, no GPU needed), for every K = 16 .. 8192, BITS 2 / 4 / 8, both lists:
  K * BITS not a whole number of words -> the prune raises; otherwise the candidates are non-empty and all divide
  the row; if a primary config divides, the candidates are exactly the dividing primary configs (the same objects,
  the same order, i.e. the pre-WP7 candidate set: with --old-kernels also checked against the old kernels.py's own
  prune, by value and order), else exactly the dividing tail configs (all SPLIT_K 1). With --old-kernels: the old
  prune returned a non-dividing config exactly at the keys where the new one takes the tail.
Part 2, kernels (GPU): K in {704, 1040, 1408, 2048, 2880} x {k4row 2/16, k4row 4/16, rtn 4/64, rtn 2/64} (K 1040
  with group 64 does not exist: K % group != 0) x the weight pool in VRAM and in pinned host memory (zero-copy) x
  {autotuned at 1, 8 and 23 pairs; every legal fixed config (WORDS divides the row; primary and tail, as the
  engine could launch them) at 8 pairs}, MODE 1 (the engine's decode launch: pairs with slot -1 are skipped and
  must stay exactly 0), N 100 rows (no BLOCK_N divides it), a pool of 6 slots, 3 tokens, random codes and scales.
  Plus the engine's hot + cold split: VRAM pool (mode 1, slots) and host pool (mode 1, expert ids) into one output,
  autotuned. Reference: float64 dequantize(W[slot]) @ float64(x[token]); error = max |y - ref| / max |ref| over
  the computed rows; pass: <= 1e-6 everywhere and skipped rows exactly 0. The autotuned picks are recorded (tail or
  primary).
--kernels FILE runs everything against another kernels.py (e.g. the pre-WP7 one, to see the test fail there).
python tests/test_kernel_tail.py [--quick] [--prune-only] [--kernels FILE] [--old-kernels FILE] [--out result.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KS = (704, 1040, 1408, 2048, 2880)
FORMATS = (("k4row", 1, 2, 16), ("k4row", 1, 4, 16), ("rtn", 0, 4, 64), ("rtn", 0, 2, 64))
AUTO_PAIRS = (1, 8, 23)
FIXED_PAIRS = 8
N, S, T = 100, 6, 3
THRESHOLD = 1e-6


def load_kernels(path: str | None, name: str):
    if path is None:
        from behemoe import kernels

        return kernels
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "engine"
    spec.loader.exec_module(mod)
    return mod


def norm(c) -> tuple:
    return (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages)


def is_tail(c) -> bool:
    return bool(getattr(c, "tail", False))


def cfg_dict(c) -> dict:
    return dict(c.kwargs, num_warps=c.num_warps)


# ------------------------------------------------------------------------------------------------- part 1: prune


def prune_property(K, old=None, k_max: int = 8192) -> dict:
    lists = {"vram": K._grouped_gemv.configs, "host": K._grouped_gemv_host.configs}
    old_lists = None if old is None else {"vram": old._grouped_gemv.configs, "host": old._grouped_gemv_host.configs}
    bad, n_keys, n_raise, tail_keys, fallback_keys = [], 0, 0, {}, {}
    for (where, configs), bits, k in itertools.product(lists.items(), (2, 4, 8), range(16, k_max + 1)):
        n_keys += 1
        if k * bits % 32:
            try:
                K._prune(configs, {"K": k}, BITS=bits)
                bad.append({"list": where, "k": k, "bits": bits, "why": "no error for a partial word"})
            except ValueError:
                n_raise += 1
            continue
        words = k * bits // 32
        try:
            got = K._prune(configs, {"K": k}, BITS=bits)
        except Exception as e:  # noqa: BLE001
            bad.append({"list": where, "k": k, "bits": bits, "why": f"raised {type(e).__name__}: {e}"})
            continue
        prim = [c for c in configs if not is_tail(c) and words % c.kwargs["WORDS"] == 0]
        tail = [c for c in configs if is_tail(c) and words % c.kwargs["WORDS"] == 0]
        want = prim or tail
        ok = bool(got) and all(words % c.kwargs["WORDS"] == 0 for c in got) and len(got) == len(want) \
            and all(a is b for a, b in zip(got, want))
        if not prim:
            ok = ok and all(c.kwargs["SPLIT_K"] == 1 for c in got)
            tail_keys.setdefault(f"{where} bits {bits}", []).append(k)
        if old_lists is not None:
            og = old._prune(old_lists[where], {"K": k}, BITS=bits)
            old_fell_back = any(words % c.kwargs["WORDS"] for c in og)
            if old_fell_back:
                fallback_keys.setdefault(f"{where} bits {bits}", []).append(k)
            if prim:  # pre-WP7 candidate set kept: same configs by value, same order
                ok = ok and not old_fell_back and [norm(c) for c in og] == [norm(c) for c in got]
            else:
                ok = ok and old_fell_back
        if not ok:
            bad.append({"list": where, "k": k, "bits": bits, "words": words, "why": "candidate set",
                        "got": [cfg_dict(c) for c in got][:6]})
    real = {}  # the real shapes this fix is for, and two that must not change
    for name, (k, bits) in {"gpt-oss 2-bit (K 2880)": (2880, 2), "gemma-4 down 2-bit (K 704)": (704, 2),
                            "gpt-oss 4-bit (K 2880)": (2880, 4), "gemma-4 down 4-bit (K 704)": (704, 4),
                            "qwen3-30b gate_up 2-bit (K 2048)": (2048, 2),
                            "qwen3-30b down 2-bit (K 768)": (768, 2)}.items():
        real[name] = {}
        for w, cl in lists.items():
            got = K._prune(cl, {"K": k}, BITS=bits)
            real[name][w] = {"n": len(got), "tail": all(is_tail(c) for c in got),
                             "words": sorted({c.kwargs["WORDS"] for c in got})}
    out = {"keys": n_keys, "raised_partial_word": n_raise, "failures": len(bad), "bad": bad[:20],
           "tail_keys_count": {k: len(v) for k, v in tail_keys.items()},
           "tail_keys_first": {k: v[:12] for k, v in tail_keys.items()}, "real_shapes": real,
           "checked_against_old": old is not None}
    if old is not None:
        out["old_fallback_keys_count"] = {k: len(v) for k, v in fallback_keys.items()}
        out["old_fallback_equals_new_tail"] = {k: fallback_keys.get(k, []) == v for k, v in tail_keys.items()} | \
            {k: False for k in fallback_keys if k not in tail_keys}
        out["primary_lists_equal_old"] = {w: [norm(c) for c in lists[w] if not is_tail(c)]
                                          == [norm(c) for c in old_lists[w]] for w in lists}
        if not all(out["old_fallback_equals_new_tail"].values()) or not all(out["primary_lists_equal_old"].values()):
            out["failures"] += 1
    return out


# ------------------------------------------------------------------------------------------------ part 2: kernels


def make_pool(fmt: int, bits: int, group: int, k: int, g: torch.Generator, dev):
    qw = torch.randint(0, 256, (S, N, k * bits // 8), generator=g, device=dev, dtype=torch.uint8)
    G = k // group
    if fmt == 0:
        scale = (torch.rand(S, N, G, generator=g, device=dev) * 1.5e-3 + 5e-4).half()
        minimum = (-scale.float() * torch.rand(S, N, G, generator=g, device=dev) * (2**bits - 1)).half()
        return {"qweight": qw, "scale": scale, "minimum": minimum}
    sc = torch.randint(1, 16, (S, N, G), generator=g, device=dev, dtype=torch.uint8)
    mc = torch.randint(0, 16, (S, N, G), generator=g, device=dev, dtype=torch.uint8)
    d = (torch.rand(S, N, generator=g, device=dev) * 1.8e-4 + 2e-5).half()
    dmin = (torch.rand(S, N, generator=g, device=dev) * 1.8e-4 + 2e-5).half()
    return {"qweight": qw, "scale": sc | (mc << 4), "minimum": None, "d": d, "dmin": dmin}


def reference(fmt: int, bits: int, group: int, p: dict, x64: torch.Tensor) -> torch.Tensor:
    """[S, T, N] float64: every slot dequantized in float64 (behemoe.quant / behemoe.kquant packing), times every token."""
    b = p["qweight"].to(torch.int32)
    q = torch.stack([(b >> (bits * i)) & (2**bits - 1) for i in range(8 // bits)], -1).reshape(S, N, -1).double()
    k = q.shape[-1]
    q = q.view(S, N, k // group, group)
    if fmt == 0:
        w = q * p["scale"][..., None].double() + p["minimum"][..., None].double()
    else:
        code = p["scale"].to(torch.int32)
        w = (p["d"][:, :, None, None].double() * (code & 15)[..., None].double() * q
             - p["dmin"][:, :, None, None].double() * (code >> 4)[..., None].double())
    return torch.einsum("snk,tk->stn", w.reshape(S, N, k), x64)


def kernel_cases(K, quick: bool, dev) -> tuple[list, list, dict]:
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    cpu_g = torch.Generator()
    cpu_g.manual_seed(1)
    tuners = {"vram": K._grouped_gemv, "host": K._grouped_gemv_host}
    rows, bad, picks = [], [], {}
    t0 = time.time()

    def check(y, ref_rows, skip, meta):
        torch.cuda.synchronize()
        yc = y.double().cpu()
        err = 0.0 if skip.all() else \
            float((yc[~skip] - ref_rows[~skip]).abs().max() / ref_rows[~skip].abs().max().clamp(min=1e-30))
        zero = bool((yc[skip] == 0).all())
        row = dict(meta, rel_err=err, skipped_rows_zero=zero, ok=err <= THRESHOLD and zero)
        rows.append(row)
        if not row["ok"]:
            bad.append(row)
            print("BAD", row, flush=True)

    for k, (fname, fmt, bits, group) in itertools.product(KS, FORMATS):
        if k % group:
            continue
        words = k * bits // 32
        x = torch.randn(T, k, generator=g, device=dev).to(torch.bfloat16)
        p = make_pool(fmt, bits, group, k, g, dev)
        ref = reference(fmt, bits, group, p, x.double()).cpu()  # [S, T, N]
        pools = {"vram": p, "host": {n: (v.cpu().pin_memory() if v is not None else None) for n, v in p.items()}}
        for where, pp in pools.items():
            kw = {"qweight": pp["qweight"], "scale": pp["scale"], "minimum": pp["minimum"], "bits": bits,
                  "group": group, "fmt": fmt, "d": pp.get("d"), "dmin": pp.get("dmin")}

            def launch(n_pairs, config=None, meta=None):
                tok = torch.randint(0, T, (n_pairs,), generator=cpu_g, dtype=torch.int32)
                slot = torch.randint(0, S, (n_pairs,), generator=cpu_g, dtype=torch.int32)
                skip = torch.rand(n_pairs, generator=cpu_g) < 1 / 3
                if n_pairs == 1:
                    skip[0] = False
                sel = torch.where(skip, torch.full_like(slot, -1), slot)
                y = torch.zeros(n_pairs, N, device=dev, dtype=torch.float32)
                K.grouped_gemv(x, tok.to(dev), sel.to(dev), mode=1, out=y, config=config, **kw)
                er = ref[slot.long(), tok.long()]
                er[skip] = 0
                check(y, er, skip, meta)

            # autotuned (the engine's launch), recording the pick
            for n_pairs in (AUTO_PAIRS if not quick else (8,)):
                launch(n_pairs, None, {"k": k, "format": fname, "bits": bits, "group": group, "words": words,
                                       "pool": where, "pairs": n_pairs, "config": "auto"})
            for key, c in tuners[where].cache.items():
                if key[1] == k and key[2] == bits and key[3] == group and key[4] == fmt:
                    picks[f"{where} K{k} {fname} {bits}/{group} P{key[5]}"] = dict(cfg_dict(c), tail=is_tail(c))
            # every legal fixed config of this pool's list (deduplicated by value)
            seen, legal = set(), []
            for c in tuners[where].configs:
                if words % c.kwargs["WORDS"] == 0 and norm(c) not in seen:
                    seen.add(norm(c))
                    legal.append(c)
            if quick:  # every tail config, plus the first 3 primary ones
                legal = [c for c in legal if is_tail(c)] + [c for c in legal if not is_tail(c)][:3]
            for c in legal:
                launch(FIXED_PAIRS, cfg_dict(c), {"k": k, "format": fname, "bits": bits, "group": group,
                                                  "words": words, "pool": where, "pairs": FIXED_PAIRS,
                                                  "config": cfg_dict(c), "tail": is_tail(c)})
        # the engine's decode split: hot pairs from the VRAM pool, the rest zero-copy from the host pool, one output
        for n_pairs in (8, 23):
            tok = torch.randint(0, T, (n_pairs,), generator=cpu_g, dtype=torch.int32)
            ex = torch.randint(0, S, (n_pairs,), generator=cpu_g, dtype=torch.int32)
            hot = torch.rand(n_pairs, generator=cpu_g) < 0.5
            y = torch.zeros(n_pairs, N, device=dev, dtype=torch.float32)
            for where, sel in (("vram", torch.where(hot, ex, -1)), ("host", torch.where(hot, -1, ex))):
                pp = pools[where]
                K.grouped_gemv(x, tok.to(dev), sel.to(torch.int32).to(dev), pp["qweight"], pp["scale"], pp["minimum"],
                               bits, group, mode=1, out=y, fmt=fmt, d=pp.get("d"), dmin=pp.get("dmin"))
            check(y, ref[ex.long(), tok.long()], torch.zeros(n_pairs, dtype=torch.bool),
                  {"k": k, "format": fname, "bits": bits, "group": group, "words": words, "pool": "vram+host",
                   "pairs": n_pairs, "config": "auto"})
        del pools, p, ref
        worst = max((r["rel_err"] for r in rows if r["k"] == k and r["format"] == fname and r["bits"] == bits), default=0)
        print(f"[{time.time() - t0:5.0f}s] K {k} {fname} {bits}/{group} ({words} words): "
              f"{sum(1 for r in rows if r['k'] == k and r['format'] == fname and r['bits'] == bits)} launches, "
              f"worst {worst:.2e}", flush=True)
    return rows, bad, picks


def config_assert(K, dev) -> dict:
    """grouped_gemv(config=...) refuses a WORDS that does not divide the row (new kernels only)."""
    k, bits, group = 2880, 2, 16
    p = make_pool(1, bits, group, k, torch.Generator(device=dev).manual_seed(3), dev)
    x = torch.zeros(1, k, device=dev, dtype=torch.bfloat16)
    one = torch.zeros(1, dtype=torch.int32, device=dev)
    try:
        K.grouped_gemv(x, one, one, p["qweight"], p["scale"], None, bits, group, fmt=1, d=p["d"], dmin=p["dmin"],
                       config={"BLOCK_N": 32, "SPLIT_K": 1, "WORDS": 8, "num_warps": 4})
        return {"raised": False, "ok": False}
    except AssertionError as e:
        return {"raised": True, "message": str(e), "ok": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", help="test this kernels.py instead of behemoe/kernels.py")
    ap.add_argument("--old-kernels", help="the pre-WP7 kernels.py: the prune property also compares with it")
    ap.add_argument("--quick", action="store_true", help="fixed configs: every tail one + 3 primary; autotuned at 8 pairs")
    ap.add_argument("--prune-only", action="store_true", help="part 1 only (no GPU needed)")
    ap.add_argument("--out")
    a = ap.parse_args()
    K = load_kernels(a.kernels, "engine._kernels_under_test")
    old = load_kernels(a.old_kernels, "engine._kernels_old") if a.old_kernels else None
    t0 = time.time()
    res = {"kernels": str(Path(K.__file__).resolve()), "torch": torch.__version__}
    import triton

    res["triton"] = triton.__version__
    res["has_tail_configs"] = hasattr(K, "_tail_configs")
    if res["has_tail_configs"]:
        res["prune"] = prune_property(K, old)
        print("prune property:", json.dumps({k: v for k, v in res["prune"].items() if k != "bad"}), flush=True)
    ok = res.get("prune", {}).get("failures", 0) == 0
    if not a.prune_only:
        dev = torch.device("cuda")
        res["gpu"] = torch.cuda.get_device_name(dev)
        rows, bad, picks = kernel_cases(K, a.quick, dev)
        res["kernels_summary"] = {
            "launches": len(rows), "failures": len(bad), "threshold": THRESHOLD, "quick": a.quick,
            "max_rel_err": max(r["rel_err"] for r in rows),
            "max_rel_err_by_pool": {w: max(r["rel_err"] for r in rows if r["pool"] == w)
                                    for w in ("vram", "host", "vram+host")},
            "max_rel_err_by_k": {str(k): max(r["rel_err"] for r in rows if r["k"] == k) for k in KS},
            "fixed_configs_run": sum(1 for r in rows if r["config"] != "auto"),
            "tail_fixed_configs_run": sum(1 for r in rows if r.get("tail")),
            "autotuned_picks": picks}
        res["failures"] = bad[:50]
        res["rows"] = rows
        if res["has_tail_configs"]:
            res["config_assert"] = config_assert(K, dev)
            ok = ok and res["config_assert"]["ok"]
        ok = ok and not bad
        print(json.dumps({k: v for k, v in res["kernels_summary"].items() if k != "autotuned_picks"}, indent=1))
    res["ok"] = bool(ok and res["has_tail_configs"])
    res["seconds"] = round(time.time() - t0, 1)
    print("RESULT", "PASS" if res["ok"] else "FAIL", f"{res['seconds']}s", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
