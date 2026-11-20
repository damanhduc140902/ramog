"""Grouped dequant-GEMM (moe_tiering.kernels.moe_gemm, the prefill path) vs a float64 reference, both weight formats.

Cases: FMT 0 (rtn, groups 32/64) and FMT 1 (k4row, groups 16/64) x bits 2, 4 x the expert matrix shapes of
Qwen3-30B (gate_up [1536, 2048], down [2048, 768]) and Qwen3-235B ([3072, 4096], [4096, 1536]) x the pool in VRAM
and in pinned host memory (zero-copy) x MODE 0, 1, 2 and 1+2 into one output x block_m 16, 32, 64 x prompts of 1,
38 and 300 tokens (top-8 of a pool of 24 experts, Zipf-skewed: some experts get many pairs, some none) x both
input uses (in_div 8: a gate_up pair reads its token's row; in_div 1: a down pair reads its own row).
Reference: float64 (dequant(W) rounded to bf16, which is what the kernel feeds tl.dot; unrounded where the GPU
takes the float32 dot, ENGINE_DOT_F32) @ float64(x); error =
max |y - ref| / max |ref| over the rows a launch must write; the rows it must skip keep their sentinel. Three fixed
launch configs (those that fit this GPU's shared memory) everywhere; the autotuned wrappers on the 38- and 300-token
prompts at the block_m the engine picks for them (kernels.block_m_for: 16, and 64 in VRAM / 128 from host memory).
python tests/test_moe_gemm.py [--quick] [--out result.json]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_tiering.kernels import (LDS_BYTES, _device_target, _gemm_smem, block_m_for, dot_f32,  # noqa: E402
                            moe_blocks, moe_gemm)
from moe_tiering.store import BlobLayout  # noqa: E402

S = 24  # experts in the pool
TOPK = 8
SHAPES = ((1536, 2048), (2048, 768), (3072, 4096), (4096, 1536))
CONFIGS = ({"BLOCK_N": 32, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
           {"BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2},
           {"BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
SENTINEL = 7.0


def random_pool(lay: BlobLayout, g: torch.Generator, dev) -> torch.Tensor:
    """[S, nbytes] blobs: random codes, valid random scale fields (as tests/test_k4row_kernel.py)."""
    pool = torch.randint(0, 256, (S, lay.nbytes), generator=g, device=dev, dtype=torch.uint8)
    v = lay.views(pool)
    for m in ("gu", "dn"):
        if lay.format == "rtn":
            s = torch.rand(v[f"{m}_s"].shape, generator=g, device=dev) * 1.5e-3 + 5e-4
            v[f"{m}_s"].copy_(s.half())
            v[f"{m}_m"].copy_((-s * torch.rand(s.shape, generator=g, device=dev) * (2**lay.bits - 1)).half())
        else:
            sc = torch.randint(1, 16, v[f"{m}_sb"].shape, generator=g, device=dev, dtype=torch.uint8)
            mc = torch.randint(0, 16, v[f"{m}_sb"].shape, generator=g, device=dev, dtype=torch.uint8)
            v[f"{m}_sb"].copy_(sc | (mc << 4))
            for f in ("d", "dm"):
                v[f"{m}_{f}"].copy_((torch.rand(v[f"{m}_{f}"].shape, generator=g, device=dev) * 1.8e-4 + 2e-5).half())
    return pool


def routing(tokens: int, g: torch.Generator) -> torch.Tensor:
    """[tokens, TOPK] distinct experts per token with Zipf-skewed popularity."""
    w = 1.0 / torch.arange(1, S + 1).float() ** 1.1
    return torch.stack([torch.multinomial(w, TOPK, replacement=False, generator=g) for _ in range(tokens)])


def reference(w64: torch.Tensor, wsel: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """y[p] = w64[wsel[p]] @ xs[p] in float64, one matmul per distinct weight."""
    ref = torch.zeros(xs.shape[0], w64.shape[1], dtype=torch.float64, device=xs.device)
    for s in wsel.unique().tolist():
        m = wsel == s
        ref[m] = xs[m] @ w64[s].t()
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Qwen3-30B shapes only")
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    cg = torch.Generator()
    cg.manual_seed(1)
    shapes = SHAPES[:2] if args.quick else SHAPES
    f32 = dot_f32(dev)
    backend, arch, limit = _device_target()
    print(f"{torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).gcnArchName} ({backend} {arch}, "
          f"{limit} B shared memory per block): tl.dot inputs {'float32' if f32 else 'bf16'}", flush=True)
    rows, bad, worst = [], [], {}
    t0 = time.time()
    for (n, k), bits, fmt in itertools.product(shapes, (2, 4), ("rtn", "k4row")):
        for group in ((16, 64) if fmt == "k4row" else (32, 64)):
            lay = BlobLayout(bits, group, (n, k), (k, n), fmt)  # matrix "gu" [n, k] is the one under test
            pool = random_pool(lay, g, dev)
            vv = lay.views(pool)
            wdt = torch.float32 if f32 else torch.bfloat16  # what the kernel feeds tl.dot
            w64 = torch.stack([lay.dequant(vv, s, "gu", torch.float32).to(wdt) for s in range(S)]).double()
            for loc in ("vram", "host"):
                p = pool if loc == "vram" else pool.cpu().pin_memory()
                v = lay.views(p)
                for tokens, in_div in itertools.product((1, 38, 300), (TOPK, 1)):
                    flat = routing(tokens, cg).to(dev).reshape(-1)
                    npairs = flat.numel()
                    x = torch.randn(tokens if in_div == TOPK else npairs, k, generator=g, device=dev).to(torch.bfloat16)
                    xs = x[torch.arange(npairs, device=dev) // in_div].double()
                    slot_of = torch.full((S,), -1, dtype=torch.int32, device=dev)
                    hot = torch.randperm(S, generator=cg)[: S // 2].to(dev)
                    slot_of[hot] = torch.randperm(S, generator=cg)[: S // 2].to(dev, torch.int32)
                    e_of = flat.long()
                    s_of = slot_of[e_of].long()
                    refs = {"0": (reference(w64, e_of, xs), torch.ones_like(e_of, dtype=torch.bool)),
                            "1": (reference(w64, s_of.clamp(min=0), xs), s_of >= 0),
                            "2": (reference(w64, e_of, xs), s_of < 0)}
                    refs["1+2"] = (torch.where((s_of >= 0)[:, None], refs["1"][0], refs["2"][0]),
                                   torch.ones_like(e_of, dtype=torch.bool))
                    elem = 4 if f32 else 2  # fixed configs that fit the shared memory (the autotuner prunes the rest)
                    cases = [(bm, c) for bm in (16, 32, 64) for c in range(len(CONFIGS))
                             if _gemm_smem(bm, CONFIGS[c]["BLOCK_N"], CONFIGS[c]["BLOCK_K"], CONFIGS[c]["num_stages"],
                                           elem, bits, backend, arch) <= min(limit, LDS_BYTES)]
                    if tokens in (38, 300):  # the autotuned wrapper (VRAM or host configs) at the engine's block_m
                        cases.append((block_m_for(npairs, S, host=loc == "host"), None))
                    for bm, cfg_i in cases:
                        cfg = CONFIGS[cfg_i] if cfg_i is not None else None
                        order, blocks, _ = moe_blocks(flat, S, bm)
                        for mode, (ref, comp) in refs.items():
                            out = torch.full((npairs, n), SENTINEL, device=dev)
                            kw = dict(num_experts=S, out=out, block_m=bm, in_div=in_div, config=cfg,
                                      **lay.gemv_args(v, "gu"))
                            for md in ((1, 2) if mode == "1+2" else (int(mode),)):
                                moe_gemm(x, order, blocks, slot_of, mode=md, **kw)
                            torch.cuda.synchronize()
                            d = (out.double() - ref).abs()
                            err = float(d[comp].max() / ref[comp].abs().max()) if comp.any() else 0.0
                            untouched = bool((out[~comp] == SENTINEL).all())
                            row = {"fmt": fmt, "bits": bits, "group": group, "n": n, "k": k, "pool": loc,
                                   "tokens": tokens, "in_div": in_div, "block_m": bm,
                                   "config": "auto" if cfg_i is None else cfg_i, "mode": mode, "rel_err": err,
                                   "skipped_untouched": untouched}
                            rows.append(row)
                            key = f"{fmt} b{bits} g{group}"
                            worst[key] = max(worst.get(key, 0.0), err)
                            if not err <= 2e-5 or not untouched:
                                bad.append(row)
                                print("BAD", row, flush=True)
                del p, v
            del pool, vv, w64
            print(f"[{time.time() - t0:5.0f}s] {n}x{k} {fmt} b{bits} g{group} done; worst so far "
                  f"{json.dumps({kk: f'{val:.1e}' for kk, val in worst.items()})}", flush=True)
    summary = {"cases": len(rows), "failures": len(bad), "max_rel_err": max(r["rel_err"] for r in rows),
               "worst": worst, "threshold": 2e-5, "seconds": round(time.time() - t0, 1), "dot_f32": f32,
               "by_pool": {pl: max(r["rel_err"] for r in rows if r["pool"] == pl) for pl in ("vram", "host")},
               "by_mode": {m: max(r["rel_err"] for r in rows if r["mode"] == m) for m in ("0", "1", "2", "1+2")}}
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "failures": bad, "rows": rows}))
    assert not bad, f"{len(bad)} failing cases"


if __name__ == "__main__":
    main()
