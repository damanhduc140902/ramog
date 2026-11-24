"""Shared memory of the prefill GEMM (ramog.kernels._gemm_body) on every GPU family the engine targets.

The autotuner benchmarks the configs ramog.kernels.gemm_candidates() keeps for a call; a config that needs more
shared memory than the GPU allows per block fails to launch (OutOfResources) and is timed as inf, so a call must
always keep at least one config that fits. This test compiles every kept config of both prefill GEMMs (the VRAM
pool, _moe_gemm, and the zero-copy host pool, _moe_gemm_host) for each target with Triton's own compiler (no GPU
needed), at every block_m the engine launches (kernels.block_m_for: 16/32/64 VRAM, 16/64/128 host), for the
weight layouts the engine uses (rtn 4-bit g64 hot / dense, k4row 2-bit g16 cold or hot-from-cold, rtn 2-bit g64
legacy cold), and checks the compiled metadata.shared against the target's limit:
  FAIL if any (target, kernel, block_m, layout) keeps no config that fits (the launch would raise OutOfResources);
  FAIL if a kept config does not fit on a target listed in DEFAULT_TARGETS (the model in kernels._gemm_smem is
  claimed for them); reports the model's error per target.
Targets: NVIDIA sm_75 (Turing: GTX 16xx, RTX 20xx, Titan RTX; float32 dot operands), sm_80, sm_86, sm_89, sm_90,
AMD gfx90a (MI250), gfx942 (MI300), gfx1100 (RX 7900), and the float32 operands forced (ENGINE_DOT_F32=1) on
gfx90a / sm_86.
--live (needs a GPU): also runs the autotuned moe_gemm on this GPU at every block_m and layout (results vs float64,
including 128-pair blocks), and checks that the kernels it compiled use exactly the shared memory this test
computes offline for the GPU's target.
python tests/test_gemm_smem.py [--targets cuda:75:f32,hip:gfx90a] [--live] [--jobs 8] [--out result.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# backend:arch[:f32] -> (warp size, max shared memory per block in bytes, as the drivers report it)
DEFAULT_TARGETS = {
    "cuda:75:f32": (32, 65536),     # Turing (pre-Ampere: no bf16 matrix units -> float32 dot operands)
    "cuda:80": (32, 166912),        # A100
    "cuda:86": (32, 101376),        # RTX 30xx, A10/A40
    "cuda:89": (32, 101376),        # RTX 40xx, L4/L40
    "cuda:90": (32, 232448),        # H100
    "hip:gfx90a": (64, 65536),      # MI250
    "hip:gfx942": (64, 65536),      # MI300
    "hip:gfx1100": (32, 65536),     # RX 7900 (RDNA3)
    "hip:gfx90a:f32": (64, 65536),  # float32 operands forced (ENGINE_DOT_F32=1), as the f32 unit tests do
    "cuda:86:f32": (32, 101376),
}
LAYOUTS = {"vram": (("rtn", 4, 64), ("k4row", 2, 16)), "host": (("k4row", 2, 16), ("rtn", 2, 64))}
BLOCK_MS = {"vram": (16, 32, 64), "host": (16, 64, 128)}
KS = (2048, 768)  # Qwen3-30B gate_up / down inputs (768: no 512-wide K step)
INT_ARGS = ("N", "K", "E", "IN_DIV", "stride_x", "stride_qw_slot", "stride_qw_row", "stride_sc_slot",
            "stride_sc_row", "stride_d_slot", "stride_y")


def parse_target(spec: str):
    parts = spec.split(":")
    backend, arch = parts[0], parts[1]
    f32 = len(parts) > 2 and parts[2] == "f32"
    return backend, (int(arch) if backend == "cuda" else arch), f32


def signature(fmt: str) -> dict:
    s = {"x_ptr": "*bf16", "pair_ptr": "*i32", "blk_ptr": "*i32", "slot_ptr": "*i32", "qw_ptr": "*i32",
         "sc_ptr": "*fp16" if fmt == "rtn" else "*u8", "mn_ptr": "*fp16" if fmt == "rtn" else "*u8",
         "d_ptr": "*fp16", "dm_ptr": "*fp16", "y_ptr": "*fp32"}
    for n in INT_ARGS:
        s[n] = "i32"
    for n in ("BITS", "GROUP", "FMT", "MODE", "BLOCK_M", "BLOCK_N", "BLOCK_K", "DOT_F32"):
        s[n] = "constexpr"
    return s


def compile_shared(job):
    """(backend, arch, warp, fmt, bits, group, mode, block_m, block_n, block_k, warps, stages, f32) ->
    (shared bytes or None, error)."""
    backend, arch, warp, fmt, bits, group, mode, bm, bn, bk, warps, stages, f32 = job
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from ramog.kernels import _gemm_body

    const = {"BITS": bits, "GROUP": group, "FMT": 0 if fmt == "rtn" else 1, "MODE": mode, "BLOCK_M": bm,
             "BLOCK_N": bn, "BLOCK_K": bk, "DOT_F32": f32}
    sig = signature(fmt)
    names = list(sig)
    # what a real launch specializes: 16-byte aligned pointers, ints divisible by 16 (all but IN_DIV here)
    attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in names
             if sig[n].startswith("*") or (sig[n] == "i32" and n != "IN_DIV")}
    try:
        k = triton.compile(ASTSource(_gemm_body, sig, constexprs=const, attrs=attrs),
                           target=GPUTarget(backend, arch, warp), options={"num_warps": warps, "num_stages": stages})
        md = k.metadata
        return int(md["shared"] if isinstance(md, dict) else md.shared), None
    except Exception as ex:  # noqa: BLE001
        return None, f"{type(ex).__name__}: {str(ex).splitlines()[0][:160] if str(ex) else ''}"


def offline(specs: list[str], jobs: int, strict: set[str]) -> tuple[list, list]:
    from ramog.kernels import _gemm_configs, _gemm_host_configs, _gemm_smem, gemm_candidates

    cfgs = {"vram": _gemm_configs(), "host": _gemm_host_configs()}
    plan = []  # (spec, kernel, layout, K, block_m, [configs kept])
    for spec in specs:
        backend, arch, f32 = parse_target(spec)
        warp, limit = DEFAULT_TARGETS.get(spec, (32 if backend == "cuda" else 64, 65536))
        for kern in ("vram", "host"):
            for (fmt, bits, group), k, bm in itertools.product(LAYOUTS[kern], KS, BLOCK_MS[kern]):
                kept = gemm_candidates(cfgs[kern], k, group, bits, bm, f32, backend, arch, limit)
                plan.append((spec, kern, (fmt, bits, group), k, bm, kept, warp, limit))
    todo = {}
    for spec, kern, (fmt, bits, group), _k, bm, kept, warp, _limit in plan:
        backend, arch, f32 = parse_target(spec)
        for c in kept:
            job = (backend, arch, warp, fmt, bits, group, 1 if kern == "vram" else 2, bm, c.kwargs["BLOCK_N"],
                   c.kwargs["BLOCK_K"], c.num_warps, c.num_stages, f32)
            todo[job] = None
    t0 = time.time()
    print(f"compiling {len(todo)} kernel variants for {len(specs)} targets ({jobs} processes)", flush=True)
    with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
        for job, res in zip(todo, ex.map(compile_shared, todo, chunksize=4)):
            todo[job] = res
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    rows, fails = [], []
    for spec, kern, (fmt, bits, group), k, bm, kept, warp, limit in plan:
        backend, arch, f32 = parse_target(spec)
        fit = []
        for c in kept:
            job = (backend, arch, warp, fmt, bits, group, 1 if kern == "vram" else 2, bm, c.kwargs["BLOCK_N"],
                   c.kwargs["BLOCK_K"], c.num_warps, c.num_stages, f32)
            shared, err = todo[job]
            model = _gemm_smem(bm, c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"], c.num_stages, 4 if f32 else 2, bits,
                               backend, arch)
            row = {"target": spec, "kernel": kern, "fmt": fmt, "bits": bits, "group": group, "K": k, "block_m": bm,
                   "BLOCK_N": c.kwargs["BLOCK_N"], "BLOCK_K": c.kwargs["BLOCK_K"], "warps": c.num_warps,
                   "stages": c.num_stages, "model": model, "shared": shared, "limit": limit,
                   "fits": shared is not None and shared <= limit, "error": err}
            rows.append(row)
            if row["fits"]:
                fit.append(row)
            elif spec in strict:
                fails.append(f"{spec} {kern} {fmt}{bits}g{group} K={k} block_m={bm}: kept config "
                             f"{c.kwargs} {c.num_warps}w {c.num_stages}st does not fit "
                             f"({shared} B > {limit} B; model {model} B){' ' + err if err else ''}")
        if not fit:
            fails.append(f"{spec} {kern} {fmt}{bits}g{group} K={k} block_m={bm}: NO kept config fits "
                         f"(kept {[(c.kwargs['BLOCK_N'], c.kwargs['BLOCK_K'], c.num_stages) for c in kept]})")
    return rows, fails


def live(jobs_rows: list) -> list:
    """Autotuned moe_gemm on this GPU for every kernel / layout / block_m (vs float64), then compare the shared
    memory of every variant Triton compiled at run time with the offline compile for this GPU's target."""
    import torch

    from ramog import kernels
    from ramog.store import BlobLayout

    fails = []
    dev = torch.device("cuda")
    backend, arch, limit = kernels._device_target()
    f32 = kernels.dot_f32(dev)
    print(f"live: {backend} {arch}, {limit} B shared memory per block, dot operands {'float32' if f32 else 'bf16'}",
          flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    e, n, k = 16, 256, 2048
    for kern, ((fmt, bits, group), bm) in itertools.chain(
            (("vram", x) for x in itertools.product(LAYOUTS["vram"], BLOCK_MS["vram"])),
            (("host", x) for x in itertools.product(LAYOUTS["host"], BLOCK_MS["host"]))):
        lay = BlobLayout(bits, group, (n, k), (k, n), fmt)
        w = torch.randn(e, n, k, device=dev, generator=g) * 0.02
        pool = lay.quantize(w, torch.randn(e, k, n, device=dev, generator=g) * 0.02)
        if kern == "host":
            pool = pool.cpu().pin_memory()
        v = lay.views(pool)
        pairs = bm * 3 + 5  # some full blocks and a partial one per expert
        flat = torch.randint(0, e, (pairs,), device=dev, generator=g)
        x = torch.randn(pairs, k, device=dev, generator=g).to(torch.bfloat16)
        slot_of = torch.full((e,), -1, dtype=torch.int32, device=dev)
        if kern == "vram":
            slot_of = torch.arange(e, dtype=torch.int32, device=dev)
        order, blocks, _ = kernels.moe_blocks(flat, e, bm)
        out = torch.zeros(pairs, n, device=dev)
        kernels.moe_gemm(x, order, blocks, slot_of, **lay.gemv_args(v, "gu"), num_experts=e, out=out, block_m=bm,
                         mode=1 if kern == "vram" else 2, in_div=1)
        torch.cuda.synchronize()
        vv = lay.views(pool.to(dev))
        ref = torch.zeros(pairs, n, dtype=torch.float64, device=dev)
        for s in flat.unique().tolist():
            wd = lay.dequant(vv, s, "gu", torch.float32)
            wd = (wd if f32 else wd.to(torch.bfloat16)).double()  # what the kernel feeds tl.dot
            m = flat == s
            ref[m] = x[m].double() @ wd.t()
        err = float((out.double() - ref).abs().max() / ref.abs().max())
        ok = err <= 2e-5
        print(f"live {kern} {fmt}{bits}g{group} block_m {bm}: rel err {err:.1e} {'ok' if ok else 'FAIL'}", flush=True)
        if not ok:
            fails.append(f"live {kern} {fmt}{bits}g{group} block_m {bm}: rel err {err:.1e}")
    # every variant compiled so far (autotuner trials included) vs the offline compile for this target
    warp = 32 if backend == "cuda" else 64
    if backend == "hip" and str(arch).startswith("gfx1"):
        warp = 32
    names = kernels._gemm_body.arg_names
    seen = []
    for cache in kernels._gemm_body.device_caches.values():
        for ck in cache[0].values():
            ck = ck.result() if hasattr(ck, "result") else ck
            const = {names[i[0]]: v for i, v in ck.src.constants.items() if len(i) == 1}
            md = ck.metadata
            shared = int(md["shared"] if isinstance(md, dict) else md.shared)
            warps = md["num_warps"] if isinstance(md, dict) else md.num_warps
            stages = md["num_stages"] if isinstance(md, dict) else md.num_stages
            fmt = "rtn" if const["FMT"] == 0 else "k4row"
            job = (backend, arch, warp, fmt, const["BITS"], const["GROUP"], const["MODE"], const["BLOCK_M"],
                   const["BLOCK_N"], const["BLOCK_K"], warps, stages, bool(const["DOT_F32"]))
            seen.append((job, shared))
    if not seen:
        fails.append("live: no compiled _gemm_body variants found")
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        offl = list(ex.map(compile_shared, [j for j, _ in seen]))
    mism = [(j, s, o) for (j, s), o in zip(seen, offl, strict=True) if o[0] != s]
    print(f"live: {len(seen)} variants compiled at run time; offline compile gives the same shared memory for "
          f"{len(seen) - len(mism)}", flush=True)
    for j, s, o in mism:
        fails.append(f"live: run-time shared {s} B != offline {o} for {j}")
    jobs_rows.extend({"live_variant": list(map(str, j)), "shared": s} for j, s in seen)
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--out")
    args = ap.parse_args()
    specs = [s for s in args.targets.split(",") if s]
    rows, fails = offline(specs, args.jobs, strict=set(DEFAULT_TARGETS))
    for spec in specs:
        rs = [r for r in rows if r["target"] == spec and r["shared"] is not None]
        over = [r for r in rs if not r["fits"]]
        errs = [r for r in rows if r["target"] == spec and r["shared"] is None]
        dev_model = max((abs(r["model"] - r["shared"]) for r in rs), default=0)
        under = sum(r["model"] < r["shared"] for r in rs)
        print(f"{spec}: {len(rs)} kept (kernel, layout, K, block_m, config) cases compiled, {len(over)} over the "
              f"limit, {len(errs)} compile errors; model vs compiled: max |diff| {dev_model} B, under-estimates "
              f"{under}", flush=True)
    for bm in (64, 128):
        for spec in specs:
            rs = [r for r in rows if r["target"] == spec and r["kernel"] == "host" and r["block_m"] == bm
                  and r["fmt"] == "k4row" and r["K"] == 2048]
            print(f"  {spec} host k4row block_m {bm}: kept {[(r['BLOCK_N'], r['BLOCK_K'], r['shared']) for r in rs]}")
    if args.live:
        fails += live(rows)
    for f in fails:
        print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"targets": specs, "fails": fails, "rows": rows}, f, indent=0)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
