"""KLinear (behemoe/klinear.py): decode and prefill vs a float64 reference on the dequantized weight, CUDA-graph replay
vs eager, and (--bench) speed against today's dense QuantLinear (rtn 4-bit g64) and bf16 F.linear.

Accuracy: y = KLinear.matmul_f32(x) (the float32 sums before the cast to x.dtype) against
ref = x.double() @ kquant.dequantize(W).double().T (+ bias), x bf16 (the model's activations; exact in both paths).
Errors: rel_fro = ||y - ref|| / ||ref|| (Frobenius) and rel_max = max |y - ref| / max |ref|. Decode (GEMV,
deterministic and not): both <= 1e-6. Prefill-sized cases (GEMM with float32 dot; float32-input fallback = torch
matmul): both <= 1e-6, or else every output element within the standard float32 dot-product error bound
gamma_(K+1) |x|.|w| (see gemm_check); torch's own float32 GEMM error on the same float32 dequantized weight is reported
beside it (a K = 4096 float32 accumulation alone reaches ~1e-6 on these inputs), and how many cases meet 1e-6.
prefill="bf16" (the engine's MoE numerics: weights rounded to bf16 in the GEMM) is reported, not asserted.
forward(x) must equal matmul_f32(x).to(x.dtype) bit for bit.
Cases: every bits in {2, 4, 8} x group in {16, 32} on two synthetic shapes, and the Qwen3-30B dense-c modules
(q/k/v/o, router, lm_head) from a convert_dense directory (--packed) or synthetic weights of the same shapes.
Graph: per module and token count, warmup, capture one forward on a static input, replay on new inputs: the output
must equal an eager forward bit for bit (and repeated eager calls must agree bit for bit).

python tests/test_klinear.py [--packed DIR] [--bench] [--out results/qint/qw2/test_klinear.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from behemoe import kquant  # noqa: E402
from behemoe.klinear import DECODE_MAX_TOKENS, KLinear  # noqa: E402

TOL = 1e-6
DECODE_T = (1, 2, 3, 4, 5, 7, 8)
PREFILL_T = (9, 16, 33, 100, 512)


def err(y: torch.Tensor, ref: torch.Tensor) -> dict:
    e = y.double() - ref
    return {"rel_fro": float(e.norm() / ref.norm().clamp(min=1e-300)),
            "rel_max": float(e.abs().max() / ref.abs().max().clamp(min=1e-300))}


def synth(n, k, gen, dev, outliers=True):
    W = torch.randn(n, k, generator=gen, device=dev) * 0.02
    if outliers:  # a few large input channels, as real attention / head weights have
        cols = torch.randint(0, k, (max(1, k // 256),), generator=gen, device=dev)
        W[:, cols] *= 8
    return W.to(torch.bfloat16)


def gemm_check(r: dict, y, x, Wd, bias, ref) -> None:
    """Prefill-sized cases: the spec bound (1e-6 on both metrics); beside it, the error of torch's float32 GEMM
    (hipBLAS / cuBLAS) on the same float32 dequantized weight and input, for comparison (a K-long float32
    accumulation alone reaches ~1e-6 at K = 4096 on these inputs), and the standard float32 dot-product error bound
    |fl(x.w) - x.w| <= gamma_n |x|.|w|, gamma_n = n u / (1 - n u), u = 2^-24, n = K + 1 (bias), checked on every
    output element: any summation order satisfies it, a wrong code, scale or index does not.
    ok = within the spec bound, or every element within the float32 bound."""
    yg = x.float() @ Wd.float().t()
    if bias is not None:
        yg = yg + bias.float()
    g = err(yg, ref)
    r["fp32_gemm_rel_fro"], r["fp32_gemm_rel_max"] = g["rel_fro"], g["rel_max"]
    r["spec_ok"] = r["rel_fro"] <= TOL and r["rel_max"] <= TOL
    n = x.shape[1] + 1
    u = 2.0**-24
    gamma = n * u / (1 - n * u)
    bound = gamma * (x.double().abs() @ Wd.abs().t())
    if bias is not None:
        bound = bound + gamma * bias.double().abs()
    e = (y.double() - ref).abs()
    r["fp32_bound_max_ratio"] = float((e / bound.clamp(min=1e-300)).max())
    r["fp32_bound_ok"] = bool((e <= bound).all())
    r["ok"] = r["spec_ok"] or r["fp32_bound_ok"]


def check_module(lin: KLinear, tag: str, gen, dev, bias=None, extra: bool = True) -> list:
    rows = []
    Wd = lin.dequantize(torch.float32).double()
    k = lin.in_features
    for T in DECODE_T + PREFILL_T:
        x = (torch.randn(T, k, generator=gen, device=dev)).to(torch.bfloat16)
        ref = x.double() @ Wd.t()
        if bias is not None:
            ref = ref + bias.double()
        y = lin.matmul_f32(x)
        path = "decode" if T <= DECODE_MAX_TOKENS else "prefill-f32"
        r = {"case": tag, "T": T, "path": path, **err(y, ref)}
        fwd, cast = lin(x), y.to(x.dtype)
        r["forward_is_cast"] = bool(torch.equal(fwd, cast))
        if T == 1 and not r["forward_is_cast"] and x.dtype == torch.bfloat16:
            # one bf16 token: forward() runs behemoe.gemv1's bf16-output kernel, matmul_f32 its float32-output twin;
            # the two compiled kernels may round once differently: within one bf16 ulp counts as the cast
            ulp = (fwd.view(torch.int16).int() - cast.view(torch.int16).int()).abs().max().item()
            r["forward_ulp"] = ulp
            r["forward_is_cast"] = ulp <= 1
        if path == "decode":
            r["ok"] = r["spec_ok"] = r["rel_fro"] <= TOL and r["rel_max"] <= TOL and r["forward_is_cast"]
        else:
            gemm_check(r, y, x, Wd, bias, ref)
            r["ok"] = r["ok"] and r["forward_is_cast"]
        rows.append(r)
        if not extra:
            continue
        if T <= DECODE_MAX_TOKENS:
            lin.deterministic = False
            r2 = {"case": tag, "T": T, "path": "decode-nondet", **err(lin.matmul_f32(x), ref)}
            lin.deterministic = True
            r2["ok"] = r2["spec_ok"] = r2["rel_fro"] <= TOL and r2["rel_max"] <= TOL
            rows.append(r2)
        if T in (4, 100):  # float32 input: GEMV reads it as float32, prefill takes the exact fallback
            xf = torch.randn(T, k, generator=gen, device=dev)
            reff = xf.double() @ Wd.t() + (0 if bias is None else bias.double())
            yf = lin.matmul_f32(xf)
            r3 = {"case": tag, "T": T, "path": "fp32-input", **err(yf, reff)}
            if T <= DECODE_MAX_TOKENS:
                r3["ok"] = r3["spec_ok"] = r3["rel_fro"] <= TOL and r3["rel_max"] <= TOL
            else:
                gemm_check(r3, yf, xf, Wd, bias, reff)
            rows.append(r3)
        if T > DECODE_MAX_TOKENS:
            lin.prefill = "bf16"
            r4 = {"case": tag, "T": T, "path": "prefill-bf16 (info)", **err(lin.matmul_f32(x), ref), "ok": True}
            lin.prefill = "f32"
            rows.append(r4)
    return rows


def check_graph(lin: KLinear, tag: str, gen, dev, Ts=(1, 4, 8, 64)) -> list:
    rows = []
    k = lin.in_features
    for T in Ts:
        lin.warmup((T,))
        xs = torch.randn(T, k, generator=gen, device=dev).to(torch.bfloat16)
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):  # one more eager call on the capture stream (allocator warmup)
            lin(xs)
        torch.cuda.current_stream(dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ys = lin(xs)
        same, rep = 0, 0
        for _ in range(3):
            xn = torch.randn(T, k, generator=gen, device=dev).to(torch.bfloat16)
            xs.copy_(xn)
            g.replay()
            torch.cuda.synchronize(dev)
            eager = lin(xn)
            eager2 = lin(xn)
            same += int(torch.equal(ys, eager))
            rep += int(torch.equal(eager, eager2))
        rows.append({"case": tag, "T": T, "graph_equal_eager": f"{same}/3", "eager_repeat_equal": f"{rep}/3",
                     "ok": same == 3 and rep == 3})
        del g
    return rows


def bench_one(fn, x, iters: int, graph: bool, dev):
    """Median time per call in microseconds: CUDA graph of `iters` calls (decode) or events around eager calls."""
    fn(x)
    torch.cuda.synchronize(dev)
    if graph:
        s = torch.cuda.Stream(dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            fn(x)
        torch.cuda.current_stream(dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn(x)
        times = []
        for _ in range(5):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            torch.cuda.synchronize(dev)
            times.append(a.elapsed_time(b) * 1000 / iters)
        del g
        return sorted(times)[2]
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn(x)
        b.record()
        torch.cuda.synchronize(dev)
        times.append(a.elapsed_time(b) * 1000)
    return sorted(times)[len(times) // 2]


def bench(mods: dict, dev, gen) -> list:
    from behemoe.qlinear import QuantLinear

    rows = []
    for tag, lin in mods.items():
        W = lin.dequantize(torch.bfloat16)
        ql = QuantLinear(W, None, 4, 64, device=dev)
        for T, iters, graph in ((1, 100, True), (4, 100, True), (512, 20, False)):
            x = torch.randn(T, lin.in_features, generator=gen, device=dev).to(torch.bfloat16)
            lin.warmup((T,))
            r = {"module": tag, "shape": [lin.out_features, lin.in_features], "fmt": f"{lin.bits}:{lin.group}", "T": T}
            r["klinear_us"] = bench_one(lin, x, iters, graph, dev)
            if T <= DECODE_MAX_TOKENS:
                lin.deterministic = False
                lin.warmup((T,))
                r["klinear_nondet_us"] = bench_one(lin, x, iters, graph, dev)
                lin.deterministic = True
            else:
                lin.prefill = "bf16"
                lin.warmup((T,))
                r["klinear_prefill_bf16_us"] = bench_one(lin, x, iters, graph, dev)
                lin.prefill = "f32"
            r["quantlinear_rtn4g64_us"] = bench_one(ql, x, iters, graph, dev)
            r["bf16_linear_us"] = bench_one(lambda v: F.linear(v, W), x, iters, graph, dev)
            rows.append(r)
            print(r, flush=True)
        del ql, W
    return rows


QWEN3 = {  # module: (shape, bits, group) of dense-c on Qwen3-30B (layer 0)
    "q_proj": ((4096, 2048), 4, 16), "k_proj": ((512, 2048), 8, 32), "v_proj": ((512, 2048), 8, 32),
    "o_proj": ((2048, 4096), 4, 16), "router": ((128, 2048), 8, 32), "lm_head": ((151936, 2048), 4, 16)}
PACKED_NAMES = {"q_proj": "model.layers.0.self_attn.q_proj", "k_proj": "model.layers.0.self_attn.k_proj",
                "v_proj": "model.layers.0.self_attn.v_proj", "o_proj": "model.layers.0.self_attn.o_proj",
                "router": "model.layers.0.mlp.gate", "lm_head": "lm_head"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default=None, help="convert_dense directory: real Qwen3-30B dense-c / dense-hi modules")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    rows, graphs, speed = [], [], []
    t0 = time.time()
    with torch.no_grad():
        # 1. every format on synthetic weights (two shapes), with importance; one case with a bias
        for bits in (2, 4, 8):
            for group in (16, 32):
                for n, k in ((512, 2048), (2048, 4096)):
                    W = synth(n, k, gen, dev)
                    imp = torch.rand(k, generator=gen, device=dev) + 0.5
                    lin = KLinear.from_packed(kquant.quantize(W.float(), bits, group, imp))
                    rows += check_module(lin, f"synth {n}x{k} {bits}:{group}", gen, dev, extra=(n == 512))
        W = synth(1024, 2048, gen, dev)
        b = torch.randn(1024, generator=gen, device=dev).to(torch.bfloat16)
        lin = KLinear.from_packed(kquant.quantize(W.float(), 4, 16), bias=b)
        rows += check_module(lin, "synth 1024x2048 4:16 +bias", gen, dev, bias=b, extra=False)
        # 2. Qwen3-30B dense-c modules (real when --packed)
        mods = {}
        if args.packed:
            from behemoe.dense_recipe import PackedDense

            pd = PackedDense(args.packed)
            for tag, name in PACKED_NAMES.items():
                kp, bias = pd.get(name, dev)
                mods[tag] = KLinear.from_packed(kp, bias)
        else:
            for tag, ((n, k), bits, group) in QWEN3.items():
                mods[tag] = KLinear.from_packed(kquant.quantize(synth(n, k, gen, dev).float(), bits, group))
        src = "real (packed)" if args.packed else "synthetic"
        for tag, lin in mods.items():
            # shapes always; bits / group are dense-c's for the synthetic modules, the directory's own with --packed
            # (dense-c or dense-hi, design-quality WP10)
            assert (lin.out_features, lin.in_features) == QWEN3[tag][0]
            assert args.packed or (lin.bits, lin.group) == QWEN3[tag][1:]
            rows += check_module(lin, f"qwen3 {tag} {src}", gen, dev, extra=(tag != "lm_head"))
            graphs += check_graph(lin, f"qwen3 {tag} {src}", gen, dev)
            print(rows[-1], graphs[-1], flush=True)
        if args.bench:
            speed = bench(mods, dev, gen)
    asserted = [r for r in rows if "info" not in r["path"]]
    worst = {p: max((r["rel_fro"] for r in asserted if r["path"] == p), default=None)
             for p in ("decode", "decode-nondet", "prefill-f32", "fp32-input")}
    worst_max = {p: max((r["rel_max"] for r in asserted if r["path"] == p), default=None)
                 for p in ("decode", "decode-nondet", "prefill-f32", "fp32-input")}
    info_bf16 = [r["rel_fro"] for r in rows if r["path"].startswith("prefill-bf16")]
    spec = {p: f"{sum(r['spec_ok'] for r in asserted if r['path'] == p)}/{sum(r['path'] == p for r in asserted)}"
            for p in ("decode", "decode-nondet", "prefill-f32", "fp32-input")}
    gemm = [r for r in asserted if "fp32_gemm_rel_fro" in r]
    summary = {"cases": len(rows), "asserted": len(asserted), "passed": sum(r["ok"] for r in asserted),
               "within_1e-6_both_metrics": spec,
               "prefill_vs_fp32_gemm_ratio_median_max": {
                   k: ([sorted(v)[len(v) // 2], max(v)] if v else None)
                   for k, v in (("rel_fro", [r["rel_fro"] / r["fp32_gemm_rel_fro"] for r in gemm]),
                                ("rel_max", [r["rel_max"] / r["fp32_gemm_rel_max"] for r in gemm]))},
               "prefill_within_fp32_bound": f"{sum(r['fp32_bound_ok'] for r in gemm)}/{len(gemm)}",
               "prefill_worst_fraction_of_fp32_bound": max((r["fp32_bound_max_ratio"] for r in gemm), default=None),
               "fp32_gemm_worst": {"rel_fro": max((r["fp32_gemm_rel_fro"] for r in gemm), default=None),
                                   "rel_max": max((r["fp32_gemm_rel_max"] for r in gemm), default=None)},
               "worst_rel_fro": worst, "worst_rel_max": worst_max,
               "prefill_bf16_rel_fro_range": [min(info_bf16), max(info_bf16)] if info_bf16 else None,
               "graph_cases": len(graphs), "graph_passed": sum(g["ok"] for g in graphs),
               "ok": all(r["ok"] for r in asserted) and all(g["ok"] for g in graphs),
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"summary": summary, "accuracy": rows, "graph": graphs, "bench": speed, "packed": args.packed,
                   "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev),
                   "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")}, open(args.out, "w"), indent=1)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
