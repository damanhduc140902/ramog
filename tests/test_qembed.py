"""QEmbedding (behemoe/qembed.py) == nn.Embedding on the dequantized table, on the GPU and from pinned host memory.

Table: the real Qwen3-30B dense-c embedding from a convert_dense directory (--packed; 151936 x 2048, k4row 4-bit
g32) or a synthetic one quantized with behemoe.kquant (same format). Reference: F.embedding(ids, T) with
T = kquant.dequantize(codes, sb, d, dmin).to(dtype), i.e. nn.Embedding holding the dequantized table. For both
placements (host=False: GPU table; host=True: page-locked host table read zero-copy), both output dtypes (bf16, the
model's; float32) and both lookups (fused Triton kernel; torch ops), the lookup must equal the reference bit for bit, for
id tensors of several shapes (1, [1, 1], [3, 17], [2, 512], with repeats and the first/last ids), int64 and int32, and
of several layouts: views whose reshape(-1) keeps a non-unit stride (x[:, ::3], a column of [T, 2], int32 too),
expanded ids (stride 0, int64 and int32, 1-D and broadcast [1, 1] -> [1, 7]), strided CPU ids and a transposed view.
Also: CUDA-graph capture of a lookup replayed with new ids equals eager (contiguous ids, and a strided view of the
captured id buffer); embed_scale (Gemma-style) multiplies like HostEmbedding; the packed record round-trips
(pack_rows / packed()). --bench: lookup time for 1 and 512 tokens vs nn.Embedding (bf16 table on the GPU), and the bytes held.

python tests/test_qembed.py [--packed DIR] [--bench] [--out results/qint/qw2/test_qembed.json]
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
from behemoe.qembed import QEmbedding  # noqa: E402


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    iv = torch.int16 if a.element_size() == 2 else torch.int32
    return bool(torch.equal(a.contiguous().view(iv), b.contiguous().view(iv)))


def id_sets(V: int, gen, dev):
    r = lambda *s: torch.randint(0, V, s, generator=gen, device=dev)  # noqa: E731
    edge = torch.tensor([[0, V - 1, 0, 5, 5, V - 1]], device=dev)
    i32 = torch.int32
    return {"[1]": r(1), "[1,1]": r(1, 1), "[3,17]": r(3, 17), "[2,512]": r(2, 512), "edges+repeats": edge,
            "int32 [4,9]": r(4, 9).to(i32),
            # views whose reshape(-1) is a view with a non-unit stride (the gather kernels need unit stride)
            "x[:, ::3] [3,6] (flat stride 3)": r(3, 18)[:, ::3],
            "column of [64,2] (flat stride 2)": r(64, 2)[:, 0],
            "int32 column of [33,2] (flat stride 2)": r(33, 2).to(i32)[:, 1],
            "expand [1] -> [4] (flat stride 0)": r(1).expand(4),
            "int32 expand [1] -> [5] (flat stride 0)": r(1).to(i32).expand(5),
            "expand [1,1] -> [1,7] (flat stride 0)": r(1, 1).expand(1, 7),
            "cpu x[:, ::2] [2,5]": r(2, 10)[:, ::2].cpu(),
            "transposed [17,3] (reshape copies)": r(3, 17).t()}


def graph_check(emb: QEmbedding, V: int, gen, dev, T: int, stride: int = 1) -> bool:
    """Capture emb(ids) once, then replay with new ids written into the captured id buffer; stride > 1 captures a
    strided view base[:, ::stride] of that buffer (its contiguous copy is part of the graph)."""
    base = torch.randint(0, V, (1, T * stride), generator=gen, device=dev)
    ids = base[:, ::stride]
    emb(ids)
    s = torch.cuda.Stream(dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        emb(ids)
    torch.cuda.current_stream(dev).wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = emb(ids)
    ok = True
    for _ in range(3):
        base.copy_(torch.randint(0, V, (1, T * stride), generator=gen, device=dev))
        g.replay()
        torch.cuda.synchronize(dev)
        ok &= bitwise_equal(out, emb(base[:, ::stride].contiguous()))
    return ok


def time_us(fn, iters=200, dev=None):
    fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize(dev)
        ts.append(a.elapsed_time(b) * 1000)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    t0 = time.time()
    if args.packed:
        from behemoe.dense_recipe import PackedDense

        kp, _ = PackedDense(args.packed).get("model.embed_tokens", dev)
        src = "real Qwen3-30B embed_tokens (packed)"
    else:
        table = (torch.randn(32000, 2048, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        kp = kquant.quantize(table.float(), 4, 32, None)
        src = "synthetic 32000x2048"
    V = kp.qweight.shape[0]
    deq32 = kquant.dequantize_packed(kp, torch.float32)
    rows = []
    mods = {}
    with torch.no_grad():
        for fused in (True, False):
            for host, dtype in ((False, torch.bfloat16), (False, torch.float32), (True, torch.bfloat16),
                                (True, torch.float32)):
                emb = QEmbedding.from_packed(kp, host=host, device=dev, dtype=dtype, fused=fused)
                mods[(host, dtype, fused)] = emb
                ref_table = deq32.to(dtype)
                nn_emb = torch.nn.Embedding(V, emb.embedding_dim, device=dev, dtype=dtype)
                nn_emb.weight.data.copy_(ref_table)
                for tag, ids in id_sets(V, gen, dev).items():
                    out = emb(ids)
                    ref_ids = ids.to(dev).long()
                    ok = (bitwise_equal(out, nn_emb(ref_ids)) and bitwise_equal(out, F.embedding(ref_ids, ref_table))
                          and out.device == dev)
                    rows.append({"host": host, "dtype": str(dtype), "fused": fused, "ids": tag, "shape": list(out.shape),
                                 "flat_stride": ids.reshape(-1).stride(0), "ok": ok})
                    if not ok:
                        print("FAIL", rows[-1], flush=True)
                rows.append({"host": host, "dtype": str(dtype), "fused": fused, "ids": "graph T=1",
                             "ok": graph_check(emb, V, gen, dev, 1)})
                rows.append({"host": host, "dtype": str(dtype), "fused": fused, "ids": "graph T=64",
                             "ok": graph_check(emb, V, gen, dev, 64)})
                rows.append({"host": host, "dtype": str(dtype), "fused": fused, "ids": "graph T=64, ids[:, ::3] view",
                             "ok": graph_check(emb, V, gen, dev, 64, stride=3)})
                print(rows[-1], flush=True)
        # full-table dequantize and packed() round trip, embed_scale
        e = mods[(True, torch.bfloat16, True)]
        rt = e.packed()
        rows.append({"check": "packed() round trip (host)",
                     "ok": all(torch.equal(getattr(rt, f).to(dev), getattr(kp, f)) for f in ("qweight", "sb", "d", "dmin"))})
        rows.append({"check": "dequantize() == kquant.dequantize (host, bf16)",
                     "ok": bitwise_equal(e.dequantize(torch.bfloat16), deq32.to(torch.bfloat16))})
        scale = torch.tensor(45.25, device=dev)
        es = QEmbedding.from_packed(kp, host=True, device=dev, embed_scale=scale)
        ids = torch.randint(0, V, (2, 7), generator=gen, device=dev)
        rows.append({"check": "embed_scale",
                     "ok": bitwise_equal(es(ids), F.embedding(ids, deq32.to(torch.bfloat16)) * scale.to(torch.bfloat16))})
        speed = {}
        if args.bench:
            nn_emb = torch.nn.Embedding(V, kp.qweight.shape[1] * 2, device=dev, dtype=torch.bfloat16)
            for T in (1, 512):
                ids = torch.randint(0, V, (1, T), generator=gen, device=dev)
                speed[f"T={T}"] = {
                    "qembed_gpu_fused_us": time_us(lambda: mods[(False, torch.bfloat16, True)](ids), dev=dev),
                    "qembed_host_fused_us": time_us(lambda: mods[(True, torch.bfloat16, True)](ids), dev=dev),
                    "qembed_gpu_torch_us": time_us(lambda: mods[(False, torch.bfloat16, False)](ids), dev=dev),
                    "qembed_host_torch_us": time_us(lambda: mods[(True, torch.bfloat16, False)](ids), dev=dev),
                    "nn_embedding_bf16_us": time_us(lambda: nn_emb(ids), dev=dev)}
            speed["bytes"] = {"qembed_rows": mods[(False, torch.bfloat16, True)].nbytes(),
                              "row_bytes": mods[(False, torch.bfloat16, True)].layout["row_bytes"],
                              "bf16_table": V * nn_emb.embedding_dim * 2}
            print(speed, flush=True)
    summary = {"table": src, "V": V, "checks": len(rows), "passed": sum(r["ok"] for r in rows),
               "ok": all(r["ok"] for r in rows), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"summary": summary, "checks": rows, "bench": speed, "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(dev), "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")},
                  open(args.out, "w"), indent=1)
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
