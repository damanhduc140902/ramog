"""Correctness and bandwidth of the grouped dequant-GEMV kernel. Run on a GPU: python tests/test_kernels.py"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_tiering.kernels import grouped_gemv  # noqa: E402
from moe_tiering.quant import dequantize, quantize  # noqa: E402


def make_pool(n_slots, n, k, bits, group, device):
    packs = [quantize(torch.randn(n, k, device=device) * 0.02, bits, group) for _ in range(n_slots)]
    qw = torch.stack([p.qweight for p in packs])
    sc = torch.stack([p.scale for p in packs])
    mn = torch.stack([p.minimum for p in packs])
    return packs, qw, sc, mn


def check(bits, n=1536, k=2048, group=64, n_slots=16, n_tok=3, pairs=10):
    dev = "cuda"
    packs, qw, sc, mn = make_pool(n_slots, n, k, bits, group, dev)
    x = torch.randn(n_tok, k, device=dev, dtype=torch.bfloat16)
    tok = torch.randint(0, n_tok, (pairs,), device=dev, dtype=torch.int32)
    slot = torch.randint(0, n_slots, (pairs,), device=dev, dtype=torch.int32)
    y = grouped_gemv(x, tok, slot, qw, sc, mn, bits, group)
    ref = torch.stack([dequantize(packs[s]) @ x[t].float() for t, s in zip(tok.tolist(), slot.tolist(), strict=True)])
    err = (y - ref).abs().max().item() / ref.abs().max().item()
    # quantization error itself, for scale: dequantized vs the original is not available here, so just print
    print(f"bits={bits} n={n} k={k}: max rel err vs torch reference {err:.2e}")
    assert err < 1e-3, err


def bench(bits, n=1536, k=2048, group=64, n_slots=64, pairs=8, iters=200):
    dev = "cuda"
    _, qw, sc, mn = make_pool(n_slots, n, k, bits, group, dev)
    x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
    tok = torch.zeros(pairs, device=dev, dtype=torch.int32)
    slot = torch.randperm(n_slots, device=dev)[:pairs].to(torch.int32)
    for _ in range(10):
        grouped_gemv(x, tok, slot, qw, sc, mn, bits, group)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        grouped_gemv(x, tok, slot, qw, sc, mn, bits, group)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    nbytes = pairs * (n * k * bits / 8 + 2 * n * (k // group) * 2)
    print(f"bits={bits} {pairs} experts {n}x{k}: {dt * 1e6:.1f} us/launch, {nbytes / dt / 1e9:.0f} GB/s effective")


def bench_graph(bits, n=1536, k=2048, group=64, n_slots=64, pairs=8, reps=100):
    """GPU time per launch without CPU launch overhead: capture `reps` launches in one graph."""
    dev = "cuda"
    _, qw, sc, mn = make_pool(n_slots, n, k, bits, group, dev)
    x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
    tok = torch.zeros(pairs, device=dev, dtype=torch.int32)
    slot = torch.randperm(n_slots, device=dev)[:pairs].to(torch.int32)
    for _ in range(5):  # autotune outside the capture
        grouped_gemv(x, tok, slot, qw, sc, mn, bits, group)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            grouped_gemv(x, tok, slot, qw, sc, mn, bits, group)
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    dt = start.elapsed_time(end) / 1e3 / (5 * reps)
    nbytes = pairs * (n * k * bits / 8 + 2 * n * (k // group) * 2)
    print(f"[graph] bits={bits} {pairs} experts {n}x{k}: {dt * 1e6:.1f} us/launch, {nbytes / dt / 1e9:.0f} GB/s")


if __name__ == "__main__":
    torch.manual_seed(0)
    for b in (4, 2):
        check(b)
        check(b, n=2048, k=768)
    for b in (4, 2):
        bench(b)
        bench(b, n=2048, k=768)
    for b in (4, 2):
        bench_graph(b)
        bench_graph(b, n=2048, k=768)
        bench_graph(b, n=3072, k=4096)  # Qwen3-235B gate_up shape
