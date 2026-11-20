"""Can the grouped GEMV kernel read expert weights straight from pinned host memory (zero-copy over PCIe)?"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_tiering.kernels import grouped_gemv  # noqa: E402
from moe_tiering.quant import dequantize, quantize  # noqa: E402


def main():
    torch.manual_seed(0)
    dev = "cuda"
    n, k, bits, group, slots, pairs = 1536, 2048, 2, 64, 256, 8
    packs = [quantize(torch.randn(n, k, device=dev) * 0.02, bits, group) for _ in range(slots)]
    qw = torch.stack([p.qweight for p in packs]).cpu().pin_memory()
    sc = torch.stack([p.scale for p in packs]).cpu().pin_memory()
    mn = torch.stack([p.minimum for p in packs]).cpu().pin_memory()
    x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
    tok = torch.zeros(pairs, device=dev, dtype=torch.int32)
    slot = torch.randperm(slots, device=dev)[:pairs].to(torch.int32)
    try:
        y = grouped_gemv(x, tok, slot, qw, sc, mn, bits, group)
    except Exception as e:  # noqa: BLE001
        print("zero-copy launch failed:", type(e).__name__, str(e)[:300])
        return
    torch.cuda.synchronize()
    ref = torch.stack([dequantize(packs[s]) @ x[0].float() for s in slot.tolist()])
    print("max rel err (host-resident weights):", ((y - ref).abs().max() / ref.abs().max()).item())

    # bandwidth: different random experts each launch so nothing is cached on the GPU side
    nbytes = pairs * (n * k * bits / 8 + 2 * n * (k // group) * 2)
    for trial in range(2):
        slots_seq = [torch.randperm(slots, device=dev)[:pairs].to(torch.int32) for _ in range(50)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in slots_seq:
            grouped_gemv(x, tok, s, qw, sc, mn, bits, group)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / len(slots_seq)
        print(f"trial {trial}: {dt * 1e6:.0f} us per launch of {pairs} host-resident experts "
              f"({nbytes / 2**20:.1f} MiB) -> {nbytes / dt / 1e9:.1f} GB/s over the host link")
    # reference: explicit H2D copy bandwidth of one big pinned buffer
    big = torch.empty(512 * 2**20, dtype=torch.uint8).pin_memory()
    dst = torch.empty_like(big, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        dst.copy_(big, non_blocking=True)
    torch.cuda.synchronize()
    print(f"H2D memcpy: {5 * big.numel() / (time.perf_counter() - t0) / 1e9:.1f} GB/s")


if __name__ == "__main__":
    main()
