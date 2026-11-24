"""Zero-copy reads of a few random experts out of a large host region: regular pinned memory vs a
transparent-huge-page (2 MiB) buffer registered with the GPU driver."""

import mmap
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramog.kernels import grouped_gemv  # noqa: E402
from ramog.quant import quantize  # noqa: E402

N, K, BITS, GROUP = 1536, 2048, 2, 64
EXPERTS = 4096  # ~5.8 GB of 2-bit blobs, like a real model's cold store


def huge_pinned(nbytes: int) -> tuple[torch.Tensor, mmap.mmap]:
    """Anonymous mmap with MADV_HUGEPAGE, touched, then registered (portable | mapped) with the driver."""
    size = (nbytes + (2 << 20) - 1) // (2 << 20) * (2 << 20)
    m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m.madvise(mmap.MADV_HUGEPAGE)
    t = torch.frombuffer(m, dtype=torch.uint8)
    t.zero_()  # fault the pages in as huge pages
    err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), size, 3)
    assert int(err) == 0, f"register failed: {err}"
    return t[:nbytes], m


def run(label, qw, sc, mn, x, reps=200):
    tok = torch.zeros(8, device="cuda", dtype=torch.int32)
    slot = torch.zeros(8, device="cuda", dtype=torch.int32)
    slot[:2] = -1
    exps = [torch.randint(0, EXPERTS, (8,), device="cuda", dtype=torch.int32) for _ in range(reps)]
    y = torch.zeros(8, N, device="cuda", dtype=torch.float32)
    for e in exps[:5]:
        grouped_gemv(x, tok, slot, qw, sc, mn, BITS, GROUP, mode=2, expert=e, out=y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for e in exps:
        grouped_gemv(x, tok, slot, qw, sc, mn, BITS, GROUP, mode=2, expert=e, out=y)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    nbytes = 2 * (N * K * BITS / 8 + 2 * N * (K // GROUP) * 2)
    print(f"{label:12s}: {dt * 1e6:6.0f} us per launch (2 random experts of {EXPERTS}), {nbytes / dt / 1e9:5.2f} GB/s", flush=True)


def main():
    if not hasattr(mmap, "MAP_PRIVATE"):  # anonymous private mappings (and huge pages): Linux only
        print("SKIP: Linux only")
        return
    torch.manual_seed(0)
    p = quantize(torch.randn(N, K, device="cuda") * 0.02, BITS, GROUP)
    q1, s1, m1 = p.qweight.cpu(), p.scale.cpu(), p.minimum.cpu()
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)

    # regular pinned memory (PyTorch's pinned allocator)
    qw = q1.unsqueeze(0).expand(EXPERTS, *q1.shape).contiguous().pin_memory()
    sc = s1.unsqueeze(0).expand(EXPERTS, *s1.shape).contiguous().pin_memory()
    mn = m1.unsqueeze(0).expand(EXPERTS, *m1.shape).contiguous().pin_memory()
    run("pin_memory", qw, sc, mn, x)
    del qw, sc, mn

    # THP-backed, registered
    keep = []
    out = []
    for src in (q1, s1, m1):
        flat, m = huge_pinned(EXPERTS * src.numel() * src.element_size())
        keep.append(m)
        t = flat.view(src.dtype).view(EXPERTS, *src.shape)
        t.copy_(src.unsqueeze(0).expand_as(t))
        out.append(t)
    run("hugepage", *out, x)
    with open("/proc/meminfo") as f:
        print([ln.strip() for ln in f if ln.startswith("AnonHugePages")])


if __name__ == "__main__":
    main()
