"""ramog.hostmem: exact-size page-locked blobs, read zero-copy by the kernels (CUDA/Windows and ROCm/Linux).

Checks: the file bytes arrive intact, the tensor is pinned at exact size (no power-of-two rounding), the grouped
GEMV reads a registered k4row pool zero-copy (inside a CUDA graph too, seeing host updates made after capture),
copy_blobs and async copies from it are right, a failed registration (a real failing runtime call) leaves no
pending error for the next torch / Triton launch, also through read_file's pin_memory() fallback, and release()
unregisters everything.
python tests/test_hostmem.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ramog import hostmem  # noqa: E402
from ramog.kernels import copy_blobs, grouped_gemv  # noqa: E402
from ramog.store import BlobLayout  # noqa: E402


def ref_gemv(lay, pool, x, tok, ids):
    v = lay.views(pool)
    return torch.stack([lay.dequant(v, int(e), "gu", torch.float64) @ x[int(t)].double() for t, e in zip(tok, ids)])


def launch_check(what, x, tok, neg, ids, lay, views, y_ref, expect_ok=True) -> int:
    """A torch kernel and a zero-copy Triton GEMV right after `what`: both must run and give the right result.
    Returns 1 on failure."""
    try:
        s = float((torch.arange(1024, device=x.device, dtype=torch.float32) * 2).sum())
        yy = grouped_gemv(x, tok, neg, **lay.gemv_args(views, "gu"), mode=2, expert=ids)
        torch.cuda.synchronize()
        err = float((yy - y_ref).abs().max() / y_ref.abs().max())
        ok = s == 1047552.0 and err < 1e-5
        msg = f"run, results right (rel diff {err:.1e})" if ok else f"WRONG RESULT (rel diff {err:.1e})"
    except Exception as ex:  # noqa: BLE001
        ok, msg = False, f"RAISED {type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}"
    ok = ok and expect_ok
    print(f"{what}: next torch + Triton launches {msg}")
    return 0 if ok else 1


def main() -> int:
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    fails = 0
    lay = BlobLayout(2, 16, (256, 512), (512, 128), "k4row")
    e = 7
    gu = torch.randn(e, 256, 512) * 0.02
    dn = torch.randn(e, 512, 128) * 0.02
    blobs = lay.quantize(gu, dn, device=dev).cpu()  # [e, nbytes]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "layer.bin")
        blobs.numpy().tofile(path)
        before = hostmem.registered_bytes()
        pool = hostmem.read_file(path, e * lay.nbytes).view(e, lay.nbytes)
        reg = hostmem.registered_bytes() - before
        exact = reg == -(-e * lay.nbytes // hostmem.PAGE) * hostmem.PAGE
        ok = torch.equal(pool, blobs) and pool.is_pinned() and exact and pool.data_ptr() % hostmem.PAGE == 0
        print(f"read_file: bytes equal {torch.equal(pool, blobs)}, is_pinned {pool.is_pinned()}, registered {reg} B "
              f"for {e * lay.nbytes} B (exact to the page: {exact}), page aligned {pool.data_ptr() % hostmem.PAGE == 0}"
              f", fallback {hostmem.fallback_bytes()} B")
        fails += not ok

        # zero-copy grouped GEMV (MODE 2: every pair reads the host pool) vs float64
        x = torch.randn(3, 512, device=dev, dtype=torch.bfloat16)
        tok = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int32, device=dev)
        ids = torch.tensor([6, 1, 3, 0, 6], dtype=torch.int32, device=dev)
        neg = torch.full_like(ids, -1)
        v = lay.views(pool)
        y = grouped_gemv(x, tok, neg, **lay.gemv_args(v, "gu"), mode=2, expert=ids)
        r = ref_gemv(lay, pool, x.cpu(), tok.cpu(), ids.cpu())
        err = float((y.double().cpu() - r).abs().max() / r.abs().max())
        print(f"zero-copy GEMV from the registered pool: max rel err {err:.2e}")
        fails += not err < 1e-5

        # graph capture, then change the host data: the replay must read the new bytes
        g = torch.cuda.CUDAGraph()
        out = torch.zeros(5, 256, device=dev)
        grouped_gemv(x, tok, neg, **lay.gemv_args(v, "gu"), mode=2, expert=ids, out=out.zero_())  # warm/tune
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out.zero_()
            grouped_gemv(x, tok, neg, **lay.gemv_args(v, "gu"), mode=2, expert=ids, out=out)
        pool[[6, 1, 3, 0]] = pool[[0, 3, 1, 6]].clone()  # permute experts in host memory after capture
        g.replay()
        torch.cuda.synchronize()
        perm = {6: 0, 1: 3, 3: 1, 0: 6}
        r2 = ref_gemv(lay, blobs, x.cpu(), tok.cpu(), torch.tensor([perm[int(i)] for i in ids.cpu()]))
        err2 = float((out.double().cpu() - r2).abs().max() / r2.abs().max())
        print(f"graph replay after a host update: max rel err {err2:.2e}")
        fails += not err2 < 1e-5

        # copy_blobs (Triton gather from host) and an async DMA copy
        dst = torch.empty(3, lay.nbytes, dtype=torch.uint8, device=dev)
        want = torch.ones(3, dtype=torch.int32, device=dev)
        sel = torch.tensor([2, 5, 0], dtype=torch.int32, device=dev)
        copy_blobs(pool, dst, sel, want)
        dma = torch.empty(e, lay.nbytes, dtype=torch.uint8, device=dev)
        dma.copy_(pool, non_blocking=True)
        torch.cuda.synchronize()
        ok = torch.equal(dst.cpu(), pool[[2, 5, 0]]) and torch.equal(dma.cpu(), pool)
        print(f"copy_blobs and async copy from the registered pool: {'equal' if ok else 'DIFFERENT'}")
        fails += not ok

        # pin_copy: exact-size pinned copy of a tensor
        t = torch.randn(1000, 33)
        pc = hostmem.pin_copy(t)
        ok = torch.equal(pc, t) and pc.is_pinned()
        print(f"pin_copy: equal and pinned {ok}")
        fails += not ok

        # A failed registration (low RAM, a pinning limit, ...) must leave no pending runtime error: torch's next
        # kernel-launch check would raise it as an unrelated "invalid argument". A real failing call: register an
        # unmapped range, then launch torch and Triton kernels.
        y_now = grouped_gemv(x, tok, neg, **lay.gemv_args(v, "gu"), mode=2, expert=ids)  # pool as permuted above
        torch.cuda.synchronize()
        bad = hostmem._register(1 << 16, 1 << 20)
        fails += launch_check(f"failed registration (unmapped range, register returned {bad})", x, tok, neg, ids,
                              lay, v, y_now, expect_ok=not bad)
        # the documented fallback: registration fails inside read_file -> pin_memory() copy, engine keeps running
        real = hostmem._register
        hostmem._register = lambda addr, nbytes: real(1 << 16, nbytes)  # the real runtime call, failing
        try:
            fb0 = hostmem.fallback_bytes()
            pool2 = hostmem.read_file(path, e * lay.nbytes).view(e, lay.nbytes)
        finally:
            hostmem._register = real
        ok = (torch.equal(pool2, blobs) and pool2.is_pinned()
              and hostmem.fallback_bytes() - fb0 == e * lay.nbytes)
        print(f"read_file fallback: bytes equal {torch.equal(pool2, blobs)}, pinned {pool2.is_pinned()}, fallback "
              f"{hostmem.fallback_bytes() - fb0} B")
        fails += not ok
        fails += launch_check("after the read_file fallback", x, tok, neg, ids, lay, lay.views(pool2), y)
        del pool2

        n = hostmem.release()
        ok = hostmem.registered_bytes() == 0 and not pool.is_pinned() and torch.equal(pool[[2, 5, 0]], dst.cpu())
        print(f"release(): {n} B unregistered; nothing left {hostmem.registered_bytes() == 0}; tensor still readable, "
              f"no longer pinned {not pool.is_pinned()}")
        fails += not ok
        del pool, v, pc
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
