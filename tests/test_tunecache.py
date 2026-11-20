"""moe_tiering.tunecache: autotuning choices saved to disk and loaded by the next process (any CUDA/ROCm GPU).

Runs in a temporary MOE_TIERING_TUNE_DIR: a grouped GEMV launch is tuned (its benchmark runs), save() writes the
choice, the in-memory cache is cleared, load() restores the same configuration and the next launch runs without
benchmarking; the result matches the tuned launch (up to split-K summation order); a different engine version
uses another file; and MOE_TIERING_TUNE_CACHE=0 turns loading and saving off.
python tests/test_tunecache.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_tiering import kernels, tunecache  # noqa: E402
from moe_tiering.store import BlobLayout  # noqa: E402


def main() -> int:
    fails = 0
    dev = torch.device("cuda")
    with tempfile.TemporaryDirectory() as d:
        os.environ["MOE_TIERING_TUNE_DIR"] = d
        tuner = kernels._grouped_gemv
        calls = {"n": 0}
        bench = tuner._bench

        def counting(*a, **k):
            calls["n"] += 1
            return bench(*a, **k)

        tuner._bench = counting
        lay = BlobLayout(4, 64, (192, 256), (256, 192))
        pool = torch.cat([lay.pack(torch.randn(192, 256, device=dev) * 0.02, torch.randn(256, 192, device=dev) * 0.02)[None]
                          for _ in range(3)])
        v = lay.views(pool)
        x = torch.randn(2, 256, device=dev, dtype=torch.bfloat16)
        tok = torch.tensor([0, 1, 1], dtype=torch.int32, device=dev)
        slot = torch.tensor([2, 0, 1], dtype=torch.int32, device=dev)
        n0 = tunecache.load(dev)
        y1 = kernels.grouped_gemv(x, tok, slot, **lay.gemv_args(v, "gu"))
        tuned = calls["n"]
        written = tunecache.save()
        path = tunecache.path()
        entries = json.load(open(path))["entries"]
        key = next(k for k in tuner.cache if k[0] == 192)
        cfg = tuner.cache[key]
        print(f"load before tuning: {n0}; tuned with {tuned} benchmark runs; save wrote {written} entries to "
              f"{os.path.basename(path)}")
        fails += not (n0 == 0 and tuned > 0 and written >= 1 and "_grouped_gemv" in entries)

        tuner.cache.clear()
        n1 = tunecache.load(dev)
        restored = tuner.cache.get(key)
        same = restored is not None and restored.kwargs == cfg.kwargs and restored.num_warps == cfg.num_warps
        calls["n"] = 0
        y2 = kernels.grouped_gemv(x, tok, slot, **lay.gemv_args(v, "gu"))
        torch.cuda.synchronize()
        diff = float((y1 - y2).abs().max() / y1.abs().max())  # split-K atomics: equal up to summation order
        print(f"after clearing: load restored {n1} entries, same config {same}, benchmark runs on the next launch "
              f"{calls['n']}, result max rel diff {diff:.1e}")
        fails += not (n1 >= 1 and same and calls["n"] == 0 and diff < 1e-5)

        old = tunecache.ENGINE_VERSION
        tunecache.ENGINE_VERSION = old + "-test"
        other, _ = tunecache._locate(dev)
        tunecache.ENGINE_VERSION = old
        print(f"another engine version uses another file: {other != path}")
        fails += not other != path

        os.environ["MOE_TIERING_TUNE_CACHE"] = "0"
        off = (tunecache.load(dev), tunecache.save())
        del os.environ["MOE_TIERING_TUNE_CACHE"]
        print(f"MOE_TIERING_TUNE_CACHE=0: load/save {off}")
        fails += off != (0, 0)
        tuner._bench = bench
        del os.environ["MOE_TIERING_TUNE_DIR"]
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
