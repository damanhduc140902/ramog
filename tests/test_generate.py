"""ramog.generate.Generator on a tiny random Qwen3-MoE store (tests/test_tiny_engine.py's checkpoint and store):
any CUDA/ROCm GPU, Linux or Windows, no download.

1. Greedy (temperature 0, no stop ids) == session.GreedyDecoder.generate, token for token, graph and eager mode.
2. A second prompt on the same Generator (graph replayed, not captured again) == a fresh GreedyDecoder on it.
3. Stop ids: generation ends at the first stop id, which is included; finish "stop".
4. on_token returning False ends it after that token; finish "cancel".
5. Sampling: the same seed gives the same tokens; temperature 0 ignores the seed.
6. pick(): top_k 1 == argmax; top_p keeps the smallest prefix whose mass reaches top_p (CPU, exact).
python tests/test_generate.py [--work DIR] [--out result.json]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ramog.build import build_converted  # noqa: E402
from ramog.convert import convert  # noqa: E402
from ramog.generate import Generator, pick  # noqa: E402
from ramog.session import GreedyDecoder  # noqa: E402
from test_tiny_engine import write_checkpoint  # noqa: E402


def check_pick() -> dict:
    x = torch.tensor([0.0, 3.0, 1.0, 2.0])
    g = torch.Generator().manual_seed(0)
    top1 = all(pick(x, 1.0, 1.0, 1, g) == 1 for _ in range(20))
    # probabilities of [3, 2, 1, 0] at T=1: 0.644, 0.237, 0.087, 0.032; top_p 0.7 keeps {1, 3} (0.644 < 0.7 <= 0.881)
    seen = {pick(x, 1.0, 0.7, 0, g) for _ in range(400)}
    return {"top_k_1_is_argmax": top1, "top_p_support": sorted(seen), "ok": top1 and seen == {1, 3}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work")
    ap.add_argument("--out")
    args = ap.parse_args()
    tmp = None if args.work else tempfile.TemporaryDirectory()
    work = Path(args.work or tmp.name)
    ckpt, store = work / "ckpt", work / "store"
    if not (store / "meta.json").exists():
        write_checkpoint(ckpt)
        convert(str(ckpt), str(store), device="cuda:0", cold_format="k4row", cold_group=16, log=lambda *a: None)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "pick": check_pick()}
    # hot from cold: every expert comes from the same cold blobs wherever it is cached, so two runs of one prompt
    # must agree token for token whatever the cache held before (with a hot tier they legitimately may not)
    model, info = build_converted(str(store), vram_gb=2.0, dense_bits=None, hot_from_cold=True, log=lambda *a: None)
    g = torch.Generator().manual_seed(1)
    p1 = torch.randint(0, 4096, (1, 24), generator=g).to(info["device"])
    p2 = torch.randint(0, 4096, (1, 31), generator=g).to(info["device"])
    N, L = 20, 96

    def greedy_ref(prompt, mode):
        dec = GreedyDecoder(model, info["manager"], max_len=L, mode=mode, device=info["device"], entry=info["entry"])
        dec.warmup_and_capture(prompt)
        dec.prefill(prompt)
        out, _ = dec.generate(N)
        return out.tolist()

    for mode in ("graph", "eager"):
        r = {}
        ref1, ref2 = greedy_ref(p1, mode), greedy_ref(p2, mode)
        gen = Generator(model, info, L, mode=mode)
        t1, f1, _ = gen.generate(p1[0].tolist(), N)
        t2, f2, _ = gen.generate(p2[0].tolist(), N)
        r["greedy_equal"] = t1 == ref1 and f1 == "length"
        r["second_prompt_equal"] = t2 == ref2 and f2 == "length"
        stop = ref1[5]
        k = ref1.index(stop)
        t3, f3, _ = gen.generate(p1[0].tolist(), N, stop_ids=[stop])
        r["stop"] = t3 == ref1[:k + 1] and f3 == "stop"
        seen = []
        t4, f4, _ = gen.generate(p1[0].tolist(), N, on_token=lambda t: seen.append(t) or len(seen) < 3)
        r["cancel"] = t4 == ref1[:3] and f4 == "cancel" and seen == ref1[:3]
        s1, _, _ = gen.generate(p1[0].tolist(), N, temperature=0.8, top_p=0.95, seed=7)
        s2, _, _ = gen.generate(p1[0].tolist(), N, temperature=0.8, top_p=0.95, seed=7)
        t5, _, _ = gen.generate(p1[0].tolist(), N, temperature=0.0, seed=123)
        r["seeded_sampling_repeats"] = s1 == s2
        r["temperature_0_ignores_seed"] = t5 == ref1
        r["sample_differs_from_greedy"] = s1 != ref1  # informative only
        r["ok"] = all(v for kk, v in r.items() if kk != "sample_differs_from_greedy")
        res[mode] = r
        print(mode, r, flush=True)
    res["ok"] = res["pick"]["ok"] and res["graph"]["ok"] and res["eager"]["ok"]
    print(json.dumps({"pick": res["pick"], "ok": res["ok"]}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    os._exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
