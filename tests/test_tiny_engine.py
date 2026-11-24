"""End-to-end engine test on a tiny random Qwen3-MoE checkpoint: no download, any CUDA/ROCm GPU, Linux or Windows.

1. Writes a random checkpoint in the Hugging Face layout of Qwen3-MoE (per-expert gate/up/down tensors + index;
   router logits spread out so that rounding rarely flips the top-k: a smooth network, hence a sharp test).
2. Converts it with ramog.convert: hot rtn 4-bit g64 + cold k4row 2-bit g16 (plain weight-MSE importance).
3. Reference: a float32 transformers forward of the same checkpoint with every expert replaced by its dequantized
   k4row blob (what the engine must reproduce), and the original weights (only to size the quantization effect).
4. Engine, --hot-from-cold (every expert served from k4row blobs: VRAM cache, pinned-host zero-copy, prefetch
   staging; a small cache so all three are used): ramog.session.TeacherForced's graph-captured one-token decode with
   cache maintenance between replays; per position KL(reference || engine) on the full vocabulary and top-1.
   Pass: the engine's KL is at the bf16 noise floor (KL between float32 and bf16 forwards of the reference,
   <= 3x floor + 2e-4) and below 5% of the quantization effect (KL original vs dequantized weights).
5. Engine, normal store (hot rtn 4-bit + cold k4row): GraphDecoder prefill + greedy generation (smoke run).
python tests/test_tiny_engine.py [--work DIR] [--out result.json]
"""

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ramog import arch  # noqa: E402
from ramog.build import build_converted, reset_stats, stats  # noqa: E402
from ramog.convert import convert  # noqa: E402
from ramog.store import BlobLayout  # noqa: E402

FUSED = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def write_checkpoint(d: Path, seed: int = 0, expert_scale: float = 1.0, router_scale: float = 16.0,
                     io_scale: float = 4.0) -> None:
    """Random Qwen3-MoE in the Hugging Face per-expert layout. router_scale widens the gaps between router logits
    (fewer top-k flips under rounding: a smoother network, so the comparison is sharp); io_scale sharpens the output
    distributions; expert_scale sizes the expert outputs."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    cfg = Qwen3MoeConfig(vocab_size=4096, hidden_size=512, intermediate_size=1024, moe_intermediate_size=256,
                         # head_dim 128 as in every Qwen3: on ROCm (torch 2.11, sdpa) a StaticCache decode with
                         # head_dim 64 and a cache length that is not a multiple of 128 does not match the
                         # whole-sequence forward (agents/k4/diag_static_cache.py), engine or not
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                         num_experts=32, num_experts_per_tok=8, norm_topk_prob=True, max_position_embeddings=1024,
                         tie_word_embeddings=False, rms_norm_eps=1e-6)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    sd = {}
    inter = cfg.moe_intermediate_size
    for k, v in model.state_dict().items():
        if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight"):
            v = v * io_scale
        if ".mlp.experts." in k:
            v = v * expert_scale
        if k.endswith(".mlp.gate.weight"):
            v = v * router_scale
        m = FUSED.match(k)
        if m is None:  # dense tensors (and per-expert tensors, if this transformers keeps them unfused)
            sd[k] = v.to(torch.bfloat16).contiguous()
            continue
        layer, kind = m.groups()
        for e in range(v.shape[0]):
            p = f"model.layers.{layer}.mlp.experts.{e}."
            if kind == "gate_up_proj":  # [E, 2I, H]: gate rows, then up rows
                assert v.shape[1:] == (2 * inter, cfg.hidden_size), v.shape
                sd[p + "gate_proj.weight"] = v[e, :inter].to(torch.bfloat16).contiguous()
                sd[p + "up_proj.weight"] = v[e, inter:].to(torch.bfloat16).contiguous()
            else:  # [E, H, I]
                assert v.shape[1:] == (cfg.hidden_size, inter), v.shape
                sd[p + "down_proj.weight"] = v[e].to(torch.bfloat16).contiguous()
    d.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in sd}}))
    cfg.save_pretrained(d)


@torch.no_grad()
def reference_logp(ckpt: Path, store: Path | None, tokens: torch.Tensor, dev, dtype=torch.float32) -> torch.Tensor:
    """Forward of the checkpoint in `dtype`; with `store`, every expert replaced by its dequantized cold blob."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=dtype).to(dev).eval()
    if store is not None:
        meta = json.loads((store / "meta.json").read_text())
        c = meta["cold"]
        lay = BlobLayout(c["bits"], c["group"], tuple(meta["gu_shape"]), tuple(meta["dn_shape"]), c["format"])
        for i, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            blobs = torch.from_numpy(np.fromfile(store / f"layer{i:03d}.cold.bin", dtype=np.uint8))
            v = lay.views(blobs.view(meta["num_experts"], lay.nbytes).to(dev))
            for e in range(meta["num_experts"]):
                ex.gate_up_proj.data[e] = lay.dequant(v, e, "gu", dtype)
                ex.down_proj.data[e] = lay.dequant(v, e, "dn", dtype)
    logits = model(input_ids=tokens[None].to(dev), use_cache=False).logits[0].float()
    del model
    torch.cuda.empty_cache()
    return F.log_softmax(logits, -1).cpu()


def kl_rows(p_logp: torch.Tensor, q_logp: torch.Tensor) -> torch.Tensor:
    return (p_logp.exp() * (p_logp - q_logp)).sum(-1)


def run_engine(path: Path, meta: dict, tokens: torch.Tensor, ref: torch.Tensor, seq_len: int, *, prefetch=True,
               manager=True, graph=True, warm_all=False) -> dict:
    """Teacher-forced decode of `tokens` through the engine built from `path` with --hot-from-cold; KL vs ref."""
    from ramog import session  # TeacherForced: graph-captured single-token decode with external input ids

    gc.collect()
    torch.cuda.empty_cache()
    dense = (path / "dense.safetensors").stat().st_size
    lay_bytes = meta["cold"]["nbytes"] * meta["num_layers"]
    slots = meta["num_experts"] + 1 if warm_all else 10.5  # about 10 cache slots per layer (32 experts)
    vram = (torch.cuda.memory_allocated() + 0.75 * 2**30 + 1.5 * dense + meta["cold"]["nbytes"]
            + slots * lay_bytes) / 2**30
    model, info = build_converted(str(path), vram_gb=vram, dense_bits=None, hot_from_cold=True, log=lambda *a: None)
    info["prefetcher"].enabled = prefetch
    if warm_all:
        for c in info["manager"].caches:
            c.warm(list(range(info["num_experts"])))
    info["manager"].enabled = manager
    runner = session.TeacherForced(model, info["manager"], max_len=seq_len + 8, device=info["device"],
                                       entry=info["entry"])
    runner.capture(tokens)  # autotunes eagerly, then captures (the graph is unused when graph=False)
    reset_stats(model)
    runner.cache.reset()
    kls, agree = [], 0
    with torch.no_grad():
        for t in range(seq_len - 1):
            runner.ids.copy_(tokens[t].view(1, 1))
            runner.pos.fill_(t)  # explicit position ids (ramog.session, WP9)
            runner.manager.step()
            if graph:
                runner.graph.replay()
                logits = runner.logits[0]
            else:
                logits = runner._forward(runner.cache)[0]
            q = F.log_softmax(logits.float(), -1).cpu()
            kls.append(float(kl_rows(ref[t], q)))
            agree += int(q.argmax() == ref[t].argmax())
    for core in arch.tiered(model):  # graph replays do not run the Python counters (hook or legacy block path)
        core.pair_count = (seq_len - 1) * core.top_k
    st = stats(model)
    pairs = max(1, st["pairs"])
    out = {"slots": info["slots"], "experts": info["num_experts"], "kl_mean": sum(kls) / len(kls), "kl_max": max(kls),
           "top1_agree": agree / len(kls), "served_cache": st["hit"] / pairs, "prefetched": st["staged"] / pairs,
           "zero_copy": 1 - (st["hit"] + st["staged"]) / pairs}
    del model, runner, info, core
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", help="work dir (default: a temporary dir, removed afterwards)")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--diag", action="store_true", help="also run with prefetch / cache maintenance / graphs off")
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = torch.device("cuda")
    tmp = None if args.work else tempfile.TemporaryDirectory()
    work = Path(args.work or tmp.name)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    t0 = time.time()

    ckpt, store, cold_only = work / "ckpt", work / "store-k4row", work / "store-k4row-coldonly"
    write_checkpoint(ckpt)
    log = lambda *a: None  # noqa: E731
    convert(str(ckpt), str(store), device="cuda:0", cold_format="k4row", cold_group=16, log=log)
    convert(str(ckpt), str(cold_only), device="cuda:0", cold_format="k4row", cold_group=16, cold_only=True, log=log)
    meta = json.loads((store / "meta.json").read_text())
    res["store"] = {"hot": meta["hot"], "cold": meta["cold"]}
    res["cold_only_meta_hot"] = json.loads((cold_only / "meta.json").read_text())["hot"]

    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 4096, (args.seq_len,), generator=g)
    ref = reference_logp(ckpt, store, tokens, dev)  # what the engine must reproduce (float32)
    ref16 = reference_logp(ckpt, store, tokens, dev, torch.bfloat16)  # same weights, bf16 arithmetic
    orig = reference_logp(ckpt, None, tokens, dev)
    res["kl_quantization_effect"] = float(kl_rows(orig[:-1], ref[:-1]).mean())  # size of what we could miss
    res["kl_bf16_floor"] = float(kl_rows(ref[:-1], ref16[:-1]).mean())  # the engine computes in bf16 too
    print({k: res[k] for k in ("kl_quantization_effect", "kl_bf16_floor")}, flush=True)

    variants = [("hot_from_cold", store, {}), ("cold_only_hot_from_cold", cold_only, {})]
    if args.diag:  # isolate a mismatch: which part of the decode path is responsible?
        variants += [(f"diag_{k}", store, kw) for k, kw in (
            ("no_prefetch", {"prefetch": False}), ("frozen_cache", {"manager": False}),
            ("eager", {"graph": False}), ("eager_no_prefetch_frozen", {"graph": False, "prefetch": False, "manager": False}),
            ("all_cached", {"warm_all": True, "manager": False, "prefetch": False}))]
    for name, path, kw in variants:
        res[name] = run_engine(path, meta, tokens, ref, args.seq_len, **kw)
        print(name, res[name], flush=True)

    # normal store: hot rtn 4-bit cache + cold k4row, prefill + greedy generation through GraphDecoder
    from ramog.runner import GraphDecoder

    dense = (store / "dense.safetensors").stat().st_size
    vram = (torch.cuda.memory_allocated() + 0.75 * 2**30 + 1.5 * dense + meta["cold"]["nbytes"]
            + 10.5 * meta["hot"]["nbytes"] * meta["num_layers"]) / 2**30
    model, info = build_converted(str(store), vram_gb=vram, dense_bits=None, hot_from_cold=False, log=log)
    dec = GraphDecoder(model, info["manager"], max_len=args.seq_len + 40)
    prompt = tokens[:24][None].to(dev)
    dec.warmup_and_capture(prompt)
    dec.prefill(prompt)
    out, dt = dec.generate(16)
    res["normal_store_generate"] = {"slots": info["slots"], "tokens": out.tolist(), "seconds": round(dt, 3)}
    print("normal store generate", res["normal_store_generate"], flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    # the engine must sit at the bf16 noise floor, far below the quantization effect it reproduces
    bound = 3 * res["kl_bf16_floor"] + 2e-4
    res["ok"] = (all(res[n]["kl_mean"] < bound and res[n]["kl_mean"] < 0.05 * res["kl_quantization_effect"]
                     for n in ("hot_from_cold", "cold_only_hot_from_cold"))
                 and res["hot_from_cold"]["served_cache"] > 0 and res["hot_from_cold"]["zero_copy"] > 0
                 and res["cold_only_meta_hot"] is None and res["normal_store_generate"]["slots"] > 0)
    res["kl_bound"] = bound
    print(json.dumps({k: res[k] for k in ("kl_quantization_effect", "kl_bf16_floor", "kl_bound", "ok", "seconds")}),
          flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    if tmp is not None:
        del model, dec
        torch.cuda.empty_cache()
        try:
            tmp.cleanup()
        except OSError:  # Windows may still hold the store files mapped (pinned numpy views)
            pass
    os._exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
