"""ramog command line.

    ramog doctor                       GPU, RAM and library versions
    ramog plan MODEL_DIR               what auto-config would run on this machine (nothing is loaded)
    ramog run MODEL_DIR                chat in the terminal (--prompt TEXT: one answer, then exit)
    ramog serve MODEL_DIR              OpenAI-compatible HTTP API (/v1/chat/completions, /v1/completions)
    ramog bench MODEL_DIR              prefill and decode speed on this machine (greedy, fixed lengths)
    ramog pull MODEL [--variant V]     download a published model from Hugging Face

MODEL_DIR is a model directory with one subdirectory per variant (H4, TQ4.0, TQ3.5, ...) or one variant directory.
--vram-gb / --ram-gb override the detected budgets (default: the GPU's free memory, the machine's RAM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__

HF_ORG_ENV = "RAMOG_HF_ORG"  # the Hugging Face user / org that publishes the stores (not decided yet)

BENCH_TEXT = (  # plain prose for `bench` (written for this tool); repeated to reach --prompt-tokens
    "The first mechanical calculators were built in the seventeenth century, long before anyone imagined a machine "
    "that could follow a stored list of instructions. Over the next three hundred years, engineers replaced gears "
    "with relays, relays with vacuum tubes, and tubes with transistors, and each step made machines smaller, cheaper "
    "and more reliable. Today a phone in a pocket performs more arithmetic in one second than all the human "
    "computers of the early twentieth century managed in their whole working lives. Explain, step by step and with "
    "concrete examples, how these changes affected science, business and everyday life, and which of them you "
    "consider the most important. ")


def _budget_args(p):
    p.add_argument("model", help="model directory (one subdirectory per variant) or a variant directory")
    p.add_argument("--vram-gb", type=float, help="GPU memory to use (default: free memory now)")
    p.add_argument("--ram-gb", type=float, help="machine RAM (default: total RAM)")
    p.add_argument("--variant", default="auto", help="auto (default) or a variant name, e.g. TQ3.5")
    p.add_argument("--prefer", default="balanced", choices=("balanced", "quality", "speed"),
                   help="auto's trade-off: balanced (default), quality, speed")
    p.add_argument("--device", default="cuda:0")


def _gen_args(p):
    p.add_argument("--ctx", type=int, default=4096, help="context length (prompt + answer), default 4096")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", choices=("graph", "eager"), help="decode mode (default: the architecture's)")
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile and the fused decode kernels")
    p.add_argument("-v", "--verbose", action="store_true", help="show the engine's build log")


def cmd_doctor(a) -> int:
    from . import app

    print(json.dumps({"ramog": __version__, **app.env_summary()}, indent=1))
    return 0


def cmd_plan(a) -> int:
    from . import app

    cfg, stores, vram = app.plan(a.model, a.vram_gb, a.ram_gb, a.variant, a.device, log=print, prefer=a.prefer)
    print(f"\nvariants found: {', '.join(s.name for s in stores)}")
    print(f"VRAM budget {vram:g} GiB -> {cfg.describe()}")
    return 0 if cfg.fit else 1


def _load(a):
    from . import app

    log = print if a.verbose else (lambda *x: None)
    t0 = time.time()
    print(f"loading {a.model} ...", flush=True)
    eng = app.load(a.model, a.vram_gb, a.ram_gb, a.variant, a.device, ctx=a.ctx, mode=a.mode, log=log,
                   compile=not a.no_compile, prefer=a.prefer)
    print(f"ready in {time.time() - t0:.0f}s: {eng.cfg.describe()}", flush=True)
    return eng


def _sampling(a) -> dict:
    return {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed}


def cmd_run(a) -> int:
    from . import app

    eng = _load(a)
    msgs = [{"role": "system", "content": a.system}] if a.system else []

    def answer() -> None:
        def show(s):
            sys.stdout.write(s)
            sys.stdout.flush()

        _text, finish, u = eng.chat(msgs, a.max_new_tokens, on_text=show, **_sampling(a))
        msgs.append({"role": "assistant", "content": _text})
        print(f"\n[{u.completion_tokens} tokens, {u.tok_per_s:.2f} tok/s, {finish}]", flush=True)

    try:
        if a.prompt is not None:
            msgs.append({"role": "user", "content": a.prompt})
            answer()
            return 0
        print("chat: /reset clears the conversation, /exit quits", flush=True)
        while True:
            try:
                line = input("\n> ")
            except EOFError:
                break
            if line.strip() in ("/exit", "/quit"):
                break
            if line.strip() == "/reset":
                msgs[:] = msgs[:1] if a.system else []
                continue
            if not line.strip():
                continue
            msgs.append({"role": "user", "content": line})
            try:
                answer()
            except ValueError as e:  # context full
                msgs.pop()
                print(f"[{e}; /reset to start over]")
            except KeyboardInterrupt:
                print("\n[interrupted]")
    finally:
        app.release()
    return 0


def cmd_serve(a) -> int:
    from . import app, serve

    eng = _load(a)
    defaults = {"max_new_tokens": a.max_new_tokens, **_sampling(a)}
    try:
        serve.serve(eng, a.host, a.port, name=a.name or eng.name, defaults=defaults)
    finally:
        app.release()
    return 0


def cmd_bench(a) -> int:
    """Prefill and decode speed: a chat prompt of about --prompt-tokens tokens (plain prose), --new-tokens greedy
    tokens without stopping at end-of-text, --repeats times after one warm-up (kernel autotuning, graph capture)."""
    import torch

    from . import app, autoconfig

    eng = _load(a)
    text = BENCH_TEXT
    while len(eng.encode_chat([{"role": "user", "content": text}])) < a.prompt_tokens:
        text += BENCH_TEXT
    ids = eng.encode_chat([{"role": "user", "content": text}])
    need = len(ids) + a.new_tokens + 1
    if need > eng.ctx:
        eng.gen = type(eng.gen)(eng.model, eng.info, need, mode=a.mode)
    eng.gen.generate(ids, 8, stop_ids=())  # warm-up
    runs = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        toks, _fin, dt = eng.gen.generate(ids, a.new_tokens, stop_ids=())
        wall = time.perf_counter() - t0
        pre = wall - dt
        runs.append({"prefill_s": round(pre, 3), "prefill_tok_s": round(len(ids) / pre, 1),
                     "decode_tok_s": round((len(toks) - 1) / dt, 2)})
        print(f"prefill {len(ids)} tokens {pre:.2f} s ({len(ids) / pre:.1f} tok/s), "
              f"decode {len(toks)} tokens {(len(toks) - 1) / dt:.2f} tok/s", flush=True)
    res = {"model": a.model, "config": eng.cfg.describe(), "variant": eng.cfg.variant, "mode": eng.gen.mode,
           "vram_gb": eng.cfg.inputs.get("vram_gib"), "ram_gb": eng.cfg.inputs.get("ram_gib"),
           "prompt_tokens": len(ids), "new_tokens": a.new_tokens, "runs": runs,
           "speedups": eng.info.get("speedups"),
           "decode_tok_s_median": sorted(r["decode_tok_s"] for r in runs)[len(runs) // 2],
           "prefill_tok_s_median": sorted(r["prefill_tok_s"] for r in runs)[len(runs) // 2],
           "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "pinned_gib": round(eng.cfg.pinned_bytes / 2**30, 2), "env": app.env_summary(),
           "mem_available_gib_now": autoconfig.mem_available_gib()}
    print(json.dumps({k: res[k] for k in ("config", "prompt_tokens", "decode_tok_s_median", "prefill_tok_s_median",
                                          "peak_vram_gib", "pinned_gib")}), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    app.release()
    return 0


def cmd_pull(a) -> int:
    org = a.org or os.environ.get(HF_ORG_ENV)
    if not org:
        print(f"no Hugging Face publisher set: pass --org or set {HF_ORG_ENV} (pre-converted models are not "
              "published yet)", file=sys.stderr)
        return 2
    repo = a.model if "/" in a.model else f"{org}/{a.model}"
    from . import app

    patterns = [f"{v}/*" for v in a.variant] if a.variant else None
    if patterns is not None:
        patterns += list(app.TOKENIZER_FILES) + ["README.md"]
    dest = a.dir or os.path.join("models", repo.split("/")[-1])
    print(f"{repo} -> {dest}" + (f" (variants {', '.join(a.variant)})" if a.variant else " (every variant)"))
    if a.dry_run:
        return 0
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir=dest, allow_patterns=patterns)
    print(f"done: ramog run {dest}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ramog", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"ramog {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="GPU, RAM and library versions")
    p = sub.add_parser("plan", help="what auto-config would run on this machine")
    _budget_args(p)
    p = sub.add_parser("run", help="chat in the terminal")
    _budget_args(p)
    _gen_args(p)
    p.add_argument("--prompt", help="answer this prompt and exit")
    p.add_argument("--system", help="system prompt")
    p = sub.add_parser("serve", help="OpenAI-compatible HTTP API")
    _budget_args(p)
    _gen_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--name", help="model name in the API (default: the model directory's name)")
    p = sub.add_parser("bench", help="prefill and decode speed on this machine")
    _budget_args(p)
    _gen_args(p)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", help="write the result as JSON")
    p = sub.add_parser("pull", help="download a published model from Hugging Face")
    p.add_argument("model", help="model name, or ORG/REPO")
    p.add_argument("--variant", action="append", help="only these variants (repeatable; default: all)")
    p.add_argument("--org", help=f"publisher (default: ${HF_ORG_ENV})")
    p.add_argument("--dir", help="destination (default: models/<name>)")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "plan": cmd_plan, "run": cmd_run, "serve": cmd_serve, "bench": cmd_bench,
            "pull": cmd_pull}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
