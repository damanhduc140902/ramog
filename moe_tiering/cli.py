"""moe-tiering command line.

    moe-tiering doctor                       GPU, RAM and library versions
    moe-tiering plan MODEL_DIR               what auto-config would run on this machine (nothing is loaded)
    moe-tiering run MODEL_DIR                chat in the terminal (--prompt TEXT: one answer, then exit)
    moe-tiering serve MODEL_DIR              OpenAI-compatible HTTP API (/v1/chat/completions, /v1/completions)
    moe-tiering pull MODEL [--variant V]     download a published model from Hugging Face

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

HF_ORG_ENV = "MOE_TIERING_HF_ORG"  # the Hugging Face user / org that publishes the stores (not decided yet)


def _budget_args(p):
    p.add_argument("model", help="model directory (one subdirectory per variant) or a variant directory")
    p.add_argument("--vram-gb", type=float, help="GPU memory to use (default: free memory now)")
    p.add_argument("--ram-gb", type=float, help="machine RAM (default: total RAM)")
    p.add_argument("--variant", default="auto", help="auto (default) or a variant name, e.g. TQ3.5")
    p.add_argument("--device", default="cuda:0")


def _gen_args(p):
    p.add_argument("--ctx", type=int, default=4096, help="context length (prompt + answer), default 4096")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int)
    p.add_argument("--mode", choices=("graph", "eager"), help="decode mode (default: the architecture's)")
    p.add_argument("-v", "--verbose", action="store_true", help="show the engine's build log")


def cmd_doctor(a) -> int:
    from . import app

    print(json.dumps({"moe_tiering": __version__, **app.env_summary()}, indent=1))
    return 0


def cmd_plan(a) -> int:
    from . import app

    cfg, stores, vram = app.plan(a.model, a.vram_gb, a.ram_gb, a.variant, a.device, log=print)
    print(f"\nvariants found: {', '.join(s.name for s in stores)}")
    print(f"VRAM budget {vram:g} GiB -> {cfg.describe()}")
    return 0 if cfg.fit else 1


def _load(a):
    from . import app

    log = print if a.verbose else (lambda *x: None)
    t0 = time.time()
    print(f"loading {a.model} ...", flush=True)
    eng = app.load(a.model, a.vram_gb, a.ram_gb, a.variant, a.device, ctx=a.ctx, mode=a.mode, log=log)
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
    print(f"done: moe-tiering run {dest}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="moe-tiering", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"moe-tiering {__version__}")
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
    p = sub.add_parser("pull", help="download a published model from Hugging Face")
    p.add_argument("model", help="model name, or ORG/REPO")
    p.add_argument("--variant", action="append", help="only these variants (repeatable; default: all)")
    p.add_argument("--org", help=f"publisher (default: ${HF_ORG_ENV})")
    p.add_argument("--dir", help="destination (default: models/<name>)")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return {"doctor": cmd_doctor, "plan": cmd_plan, "run": cmd_run, "serve": cmd_serve, "pull": cmd_pull}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
