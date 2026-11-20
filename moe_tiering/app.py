"""The user-facing engine: open a model directory, let auto-config pick a variant for this machine, chat.

    eng = load(path, vram_gb=None, ram_gb=None, variant="auto", device="cuda:0", ctx=4096)
    text, finish, usage = eng.chat(messages, max_new_tokens=512, temperature=0.7, top_p=0.95, on_text=print)

path: a store-v2 variant directory (meta.json) or a model directory whose subdirectories are variants (the published
layout <model>/<variant>/, e.g. <model>/TQ3.5). VRAM defaults to the GPU's free memory at start, RAM to the
machine's total (auto-config also caps the pinned budget by what is available now). The tokenizer is read from the
first of: the variant directory, the model directory, meta.json "source" (the checkpoint the store was made from).

Engine serializes generation (one GPU, batch 1): chat / complete hold a lock, so a server can call them from several
threads. on_text receives the new text as it is decoded (held back while the last token ends inside a multi-byte
character); returning False from it cancels the generation.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "generation_config.json")


def find_stores(path) -> list[Path]:
    """The variant directories of a model directory, or [path] for a variant directory."""
    p = Path(path)
    if (p / "meta.json").exists():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"{p}: no such directory")
    out = sorted(x for x in p.iterdir() if (x / "meta.json").exists())
    if not out:
        raise FileNotFoundError(f"{p}: neither a store (meta.json) nor a directory of stores")
    return out


def tokenizer_dir(stores: list[Path]) -> Path:
    """Where the tokenizer files are (see the module docstring)."""
    cands = []
    for s in stores:
        cands += [s, s.parent]
    for s in stores:
        src = json.loads((s / "meta.json").read_text(encoding="utf-8")).get("source")
        if src:
            cands.append(Path(src))
    for c in cands:
        if (c / "tokenizer.json").exists() or (c / "tokenizer_config.json").exists():
            return c
    raise FileNotFoundError("no tokenizer found next to the store(s) or at meta.json 'source'; copy the model's "
                            f"tokenizer files ({', '.join(TOKENIZER_FILES[:3])}, ...) into {stores[0].parent}")


def gpu_free_gib(device="cuda:0") -> float:
    free, _total = torch.cuda.mem_get_info(torch.device(device))
    return int(free / 2**30 * 10) / 10  # rounded down to 0.1 GiB


def stop_ids_of(tok, tdir: Path) -> list[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    gc = tdir / "generation_config.json"
    if gc.exists():
        e = json.loads(gc.read_text(encoding="utf-8")).get("eos_token_id")
        ids.update(int(x) for x in (e if isinstance(e, list) else [e]) if x is not None)
    return sorted(ids)


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    seconds: float

    @property
    def tok_per_s(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


class Engine:
    def __init__(self, model, info: dict, cfg, tok, tdir: Path, name: str, ctx: int, mode: str | None = None):
        from .generate import Generator

        self.model, self.info, self.cfg, self.tok, self.name, self.ctx = model, info, cfg, tok, name, ctx
        self.stop_ids = stop_ids_of(tok, tdir)
        self.gen = Generator(model, info, ctx, mode=mode)
        self.lock = threading.Lock()

    def encode_chat(self, messages: list[dict], tools=None) -> list[int]:
        ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return list(ids)

    def encode_text(self, text: str) -> list[int]:
        return self.tok(text)["input_ids"]

    def complete(self, ids: list[int], max_new_tokens: int = 512, *, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 0, seed: int | None = None, stop: list[str] | None = None, on_text=None):
        """Generate after the token ids; returns (text, finish "stop" | "length" | "cancel", Usage)."""
        room = self.ctx - len(ids)
        if room <= 0:
            raise ValueError(f"the prompt has {len(ids)} tokens; the context is {self.ctx} (raise --ctx)")
        n = min(max_new_tokens, room)
        out, state = [], {"sent": 0, "text": "", "stopped": False}
        stops = [s for s in (stop or []) if s]

        def on_token(t: int):
            out.append(t)
            if t in self.stop_ids:
                return True
            text = self.tok.decode(out, skip_special_tokens=True)
            if stops:
                cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
                if cut >= 0:
                    text, state["stopped"] = text[:cut], True
            state["text"] = text
            if state["stopped"]:
                ready = len(text)
            else:  # hold back a partial character and a tail that may still grow into a stop string
                ready = len(text.rstrip("�"))
                hold = max((k for s in stops for k in range(1, len(s)) if text[:ready].endswith(s[:k])), default=0)
                ready -= hold
            if on_text is not None and ready > state["sent"]:
                if on_text(text[state["sent"]:ready]) is False:
                    return False
                state["sent"] = ready
            return not state["stopped"]

        with self.lock:
            toks, finish, dt = self.gen.generate(ids, n, temperature=temperature, top_p=top_p, top_k=top_k,
                                                 stop_ids=self.stop_ids, seed=seed, on_token=on_token)
        if state["stopped"]:
            finish = "stop"
        text = state["text"]
        if on_text is not None and len(text) > state["sent"] and finish != "cancel":
            on_text(text[state["sent"]:])
        return text, finish, Usage(len(ids), len(toks), dt)

    def chat(self, messages: list[dict], max_new_tokens: int = 512, tools=None, **kw):
        return self.complete(self.encode_chat(messages, tools=tools), max_new_tokens, **kw)


def plan(path, vram_gb: float | None = None, ram_gb: float | None = None, variant: str = "auto",
         device="cuda:0", log=print):
    """The auto-config choice for this machine (no model load). Returns (cfg, stores, vram_gb)."""
    from . import autoconfig

    stores = find_stores(path)
    if vram_gb is None:
        vram_gb = gpu_free_gib(device)
    cfg = autoconfig.plan([str(s) for s in stores], vram_gb, ram_gb, variant=variant, log=log)
    return cfg, stores, vram_gb


def load(path, vram_gb: float | None = None, ram_gb: float | None = None, variant: str = "auto", device="cuda:0",
         ctx: int = 4096, mode: str | None = None, log=print) -> Engine:
    from transformers import AutoTokenizer

    from . import autoconfig

    cfg, stores, vram_gb = plan(path, vram_gb, ram_gb, variant, device, log)
    if not cfg.fit:
        raise SystemExit(f"this machine cannot run {path}: {cfg.why}")
    tdir = tokenizer_dir(stores)
    tok = AutoTokenizer.from_pretrained(str(tdir))
    model, info = autoconfig.build(cfg, device, vram_gb, log=log)
    name = Path(path).resolve().name if len(stores) > 1 else stores[0].parent.name
    return Engine(model, info, cfg, tok, tdir, name, ctx, mode)


def release() -> None:
    """Free the page-locked expert memory (call before exit on Windows)."""
    from . import hostmem

    hostmem.release()


def env_summary() -> dict:
    out = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    try:
        import transformers
        import triton

        out["transformers"], out["triton"] = transformers.__version__, triton.__version__
    except ImportError as e:
        out["missing"] = str(e)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = torch.cuda.get_device_name(0)
        out["vram_total_gib"], out["vram_free_gib"] = round(total / 2**30, 1), round(free / 2**30, 1)
    from . import autoconfig

    out["ram_total_gib"] = autoconfig.mem_total_gib()
    out["ram_available_gib"] = autoconfig.mem_available_gib()
    out["os"] = os.name
    return out
