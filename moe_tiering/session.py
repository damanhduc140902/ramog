"""Decode sessions for engines built by moe_tiering.loader (design-core A8/B6; work package WP4).

Two modes, both with explicit device-side position ids (the value the model would derive itself; passing it keeps
the decode step independent of how a cache reports its length):
- "eager": one forward per token with a fresh DynamicCache per sequence (transformers' own default cache for the
  model, DynamicCache(config=model.config)); the Manager is stepped by the first TieredExperts (manager.external is
  False). This is the v0.1 path of every new architecture.
- "graph": a StaticCache and one captured decode step, replayed per token; the host steps the Manager between
  replays (manager.external is True). Capture happens after cache.reset() plus one eager step, so a linear-attention
  model records its recurrent branch (cache_utils 5.19 :996-1003 reset() clears has_previous_state; qwen3_5_moe picks
  its branch on it). The entry defaults to the model's own (resolve_entry: arch.arch_for(model.config)), so the
  refusal holds for the v0.1 signatures too. graph_ok(entry) refuses the eager_only architectures (gpt_oss, gemma4_text: sliding-window
  layers branch on a Python int; glm5_next_text: data-dependent indexer) and, below transformers 5.19, glm_moe_dsa
  and deepseek_v32 (GRAPH_MIN_TF); the default mode is graph only for qwen3_moe (arch.ArchEntry.decode == "graph");
  the others need an eager == graph test first (first_run R8).

TeacherForced has v0.1 eval_engine's KL / top-1 arithmetic, operation for operation, and GreedyDecoder in graph mode
is v0.1 moe_tiering.runner.GraphDecoder's decode plus explicit positions. Since the WP9 switch-over (design-core B6) they
are what eval_engine.py, bench_graph.py and moe_tiering.runner.GraphDecoder run (output JSON keys unchanged).
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from . import arch, hfcompat, tunecache

# Graph decode needs a StaticCache, and transformers' own StaticCache decode of these architectures is wrong below
# 5.19: the 5.15 DSA indexer adds the static cache's mask instead of masking with it (5.19 added a bool branch);
# plain transformers, StaticCache vs DynamicCache decode: rel 1.079 / 1.094 on 5.15, 0.003 / 0.0 on 5.19
# (results/coregen/verify_wp4/adv/diag_dsa_static.*, WP4 verification minor issue 1).
GRAPH_MIN_TF = {"glm_moe_dsa": "5.19.0", "deepseek_v32": "5.19.0"}


def graph_refusal(entry: arch.ArchEntry) -> str | None:
    """Why this architecture cannot be graph-captured under this transformers; None when it can in principle."""
    if entry.decode == "eager_only":
        return entry.note or entry.decode
    need = GRAPH_MIN_TF.get(entry.model_type)
    if need is not None and not hfcompat.tf_at_least(need):
        return (f"transformers {hfcompat.TF_VERSION}'s StaticCache decode of {entry.model_type} is wrong below {need} "
                f"(pip install 'transformers>={need}')")
    return None


def graph_ok(entry: arch.ArchEntry) -> bool:
    """Graph capture is possible in principle (not eager_only, and not refused for this transformers version)."""
    return graph_refusal(entry) is None


def default_mode(entry: arch.ArchEntry) -> str:
    return "graph" if arch.graph_capable(entry) else "eager"


def check_mode(mode: str, entry: arch.ArchEntry | None) -> str:
    if mode == "auto":
        return default_mode(entry) if entry is not None else "graph"
    if mode not in ("eager", "graph"):
        raise ValueError(f"decode mode {mode!r}: expected eager, graph or auto")
    if mode == "graph" and entry is not None and not graph_ok(entry):
        raise ValueError(f"{entry.model_type} cannot be graph-captured ({graph_refusal(entry)}); use eager")
    return mode


def resolve_entry(model, entry: arch.ArchEntry | None) -> arch.ArchEntry | None:
    """entry, or when the caller passed none (the v0.1 signatures GraphDecoder(model, manager, max_len) and
    eval_engine.TeacherForced(model, manager, max_len)) the model's own ArchEntry, so the graph refusal and the "auto"
    mode never depend on the caller. None only for a model without a config or of a type outside arch.ARCHES (no
    engine architecture: nothing to refuse, v0.1 behaviour)."""
    if entry is not None:
        return entry
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    try:
        return arch.arch_for(cfg)
    except ValueError:
        return None


def model_device(model, info: dict | None = None) -> torch.device:
    if info is not None and info.get("device") is not None:
        return torch.device(info["device"])
    return next(p.device for p in model.parameters() if p.device.type != "meta")


def new_cache(model):
    """The cache transformers would create for this model in forward (DynamicCache(config=self.config))."""
    from transformers import DynamicCache

    return DynamicCache(config=model.config)


class TeacherForced:
    """Single-token forwards with externally supplied input ids (no argmax feedback): KL / top-1 per window against
    make_refs-format references (top-k log-probs + tail mass)."""

    def __init__(self, model, manager, max_len: int, mode: str = "graph", device=None, entry=None):
        self.entry = resolve_entry(model, entry)
        self.mode = check_mode(mode, self.entry)
        self.model = model
        self.manager = manager
        manager.external = self.mode == "graph"
        self.dev = torch.device(device) if device is not None else model_device(model)
        self.ids = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.cache = None
        if self.mode == "graph":
            from transformers import StaticCache

            self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.graph = None
        self.logits = None

    @torch.no_grad()
    def _forward(self, cache):
        return self.model(input_ids=self.ids, position_ids=self.pos, past_key_values=cache,
                          use_cache=True).logits[:, -1]

    @torch.no_grad()
    def capture(self, warm_tokens: torch.Tensor) -> None:
        """Run 4 eager steps (kernel autotuning). Graph mode then resets the cache, runs one eager step and captures
        one decode step; eager mode only saves the tuned kernel choices."""
        cache = self.cache if self.mode == "graph" else new_cache(self.model)
        if self.mode == "graph":
            cache.reset()
        for i, t in enumerate(warm_tokens[:4]):
            self.ids.fill_(int(t))
            self.pos.fill_(i)
            self._forward(cache)
        torch.cuda.synchronize(self.dev)
        if self.mode == "graph":
            cache.reset()
            self.ids.fill_(int(warm_tokens[0]))
            self.pos.fill_(0)
            self._forward(cache)  # recorded branch = a step with a previous state (linear attention)
            torch.cuda.synchronize(self.dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.logits = self._forward(cache)
            self.graph = g
        tunecache.save()

    @torch.no_grad()
    def step(self, token: torch.Tensor, t: int, cache=None) -> torch.Tensor:
        """Logits [V] of position t given input token (a 0-dim/1-element device tensor)."""
        self.ids.copy_(token.view(1, 1))
        self.pos.fill_(t)
        if self.mode == "graph":
            self.manager.step()
            self.graph.replay()
            return self.logits[0]
        return self._forward(cache)[0]

    @torch.no_grad()
    def window(self, tokens: torch.Tensor, ref: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Returns (sum of KL, number of top-1 agreements) as GPU scalars, and the number of positions
        (eval_engine.TeacherForced.window's arithmetic)."""
        cache = None
        if self.mode == "graph":
            self.cache.reset()
        else:
            cache = new_cache(self.model)
        toks = tokens.to(self.dev)
        p_v = ref["top_v"].to(self.dev).float()
        p_i = ref["top_i"].to(self.dev).long()
        p_tail = ref["tail"].to(self.dev)
        kl = torch.zeros((), device=self.dev, dtype=torch.float64)
        agree = torch.zeros((), device=self.dev, dtype=torch.float64)
        n = toks.numel() - 1
        for t in range(n):
            logits = self.step(toks[t], t, cache)
            logp = F.log_softmax(logits.float(), -1)
            q_v = logp[p_i[t]]
            q_tail = torch.log((1 - q_v.exp().sum()).clamp(min=1e-12))
            kl += (p_v[t].exp() * (p_v[t] - q_v)).sum() + p_tail[t].exp() * (p_tail[t] - q_tail)
            agree += (logp.argmax() == p_i[t, 0]).double()
        return kl, agree, n


class GreedyDecoder:
    """Greedy generation: eager prefill, then per-token decode (graph: one captured step that also advances the
    position; eager: one forward per token on a DynamicCache)."""

    def __init__(self, model, manager, max_len: int, mode: str = "graph", device=None, entry=None):
        self.entry = resolve_entry(model, entry)
        self.mode = check_mode(mode, self.entry)
        self.model = model
        self.manager = manager
        manager.external = self.mode == "graph"
        self.dev = torch.device(device) if device is not None else model_device(model)
        self.max_len = max_len
        self.cache = None
        if self.mode == "graph":
            from transformers import StaticCache

            self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.ids = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.next_id = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.pos = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.replays = 0

    @torch.no_grad()
    def _forward(self, input_ids, position_ids):
        return self.model(input_ids=input_ids, position_ids=position_ids, past_key_values=self.cache, use_cache=True,
                          logits_to_keep=1).logits[:, -1]

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "graph":
            self.cache.reset()
        else:
            self.cache = new_cache(self.model)
        n = input_ids.shape[1]
        pos = torch.arange(n, device=self.dev).unsqueeze(0)
        logits = self._forward(input_ids, pos)
        self.next_id.copy_(logits.argmax(-1, keepdim=True))
        self.pos.fill_(n)
        return self.next_id

    @torch.no_grad()
    def _decode_step(self) -> None:
        self.ids.copy_(self.next_id)
        logits = self._forward(self.ids, self.pos)
        self.next_id.copy_(logits.argmax(-1, keepdim=True))
        self.pos.add_(1)

    @torch.no_grad()
    def warmup_and_capture(self, input_ids: torch.Tensor, warm_steps: int = 3) -> None:
        """A few eager steps (kernel autotuning); graph mode then re-prefills and captures one decode step
        (moe_tiering.runner.GraphDecoder.warmup_and_capture)."""
        self.prefill(input_ids)
        for _ in range(warm_steps):
            self._decode_step()
        torch.cuda.synchronize(self.dev)
        if self.mode == "graph":
            self.prefill(input_ids)
            torch.cuda.synchronize(self.dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._decode_step()
            self.graph = g
        tunecache.save()

    @torch.no_grad()
    def generate(self, n: int) -> tuple[torch.Tensor, float]:
        """Greedy-decode n tokens after the last prefill. Returns (token ids [n], seconds)."""
        out = torch.empty(n, dtype=torch.long, device=self.dev)
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        for i in range(n):
            out[i] = self.next_id[0, 0]
            if self.mode == "graph":
                self.manager.step()
                self.graph.replay()
            else:
                self._decode_step()
        torch.cuda.synchronize(self.dev)
        dt = time.perf_counter() - t0
        if self.mode == "graph":
            self.replays += n
            for m in arch.tiered(self.model):  # graph replays do not run the Python counters
                m.pair_count += n * m.top_k
        return out, dt
