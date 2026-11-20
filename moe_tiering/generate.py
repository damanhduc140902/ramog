"""Text generation for a tiered engine model: sampling, stop tokens and streaming (the CLI / API path).

    gen = Generator(model, info, max_len)          info: moe_tiering.loader.build_engine's (manager, device, entry)
    ids, finish, seconds = gen.generate(prompt_ids, max_new_tokens, temperature=0.0, top_p=1.0, top_k=0,
                                        stop_ids=(), seed=None, on_token=None)

Generator is a session.GreedyDecoder whose decode step writes the last logits into a static buffer instead of taking
their argmax; the next token is picked outside the step (and outside the captured graph in graph mode) and copied into
the step's input. Picking on the host costs one device sync per token, which streaming needs anyway. Greedy
(temperature 0) gives GreedyDecoder.generate's tokens (tests/test_generate.py).
- finish is "stop" (a stop id was produced; it is included in ids, as an OpenAI-style client expects the text before
  it), "length" (max_new_tokens reached) or "cancel" (on_token returned False).
- Every call prefills the whole prompt again: a chat turn resends the conversation (no prefix reuse yet).
- The graph (graph mode) is captured on the first call, on that call's prompt; later calls replay it.
"""

from __future__ import annotations

import time

import torch

from . import arch, session


def pick(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
         generator: torch.Generator | None = None) -> int:
    """The next token from one row of logits: argmax at temperature <= 0, else a sample after top-k and top-p."""
    x = logits.float()
    if temperature <= 0:
        return int(x.argmax())
    x = x / temperature
    if top_k and 0 < top_k < x.numel():
        kth = torch.topk(x, top_k).values[-1]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p < 1.0:
        srt, idx = torch.sort(x, descending=True)
        p = torch.softmax(srt, dim=-1)
        drop = torch.cumsum(p, dim=-1) - p > top_p  # keep the smallest prefix whose mass reaches top_p
        x = x.scatter(0, idx, srt.masked_fill(drop, float("-inf")))
    probs = torch.softmax(x, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


class Generator(session.GreedyDecoder):
    """One decode session (cache, captured step) with sampling and stop handling; see the module docstring."""

    def __init__(self, model, info: dict, max_len: int, mode: str | None = None):
        entry = info.get("entry")
        if mode is None:
            mode = session.default_mode(entry) if entry is not None else "eager"
        super().__init__(model, info["manager"], max_len, mode=mode, device=info["device"], entry=entry)
        self.logits: torch.Tensor | None = None  # static [1, vocab] buffer, written by prefill and every decode step

    def _keep(self, logits: torch.Tensor) -> None:
        if self.logits is None or self.logits.shape != logits.shape or self.logits.dtype != logits.dtype:
            if self.graph is not None:
                raise RuntimeError("the logits buffer changed after the graph was captured")
            self.logits = torch.empty_like(logits)
        self.logits.copy_(logits)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "graph":
            self.cache.reset()
        else:
            self.cache = session.new_cache(self.model)
        n = input_ids.shape[1]
        pos = torch.arange(n, device=self.dev).unsqueeze(0)
        self._keep(self._forward(input_ids, pos))
        self.pos.fill_(n)
        return self.logits

    @torch.no_grad()
    def _decode_step(self) -> None:
        self.logits.copy_(self._forward(self.ids, self.pos))
        self.pos.add_(1)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int, *, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 0, stop_ids=(), seed: int | None = None, on_token=None):
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=self.dev).reshape(1, -1)
        n = ids.shape[1]
        if n == 0:
            raise ValueError("empty prompt")
        if n + max_new_tokens > self.max_len:
            raise ValueError(f"prompt {n} + {max_new_tokens} new tokens > context {self.max_len}")
        if self.mode == "graph" and self.graph is None:
            self.warmup_and_capture(ids)
        self.prefill(ids)
        gen = None
        if seed is not None and temperature > 0:
            gen = torch.Generator(device=self.dev)
            gen.manual_seed(seed)
        stop = set(int(s) for s in stop_ids)
        out, finish, steps = [], "length", 0
        torch.cuda.synchronize(self.dev)
        t0 = time.perf_counter()
        for i in range(max_new_tokens):
            tok = pick(self.logits[0], temperature, top_p, top_k, gen)
            out.append(tok)
            if on_token is not None and on_token(tok) is False:
                finish = "cancel"
                break
            if tok in stop:
                finish = "stop"
                break
            if i == max_new_tokens - 1:
                break
            self.ids.fill_(tok)
            if self.mode == "graph":
                self.manager.step()
                self.graph.replay()
            else:
                self._decode_step()
            steps += 1
        torch.cuda.synchronize(self.dev)
        dt = time.perf_counter() - t0
        if self.mode == "graph":
            self.replays += steps
            for m in arch.tiered(self.model):  # graph replays do not run the Python counters
                m.pair_count += steps * m.top_k
        return out, finish, dt
