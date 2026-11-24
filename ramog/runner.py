"""Graph-captured greedy decoding for a tiered engine model.

The decode step (embedding -> decoder layers of attention + MoE -> logits -> argmax) has static shapes and no host
synchronization, so it is captured once as a CUDA/HIP graph and replayed per token. Cache maintenance
(Manager.step) runs on the host between replays, outside the graph. The KV cache is a transformers StaticCache,
whose write position lives in a device tensor, so replays advance it without host involvement.

Since the WP9 switch-over (design-core B6) GraphDecoder is ramog.session.GreedyDecoder in graph mode: explicit
device-side position ids (advanced inside the captured step), graph capture refused for the eager_only
architectures (session.check_mode). The v0.1 interface is kept: GraphDecoder(model, manager, max_len), prefill,
warmup_and_capture, generate, and the attributes cache, ids, next_id, graph, replays, dev and moes (the TieredMoE
cores, ramog.arch.tiered: hook or legacy block path).
"""

from __future__ import annotations

import torch

from . import arch
from .session import GreedyDecoder


class GraphDecoder(GreedyDecoder):
    def __init__(self, model, manager, max_len: int, *, device=None, entry=None):
        """device: the engine's device (build info["device"]; default: that of the model's first parameter);
        entry: the ramog.arch.ArchEntry (build info["entry"]), for the graph refusal; default: the model's own
        (session.resolve_entry), so GraphDecoder(model, manager, max_len) refuses the eager_only architectures too."""
        super().__init__(model, manager, max_len, mode="graph", device=device, entry=entry)
        self.moes = arch.tiered(model)

    @torch.no_grad()
    def _forward(self, input_ids, position_ids=None):
        """One forward on the decoder's cache (last position's logits). position_ids None lets the model derive them
        from the cache, as the v0.1 GraphDecoder did (diagnostic scripts call _forward(ids) between replays)."""
        kw = {} if position_ids is None else {"position_ids": position_ids}
        return self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1,
                          **kw).logits[:, -1]
