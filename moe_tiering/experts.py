"""TieredExperts: the module that replaces one transformers experts module (design-core A1/B1; work package WP4).

transformers keeps all model semantics: the MoE block (or, for Gemma-4, the decoder layer) still runs its own router,
shared experts, routed scaling, per-expert scale and output reshapes, and calls

    self.experts(hidden_states [T, H], top_k_index [T, k], top_k_weights [T, k])

positionally (verified for every target architecture, design-core verified fact 1). Only that experts module is
replaced, by a TieredExperts that runs the engine's TieredMoE core on the routing it is given:
- T <= DECODE_MAX_TOKENS: core._decode (grouped GEMV over the VRAM cache, the pinned-host cold blobs and the
  prefetch staging), after the shared Manager's step when this is the first MoE layer and no graph runner drives
  the manager (manager.external);
- otherwise core._prefill (grouped dequant-GEMM).
That is exactly what TieredMoE.forward does after its own router call (moe.py :177-187), so for Qwen3-MoE the new
hook and the legacy block are the same computation (tests/test_experts_equiv.py checks it bit for bit).

The core is a TieredMoE whose router (the transformers router module, still owned by its block) and activation are
held unregistered: state_dict() and named_parameters() gain no keys. The original experts module `ref` stays on the
meta device, unregistered (object.__setattr__) and with its parameters deleted; its class carries the math facts
(`_apply_gate`, `act_fn`, the decorator's layout flags), and materialize() rebuilds it with given weights for checks.

Phase A ran only experts whose `_apply_gate` is the decorator's `_default_apply_gate` and that have no biases. WP8
(design-core B1, Phase B) lifts that restriction through TieredMoE's hooks (moe_tiering/moe.py):
- default gate (qwen3_moe, qwen3_5_moe, gemma4 with act_fn GELU-tanh, glm4_moe_lite, glm_moe_dsa, deepseek_v32,
  minimax_m2): core.act = ref.act_fn and core.glu = None, the unchanged Qwen3 code path;
- a class with its own `_apply_gate` (gpt-oss: interleaved gate/up rows, clamp at limit, alpha, (up + 1);
  GLM-5.3-Flash: clamp-10 SwiGLU): core.glu = ref._apply_gate, run on the float32 GEMV / GEMM output. Blob rows are
  in transformers' output order, so gpt-oss's interleave needs no permutation anywhere (design-core A2);
- expert biases (gpt-oss): core.gu_bias / core.dn_bias, float32 [E, 2I] / [E, H] tables on the GPU, added per pair
  before the gate and before the routing weight (modeling_gpt_oss 5.19 :107-110).
core_kwargs(ref, biases, device) gives those TieredMoE keyword arguments; the constructor checks that the core uses
exactly them. Non-gated experts (up_proj only) are still refused: no target architecture has them.

Trellis blobs (design-quality WP15): rotation_kwargs(layout, device) gives a trellis layer's core its input rotation,
rot = tcq_rotation(layout, device) = (signs_gu, b_gu, signs_dn, b_dn) from moe_tiering.tcq.layer_rotation (seed
layer * 1000; moe_tiering.store_v2.check has verified the store's sign hashes against the same function), the signs'
upload to the GPU checked (amendment 6); {} for every other layout (today's call, unchanged).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from . import hfcompat
from .moe import DECODE_MAX_TOKENS, TieredMoE


def hold_unregistered(module: nn.Module, name: str) -> None:
    """Keep `module.<name>` as a plain attribute: no longer a registered submodule, parameter or buffer (so it adds no
    state-dict keys and is not moved by module.to()), still reachable as module.<name>."""
    for table in (module._modules, module._parameters, module._buffers):
        if name in table:
            value = table.pop(name)
            object.__setattr__(module, name, value)
            return


def phase_a_reason(ref: nn.Module) -> str | None:
    """Why Phase A's TieredMoE (default gate, no biases) could not run this experts module; None when it could. WP8
    lifted the restriction: the engine checks unsupported_reason() now; this stays for Phase-A records and tools."""
    if not hfcompat.is_experts_module(ref):
        return f"{type(ref).__name__} is not a @use_experts_implementation experts module"
    if type(ref)._apply_gate is not hfcompat.default_apply_gate:
        return f"{type(ref).__name__} has its own _apply_gate (needs TieredMoE.glu, WP8)"
    if ref.has_bias:
        return f"{type(ref).__name__} has expert biases (needs TieredMoE gu_bias/dn_bias, WP8)"
    if not ref.has_gate or not ref.is_concatenated or ref.is_transposed:
        return (f"{type(ref).__name__}: gated={ref.has_gate} concatenated={ref.is_concatenated} "
                f"transposed={ref.is_transposed}; TieredMoE expects gated, gate rows then up rows, [out, in]")
    if getattr(ref, "act_fn", None) is None:
        return f"{type(ref).__name__} has no act_fn"
    return None


def unsupported_reason(ref: nn.Module) -> str | None:
    """Why TieredMoE, with WP8's glu / gu_bias / dn_bias hooks, cannot run this experts module; None when it can."""
    if not hfcompat.is_experts_module(ref):
        return f"{type(ref).__name__} is not a @use_experts_implementation experts module"
    if not ref.has_gate:
        return f"{type(ref).__name__} has no gate (up_proj only); TieredMoE serves gate_up experts"
    if hfcompat.apply_gate_kind(ref) == "default":
        if not ref.is_concatenated:
            return (f"{type(ref).__name__}: the default gate with interleaved gate/up rows (is_concatenated=False) is "
                    "not a layout any target architecture has")
        if getattr(ref, "act_fn", None) is None:
            return f"{type(ref).__name__} has no act_fn"
    return None


def core_kwargs(ref: nn.Module, biases=None, device=None) -> dict:
    """The TieredMoE keyword arguments that give it the replaced module's expert math (module docstring): act / glu,
    and for a module with expert biases gu_bias / dn_bias made from biases = (gate_up_proj_bias [E, 2I],
    down_proj_bias [E, H]) in transformers' layout and row order, as float32 tables on `device`."""
    why = unsupported_reason(ref)
    if why is not None:
        raise NotImplementedError(f"TieredExperts: {why}")
    if hfcompat.apply_gate_kind(ref) == "default":
        kw = {"act": ref.act_fn, "glu": None}  # the Qwen3 code path, unchanged
    else:
        kw = {"act": None, "glu": ref._apply_gate}
    if ref.has_bias:
        if biases is None:
            raise ValueError(f"{type(ref).__name__} has expert biases: pass (gate_up_proj_bias, down_proj_bias)")
        shapes = {n: tuple(p.shape) for n, p in ref.named_parameters(recurse=False)}
        for name, t in zip(("gate_up_proj_bias", "down_proj_bias"), biases, strict=True):
            if name in shapes and tuple(t.shape) != shapes[name]:
                raise ValueError(f"{name}: got {tuple(t.shape)}, the experts module has {shapes[name]}")
        kw["gu_bias"] = biases[0].to(device=device, dtype=torch.float32).contiguous()
        kw["dn_bias"] = biases[1].to(device=device, dtype=torch.float32).contiguous()
    elif biases is not None:
        raise ValueError(f"{type(ref).__name__} has no expert biases, got some")
    return kw


def tcq_rotation(layout, device) -> tuple:
    """(signs_gu float32 [H], b_gu, signs_dn float32 [I], b_dn) of a trellis layer layout (moe_tiering.store_v2.
    TcqStoreLayout: its .layer picks the seed), on `device`; each upload compared with the CPU copy."""
    from . import tcq

    out = []
    for name, n in (("gate_up", layout.gu_shape[1]), ("down", layout.dn_shape[1])):
        signs, _, b = tcq.layer_rotation(n, layout.layer, "cpu")
        dev_signs = signs.to(device)
        if not torch.equal(dev_signs.cpu(), signs):
            raise RuntimeError(f"layer {layout.layer} {name}: rotation signs differ after the upload to {device}")
        out += [dev_signs, int(b)]
    return tuple(out)


def rotation_kwargs(layout, device) -> dict:
    """{"rot": tcq_rotation(...)} for a trellis layout (moe_tiering.tcq.TcqLayout), {} otherwise."""
    from . import tcq

    return {"rot": tcq_rotation(layout, device)} if isinstance(layout, tcq.TcqLayout) else {}


class TieredExperts(nn.Module):
    """Drop-in for a transformers experts module, backed by a TieredMoE core (see the module docstring)."""

    def __init__(self, core: TieredMoE, ref: nn.Module, layer_idx: int | None = None):
        super().__init__()
        why = unsupported_reason(ref)
        if why is not None:
            raise NotImplementedError(f"TieredExperts: {why}")
        if hfcompat.apply_gate_kind(ref) == "default":
            if core.act is not ref.act_fn or core.glu is not None:
                raise ValueError("default gate: core.act must be the replaced module's own act_fn and core.glu None "
                                 "(design-core B1)")
        elif (getattr(core.glu, "__func__", None) is not type(ref)._apply_gate
              or getattr(core.glu, "__self__", None) is not ref):
            raise ValueError("core.glu must be the replaced module's own _apply_gate, bound to it (design-core B1)")
        if (core.gu_bias is not None) != bool(ref.has_bias) or (core.dn_bias is not None) != bool(ref.has_bias):
            raise ValueError(f"{type(ref).__name__} has_bias={ref.has_bias}, the core has gu_bias "
                             f"{core.gu_bias is not None} and dn_bias {core.dn_bias is not None} (core_kwargs)")
        hold_unregistered(core, "router")  # the block still owns and calls its router; no duplicate state keys
        hold_unregistered(core, "act")
        hold_unregistered(core, "glu")
        self.core = core  # registered for printing and tiered(); it has no parameters or buffers
        self.layer_idx = layer_idx
        self.ref_params = {n: (tuple(p.shape), p.dtype) for n, p in ref.named_parameters(recurse=False)}
        for n in list(ref._parameters):  # the meta weights are never used; materialize() rebuilds them
            del ref._parameters[n]
        object.__setattr__(self, "ref", ref)  # unregistered: no state, not moved by .to()

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        core = self.core
        x = hidden_states
        if x.dim() != 2:
            raise ValueError(f"TieredExperts expects 2-D hidden states [T, H], got {tuple(x.shape)}")
        if x.shape[0] <= DECODE_MAX_TOKENS:
            if core.first and not core.manager.external:
                core.manager.step()
            return core._decode(x, top_k_weights, top_k_index)
        return core._prefill(x, top_k_weights, top_k_index)

    def materialize(self, gate_up: torch.Tensor, down: torch.Tensor, gate_up_bias: torch.Tensor | None = None,
                    down_bias: torch.Tensor | None = None) -> nn.Module:
        """A copy of the original transformers experts module holding the given weights (gate_up [E, 2I, H], down
        [E, H, I], nn.Linear layout as the store holds them, rows in transformers' order; turned into the module's
        own layout, [E, in, out] for gpt-oss), e.g. the store's blobs dequantized to bf16, for one-layer checks
        (first_run layercheck). A module with expert biases also gets gate_up_bias [E, 2I] / down_bias [E, H]
        (default: the core's tables), in gate_up's dtype. Runs transformers' own experts forward."""
        m = copy.deepcopy(self.ref)
        tr = bool(self.ref.is_transposed)
        tensors = {"gate_up_proj": gate_up.transpose(1, 2) if tr else gate_up,
                   "down_proj": down.transpose(1, 2) if tr else down}
        if self.ref.has_bias:
            gub = self.core.gu_bias if gate_up_bias is None else gate_up_bias
            dnb = self.core.dn_bias if down_bias is None else down_bias
            tensors["gate_up_proj_bias"] = gub.to(device=gate_up.device, dtype=gate_up.dtype)
            tensors["down_proj_bias"] = dnb.to(device=gate_up.device, dtype=gate_up.dtype)
        elif gate_up_bias is not None or down_bias is not None:
            raise ValueError(f"{type(self.ref).__name__} has no expert biases")
        for name, t in tensors.items():
            want, _ = self.ref_params[name]
            if tuple(t.shape) != want:
                raise ValueError(f"{name}: got {tuple(t.shape)}, the experts module has {want}")
            setattr(m, name, nn.Parameter(t.contiguous(), requires_grad=False))
        return m

    def extra_repr(self) -> str:
        c = self.core
        ex = c.experts
        def name(la):
            return f"{la.format}{la.bits}" if la.bits is not None else f"{la.format}-{la.describe()['gu_pattern']}"

        return (f"layer={self.layer_idx}, experts={ex.num_experts}, top_k={c.top_k}, ref={type(self.ref).__name__}, "
                f"hot={name(ex.hot_layout)}, cold={name(ex.cold_layout)}, slots={c.cache.slots}")
