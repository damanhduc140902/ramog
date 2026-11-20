"""Architecture table and model introspection for the generic engine (design-core B2; work package WP2).

The engine replaces only each MoE layer's `experts` module. Everything else (routers, shared experts, routed scaling,
attention, mHC, ...) stays transformers code. This module answers the questions the loader, converter and runtime
ask about a model, from the built transformers model itself rather than from config keys:
- ARCHES / arch_for(cfg): the supported text model types; anything else is refused;
- build_meta(cfg): the text-only model on the meta device (no memory, no weights), routers with a deterministic
  top-k order where the architecture needs it (sort_router_topk);
- find_moe_sites(model): every experts module, its router and its layout, with three checks;
- make_probe / pred_k: next-layer expert prediction for prefetch, through the next layer's own router;
- dense_quant_plan: which dense nn.Linear modules the loader quantizes to 4 bits;
- SkipRules / apply_prefix_rules: how checkpoint keys outside the text model are recognized;
- tiered(model): the TieredMoE cores of a built engine, in layer order.
Nothing here imports the GPU kernels: it runs on any CPU with torch and transformers (moe_tiering.hfcompat pins those).
"""

from __future__ import annotations

import contextlib
import re
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hfcompat
from . import models as _ported  # registers the engine's own ports (moe_tiering/models) with transformers' Auto classes
from .textlm import TextLM

DENSE_GROUP = 64  # group size of the dense 4-bit QuantLinear (moe_tiering.qlinear); in_features must be a multiple
DENSE_MIN_OUT = 64  # smaller outputs (per-head gates, 1-wide shared-expert gates) stay bf16


@dataclass(frozen=True)
class Probe:
    """Next-layer expert prediction for prefetch (design-core A3).

    call:
      "linear"      topk(F.linear(x, router.weight[, router.bias]).float(), k): today's TieredMoE predictor (moe.py
                    :158-160), kept for Qwen3-MoE (same ranking as its softmax router; zero numeric risk);
      "router"      router.forward(x)[2]: the router's own selection; .forward bypasses forward hooks (output
                    recorders), and covers sigmoid + correction bias + group top-k, per_expert_scale, ...;
      "router_bias" router.forward(x, block.e_score_correction_bias)[2]: MiniMax-M2 keeps the bias on the MoE block
                    and passes it as the router's second argument (modeling_minimax_m2 :113, :120).
    k_from: where the number of routed experts per token lives: "router.top_k", or "config.top_k_experts" for
      Gemma-4, whose router has no top_k attribute (modeling_gemma4 :1333).
    """

    call: str
    k_from: str = "router.top_k"


@dataclass(frozen=True)
class ArchEntry:
    """One supported text architecture.

    build         "causal_lm": AutoModelForCausalLM.from_config(text config); "textlm": TextLM(AutoModel.from_config)
                  for models without a CausalLM class (glm5_next).
    prefix_rules  (old, new) checkpoint-key prefix renames applied before transformers' own conversion mapping, for
                  VLM checkpoints whose mapping does not strip `model.language_model.` itself (gemma4, glm5_next;
                  qwen3_5_moe_text's mapping has its own PrefixChange, conversion_mapping 5.19 :1054, :1903).
    probe         see Probe.
    deny_quant    regexes (re.search on the module name) of dense nn.Linear modules this architecture keeps in bf16,
                  on top of the generic rules of dense_quant_plan.
    decode        "graph": CUDA-graph decode (Qwen3-MoE, as today); "eager": eager in v0.1, graph capture is a later
                  step behind an eager==graph test; "eager_only": graph capture refused (sliding-window layers branch
                  on a Python int, cache_utils 5.19 :535-636; glm5_next's indexer uses data-dependent boolean
                  indexing, modeling_glm5_next :1007-1009).
    min_tf        oldest transformers that has the architecture (new models run on hfcompat.PINNED regardless).
    fp32_keys     model-key patterns (transformers' dtype-plan syntax: a regex body, `*` -> `.*`, re.search) that load
                  as float32 on top of the class's own strict-fp32 list (hfcompat.DtypePlan adds them for bf16 / fp16
                  loads, so KeyMap targets, the streamer's references, the converter and moe_tiering.loader all follow).
                  For tensors the release stores in F32 and computes in their own dtype, where transformers' plain
                  bf16 load would round them (MiniMax-M2's router weight and correction bias: modeling_minimax_m2
                  5.19 :55 runs F.linear in the weight's dtype; GLM-5.3-Flash's mHC base / scale). transformers' own
                  pre-quantized load of those releases keeps them F32 (core_model_loading 5.19 :1680-1692).
    sorted_topk   the routers take torch.topk(..., sorted=False) (glm4_moe_lite, glm_moe_dsa, deepseek_v32,
                  minimax_m2, glm5_next); build_meta makes them run it as a stable descending sort (topk_stable via
                  sort_router_topk), so the selected set and the slot order the experts sum in are a function of
                  the scores: ties go by expert index. On MI250 torch.topk over rows wider than 256 (GLM-5.3-Flash's
                  288 experts) returns another slot order on every call with sorted=False, and with sorted=True
                  still another set when the k-th score is tied (exact fp32 ties occur at full depth: WP6 fix
                  router_diag). Without ties the result is bitwise topk(sorted=True); up to 256 columns that is also
                  what sorted=False returns there (topk_probe), so other releases are unchanged.
    host_keys     model-key regexes (re.search) of tensors that stay in host memory in their release dtype: moe_tiering.ckpt
                  maps them as "host_raw" targets (the stored bytes, never decoded or cast), the streamer binds them
                  once as zero-copy memory maps of the checkpoint, and they are never read ahead or put on the device
                  (DeepSeek-V4.1's engram n-gram tables: 2 x 101 GB of FP8 rows + E8M0 scales, a sparse lookup).
    skip_keys     ((rule, regex), ...) on checkpoint keys: extra named skip rules of this release (moe_tiering.arch.SkipRules
                  tries them after the generic tower rule), e.g. a vision tower under names outside TOWERS.
    """

    model_type: str
    build: str
    prefix_rules: tuple[tuple[str, str], ...]
    probe: Probe
    deny_quant: tuple[str, ...]
    decode: str
    min_tf: str
    note: str = ""
    fp32_keys: tuple[str, ...] = ()
    sorted_topk: bool = False
    host_keys: tuple[str, ...] = ()
    skip_keys: tuple[tuple[str, str], ...] = ()


_LM = (("model.language_model.", "model."),)

ARCHES: dict[str, ArchEntry] = {e.model_type: e for e in (
    ArchEntry("qwen3_moe", "causal_lm", (), Probe("linear"), (), "graph", "5.15.0",
              "Qwen3-30B-A3B / Qwen3-235B-A22B: the v0.1 engine's model; per-expert BF16 checkpoints"),
    ArchEntry("qwen3_5_moe_text", "causal_lm", (), Probe("router"),
              # GatedDeltaNet per-head decay (a) and beta (b) inputs: gate-like scalars that drive the recurrent
              # state; 32-wide on Qwen3.6-35B (already bf16 by DENSE_MIN_OUT), 64-wide on Qwen3.5-122B. Denied so
              # both sizes of the architecture keep the same bf16 set (computed: 28 MB of bf16 on the 122B, about
              # 20 MB more than at 4 bits).
              (r"\.linear_attn\.in_proj_a$", r"\.linear_attn\.in_proj_b$"),
              "eager", "5.15.0", "Qwen3.6-35B-A3B, Qwen3.5-122B-A10B: GatedDeltaNet + gated attention, shared expert"),
    ArchEntry("gpt_oss", "causal_lm", (), Probe("router"), (), "eager_only", "5.15.0",
              "gpt-oss-20b/120b: interleaved clamp/alpha _apply_gate and expert biases (Phase B, WP8); sliding layers"),
    ArchEntry("gemma4_text", "causal_lm", _LM, Probe("router", "config.top_k_experts"), (), "eager_only", "5.15.0",
              "Gemma-4-26B-A4B: experts on the decoder layer, router on the residual, dense MLP in every layer"),
    ArchEntry("glm4_moe_lite", "causal_lm", (), Probe("router"), (), "eager", "5.15.0",
              "GLM-4.7-Flash: MLA, sigmoid + correction-bias group top-k router, shared expert", sorted_topk=True),
    ArchEntry("glm_moe_dsa", "causal_lm", (), Probe("router"), (), "eager", "5.15.0",
              "GLM-5.3: MLA + DSA indexer (IndexShare); run on 5.19, which caches the MLA latent", sorted_topk=True),
    ArchEntry("deepseek_v32", "causal_lm", (), Probe("router"), (), "eager", "5.15.0",
              "DeepSeek-V3.2: MLA + DSA indexer; run on 5.19, which caches the MLA latent", sorted_topk=True),
    ArchEntry("minimax_m2", "causal_lm", (), Probe("router_bias"), (), "eager", "5.15.0",
              "MiniMax-M2: sigmoid router with the correction bias on the MoE block",
              # the release stores both F32 (and lists them in modules_to_not_convert); the class has no strict-fp32
              # list, so a plain bf16 load would round them. Full depth, 4 x 512 tokens: F32 vs bf16 router moves
              # the logits by KL 0.0203 (top-1 0.936) against a bit-identical run-to-run floor (verify_wp6
              # dtype_depth; WP6 fix depth_cast). Computed cost: 62 layers x 256 x 3072 x 2 B = 98 MB more than bf16.
              fp32_keys=(r"\.mlp\.gate\.weight$", r"\.mlp\.e_score_correction_bias$"), sorted_topk=True),
    ArchEntry("glm5_next_text", "textlm", _LM, Probe("router"),
              # KDA per-head beta (b_proj, 64 wide = heads) and the low-rank forget gate (decay): the same role as
              # Qwen3.5's in_proj_b / in_proj_a. The output gate (g_a/g_b) is quantized like Qwen3.5's in_proj_z.
              # Computed cost: 125 MB of bf16 over the 34 KDA layers, about 90 MB more than at 4 bits.
              (r"\.self_attn\.b_proj$", r"\.forget_gate\.f_a_proj$", r"\.forget_gate\.f_b_proj$"),
              "eager_only", "5.19.0",
              "GLM-5.3-Flash: mHC streams, KDA + indexed MLA, clamp-10 _apply_gate (Phase B, WP8); no CausalLM",
              # mHC base / scale: stored F32 (hc_attn_* / hc_ffn_*); bf16 vs F32 moves full-depth logits (4 x 512
              # tokens) by KL 0.0359 (top-1 0.906) against a bit-identical floor with the stable top-k (WP6 fix
              # depth_cast). The mixing weights (fn) are stored BF16 and stay bf16. conv1d keeps the class's strict
              # fp32 (transformers' FP8 load holds the stored BF16 instead: KL 0.0336, top-1 0.913 from F32; F32 is
              # what the fused causal_conv1d kernel computes in, and the torch fallback with an F32 weight).
              fp32_keys=(r"_hc\.base$", r"_hc\.scale$"), sorted_topk=True),
    ArchEntry("deepseek_v41_text", "causal_lm", (), Probe("router"),
              # o_a_proj is applied as a grouped einsum over its .weight (Attention.forward), never through forward;
              # the compressor's projections compute in float32 at ratio 2 (float32 weights, float32 input).
              (r"\.self_attn\.o_a_proj$", r"\.compressor\."),
              "eager_only", "5.19.0",
              "DeepSeek-V4.1-Flash: our port of the release's inference/model.py (moe_tiering/models/deepseek_v41.py): CSA2 "
              "sparse attention with a two-level indexer, mHC, engram n-gram memory, 384 FP4 experts",
              # The dtype plan is the port's meta dtypes (the reference's): float32 for the tensors the release stores
              # F32 (listed here as well, so DtypePlan reports them), the head (the reference's ParallelHead holds it
              # in float32 and returns float32 logits) and the ratio-2 compressor projections (float32 meta params,
              # per layer); bf16 for the rest; the engram tables in their release FP8 / E8M0 (host_keys).
              fp32_keys=(r"\.sinks$", r"_hc\.(fn|base|scale)$", r"\.gate\.e_score_correction_bias(_vl)?$",
                         r"^lm_head\.weight$"),
              sorted_topk=True,
              host_keys=(r"\.engram\.embed\.(weight|scale)$",),
              skip_keys=(("tower", r"^(vision|aligner)\.|^image_(start|end|newline)$"),)),
)}

GRAPH_DECODE = {"graph"}
PORTS_PREFIX = __name__.rsplit(".", 1)[0] + ".models."  # moe_tiering/models: architectures the engine ports itself


def text_config(cfg):
    """The decoder's text config (the config itself for text-only models)."""
    return cfg.get_text_config(decoder=True) if hasattr(cfg, "get_text_config") else cfg


def arch_for(cfg) -> ArchEntry:
    """The ArchEntry of a (possibly multimodal) config; unknown text model types are refused."""
    text = text_config(cfg)
    entry = ARCHES.get(text.model_type)
    if entry is None:
        raise ValueError(f"unsupported architecture: text model_type {text.model_type!r} (config model_type "
                         f"{cfg.model_type!r}); supported: {sorted(ARCHES)}")
    return entry


def graph_capable(entry: ArchEntry) -> bool:
    return entry.decode in GRAPH_DECODE


def build_meta(cfg, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """The text-only model of `cfg` on the meta device (allocates nothing, reads no weights).

    get_text_config(decoder=True), then AutoModelForCausalLM.from_config (gemma4_text gives Gemma4ForCausalLM,
    qwen3_5_moe_text gives Qwen3_5MoeForCausalLM), or TextLM(AutoModel.from_config) for "textlm" entries. Remote code
    is never used (MiniMax-M2's config carries an auto_map; the native class is built), and hub kernels must be off.
    """
    from transformers import AutoModel, AutoModelForCausalLM

    text = text_config(cfg)
    entry = arch_for(text)
    hfcompat.require_tf(entry.min_tf, f"architecture {entry.model_type}")
    with torch.device("meta"):
        if entry.build == "causal_lm":
            model = AutoModelForCausalLM.from_config(text, dtype=dtype, trust_remote_code=False)
        elif entry.build == "textlm":
            model = TextLM(AutoModel.from_config(text, dtype=dtype, trust_remote_code=False))
        else:
            raise ValueError(f"{entry.model_type}: unknown build kind {entry.build!r}")
    if getattr(model, "use_kernels", False):
        raise RuntimeError(f"{type(model).__name__} was built with hub kernels enabled; the engine needs plain modules")
    root = hfcompat.pretrained_root(model)
    is_remote = getattr(root, "is_remote_code", lambda: False)()
    if is_remote or not type(root).__module__.startswith(("transformers.", PORTS_PREFIX)):
        raise RuntimeError(f"{type(model).__name__} comes from remote code; only native transformers classes and the "
                           f"engine's own ports ({PORTS_PREFIX}*) are used")
    model.eval()
    if entry.sorted_topk:
        sort_router_topk(model)
    return model


# ---------------------------------------------------------------------------------------------------------------------
# deterministic router top-k order

_orig_topk = torch.topk
_topk_lock = threading.RLock()


def topk_stable(input, k, dim=-1, largest=True, sorted=True, *, out=None):  # noqa: A002 - torch.topk's names
    """torch.topk as a stable sort: the k largest (smallest) values in descending (ascending) order, equal values in
    index order. Without exact ties it is bitwise topk(sorted=True) (same values, same indices). With ties torch.topk
    leaves the slot order, and at the k-th value the selected set, to the kernel; on MI250 rows wider than 256 then
    give another answer on almost every call, sorted=True or not (agents/coregen/wp6/fix/router_probe.py)."""
    v, i = torch.sort(input, dim=dim, descending=largest, stable=True)
    v, i = v.narrow(dim, 0, k).contiguous(), i.narrow(dim, 0, k).contiguous()
    if out is not None:
        out[0].resize_as_(v).copy_(v)
        out[1].resize_as_(i).copy_(i)
        return torch.return_types.topk((out[0], out[1]))
    return torch.return_types.topk((v, i))


@contextlib.contextmanager
def topk_sorted():
    """Inside the block torch.topk runs as topk_stable: same selected set as transformers' sorted=False call whenever
    the k-th score is not tied, slots in descending score order, ties by expert index. Held under a lock: the swap is
    process-wide while the block runs."""
    with _topk_lock:
        prev = torch.topk
        torch.topk = topk_stable
        try:
            yield
        finally:
            torch.topk = prev


def router_modules(model: nn.Module) -> list[nn.Module]:
    """The router next to every experts module of any model (a text model, a VLM class, an engine whose experts were
    replaced keeps its routers): the one module among ROUTER_ATTRS beside `experts`."""
    out = []
    for name, parent in model.named_modules():
        ex = getattr(parent, "experts", None)
        if isinstance(ex, nn.Module) and (hfcompat.is_experts_module(ex) or hasattr(ex, "core")):
            out.append(_router_of(parent, name or type(parent).__name__)[1])
    return out


def _sorted_router_forward(self, *args, **kwargs):
    with topk_sorted():
        return type(self).forward(self, *args, **kwargs)


def sort_router_topk(model: nn.Module) -> int:
    """Make every router of `model` run its own class forward with torch.topk as topk_stable (topk_sorted), as a
    method bound on the instance (the class and other models are untouched; copy.deepcopy rebinds it to the copy).
    Idempotent; returns the number of routers patched now. build_meta applies it to the architectures whose routers
    take sorted=False (ArchEntry.sorted_topk), so the streamer, the converter's model and moe_tiering.loader's engine share
    it; a transformers oracle compared with them needs the same call."""
    n = 0
    for r in router_modules(model):
        if getattr(r, "_mt_sorted_topk", False):
            continue
        r.forward = types.MethodType(_sorted_router_forward, r)
        r._mt_sorted_topk = True
        n += 1
    return n


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"{type(model).__name__}: no decoder layers at .model.layers")
    return layers


def _module_name(model: nn.Module, target: nn.Module) -> str:
    for name, m in model.named_modules():
        if m is target:
            return name
    raise LookupError(f"{type(target).__name__} is not a submodule of {type(model).__name__}")


# ---------------------------------------------------------------------------------------------------------------------
# experts layout


@dataclass(frozen=True)
class ExpertsLayout:
    """Shape and math facts of one experts module, from the decorator attributes and the parameter shapes.

    E, H, I: experts, hidden size, expert intermediate size. gated: gate_up_proj (else up_proj); bias: per-expert
    biases (gate_up_proj_bias/down_proj_bias); transposed: weights stored [E, in, out] (gpt-oss) instead of
    [E, out, in]; concatenated: gate rows then up rows (else interleaved, gpt-oss). gate: "default" when the class
    uses the decorator's _default_apply_gate, else "custom". act: name of act_fn (a function's name or a module's
    class name; None when the module has none: gpt-oss and GLM-5.3-Flash compute it inside their own _apply_gate).
    gu_shape/dn_shape are the [out, in] (nn.Linear) shapes a blob stores, rows in transformers' output order.
    """

    E: int
    H: int
    I: int  # noqa: E741
    gated: bool
    bias: bool
    transposed: bool
    concatenated: bool
    gate: str
    act: str | None

    @property
    def gu_shape(self) -> tuple[int, int]:
        return (2 * self.I if self.gated else self.I, self.H)

    @property
    def dn_shape(self) -> tuple[int, int]:
        return (self.H, self.I)

    @classmethod
    def from_module(cls, m: nn.Module) -> ExpertsLayout:
        if not hfcompat.is_experts_module(m):
            raise TypeError(f"{type(m).__name__} is not a @use_experts_implementation experts module")
        gated, bias = bool(m.has_gate), bool(m.has_bias)
        transposed, concatenated = bool(m.is_transposed), bool(m.is_concatenated)
        wname = "gate_up_proj" if gated else "up_proj"
        gu, dn = getattr(m, wname), m.down_proj
        if gu.dim() != 3 or dn.dim() != 3:
            raise ValueError(f"{type(m).__name__}: expected 3-D {wname}/down_proj, got {tuple(gu.shape)}/"
                             f"{tuple(dn.shape)}")
        E = gu.shape[0]
        if transposed:
            H, n_gu = gu.shape[1], gu.shape[2]
            I_dn, H_dn = dn.shape[1], dn.shape[2]
        else:
            n_gu, H = gu.shape[1], gu.shape[2]
            H_dn, I_dn = dn.shape[1], dn.shape[2]
        if gated and n_gu % 2:
            raise ValueError(f"{type(m).__name__}: odd gate_up width {n_gu}")
        I = n_gu // 2 if gated else n_gu  # noqa: E741
        if dn.shape[0] != E or H_dn != H or I_dn != I:
            raise ValueError(f"{type(m).__name__}: {wname} {tuple(gu.shape)} and down_proj {tuple(dn.shape)} "
                             f"disagree (transposed={transposed})")
        if bias:
            gb, db = getattr(m, f"{wname}_bias"), m.down_proj_bias
            if tuple(gb.shape) != (E, n_gu) or tuple(db.shape) != (E, H):
                raise ValueError(f"{type(m).__name__}: bias shapes {tuple(gb.shape)}, {tuple(db.shape)}")
        act = getattr(m, "act_fn", None)  # a module (e.g. GELUTanh) or a plain function (F.silu)
        act_name = None if act is None else getattr(act, "__name__", None) or type(act).__name__
        return cls(E, H, I, gated, bias, transposed, concatenated, hfcompat.apply_gate_kind(m), act_name)

    def cross_check(self, cfg) -> dict:
        """Compare with the config's keys (logged, never used for sizes): {field: (layout, config)} on mismatch."""
        text = text_config(cfg)
        e_cfg = next((getattr(text, k) for k in ("num_experts", "num_local_experts", "n_routed_experts")
                      if getattr(text, k, None) is not None), None)
        i_cfg = getattr(text, "moe_intermediate_size", None)
        if i_cfg is None:  # gpt-oss and MiniMax-M2 size their experts with intermediate_size
            i_cfg = getattr(text, "intermediate_size", None)
        bad = {}
        for name, mine, theirs in (("E", self.E, e_cfg), ("H", self.H, text.hidden_size), ("I", self.I, i_cfg)):
            if theirs is not None and mine != theirs:
                bad[name] = (mine, theirs)
        return bad


def config_sparse_layers(cfg) -> list[int]:
    """The decoder layers the config declares MoE, read the way each modeling file decides:
    mlp_layer_types == "sparse" (glm4_moe_lite, glm_moe_dsa, deepseek_v32, glm5_next; derived from
    first_k_dense_replace when absent); enable_moe_block (gemma4: all layers or none); mlp_only_layers and
    decoder_sparse_step (qwen3_moe :308-310); first_k_dense_replace/moe_layer_freq (older DeepSeek-style configs);
    otherwise every layer (gpt_oss, minimax_m2, qwen3_5_moe_text build an MoE block in every layer)."""
    text = text_config(cfg)
    n = text.num_hidden_layers
    types = getattr(text, "mlp_layer_types", None)
    if types is not None:
        return [i for i, t in enumerate(types[:n]) if t == "sparse"]
    if hasattr(text, "enable_moe_block"):
        return list(range(n)) if text.enable_moe_block else []
    if hasattr(text, "decoder_sparse_step") or hasattr(text, "mlp_only_layers"):
        only = set(getattr(text, "mlp_only_layers", None) or [])
        step = getattr(text, "decoder_sparse_step", 1) or 1
        n_exp = getattr(text, "num_experts", None) or getattr(text, "num_local_experts", 0) or 0
        return [i for i in range(n) if i not in only and n_exp > 0 and (i + 1) % step == 0]
    if hasattr(text, "first_k_dense_replace"):
        freq = getattr(text, "moe_layer_freq", 1) or 1
        return [i for i in range(n) if i >= text.first_k_dense_replace and i % freq == 0]
    return list(range(n))


# ---------------------------------------------------------------------------------------------------------------------
# MoE sites

ROUTER_ATTRS = ("gate", "router")


@dataclass
class MoESite:
    """One MoE layer: `parent.<attr>` is the experts module `ref`; `router` is the module that picks its experts."""

    layer_idx: int
    parent: nn.Module
    attr: str
    ref: nn.Module
    router: nn.Module
    layout: ExpertsLayout
    name: str = ""  # dotted module name of the experts module
    router_name: str = ""
    top_k: int = 0
    top_k_source: str = ""
    extra: dict = field(default_factory=dict)


def _router_of(parent: nn.Module, where: str) -> tuple[str, nn.Module]:
    found = [(a, getattr(parent, a)) for a in ROUTER_ATTRS
             if isinstance(getattr(parent, a, None), nn.Module) and not hfcompat.is_experts_module(getattr(parent, a))]
    if len(found) != 1:
        raise ValueError(f"{where}: expected exactly one router among {ROUTER_ATTRS} next to the experts, found "
                         f"{[a for a, _ in found]}")
    return found[0]


def _top_k(router: nn.Module, cfg, probe: Probe) -> int:
    text = text_config(cfg)
    if probe.k_from == "router.top_k":
        if not hasattr(router, "top_k"):
            raise ValueError(f"{type(router).__name__} has no top_k (the arch table says router.top_k)")
        return int(router.top_k)
    if probe.k_from.startswith("config."):
        key = probe.k_from.split(".", 1)[1]
        if getattr(text, key, None) is None:
            raise ValueError(f"config has no {key} (the arch table says {probe.k_from})")
        return int(getattr(text, key))
    raise ValueError(f"unknown top-k source {probe.k_from!r}")


def find_moe_sites(model: nn.Module, cfg=None) -> list[MoESite]:
    """Every MoE layer of a built (meta or real) text model, with three checks (design-core B2):
    1. each layer's experts module is found under layer.mlp or on the layer itself (Gemma-4), never both;
    2. the layers holding experts are exactly the config's sparse layers (config_sparse_layers);
    3. every @use_experts_implementation module in model.named_modules() is one of the sites, so an MTP layer that was
       built by mistake (or any other unexpected experts module) fails here instead of being silently ignored."""
    cfg = cfg if cfg is not None else model.config
    text = text_config(cfg)
    entry = arch_for(text)
    layers = decoder_layers(model)
    if len(layers) != text.num_hidden_layers:
        raise ValueError(f"{len(layers)} decoder layers built, config says {text.num_hidden_layers}")
    prefix = _module_name(model, layers)
    sites = []
    for i, layer in enumerate(layers):
        found = []
        for sub in ("", "mlp"):
            parent = layer if sub == "" else getattr(layer, sub, None)
            if isinstance(parent, nn.Module) and hfcompat.is_experts_module(getattr(parent, "experts", None)):
                found.append((sub, parent))
        if len(found) > 1:
            raise ValueError(f"layer {i}: experts found both on the layer and under .mlp")
        if not found:
            continue
        sub, parent = found[0]
        base = f"{prefix}.{i}" + (f".{sub}" if sub else "")
        rattr, router = _router_of(parent, base)
        site = MoESite(i, parent, "experts", parent.experts, router, ExpertsLayout.from_module(parent.experts),
                       name=f"{base}.experts", router_name=f"{base}.{rattr}")
        site.top_k = _top_k(router, text, entry.probe)
        site.top_k_source = entry.probe.k_from
        sites.append(site)
    want = config_sparse_layers(text)
    got = [s.layer_idx for s in sites]
    if got != want:
        raise ValueError(f"MoE layers {got} != the config's sparse layers {want}")
    all_experts = {n for n, m in model.named_modules() if hfcompat.is_experts_module(m)}
    extra = sorted(all_experts - {s.name for s in sites})
    if extra:
        raise ValueError(f"experts modules outside the decoder's MoE sites (an MTP layer built?): {extra[:5]}"
                         + (f" (+{len(extra) - 5} more)" if len(extra) > 5 else ""))
    return sites


def phase_a_ok(site: MoESite) -> bool:
    """Phase A (moe.py untouched) runs a site only with the default gate (TieredMoE's act(h[:, :I]) * h[:, I:]) and
    no expert biases (design-core B1): every architecture but gpt_oss and glm5_next."""
    lay = site.layout
    return lay.gate == "default" and not lay.bias and lay.gated and lay.concatenated and not lay.transposed


# ---------------------------------------------------------------------------------------------------------------------
# prefetch probes


PHASE_A_PREFETCH = frozenset({"qwen3_moe", "qwen3_5_moe_text"})


def phase_a_prefetch(entry: ArchEntry) -> bool:
    """Phase A keeps TieredMoE's built-in predictor, topk(F.linear(x, router.weight[, router.bias])) (moe.py :158). It
    ranks experts like the router only for plain softmax-top-k routers (Qwen3, Qwen3.5); sigmoid + correction-bias
    routers rank differently and Gemma-4's router has no .weight at all. So prefetch is on only there until WP8's
    `predict` takes make_probe (design-core B1)."""
    return entry.model_type in PHASE_A_PREFETCH


def pred_k(site: MoESite) -> int:
    return site.top_k


def make_probe(site: MoESite, entry: ArchEntry) -> Callable[[torch.Tensor], torch.Tensor]:
    """x [T, H] -> predicted expert ids [T, k] of `site`'s layer, without host synchronization."""
    router, k, call = site.router, site.top_k, entry.probe.call
    if call == "linear":
        def probe(x: torch.Tensor) -> torch.Tensor:
            logits = F.linear(x, router.weight, getattr(router, "bias", None))
            return torch.topk(logits.float(), k, dim=-1).indices
    elif call == "router":
        def probe(x: torch.Tensor) -> torch.Tensor:
            return router.forward(x)[2]
    elif call == "router_bias":
        block = site.parent

        def probe(x: torch.Tensor) -> torch.Tensor:
            return router.forward(x, block.e_score_correction_bias)[2]
    else:
        raise ValueError(f"unknown probe call {call!r}")
    return probe


# ---------------------------------------------------------------------------------------------------------------------
# dense 4-bit policy (design-core B3)

GENERIC_DENY = (
    # the output is a single gating scalar; also too small to matter
    ("shared_expert_gate", re.compile(r"(^|\.)shared_expert_gate$")),
    # forward reads .weight.dtype (glm_moe_dsa :232, deepseek_v32 :231, glm5_next :866): QuantLinear would dequantize
    # the whole matrix on every call
    ("indexer.weights_proj", re.compile(r"(^|\.)indexer\.weights_proj$")),
)


def dense_quant_plan(model: nn.Module, sites: list[MoESite], entry: ArchEntry,
                     group: int = DENSE_GROUP) -> tuple[list[str], dict[str, str]]:
    """(names of the nn.Linear modules to quantize, {every other nn.Linear: reason}).

    Rules, first match wins: any Linear inside a site's router (e.g. Gemma-4's router.proj) -> "router";
    shared_expert_gate; indexer.weights_proj; the architecture's deny_quant -> "deny:<pattern>"; in_features not a
    multiple of `group` -> "in_features"; out_features < DENSE_MIN_OUT -> "out_features"; otherwise quantized. Experts
    are not nn.Linear and never appear here. On Qwen3-30B this selects exactly v0.1's 193 modules (q/k/v/o + lm_head)."""
    router_names = set()
    for s in sites:
        for n, _ in s.router.named_modules():
            router_names.add(f"{s.router_name}.{n}" if n else s.router_name)
    deny = [(p, re.compile(p)) for p in entry.deny_quant]
    selected, excluded = [], {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        reason = None
        if name in router_names:
            reason = "router"
        if reason is None:
            reason = next((r for r, rx in GENERIC_DENY if rx.search(name)), None)
        if reason is None:
            reason = next((f"deny:{p}" for p, rx in deny if rx.search(name)), None)
        if reason is None and m.in_features % group:
            reason = "in_features"
        if reason is None and m.out_features < DENSE_MIN_OUT:
            reason = "out_features"
        if reason is None:
            selected.append(name)
        else:
            excluded[name] = reason
    return selected, excluded


# ---------------------------------------------------------------------------------------------------------------------
# checkpoint keys outside the text model

TOWERS = ("visual", "vision_tower", "embed_vision", "audio_tower", "embed_audio", "multi_modal_projector")
_TOWER_RX = re.compile(r"(^|\.)(" + "|".join(TOWERS) + r")\.")
_LAYER_RX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def apply_prefix_rules(key: str, entry: ArchEntry) -> str:
    for old, new in entry.prefix_rules:
        if key.startswith(old):
            return new + key[len(old):]
    return key


class SkipRules:
    """Named rules for checkpoint keys the text model does not load (design-core B4), tried in this order:
    "tower"         a vision/audio tower or projector (TOWERS as a path component);
    <arch rules>    the architecture's own ArchEntry.skip_keys (DeepSeek-V4.1: vision.*, aligner.*, image_start/end/newline);
    "mtp"           a multi-token-prediction layer: `mtp.*`, or decoder layer index >= num_hidden_layers;
    "class_ignore"  the model classes' own _keys_to_ignore_on_load_unexpected (e.g. Gemma-4's k/v of KV-shared
                    layers), collected from every PreTrainedModel in the model as from_pretrained does.
    A key is only ever skipped after it failed to map to a model key."""

    def __init__(self, model: nn.Module, cfg=None):
        text = text_config(cfg if cfg is not None else model.config)
        self.num_layers = text.num_hidden_layers
        patterns: set[str] = set()
        for m in model.modules():
            if isinstance(m, hfcompat.PreTrainedModel):
                patterns |= set(m._keys_to_ignore_on_load_unexpected or ())
        self.class_patterns = sorted(patterns)
        self._class_rx = re.compile("|".join(f"({p})" for p in self.class_patterns)) if patterns else None
        entry = ARCHES.get(text.model_type)
        self.arch_rules = [(rule, re.compile(rx)) for rule, rx in (entry.skip_keys if entry is not None else ())]

    def __call__(self, *forms: str) -> str | None:
        """The rule that skips a key, given its forms (as stored, after prefix rules, after renaming); else None."""
        if any(_TOWER_RX.search(k) for k in forms):
            return "tower"
        for rule, rx in self.arch_rules:
            if any(rx.search(k) for k in forms):
                return rule
        for k in forms:
            m = _LAYER_RX.search(k)
            if k.startswith("mtp.") or (m and int(m.group(1)) >= self.num_layers):
                return "mtp"
        if self._class_rx is not None and any(self._class_rx.search(k) for k in forms):
            return "class_ignore"
        return None


# ---------------------------------------------------------------------------------------------------------------------
# a built engine


def tiered(model: nn.Module, cls: type | None = None) -> list:
    """The TieredMoE cores of an engine model in layer order: a legacy TieredMoE block at layer.mlp, or the `core` of
    the module that replaced a site's experts (TieredExperts, WP4) at layer.mlp.experts or layer.experts. Replaces the
    `model.model.layers[*].mlp` loops of build.py:239/:255, runner.py:32 and eval_engine.py:132."""
    if cls is None:
        from .moe import TieredMoE as cls  # imported lazily: moe_tiering.moe loads the Triton kernels
    out = []
    for layer in decoder_layers(model):
        mlp = getattr(layer, "mlp", None)
        for cand in (mlp, getattr(mlp, "experts", None), getattr(layer, "experts", None)):
            core = getattr(cand, "core", cand)
            if isinstance(core, cls):
                out.append(core)
                break
    return out
