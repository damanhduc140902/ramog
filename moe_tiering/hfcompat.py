"""The engine's single doorway into transformers internals, pinned to the versions it was read and tested on.

transformers keeps all model semantics; the engine only borrows a few of its private helpers so that checkpoint keys
get the names, the dtypes and the buffer initialization that `from_pretrained` would give them. Private helpers can
change in any release, so every use goes through this module, and importing it checks the installed version:
- PINNED (5.19.0) is the version new models are converted, referenced and run with (design-core A5);
- SUPPORTED also keeps 5.15.0 (the container's python3) until the WP9 switch-over;
- MOE_TIERING_ALLOW_TF=<v1,v2|any> accepts other versions at the user's own risk.

Wrapped (signatures identical in 5.15.0 and 5.19.0; file:line refer to 5.19.0):
- conversion_mapping.get_model_conversion_mapping(model)             (:2049) the model's WeightRenaming/Converter list
- core_model_loading.rename_source_key(key, renamings, converters)   (:1412) checkpoint key -> model key
- core_model_loading.dot_natural_key                                 (:1272) the loader's key order
- core_model_loading.build_glob_alternation                          (:49)   the loader's dtype-plan matching
- PreTrainedModel._get_dtype_plan(dtype)                             (modeling_utils :3765)
- PreTrainedModel.initialize_weights()                               (modeling_utils :2389)
- integrations.finegrained_fp8.Fp8Dequantize._dequantize_one         (:1040) reference for the FP8 decoder tests
- integrations.mxfp4.convert_moe_packed_tensors                      (:316)  reference for the MXFP4 decoder tests
- integrations.moe._default_apply_gate and the use_experts_implementation attributes (:528-597)
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping

import torch
import torch.nn as nn
import transformers

TF_VERSION: str = transformers.__version__
PINNED = "5.19.0"
SUPPORTED = ("5.15.0", "5.19.0")
ALLOW_ENV = "MOE_TIERING_ALLOW_TF"


def version_tuple(v: str) -> tuple[int, ...]:
    """'5.19.0' -> (5, 19, 0); pre-release or local suffixes ('5.20.0.dev0', '5.19.0+x') are ignored."""
    out = []
    for part in v.split("+")[0].split("."):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group()))
    return tuple(out)


def tf_at_least(v: str, version: str | None = None) -> bool:
    return version_tuple(version or TF_VERSION) >= version_tuple(v)


def check_supported(version: str | None = None) -> None:
    """Raise unless `version` (default: the installed one) is one this module was checked against."""
    version = version or TF_VERSION
    allow = [a.strip() for a in os.environ.get(ALLOW_ENV, "").split(",") if a.strip()]
    if version in SUPPORTED or "any" in allow or version in allow:
        return
    raise RuntimeError(
        f"transformers {version} is not one of the versions moe_tiering.hfcompat was checked against {SUPPORTED} "
        f"(pinned: {PINNED}). Its private loading APIs may differ; set {ALLOW_ENV}={version} to accept it anyway.")


def require_tf(min_tf: str, what: str) -> None:
    if not tf_at_least(min_tf):
        raise RuntimeError(f"{what} needs transformers >= {min_tf}; this interpreter has {TF_VERSION} "
                           f"(pip install 'transformers>={min_tf}')")


check_supported()

# Imported only after the version check: these are the private names this module exists to contain.
from transformers.conversion_mapping import get_model_conversion_mapping  # noqa: E402
from transformers.core_model_loading import (  # noqa: E402
    Concatenate,
    MergeModulelist,
    WeightConverter,
    WeightRenaming,
    build_glob_alternation,
    dot_natural_key,
    rename_source_key,
)
from transformers.integrations.moe import _default_apply_gate  # noqa: E402
from transformers.modeling_utils import PreTrainedModel  # noqa: E402

default_apply_gate = _default_apply_gate

# ---------------------------------------------------------------------------------------------------------------------
# experts modules (integrations/moe.py: use_experts_implementation)

DECORATOR_ATTRS = ("has_gate", "has_bias", "is_transposed", "is_concatenated")


def is_experts_module(m: nn.Module) -> bool:
    """True for an instance of a class decorated with @use_experts_implementation: its __init__ sets the four layout
    attributes on the instance and the class carries `_apply_gate` (its own, or the default the decorator installs)."""
    return (isinstance(m, nn.Module) and all(hasattr(m, a) for a in DECORATOR_ATTRS)
            and hasattr(type(m), "_apply_gate"))


def apply_gate_kind(m: nn.Module) -> str:
    """'default' when the class uses the decorator's _default_apply_gate (chunk(2) -> act_fn(gate) * up), which is
    exactly today's TieredMoE activation; 'custom' when the class defines its own (gpt-oss, GLM-5.3-Flash)."""
    return "default" if type(m)._apply_gate is default_apply_gate else "custom"


# ---------------------------------------------------------------------------------------------------------------------
# key naming, exactly as from_pretrained's loader does it


def pretrained_root(model: nn.Module) -> PreTrainedModel:
    """The PreTrainedModel that owns the config-level attributes: the model itself, or the base model a TextLM wraps."""
    if isinstance(model, PreTrainedModel):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, PreTrainedModel):
        return inner
    raise TypeError(f"{type(model).__name__} is neither a PreTrainedModel nor a wrapper of one (.model)")


def sort_keys(keys) -> list[str]:
    """Checkpoint keys in the loader's order (MergeModulelist stacks experts in this order)."""
    return sorted(keys, key=dot_natural_key)


class KeyRenamer:
    """Checkpoint key -> model key the way the 5.19 loader names it (core_model_loading :1662-1670): all WeightRenamings
    in order, then at most one WeightConverter, then the base_model_prefix add/strip against the meta state dict; a key
    that the transforms rename away from a name the model already has is named as-is. Returns (model_key, the
    converter source pattern that matched or None). Feed keys in sort_keys() order: some renamings are stateful
    (GroupWeightRename) and converters collect experts in that order.

    The model may be a TextLM (an nn.Module with `.model` and `base_model_prefix`): transformers then scopes the
    inner model's transforms under that prefix, the same way it scopes a VLM's language model."""

    def __init__(self, model: nn.Module, meta_keys: Mapping | None = None):
        if not isinstance(model, PreTrainedModel) and not hasattr(model, "base_model_prefix"):
            raise TypeError(f"{type(model).__name__}: a wrapper needs a base_model_prefix to scope transforms")
        transforms = get_model_conversion_mapping(model)  # fresh copies on every call
        self.renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
        self.converters = [t for t in transforms if isinstance(t, WeightConverter)]
        unknown = [t for t in transforms if not isinstance(t, (WeightRenaming, WeightConverter))]
        if unknown:
            raise TypeError(f"unexpected transform types {[type(t).__name__ for t in unknown]}")
        self.base_model_prefix = model.base_model_prefix
        self.meta = dict(meta_keys) if meta_keys is not None else dict.fromkeys(model.state_dict().keys(), True)
        self.by_pattern = {p: c for c in self.converters for p in c.source_patterns}

    def __call__(self, key: str) -> tuple[str, str | None]:
        renamed, pattern = rename_source_key(key, self.renamings, self.converters, self.base_model_prefix, self.meta)
        if renamed not in self.meta and key in self.meta:
            renamed, pattern = rename_source_key(key, [], [], base_model_prefix=self.base_model_prefix,
                                                 meta_state_dict=self.meta)
        return renamed, pattern

    def converter(self, pattern: str) -> WeightConverter:
        return self.by_pattern[pattern]

    def describe(self) -> list[str]:
        return [repr(t) for t in self.renamings + self.converters]


def converter_ops(conv: WeightConverter) -> list[tuple[str, int]]:
    """A converter's operations as [('merge'|'concat', dim)]. The engine supports only these two (expert merges and
    GLM-5.3-Flash's conv1d concat); anything else raises, so a new op is never silently mis-handled."""
    out = []
    for op in conv.operations:
        if isinstance(op, MergeModulelist):
            out.append(("merge", op.dim))
        elif isinstance(op, Concatenate):
            out.append(("concat", op.dim))
        else:
            raise NotImplementedError(f"conversion op {type(op).__name__} in {conv!r} is not supported")
    return out


def converted_shape(conv: WeightConverter, shapes: Mapping[str, list[tuple[int, ...]]]) -> tuple[int, ...]:
    """The shape the converter's ops produce from source tensors of the given shapes ({source pattern: [shape per
    collected tensor, in collection order]}), the way MergeModulelist/Concatenate combine them."""
    state: dict[str, list[tuple[int, ...]] | tuple[int, ...]] = {p: list(v) for p, v in shapes.items()}
    for name, dim in converter_ops(conv):
        if name == "merge":
            for p, v in list(state.items()):
                if isinstance(v, tuple):
                    continue
                if len(set(v)) != 1:
                    raise ValueError(f"{conv!r}: cannot stack unequal shapes {sorted(set(v))} of {p}")
                s = list(v[0])
                s.insert(dim if dim >= 0 else len(s) + dim + 1, len(v))
                state[p] = tuple(s)
        else:
            parts: list[tuple[int, ...]] = []
            for p in conv.source_patterns:  # Concatenate keeps the source-pattern order
                if p in state:
                    v = state.pop(p)
                    parts.extend([v] if isinstance(v, tuple) else v)
            if not parts:
                raise ValueError(f"{conv!r}: nothing to concatenate")
            d = dim if dim >= 0 else len(parts[0]) + dim
            for s in parts[1:]:
                if len(s) != len(parts[0]) or any(a != b for i, (a, b) in enumerate(zip(s, parts[0])) if i != d):
                    raise ValueError(f"{conv!r}: cannot concatenate {parts} along {dim}")
            out = list(parts[0])
            out[d] = sum(s[d] for s in parts)
            state = {"": tuple(out)}
    if len(state) != 1:
        raise ValueError(f"{conv!r}: ops leave {len(state)} tensors, expected 1")
    (v,) = state.values()
    if not isinstance(v, tuple):
        if len(v) != 1:
            raise ValueError(f"{conv!r}: {len(v)} tensors left unmerged")
        v = v[0]
    return v


# ---------------------------------------------------------------------------------------------------------------------
# dtypes and initialization


def dtype_plan(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.dtype]:
    """{module-name pattern: dtype} that from_pretrained(dtype=...) forces (the strict-fp32 list for bf16 loads,
    plus _keep_in_fp32_modules for fp16 loads; modeling_utils :3765-3779)."""
    return dict(pretrained_root(model)._get_dtype_plan(dtype))


def arch_fp32_keys(model: nn.Module) -> tuple[str, ...]:
    """The architecture's own float32 patterns (moe_tiering.arch ArchEntry.fp32_keys) for a model built from its config;
    () for a model type outside the table."""
    from . import arch  # moe_tiering.arch imports this module

    cfg = getattr(pretrained_root(model), "config", None)
    if cfg is None:
        return ()
    entry = arch.ARCHES.get(arch.text_config(cfg).model_type)
    return entry.fp32_keys if entry is not None else ()


class DtypePlan:
    """key -> forced dtype or None, matched the way the loader matches the plan (core_model_loading :1673, :1697).
    The plan is transformers' (dtype_plan) plus, for bf16 / fp16 loads, the architecture's fp32_keys (`extra`): tensors
    the release stores in F32 that the class has no strict-fp32 entry for (MiniMax-M2's router, GLM-5.3-Flash's mHC
    base / scale). KeyMap, moe_tiering.loader and the converter all take their dtypes from this one plan."""

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.bfloat16):
        self.plan = dtype_plan(model, dtype)
        self.extra: dict[str, torch.dtype] = {}
        if dtype in (torch.float16, torch.bfloat16):
            self.extra = {p: torch.float32 for p in arch_fp32_keys(model) if p not in self.plan}
            self.plan.update(self.extra)
        if self.plan:
            self._alt, self._by_group, _ = build_glob_alternation(list(self.plan.keys()))

    def __call__(self, key: str) -> torch.dtype | None:
        if not self.plan:
            return None
        m = self._alt.search(key)
        return None if m is None else self.plan[self._by_group[m.lastgroup]]


def load_dtype(plan: DtypePlan, key: str, meta_dtype: torch.dtype) -> torch.dtype:
    """The dtype the loader gives a non-quantized tensor: the plan's dtype if it matches, else the meta parameter's
    (core_model_loading 5.19 :1698-1703; design-core A7)."""
    forced = plan(key)
    return meta_dtype if forced is None else forced


def initialize_weights(model: nn.Module) -> None:
    """PreTrainedModel.initialize_weights() of the model (or of the base model a TextLM wraps). It skips every module
    and tensor flagged `_is_hf_initialized`, so after flagging the loaded tensors it only (re)builds the rest."""
    pretrained_root(model).initialize_weights()


def named_non_persistent_buffers(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Like PreTrainedModel.named_non_persistent_buffers (modeling_utils 5.19 :4903), for any nn.Module."""
    for name, tensor in model.named_buffers():
        parent, buf = name.rsplit(".", 1) if "." in name else ("", name)
        if buf in model.get_submodule(parent)._non_persistent_buffers_set:
            yield name, tensor


# ---------------------------------------------------------------------------------------------------------------------
# transformers' own decoders, as references for moe_tiering.dequant's tests (WP1)


def fp8_dequantize_reference(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Fp8Dequantize._dequantize_one(q, scale, dtype): raises on partial blocks (shape not divisible by the grid)."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    return Fp8Dequantize(None)._dequantize_one(q, scale, dtype)


def mxfp4_reference(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """convert_moe_packed_tensors(blocks, scales): [..., n, k/32, 16] u8 + [..., n, k/32] u8 -> [..., k, n] (it ends
    with a transpose(1, 2); the engine's decoder returns the out-major [..., n, k])."""
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    return convert_moe_packed_tensors(blocks, scales, dtype=dtype)
