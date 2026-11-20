"""Storage formats by name: a store's meta.json names the format of its expert blobs (and, later, of its dense part);
the loader (moe_tiering.loader) and the converters look the format up here instead of hard-coding k4row or rtn.

Expert formats. A format is a factory `(meta entry, gu_shape, dn_shape) -> layout`; a format registered with
per_layer=True has one layout per MoE layer and its factory takes `(meta entry, gu_shape, dn_shape, layer)` (the
trellis rungs: the meta entry holds every layer's rate pattern, design-quality 2.3; WP14). The layout is what
TieredMoE and ExpertCache use (moe_tiering.store.BlobLayout is the reference implementation):
    nbytes                        bytes of one expert's blob (gate_up, then down)
    gu_shape, dn_shape            (2I, H) and (H, I): matrices in nn.Linear [out, in] layout, rows in transformers'
                                  output order
    format, bits, group           the format's name and parameters (describe() returns them for meta.json)
    views(pool)                   uint8 [S, nbytes] pool (VRAM or pinned host) -> per-array strided views
    gemv_args(views, which)       keyword arguments of moe_tiering.kernels.grouped_gemv / moe_gemm for "gu" or "dn"
    dequant(views, idx, which, dtype)  one matrix of one slot, dequantized (references, store checks)
    quantize(gate_up, down, importance=None, device=None, info=None) -> uint8 [E, nbytes]   (converters)
Registered here: "rtn" (moe_tiering.quant: fp16 step/min per group) and "k4row" (moe_tiering.kquant). Both run on today's
kernels (FMT 0 / 1). "tcq" (design-quality WP14): the trellis rungs' blobs, moe_tiering.store_v2.TcqStoreLayout (an
moe_tiering.tcq.TcqLayout per layer, built from cold["layers"][str(layer)] with the codec checked), served since WP15 by
moe_tiering.tcq_kernels through TieredMoE's rotated-domain path (moe_tiering.moe; one tier, hot from cold; host_words=None: the
kernels read whole 16x16 tiles, every K is a multiple of 16, so there is no row tail). A new format (e.g. the quality sprint's trellis-coded 2-bit experts with a per-layer Hadamard
rotation, quant_methods/t4_tcq.py) registers its own layout class with register_expert_format(); its gemv_args must
then name a kernel that decodes it, which is a kernels.py change (Phase B), so `kernels=False` marks a format that
can be stored and checked (dequant) but not yet served: the loader refuses to build an engine on it.

Dense formats. A factory `(weight [out, in], bias | None, bits, device) -> nn.Module` that replaces one nn.Linear
the dense plan selects (moe_tiering.arch.dense_quant_plan). Registered: "rtn" = moe_tiering.qlinear.QuantLinear group 64
(today's dense 4-bit, quantized from the bf16 weight in row blocks on the GPU).

Dense-q kinds (design-quality WP10): the packed dense part a store ships (moe_tiering.dense_recipe "moe-tiering/dense-q"
v1 directories, meta.json "dense_q"), already quantized: a factory `(role, KPacked, bias | None, device, *, host,
**kw) -> nn.Module` builds the module of one packed tensor. Registered: "k4row" = moe_tiering.klinear.KLinear (every
linear role) and moe_tiering.qembed.QEmbedding (role "embed"; host=True keeps its table in pinned host memory, read
zero-copy). The kind of a tensor is its recipe format's kind (Fmt.kind in the directory's recipe_spec).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from .store import BlobLayout


@dataclass(frozen=True)
class ExpertFormat:
    name: str
    factory: Callable  # (entry: dict, gu_shape, dn_shape) -> layout
    kernels: bool = True  # the decode/prefill kernels can read it (served by TieredMoE)
    host_words: int | None = 8  # the pinned-host GEMV reads WORDS of this many int32 per row step (None: n/a)
    note: str = ""
    per_layer: bool = False  # one layout per MoE layer: the factory takes the layer index (store v2 trellis rungs)


_EXPERT: dict[str, ExpertFormat] = {}
_DENSE: dict[str, Callable] = {}
_DENSE_Q: dict[str, Callable] = {}


def register_expert_format(name: str, factory: Callable, *, kernels: bool = True, host_words: int | None = 8,
                           note: str = "", replace: bool = False, per_layer: bool = False) -> ExpertFormat:
    if name in _EXPERT and not replace:
        raise ValueError(f"expert format {name!r} is already registered")
    fmt = ExpertFormat(name, factory, kernels, host_words, note, per_layer)
    _EXPERT[name] = fmt
    return fmt


def expert_formats() -> dict[str, ExpertFormat]:
    return dict(_EXPERT)


def canonical_name(entry: dict) -> str:
    """The registry name of a meta.json "hot"/"cold" entry; no "format" (or the old "rtn-g64") means rtn."""
    name = entry.get("format", "rtn")
    return "rtn" if name.startswith("rtn") else name


def expert_format(entry_or_name) -> ExpertFormat:
    name = entry_or_name if isinstance(entry_or_name, str) else canonical_name(entry_or_name)
    if name not in _EXPERT:
        raise ValueError(f"unknown expert blob format {name!r}; registered: {sorted(_EXPERT)}")
    return _EXPERT[name]


def expert_layout(entry: dict, gu_shape, dn_shape, layer: int | None = None):
    """meta.json "hot"/"cold" entry -> the layout object (moe_tiering.build.layout_from_meta, through the registry).
    layer: the MoE layer (decoder layer index); required for a per_layer format, ignored by the others."""
    fmt = expert_format(entry)
    if fmt.per_layer:
        if layer is None:
            raise ValueError(f"expert format {fmt.name!r} has one layout per MoE layer: pass layer=")
        lay = fmt.factory(entry, tuple(gu_shape), tuple(dn_shape), int(layer))
    else:
        lay = fmt.factory(entry, tuple(gu_shape), tuple(dn_shape))
    if "nbytes" in entry and entry["nbytes"] != lay.nbytes:
        raise ValueError(f"meta says {entry['nbytes']} bytes per expert, layout {lay} has {lay.nbytes}")
    return lay


def layer_layouts(entry: dict, gu_shape, dn_shape, layers) -> dict:
    """{layer: layout} for the given MoE layers: one shared layout object for a uniform format (k4row, rtn: exactly
    what expert_layout returns today), one per layer for a per_layer format (trellis)."""
    layers = [int(x) for x in layers]
    if expert_format(entry).per_layer:
        return {li: expert_layout(entry, gu_shape, dn_shape, li) for li in layers}
    lay = expert_layout(entry, gu_shape, dn_shape)
    return {li: lay for li in layers}


def new_expert_layout(name: str, bits: int, group: int, gu_shape, dn_shape, **extra):
    """A layout for a converter: format name and parameters -> layout (the same factory the loader uses)."""
    return expert_layout({"format": name, "bits": bits, "group": group, **extra}, gu_shape, dn_shape)


def servable(layout) -> bool:
    return expert_format(layout.format).kernels


def host_row_words(layout) -> dict[str, int]:
    """int32 words per row of each matrix (K * bits / 32), as the GEMV kernels read them."""
    return {"gu": layout.gu_shape[1] * layout.bits // 32, "dn": layout.dn_shape[1] * layout.bits // 32}


def host_tail_ok(layout) -> bool:
    """False when a matrix of this layout, read from pinned host memory, would hit the kernel tail bug (design-core
    verified fact 16: the host GEMV offered WORDS {8, 16} only and its prune fell back to WORDS 8, dropping the last
    words of a row when 8 does not divide them: gpt-oss 180 words, Gemma-4 down 44); the loader then refuses host
    placement unless every expert is resident in VRAM (design-core B3's interim guard). WP7 fixed the kernels (tail
    configs WORDS 4 / 2 / 1, tests/test_kernel_tail.py), so rtn and k4row register host_words=1: any whole number
    of words per row is read right and the guard no longer fires for them."""
    w = expert_format(layout.format).host_words
    return w is None or all(n % w == 0 for n in host_row_words(layout).values())


def _blob_layout(entry: dict, gu_shape, dn_shape) -> BlobLayout:
    return BlobLayout(entry["bits"], entry["group"], tuple(gu_shape), tuple(dn_shape), canonical_name(entry))


register_expert_format("rtn", _blob_layout, host_words=1,
                       note="moe_tiering.quant: per group fp16 step and minimum (FMT 0)")
register_expert_format("k4row", _blob_layout, host_words=1,
                       note="moe_tiering.kquant: 4-bit (sc, mc) per group, fp16 d/dmin per row (FMT 1)")


def _tcq_layout(entry: dict, gu_shape, dn_shape, layer: int):
    from .store_v2 import tcq_layer_layout  # moe_tiering.tcq (torch only, no kernels)

    return tcq_layer_layout(entry, gu_shape, dn_shape, layer)


register_expert_format("tcq", _tcq_layout, kernels=True, host_words=None, per_layer=True,
                       note="moe_tiering.tcq: trellis codec 1 (L12 V2, 16x16 tail-biting tiles, fp16 row scales), rotated "
                            "input domain, one rate pattern per layer; moe_tiering.tcq_kernels through TieredMoE (WP15)")


# ---------------------------------------------------------------------------------------------------------------------
# dense formats


def register_dense_format(name: str, factory: Callable, replace: bool = False) -> None:
    if name in _DENSE and not replace:
        raise ValueError(f"dense format {name!r} is already registered")
    _DENSE[name] = factory


def dense_formats() -> list[str]:
    return sorted(_DENSE)


def dense_module(name: str, weight, bias, bits: int, device) -> nn.Module:
    if name not in _DENSE:
        raise ValueError(f"unknown dense format {name!r}; registered: {sorted(_DENSE)}")
    return _DENSE[name](weight, bias, bits, device)


def _rtn_dense(weight, bias, bits, device):
    from .qlinear import QuantLinear  # imports the Triton kernels

    return QuantLinear(weight, bias, bits, device=device)


register_dense_format("rtn", _rtn_dense)


# ---------------------------------------------------------------------------------------------------------------------
# dense-q kinds (packed dense parts shipped in a store)


def register_dense_q_kind(name: str, factory: Callable, replace: bool = False) -> None:
    if name in _DENSE_Q and not replace:
        raise ValueError(f"dense-q kind {name!r} is already registered")
    _DENSE_Q[name] = factory


def dense_q_kinds() -> list[str]:
    return sorted(_DENSE_Q)


def dense_q_module(kind: str, role: str, packed, bias, device, *, host: bool = False, **kw) -> nn.Module:
    """The engine module for one packed dense tensor (kquant.KPacked on `device`, or on the CPU for host=True)."""
    if kind not in _DENSE_Q:
        raise ValueError(f"unknown dense-q kind {kind!r}; registered: {sorted(_DENSE_Q)}")
    return _DENSE_Q[kind](role, packed, bias, device, host=host, **kw)


def _k4row_dense_q(role, packed, bias, device, *, host=False, **kw):
    if role == "embed":
        from .qembed import QEmbedding

        if bias is not None:
            raise ValueError("a packed embedding has no bias")
        return QEmbedding.from_packed(packed, host=host, device=device, **kw)
    if host:
        raise NotImplementedError(f"dense-q role {role!r}: KLinear has no host-memory variant (only the embedding "
                                  "can live in host memory)")
    from .klinear import KLinear

    return KLinear.from_packed(packed, bias, device=device, **kw)


register_dense_q_kind("k4row", _k4row_dense_q)
