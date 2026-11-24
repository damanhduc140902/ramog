"""Dense-part recipes: which k4row format (and GPTQ or not) each role of a model's non-expert tensors gets, a
role resolver per architecture, and the converter that writes the packed dense part one layer at a time.

Recipe "dense-c" (track T1 of the quality sprint, agents/quality/t1/cfg/dense-c.json; results/quality/REPORT.md):
every format is one behemoe.kernels already decodes (FMT 1: k4row, BITS 2/4/8, GROUP 16/32):
  q_proj, o_proj, lm_head   k4row 4-bit g16, importance-weighted grid, GPTQ (damp 0.01, act-order, static groups)
  k_proj, v_proj, router    k4row 8-bit g32, importance-weighted grid
  embed                     k4row 4-bit g32, plain MSE (no importance)
  norm                      bf16 (not converted here; the loader keeps them from the bf16 dense file)
  default                   any other dense linear a resolver names: T1's config "default" (= q/o's format)
Recipe "dense-hi" (QD, design-quality 0.2): every dense linear and the router k4row 8-bit g32 with importance, the
embedding as dense-c's, norms bf16. recipe_spec() applies per-architecture floors (ARCH_FLOORS: gpt-oss attention and
input embedding stay bf16).
Importance = behemoe.kquant.importance_from_stats(sumsq, count) of the module's input statistics and GPTQ's H = xtx /
count, both from a tools/dense_stats.py directory (layerNNN.pt: q_proj = k_proj = v_proj, o_proj, gate entries;
model.pt: lm_head, embed_tokens), looked up by the module's short name (as qbench passes them to T1's plugin).

Roles come from a resolver: name (module name, no ".weight") -> role | "expert" (skipped: expert weights are not
dense) | None (unknown: the converter refuses). qwen3_moe_role() reads Qwen3-MoE's module names; roles_from_arch()
builds one from core Phase A's architecture table (behemoe.arch: find_moe_sites + dense_quant_plan) for any model.

Converter output (convert_dense): out_dir/
  layerNNN.safetensors   per decoder layer: "<module>.qweight" uint8 [out, in*bits/8], ".sb" uint8 [out, in/group]
                         (sc | mc << 4), ".d" / ".dmin" fp16 [out] (behemoe.kquant.KPacked), ".bias" when present
  model.safetensors      the same for the tensors outside the layers (embed_tokens, lm_head)
  meta.json              format "moe-tiering/dense-q" v1: recipe, per-tensor role / format / shape / file / bits /
                         timing / GPTQ path, the bf16 tensors left to the bf16 dense file, stats provenance
Dequantizing these (kquant.dequantize) gives T1's dense-c weights bit for bit with gptq_path="cpu"
(tests/test_gptq.py), and qbench_plugin() feeds them to tools/qbench.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from . import gptq as _gptq
from . import kquant

FORMAT = "moe-tiering/dense-q"
VERSION = 1


@dataclass(frozen=True)
class Fmt:
    """kind "k4row" (bits, group, gptq, imp, damp, act_order) or "bf16" (kept as is)."""

    kind: str
    bits: int = 0
    group: int = 0
    gptq: bool = False
    imp: bool = True
    damp: float = _gptq.DAMP
    act_order: bool = True

    def bits_total(self, n: int, k: int) -> int:
        """Every stored bit: codes + one byte per group + fp16 d and dmin per row (= t1_dense.grid_bits, sbits 4)."""
        if self.kind == "bf16":
            return 16 * n * k
        return n * k * self.bits + n * (k // self.group) * 8 + n * 32


BF16 = Fmt("bf16")
K4_16_GPTQ = Fmt("k4row", 4, 16, gptq=True, imp=True)
K8_32 = Fmt("k4row", 8, 32, gptq=False, imp=True)
K4_32_MSE = Fmt("k4row", 4, 32, gptq=False, imp=False)

RECIPES: dict[str, dict[str, Fmt]] = {
    "dense-c": {"q_proj": K4_16_GPTQ, "o_proj": K4_16_GPTQ, "lm_head": K4_16_GPTQ,
                "k_proj": K8_32, "v_proj": K8_32, "router": K8_32,
                "embed": K4_32_MSE, "norm": BF16, "default": K4_16_GPTQ},
    # QD's dense-hi (design-quality 0.2; agents/qint/qd/cfg/dense-hi.json, T1's kq with sbits 4): every dense linear
    # (q/k/v/o, other linears, lm_head) and the router k4row 8-bit g32 with importance, no GPTQ; embeddings k4row
    # 4-bit g32 plain MSE; norms bf16. Registered here since WP10 (tools/build_h4.py and behemoe/convert_tcq.py use
    # this entry; build_h4 checks it against the cfg file).
    "dense-hi": {"q_proj": K8_32, "k_proj": K8_32, "v_proj": K8_32, "o_proj": K8_32, "lm_head": K8_32,
                 "router": K8_32, "embed": K4_32_MSE, "norm": BF16, "default": K8_32},
}
ROLES = ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "router", "embed", "norm")

# Per-architecture floors on top of every recipe: a role listed here never gets less precision than this format (bf16
# counts as 16 bits; a recipe entry with at least as many bits is kept). gpt-oss: its attention is the part dense
# quantization breaks (results/coregen/gpt-oss-20b/summary.txt, FINDING A: q/k/v at rtn 4-bit give top-1 0.71; the
# README budget passes only with the attention in bf16, top-1 0.753). WP10 measured k4row on it in transformers alone
# (results/qlane/wp10, gptoss-attn): q/k/v/o at k4row 8-bit g32 still cost KL 0.018 and top-1 0.980 -> 0.916, dense-c's
# 4-bit q/o 0.064 / 0.836, so q/k/v/o stay bf16 (not packed; the loader keeps them from dense.safetensors) in every
# recipe. The input embedding too (WP10 verifier, results/qlane/verify/wp10/gptoss-head-r1/r2.json, gptoss-emb.json;
# re-run results/qlane/wp10/fix1/): both recipes' k4row 4-bit g32 embedding gives KL 0.497 / top-1 0.590 (round-trip
# error 10.6%, normal for 4-bit), 8-bit g32 still 0.033 / 0.888 (worse than the 8-bit attention rejected above), so it
# stays bf16 (dense.safetensors; embed_host moves the 1.16 GB table to pinned host memory). The heads pass: lm_head
# 8-bit g32 KL 0.00015 / 0.978, 4-bit g16 (no GPTQ) 0.0037 / 0.944.
ARCH_FLOORS: dict[str, dict[str, Fmt]] = {
    "gpt_oss": {"q_proj": BF16, "k_proj": BF16, "v_proj": BF16, "o_proj": BF16, "embed": BF16},
}


def _precision(f: Fmt) -> int:
    return 16 if f.kind == "bf16" else f.bits


def recipe_spec(recipe: str | dict, model_type: str | None = None) -> dict[str, Fmt]:
    """The recipe's role -> format table for an architecture (text model_type): RECIPES[recipe] (or the dict given)
    with ARCH_FLOORS[model_type] applied. Unchanged for every architecture without floors."""
    r = dict(RECIPES[recipe] if isinstance(recipe, str) else recipe)
    for role, floor in ARCH_FLOORS.get(model_type or "", {}).items():
        cur = r.get(role, r.get("default"))
        if cur is not None and _precision(cur) < _precision(floor):
            r[role] = floor
    return r


def recipe_format(recipe: str | dict, role: str) -> Fmt:
    r = RECIPES[recipe] if isinstance(recipe, str) else recipe
    if role in r:
        return r[role]
    if "default" in r:
        return r["default"]
    raise KeyError(f"recipe has no format for role {role!r} and no default")


# ------------------------------------------------------------------------------------------------ role resolvers

Resolver = Callable[[str], "str | None"]

_QWEN3 = (
    (re.compile(r"(^|\.)embed_tokens$"), "embed"),
    (re.compile(r"^lm_head$"), "lm_head"),
    (re.compile(r"\.self_attn\.q_proj$"), "q_proj"),
    (re.compile(r"\.self_attn\.k_proj$"), "k_proj"),
    (re.compile(r"\.self_attn\.v_proj$"), "v_proj"),
    (re.compile(r"\.self_attn\.o_proj$"), "o_proj"),
    (re.compile(r"\.mlp\.gate$"), "router"),
    (re.compile(r"\.mlp\.experts(\.|$)"), "expert"),
    (re.compile(r"(^|\.)(input_layernorm|post_attention_layernorm|q_norm|k_norm|norm)$"), "norm"),
)


def qwen3_moe_role(name: str) -> str | None:
    """Role of a Qwen3-MoE module name (checkpoint or module name, with or without .weight / .bias)."""
    name = re.sub(r"\.(weight|bias)$", "", name)
    for rx, role in _QWEN3:
        if rx.search(name):
            return role
    return None


def roles_from_arch(model, sites=None, entry=None) -> Resolver:
    """A resolver from core Phase A's architecture table, for any supported model (meta-built is enough):
    routers of the MoE sites -> "router"; nn.Linear modules dense_quant_plan selects -> their short name when it is
    a known role (q_proj, k_proj, v_proj, o_proj, lm_head) else "default"; nn.Embedding -> "embed"; modules whose
    class name ends in "Norm" -> "norm"; the experts modules -> "expert". Linear modules the plan keeps in bf16 for
    other reasons (shared_expert_gate, indexer.weights_proj, the arch's deny list) -> "bf16". Everything else
    (rotary buffers, ...) -> None. On Qwen3-30B it agrees with qwen3_moe_role on every checkpoint tensor."""
    import torch.nn as nn

    from . import arch

    if entry is None:
        entry = arch.arch_for(model.config)
    if sites is None:
        sites = arch.find_moe_sites(model, model.config)
    selected, excluded = arch.dense_quant_plan(model, sites, entry)
    table: dict[str, str] = {}
    for s in sites:
        table[s.name] = "expert"
        table[s.router_name] = "router"
        for n, _ in s.router.named_modules():
            if n:
                table[f"{s.router_name}.{n}"] = "router"
    for n in selected:
        short = n.rsplit(".", 1)[-1]
        table[n] = short if short in ("q_proj", "k_proj", "v_proj", "o_proj", "lm_head") else "default"
    for n, reason in excluded.items():
        table.setdefault(n, "router" if reason == "router" else "bf16")
    for n, m in model.named_modules():
        if n in table:
            continue
        if isinstance(m, nn.Embedding):
            table[n] = "embed"
        elif type(m).__name__.endswith("Norm") or type(m).__name__.endswith("RMSNorm"):
            table[n] = "norm"
    experts = [s.name for s in sites]

    def resolve(name: str) -> str | None:
        name = re.sub(r"\.(weight|bias)$", "", name)
        if name in table:
            return table[name]
        for e in experts:  # per-expert checkpoint names below an experts module
            if name.startswith(e + "."):
                return "expert"
        return None

    return resolve


# ------------------------------------------------------------------------------------------------ quantization

def short_name(module: str) -> str:
    return module.rsplit(".", 1)[-1]


@torch.no_grad()
def quantize_tensor(W: torch.Tensor, fmt: Fmt, entry: dict | None, *, device="cuda", gptq_path: str = "cpu",
                    prep_cache: dict | None = None, info: dict | None = None) -> kquant.KPacked:
    """One dense weight [out, in] -> KPacked under `fmt`, as T1's plugin computes it (t1_dense._quant_2d with sbits 4
    and radius 2): importance from entry (when fmt.imp and the entry has sumsq), GPTQ with entry's xtx when fmt.gptq.
    prep_cache: {id(xtx): Prep} shared by modules with one input (q/k/v), like T1's per-process cache."""
    if fmt.kind != "k4row":
        raise ValueError(f"quantize_tensor: format {fmt.kind} is not quantized")
    dev = torch.device(device)
    Wd = W.to(dev).float()
    n, k = Wd.shape
    a = _gptq.importance(entry, k, dev) if fmt.imp else None
    if info is not None:
        info["importance"] = a is not None
    if not fmt.gptq:
        return _gptq.quantize(Wd, fmt.bits, fmt.group, a, None, info=info)
    if entry is None or entry.get("xtx") is None:
        raise ValueError("GPTQ needs xtx statistics (tools/dense_stats.py) for this tensor")
    prep = None
    key = (id(entry["xtx"]), fmt.damp, fmt.act_order, gptq_path)
    if prep_cache is not None:
        prep = prep_cache.get(key)
    if prep is None:
        prep = _gptq.factorize(_gptq.hessian(entry), fmt.damp, fmt.act_order, dev, gptq_path)
        if prep_cache is not None:
            prep_cache[key] = prep
    elif info is not None:
        info["prep_shared"] = True
    return _gptq.quantize(Wd, fmt.bits, fmt.group, a, None, prep=prep, info=info)


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class _Checkpoint:
    """Read-one-tensor-at-a-time access to a safetensors checkpoint (sharded or single file)."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open

        self._open = safe_open
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            self.where = {k: os.path.join(model_dir, v) for k, v in wm.items()}
        else:
            f = os.path.join(model_dir, "model.safetensors")
            with safe_open(f, "pt") as h:
                self.where = {k: f for k in h.keys()}
        self._handles: dict = {}

    def names(self) -> list[str]:
        return list(self.where)

    def get(self, name: str) -> torch.Tensor:
        f = self.where[name]
        if f not in self._handles:
            self._handles[f] = self._open(f, "pt")
        return self._handles[f].get_tensor(name)


_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def _save(tensors: dict, path: str, metadata: dict) -> None:
    from safetensors.torch import save_file

    tmp = f"{path}.{os.getpid()}.tmp"
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata=metadata)
    os.replace(tmp, path)


@torch.no_grad()
def convert_dense(model_dir: str, stats_dir: str, out_dir: str, recipe: str = "dense-c",
                  roles: Resolver = qwen3_moe_role, device="cuda", gptq_path: str = "cpu", layers=None,
                  model_level: bool = True, tie_word_embeddings: bool | None = None, log=print) -> dict:
    """Quantize the dense part of the checkpoint in model_dir under `recipe`, streaming: one decoder layer's dense
    tensors and statistics at a time (then the model-level tensors: embedding, lm_head), each written to its own
    safetensors file in out_dir before the next is read. Peak memory: one layer's dense tensors + statistics.
    layers: optional iterable of layer indices (tests); None = all. model_level: also the tensors outside the
    layers. Returns (and writes) meta.json."""
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    ck = _Checkpoint(model_dir)
    cfgp = os.path.join(model_dir, "config.json")
    cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    rspec = recipe_spec(recipe, (cfg.get("text_config") or {}).get("model_type") or cfg.get("model_type"))
    if tie_word_embeddings is None:
        tie_word_embeddings = bool(cfg.get("tie_word_embeddings", False))
    # group the checkpoint by layer; unknown names are refused before any work
    by_layer: dict[int, list[str]] = {}
    top: list[str] = []
    role_of: dict[str, str] = {}
    unknown = []
    for name in ck.names():
        r = roles(name)
        if r is None:
            unknown.append(name)
            continue
        role_of[name] = r
        m = _LAYER.match(name)
        (by_layer.setdefault(int(m.group(1)), []) if m else top).append(name)
    if unknown:
        raise ValueError(f"no role for {len(unknown)} checkpoint tensors, e.g. {unknown[:5]}")
    want = sorted(by_layer) if layers is None else [int(x) for x in layers]
    meta = {"format": FORMAT, "version": VERSION, "recipe": recipe,
            "recipe_spec": {k: asdict(v) for k, v in rspec.items()}, "source": os.path.abspath(model_dir),
            "stats": {"dir": os.path.abspath(stats_dir),
                      "meta_sha256": (_file_sha(os.path.join(stats_dir, "meta.json"))
                                      if os.path.exists(os.path.join(stats_dir, "meta.json")) else None)},
            "gptq_path": gptq_path, "device": str(device), "layers": want, "tensors": {}, "bf16": [],
            "skipped_expert_tensors": 0, "tie_word_embeddings": tie_word_embeddings,
            "engine_sha256": {f: _file_sha(os.path.join(os.path.dirname(__file__), f))[:16]
                              for f in ("gptq.py", "kquant.py", "dense_recipe.py")}}
    times: dict[str, float] = {}

    def do_group(names: list[str], stats: dict | None, fname: str, extra: list | None = None):
        out: dict[str, torch.Tensor] = {}
        prep_cache: dict = {}
        weights = [n for n in names if n.endswith(".weight")]
        biases = {n[:-5]: n for n in names if n.endswith(".bias")}
        for name in sorted(weights) + (extra or []):
            if isinstance(name, tuple):  # (module, role, source tensor name): a tied head
                module, role, src = name
            else:
                module, role, src = name[:-7], role_of[name], name
            if role == "expert":
                meta["skipped_expert_tensors"] += 1
                continue
            fmt = BF16 if role == "bf16" else recipe_format(rspec, role)
            if fmt.kind == "bf16":
                meta["bf16"].append(module)
                continue
            W = ck.get(src)
            if W.dim() != 2:
                raise ValueError(f"{src}: {W.dim()}-D weight for role {role}")
            entry = None if stats is None else stats.get(short_name(module))
            info: dict = {}
            torch.cuda.synchronize(device)
            t0 = time.time()
            kp = quantize_tensor(W, fmt, entry, device=device, gptq_path=gptq_path, prep_cache=prep_cache, info=info)
            torch.cuda.synchronize(device)
            dt = time.time() - t0
            times[role] = times.get(role, 0.0) + dt
            for f in ("qweight", "sb", "d", "dmin"):
                out[f"{module}.{f}"] = getattr(kp, f).cpu()
            if module in biases:
                out[f"{module}.bias"] = ck.get(biases[module])
            n, k = W.shape
            meta["tensors"][module] = {"role": role, "file": fname, "shape": [n, k], "bits": fmt.bits,
                                       "group": fmt.group, "gptq": fmt.gptq, "imp": fmt.imp,
                                       "bits_total": fmt.bits_total(n, k), "fallback_groups": kp.fallback,
                                       "bias": module in biases, "seconds": round(dt, 3),
                                       "info": {kk: (round(v, 4) if isinstance(v, float) else v)
                                                for kk, v in info.items()}}
            del W, kp
        _save(out, os.path.join(out_dir, fname), {"format": FORMAT, "version": str(VERSION), "recipe": recipe})
        return len(out)

    for i, li in enumerate(want):
        t0 = time.time()
        p = os.path.join(stats_dir, f"layer{li:03d}.pt")
        stats = torch.load(p, map_location="cpu") if os.path.exists(p) else None
        do_group(by_layer[li], stats, f"layer{li:03d}.safetensors")
        del stats
        log(f"dense layer {li} ({i + 1}/{len(want)}) in {time.time() - t0:.1f}s; elapsed {time.time() - t_start:.0f}s")
    if model_level and top:
        t0 = time.time()
        p = os.path.join(stats_dir, "model.pt")
        mstats = torch.load(p, map_location="cpu") if os.path.exists(p) else None
        extra = []
        roles_top = {role_of[n] for n in top}
        if tie_word_embeddings and "lm_head" not in roles_top:
            emb = next(n for n in top if role_of[n] == "embed" and n.endswith(".weight"))
            extra.append(("lm_head", "lm_head", emb))
        do_group(top, mstats, "model.safetensors", extra)
        log(f"model-level tensors in {time.time() - t0:.1f}s")
    meta["timing"] = {"total_s": round(time.time() - t_start, 1),
                      "quant_s_per_role": {k: round(v, 2) for k, v in sorted(times.items())}}
    tmp = os.path.join(out_dir, f"meta.json.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "meta.json"))
    return meta


# ------------------------------------------------------------------------------------------------ reading

class PackedDense:
    """Reader of a convert_dense directory: get(module) -> (KPacked, bias or None), one file cached at a time."""

    def __init__(self, path: str):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} directory")
        self._file = None
        self._tensors: dict = {}

    def modules(self) -> list[str]:
        return list(self.meta["tensors"])

    def __contains__(self, module: str) -> bool:
        return module in self.meta["tensors"]

    def get(self, module: str, device="cpu") -> tuple[kquant.KPacked, torch.Tensor | None]:
        from safetensors.torch import load_file

        e = self.meta["tensors"][module]
        if self._file != e["file"]:
            self._tensors = load_file(os.path.join(self.path, e["file"]))
            self._file = e["file"]
        t = {f: self._tensors[f"{module}.{f}"].to(device) for f in ("qweight", "sb", "d", "dmin")}
        bias = self._tensors.get(f"{module}.bias")
        kp = kquant.KPacked(t["qweight"], t["sb"], t["d"], t["dmin"], e["bits"], e["group"], e["fallback_groups"])
        n, k = e["shape"]
        if kp.qweight.shape != (n, k * kp.bits // 8) or kp.sb.shape != (n, k // kp.group):
            raise ValueError(f"{self.path}: {module} packed shapes do not match {e['shape']}")
        return kp, (None if bias is None else bias.to(device))

    def module(self, name: str, device="cuda", host_embed: bool = False, **kw):
        """The engine module for `name`: QEmbedding for the embed role, else KLinear (kw: KLinear options)."""
        kp, bias = self.get(name, device)
        if self.meta["tensors"][name]["role"] == "embed":
            from .qembed import QEmbedding

            return QEmbedding.from_packed(kp, host=host_embed, device=device)
        from .klinear import KLinear

        return KLinear.from_packed(kp, bias, device=device, **kw)


_PLUGIN: dict = {}


@torch.no_grad()
def qbench_plugin(layer, name, w, stats, packed: str, **_kw):
    """tools/qbench.py plugin: the dequantized weight of `name` from a convert_dense directory (float32, the
    format's math) and its stored bits. --attn/--lm-head/--embed/--router py:behemoe.dense_recipe:qbench_plugin@
    '{"packed": DIR}'."""
    if _PLUGIN.get("path") != packed:
        _PLUGIN.update(path=packed, reader=PackedDense(packed))
    rd = _PLUGIN["reader"]
    module = re.sub(r"\.weight$", "", name)
    if module not in rd:
        raise KeyError(f"{packed} has no packed tensor for {name}")
    kp, _bias = rd.get(module, w.device)
    w_hat = kquant.dequantize_packed(kp, torch.float32)
    if tuple(w_hat.shape) != tuple(w.shape):
        raise RuntimeError(f"{name}: packed {tuple(w_hat.shape)} vs weight {tuple(w.shape)}")
    return w_hat, 8 * kp.nbytes()


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Convert a checkpoint's dense part under a recipe (streaming, per layer)")
    ap.add_argument("--model", required=True, help="checkpoint directory (safetensors)")
    ap.add_argument("--stats", required=True, help="tools/dense_stats.py output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="dense-c", choices=sorted(RECIPES))
    ap.add_argument("--gptq-path", default="cpu", choices=("cpu", "gpu"))
    ap.add_argument("--layers", default=None, help="e.g. 0-3,47 (default: all)")
    ap.add_argument("--no-model-level", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    layers = None
    if args.layers:
        layers = []
        for part in args.layers.split(","):
            a, _, b = part.partition("-")
            layers += list(range(int(a), int(b) + 1)) if b else [int(a)]

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    meta = convert_dense(args.model, args.stats, args.out, args.recipe, device=args.device, gptq_path=args.gptq_path,
                         layers=layers, model_level=not args.no_model_level, log=log)
    paths = {}
    for e in meta["tensors"].values():
        p = e["info"].get("path")
        if p:
            paths[p] = paths.get(p, 0) + 1
    log(f"done: {len(meta['tensors'])} packed tensors, {len(meta['bf16'])} bf16, "
        f"{meta['skipped_expert_tensors']} expert tensors skipped; GPTQ factorization paths {paths}; "
        f"timing {meta['timing']}")


if __name__ == "__main__":
    main()
