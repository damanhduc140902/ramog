"""Persistent Triton autotuning results: the second run of the engine skips the kernel benchmarking.

Every autotuned kernel of moe_tiering.kernels picks its launch configuration per key (shape, bit width, format, pairs,
mode) by timing all candidate configurations on its first call, which made warm-up take about a minute. The
choices are saved to a JSON file in the Triton cache directory (TRITON_CACHE_DIR, default ~/.triton/cache) under
moe-tiering/, one file per GPU and software version: the file name and the tag inside it carry the GPU (name, or
architecture, compute units and memory where the driver reports no name), the engine version, the Triton and
torch versions and a hash of moe_tiering/kernels.py, so another GPU, an engine update or a kernel edit starts afresh.
Loading puts the saved choices into the autotuners' in-memory caches, so a known key launches its saved
configuration at once (its compiled binary comes from Triton's own cache); keys never seen are tuned as before and
saved by the next save(). The choice only affects speed, never results.

MOE_TIERING_TUNE_DIR overrides the directory; MOE_TIERING_TUNE_CACHE=0 disables loading and saving. The host
link is not part of the key: after moving the same GPU to a machine with a different PCIe link, delete the file
(or set the variable) to retune the host-memory kernels.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import threading

import torch
import triton

from . import __version__ as ENGINE_VERSION
from . import kernels

_lock = threading.Lock()
_state: dict = {"path": None, "tag": None, "saved": 0, "loaded": 0}


def _enabled() -> bool:
    return os.environ.get("MOE_TIERING_TUNE_CACHE", "1") != "0"


def tuners() -> dict:
    """name -> triton Autotuner of every autotuned kernel the engine launches."""
    out = {}
    for name in ("_grouped_gemv", "_grouped_gemv_host", "_moe_gemm", "_moe_gemm_host"):
        t = getattr(kernels, name, None)
        if t is not None and hasattr(t, "cache"):
            out[name] = t
    return out


def _cache_dir() -> str:
    d = os.environ.get("MOE_TIERING_TUNE_DIR")
    if d:
        return d
    base = None
    try:
        from triton import knobs

        base = knobs.cache.dir
    except Exception:  # older Triton: the environment variable or its default
        base = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    return os.path.join(base, "moe-tiering")


def device_name(device=None) -> str:
    """The GPU's name; where the driver reports none (some ROCm builds), architecture, compute units and memory."""
    p = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    name = (p.name or "").strip()
    if name:
        return name
    arch = (getattr(p, "gcnArchName", "") or f"sm_{p.major}{p.minor}").split(":")[0]
    return f"{arch} {p.multi_processor_count}CU {round(p.total_memory / 2**30)}GiB"


def _kernels_hash() -> str:
    with open(kernels.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _locate(device=None) -> tuple[str, str]:
    gpu = device_name(device)
    tag = (f"gpu={gpu}; engine={ENGINE_VERSION}; triton={triton.__version__}; torch={torch.__version__}; "
           f"kernels={_kernels_hash()}")
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", gpu).strip("-")[:48]
    h = hashlib.sha256(tag.encode()).hexdigest()[:10]
    return os.path.join(_cache_dir(), f"autotune-{slug}-engine{ENGINE_VERSION}-{h}.json"), tag


def _config_dict(c) -> dict:
    out = {"kwargs": dict(c.kwargs), "num_warps": c.num_warps, "num_stages": c.num_stages}
    for k in ("num_ctas", "maxnreg"):
        if getattr(c, k, None) is not None:
            out[k] = getattr(c, k)
    return out


def _config(d: dict):
    extra = {k: d[k] for k in ("num_ctas", "maxnreg") if k in d}
    try:
        return triton.Config(dict(d["kwargs"]), num_warps=d["num_warps"], num_stages=d["num_stages"], **extra)
    except TypeError:  # a Triton without one of the extra fields
        return triton.Config(dict(d["kwargs"]), num_warps=d["num_warps"], num_stages=d["num_stages"])


def _read(path: str, tag: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data.get("entries", {}) if data.get("tag") == tag else {}


def _count() -> int:
    return sum(len(t.cache) for t in tuners().values())


def load(device=None) -> int:
    """Put the saved choices for this GPU and versions into the autotuners. Returns how many were loaded."""
    if not _enabled():
        return 0
    with _lock:
        path, tag = _locate(device)
        _state.update(path=path, tag=tag)
        n = 0
        tn = tuners()
        for name, items in _read(path, tag).items():
            t = tn.get(name)
            if t is None:
                continue
            for key, cfg in items:
                k = tuple(key)
                if k not in t.cache:
                    t.cache[k] = _config(cfg)
                    n += 1
        _state["loaded"] += n
        _state["saved"] = _count()
        return n


def save() -> int:
    """Merge the autotuners' current choices into the file (atomic replace). Returns the number of entries
    written, 0 when nothing new was tuned since the last load/save (or the cache is disabled)."""
    if not _enabled() or _state["path"] is None:
        return 0
    with _lock:
        if _count() <= _state["saved"]:
            return 0
        path, tag = _state["path"], _state["tag"]
        entries = _read(path, tag)
        merged = {name: {json.dumps(list(k)): cfg for k, cfg in items} for name, items in entries.items()}
        for name, t in tuners().items():
            m = merged.setdefault(name, {})
            for k, c in t.cache.items():
                m[json.dumps(list(k))] = _config_dict(c)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        nl = "\n"
        with open(tmp, "w") as f:  # one entry per line: [key, config]
            f.write('{"tag": ' + json.dumps(tag) + ', "entries": {')
            for i, (name, m) in enumerate(sorted(merged.items())):
                rows = ("," + nl + "  ").join(f"[{k}, {json.dumps(cfg)}]" for k, cfg in sorted(m.items()))
                f.write(("," if i else "") + nl + " " + json.dumps(name) + ": [" + nl + "  " + rows + "]")
            f.write(nl + "}}" + nl)
        os.replace(tmp, path)
        _state["saved"] = _count()
        return sum(len(m) for m in merged.values())


def path() -> str | None:
    return _state["path"]


atexit.register(save)
