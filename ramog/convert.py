"""Streaming converter entry point: Hugging Face checkpoint -> tiered engine store, one layer at a time.

Since the WP9 switch-over (design-core C9) this module delegates to ramog.convert_any, the generic converter of
every architecture in ramog.arch.ARCHES (design-core B4; WP5). For Qwen3-MoE with the same options convert_any
writes the same bytes as this module's v0.1 converter did: the same chunks, the same BlobLayout.quantize calls on
the same bf16 values, the same importance arithmetic (tests/test_convert_equiv_qwen3.py checks layer blobs and dense
tensors against a store the v0.1 converter wrote). The only difference in the output is meta.json, now format 2 (v1's keys
plus arch, transformers, moe_layers, experts, ...); ramog.loader reads both, and so does the legacy block builder.
Output layout:
    <out>/meta.json                 format 2; hot / cold blob layouts {"hot": {format, bits, group, nbytes} | null, ...}
    <out>/dense.safetensors         non-expert weights under the model's key names
    <out>/layer{L:03d}.hot.bin      uint8 [E, hot_nbytes]   (not written with --cold-only)
    <out>/layer{L:03d}.cold.bin     uint8 [E, cold_nbytes]
    <out>/layer{L:03d}.json         how the layer's blobs were made (layouts, stats, seconds); checked on resume
Cold formats: rtn (ramog.quant) or k4row (ramog.kquant, importance-weighted with --stats). A --cold-only store
holds no hot blobs and must be built with hot_from_cold (--hot-from-cold).

One process (writes everything, resumable):
    python -m ramog.convert --model models/Qwen3-235B-A22B-Instruct-2507 --out stores/qwen3-235b
    python -m ramog.convert --model M --out D --cold-format k4row --cold-group 16 --stats STATS_DIR
Split across processes / GPUs (no shared file is written until the finalize step, so nothing races):
    HIP_VISIBLE_DEVICES=0 python -m ramog.convert <same options> --layers 0-46  &
    HIP_VISIBLE_DEVICES=1 python -m ramog.convert <same options> --layers 47-93 &
    wait; python -m ramog.convert <same options> --finalize    (checks every layer, writes dense + meta once)
The command line is ramog.convert_any's (a superset of the v0.1 flags: --arch, --hot-format, --dense-format,
--plugin). Shards, parse_layers, load_importance and EXPERT_RE stay importable for the tools that use them.
"""

from __future__ import annotations

import json
import os
import re

from safetensors import safe_open

from .convert_any import convert as _convert_any
from .convert_any import load_importance, parse_layers  # noqa: F401  (ramog.convert's names, same arithmetic)
from .convert_any import main as _main

EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class Shards:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def get(self, name: str):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.dir, f), "pt")
        return self.handles[f].get_tensor(name)


def convert(model_dir: str, out_dir: str, device="cuda:0", hot_bits=4, cold_bits=2, group=64, chunk=16,
            layers: list[int] | None = None, log=print, *, cold_format: str = "rtn", cold_group: int | None = None,
            stats_dir: str | None = None, cold_only: bool = False, finalize: bool | None = None) -> dict:
    """Convert `layers` (default: all) of the checkpoint (ramog.convert_any.convert with the v0.1 arguments); group
    is the hot group size (and the rtn cold one unless cold_group is given; k4row defaults to group 16). finalize:
    write dense.safetensors and meta.json after checking that every layer is present (default: True for a full run,
    False when `layers` is given). Returns convert_any's summary."""
    return _convert_any(model_dir, out_dir, device=device, hot_bits=hot_bits, cold_bits=cold_bits, group=group,
                        chunk=chunk, layers=layers, log=log, cold_format=cold_format, cold_group=cold_group,
                        stats_dir=stats_dir, cold_only=cold_only, finalize=finalize)


def main(argv=None):
    _main(argv)


if __name__ == "__main__":
    main()
