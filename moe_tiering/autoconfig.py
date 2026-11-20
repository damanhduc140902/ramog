"""Auto-config, rule v3 (design-quality WP13, section 4; coordinator amendments 3-5).

    cfg = choose(facts, variants, vram_gib, ram_gib, mem_available_gib=None, nvme=None)
        -> Config(variant, dense, mode, slots, frozen, hot_placement, ...)
    cfg = plan(stores, vram_gib, ram_gib=None, ...)        (facts and variants read from store-v2 directories)
    model, info = build(cfg, device, vram_gib)              (moe_tiering.loader.build_engine with the chosen placement)

Rule v3 is agents/qint/qd/autoconfig.py decide() (the reference; results/qint/qd/autoconfig.json is its decision
table) with its evidence frozen below (EVIDENCE: the four set KLs of every configuration it can pick, measured on
Qwen3-30B, with the result file each came from):
  1. Budgets, per dense recipe d the store ships (dense-hi, dense-c): the VRAM cache is the ENGINE'S OWN sizing
     (amendment 5): moe_tiering.loader.cache_slots(VRAM, dense_d, blobs) = (VRAM - dense part - one staging blob (the
     largest cold blob) - max(0.75 GiB, 8% VRAM)) // (sum over the MoE layers of the blob size). The pinned budget is
     RAM - RAM_RESERVE (5.5 GiB = 3 of OS + ENGINE_RSS_GIB 2.5 of engine process outside the pinned blobs), capped at
     MemAvailable - ENGINE_RSS_GIB when it is known (read at start: /proc/meminfo on Linux, GlobalMemoryStatusEx
     ullAvailPhys on Windows; mem_available_gib()). RAM is the machine's size in GiB (--ram-gb; default MemTotal).
     The design's 4.5 / 1.5 (a laptop measurement) are refitted to the real loader's peak (WP13 fix: at 4.5 both
     16 GB cells were OOM-killed in a 13 GiB cgroup; results/qlane/wp13/fix/).
  2. Candidates: every variant V present (trellis rungs TQ2.0..TQ4.0, H4) x every recipe d, placed by capacity
     (hot from cold, VRAM and pinned RAM additive):
       resident  slots >= E: every expert in VRAM, nothing pinned (H4: "all-hot"; build_engine(frozen=E))
       cached    experts <= pinned budget: every expert pinned, the cache dynamic (build_engine(frozen=None))
       split     otherwise, F = ceil((experts - pinned) / slot cost) <= slots: F experts per layer frozen in VRAM,
                 E - F pinned, the other slots dynamic (build_engine(frozen=F))
     each carrying V + d's measured one-tier KL (H4: the every-expert-H4 run). Two tiers (WP18, optional, NOT
     BUILT): a rung wholly pinned + H4 copies of the cached experts, only where the facts say a hot tier exists
     (the decision grid; no store has one).
  3. Choice on the tuning sets (DEV + OOD-DEV mean); HELD and OOD-TEST are reported. The lowest score; candidates
     within 2% are tied and go to the simpler placement (resident < all-hot < cached < split; pinned H4 < H4 from
     NVMe), then the higher variant. Two tiers only when >= 10% better (hot "auto"); hot "on" drops the margin,
     "off" never builds them.
  4. Host fallback (amendment 4): when no candidate exists with the dense part in VRAM, the input embedding (and the
     lm_head where the engine has a host variant for it: the rtn QuantLinear head, not KLinear, so never for the
     dense-q recipes today) moves to pinned host memory (embed_host / head_host: VRAM - those bytes, pinned budget -
     those bytes), and the rule runs again. Only then it refuses.
  5. Refusal: names the RAM that would fit the smallest variant at this VRAM (or the VRAM the dense part needs).

sizing="qd" reproduces the reference's own arithmetic (GiB sizes rounded as in sizes.json, cache = VRAM - dense -
reserve without the staging blob, no host fallback, the reference's 4.5 / 1.5 GiB RAM reserves), so
tests/test_autoconfig.py can check this module against the reference's decision table cell for cell;
sizing="engine" (the default) is the rule the engine runs.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import asdict, dataclass, field

GiB = 1024**3
OS_RESERVE_GIB = 3.0       # GiB of OS + desktop kept free on the machine (design-quality 1.1, an assumption)
# GiB of engine process outside the pinned blobs at PEAK (refit, WP13 fix): process RSS minus page-locked bytes on the
# real loader + runners (MI250 / ROCm, 30B): 1.34-1.48 right after the build, 2.07-2.09 at the end of the 4-set
# quality runs, 2.35 in bench_graph (results/qlane/wp13/fix/). The design's 1.5 (laptop, older engine) let the 8 / 16
# and 4 / 16 cells be OOM-killed on a 16 GB machine (a 13 GiB cgroup = 16 GB less the 3 GiB of OS).
ENGINE_RSS_GIB = 2.5       # also subtracted from MemAvailable when it caps the pinned budget
RAM_RESERVE = OS_RESERVE_GIB + ENGINE_RSS_GIB   # 5.5: pinned budget = RAM - RAM_RESERVE
QD_RAM_RESERVE, QD_ENGINE_RSS_GIB = 4.5, 1.5    # the reference's (agents/qint/qd/autoconfig.py): sizing "qd" only
NOISE_BAND = 0.02          # score differences below 2% are noise (encoder re-runs differ by up to ~2-3%)
HOT_MARGIN = 0.10          # a two-tier candidate must score >= 10% below the best one-tier one
HOT_MIN_FRACTION = 0.10    # two tiers only when the 4-bit cache holds >= 10% of a layer's experts
RUNGS = ("TQ2.0", "TQ2.26", "TQ2.57", "TQ3.0", "TQ3.5", "TQ4.0")
VARIANTS = RUNGS + ("H4",)
DENSES = ("dense-hi", "dense-c")     # the dense recipes Phase B ships (dense-a waits for WP10b's NL4 kernels)
SETS = ("dev", "held", "ood-dev", "ood-test")
SCORE_SETS = ("dev", "ood-dev")      # the tuning sets: the rule chooses on these
_SIMPLE = {"resident": 0, "all-hot": 1, "cached": 2, "split": 3}

# rule v3's evidence, frozen from the committed result JSONs (agents/qlane/wp13/freeze_evidence.py prints this from
# agents/qint/qd/autoconfig.py evidence(); tests/test_autoconfig.py checks it is still equal to the bit)
EVIDENCE = {
    # one tier, whole model (trellis rung + dense recipe): (DEV, HELD, OOD-DEV, OOD-TEST) KL, source
    "one": {
        ('TQ2.0', 'dense-a'): ((0.14375978214673338, 0.1387486004170446, 0.16957867037639693, 0.13318305923921134), 'results/qint/qd/qb/e-T20-densea.json'),
        ('TQ2.26', 'dense-a'): ((0.10078491742303228, 0.08652969223277346, 0.11149497867116637, 0.08274497942754482), 'results/quality/t4/qb/fin-C2262-t1densea.json'),
        ('TQ2.57', 'dense-a'): ((0.06698518014048602, 0.060030202158804016, 0.058434847591082265, 0.04756569484252201), 'results/quality/combine/qb/cmb-A-densea.json'),
        ('TQ3.0', 'dense-a'): ((0.041920513281964854, 0.03575111264740914, 0.03512605924795734, 0.02918191287999893), 'results/qint/qd/qb/e-T30-densea.json'),
        ('TQ3.5', 'dense-a'): ((0.02511173456789608, 0.019226422942924974, 0.019044174876150666, 0.017516909980435952), 'results/qint/qd/qb/e-T35-densea.json'),
        ('TQ4.0', 'dense-a'): ((0.017191771351605677, 0.013624238953260728, 0.014662203973292365, 0.013858007954001998), 'results/qint/qd/qb/e-T40-densea.json'),
        ('TQ2.0', 'dense-hi'): ((0.1430341985393892, 0.13700765875741466, 0.1686677047085447, 0.1319390036769355), 'results/qint/qd/qb/e-T20-densehi.json'),
        ('TQ2.26', 'dense-hi'): ((0.09932475156551279, 0.08530962343834918, 0.10774974294315479, 0.08111237773773672), 'results/qint/qd/qb/e-C-densehi.json'),
        ('TQ2.57', 'dense-hi'): ((0.06672146779740931, 0.05677131000132256, 0.05613193337032856, 0.0455613364002673), 'results/qint/qd/qb/e-A-densehi.json'),
        ('TQ3.0', 'dense-hi'): ((0.03995895527494681, 0.03248641600676567, 0.03257244889916899, 0.026592114191724643), 'results/qint/qd/qb/e-T30-densehi.json'),
        ('TQ3.5', 'dense-hi'): ((0.021510247567932485, 0.017388936993660998, 0.017172611456073853, 0.015471001279561957), 'results/qint/qd/qb/e-T35-densehi.json'),
        ('TQ4.0', 'dense-hi'): ((0.014834724232684723, 0.010414417220569603, 0.012565488962691804, 0.011674037413336754), 'results/qint/qd/qb/e-T40-densehi.json'),
        ('TQ2.0', 'dense-c'): ((0.1423222867457798, 0.14060406550446072, 0.17086443515980368, 0.13392792166743842), 'results/qint/qd/qb/e-T20-densec.json'),
        ('TQ2.26', 'dense-c'): ((0.10131981054304388, 0.0876979866553641, 0.11229863230257439, 0.08393532982052299), 'results/qint/qd/qb/e-C-densec.json'),
        ('TQ2.57', 'dense-c'): ((0.06793031425413604, 0.06101282006564317, 0.060477086442170855, 0.04940123371129234), 'results/quality/combine/qb/cmb-A-densec.json'),
        ('TQ3.0', 'dense-c'): ((0.04253271163691812, 0.03634208689663305, 0.03629714979507758, 0.030524818785670073), 'results/qint/qd/qb/e-T30-densec.json'),
        ('TQ3.5', 'dense-c'): ((0.023596434741718686, 0.020970456088284333, 0.021227115781087616, 0.019520235070321433), 'results/qint/qd/qb/e-T35-densec.json'),
        ('TQ4.0', 'dense-c'): ((0.018427423874522564, 0.013230611734525735, 0.01604782878686442, 0.015171436691864515), 'results/qint/qd/qb/e-T40-densec.json'),
    },
    # H4 one tier = every expert at H4 (hot-U-hess-*), wherever its blobs sit
    "all_hot": {
        'dense-a': ((0.013318808208710578, 0.012245805559946813, 0.013383642820479207, 0.012455075657380849), 'results/qint/qd/qb/hot-U-hess-densea.json'),
        'dense-hi': ((0.012346343213897085, 0.010336734660919958, 0.011182507731732192, 0.010443356314151567), 'results/qint/qd/qb/hot-U-hess-densehi.json'),
        'dense-c': ((0.014088450147491036, 0.014222890375383313, 0.015330198640774126, 0.013952999529246439), 'results/qint/qd/qb/hot-U-hess-densec.json'),
    },
    # every expert at the combine store's k4row-4 blobs (the f = 1 end of the k4 two-tier replays)
    "all_k4": {
        'dense-a': ((0.014171306145915339, 0.014168321634252148, 0.013735571593233234, 0.012835826293363155), 'results/quality/combine/qb/cmb-allhot-densea.json'),
        'dense-hi': ((0.010620667275469247, 0.011049145289773757, 0.011854381434307195, 0.010746949979589197), 'results/qint/qd/qb/allhot-k4-dh.json'),
    },
    # two tiers (WP18, optional, not built): replays at 19 / 53 / 84 of 128 slots; hot = "H4" or "k4" blobs
    "two": {
        ('TQ2.0', 'dense-a'): {"hot": 'k4', "pts": {
            19: ((0.07370395452710407, 0.06824996114932552, 0.055425908082568025, 0.03904113507453905), 'results/qint/qd/qb/rp-T20-s19.json'),
            53: ((0.0427999429149971, 0.035042713366250366, 0.023599672330522584, 0.022027068286727874), 'results/qint/qd/qb/rp-T20-s53.json'),
            84: ((0.024922837563658585, 0.01943462452427152, 0.016879605586946598, 0.01604965942315971), 'results/qint/qd/qb/rp-T20-s84.json'),
        }},
        ('TQ2.26', 'dense-a'): {"hot": 'k4', "pts": {
            19: ((0.05848373903633604, 0.05159173823015214, 0.03788990211135685, 0.031389239567729224), 'results/qint/qd/qb/rp-C-s19.json'),
            53: ((0.03373152879125093, 0.02829068267401564, 0.020569278627120283, 0.018957283502368735), 'results/qint/qd/qb/rp-C-s53.json'),
            84: ((0.02155309519300385, 0.018461589737177268, 0.016332538647964148, 0.014663704852997112), 'results/qint/qd/qb/rp-C-s84.json'),
        }},
        ('TQ2.57', 'dense-a'): {"hot": 'k4', "pts": {
            19: ((0.036626697340685854, 0.038416407088064715, 0.025534720058318465, 0.021694168070260277), 'results/qint/qd/qb/rp-A-s19.json'),
            53: ((0.02422057791468852, 0.02263544960784481, 0.016455004500376777, 0.015831849812825118), 'results/qint/qd/qb/rp-A-s53.json'),
            84: ((0.01572491016358609, 0.015926466373453563, 0.015096026188091543, 0.013626342913629783), 'results/qint/qd/qb/rp-A-s84.json'),
        }},
        ('TQ3.0', 'dense-a'): {"hot": 'k4', "pts": {
            19: ((0.026922116735294717, 0.02561180905392415, 0.020521683107328645, 0.01649056681050958), 'results/qint/qd/qb/rp-T30-s19.json'),
            53: ((0.01933883000888269, 0.01701382270052918, 0.015420348323178297, 0.013759788889966117), 'results/qint/qd/qb/rp-T30-s53.json'),
            84: ((0.014495788997817713, 0.01449319046382654, 0.014359457853380454, 0.012987928031440159), 'results/qint/qd/qb/rp-T30-s84.json'),
        }},
        ('TQ3.0', 'dense-hi'): {"hot": 'H4', "pts": {
            19: ((0.02278300485285628, 0.021506086683944332, 0.01788920322679449, 0.014230824722823242), 'results/qint/qd/qb/rp-T30-s19-unrot-dh.json'),
            53: ((0.015246682702828538, 0.015296389630878305, 0.013134573013683055, 0.011329397266291574), 'results/qint/qd/qb/rp-T30-s53-unrot-dh.json'),
            84: ((0.013914721243059525, 0.012297328585218929, 0.011669185160784135, 0.01067571082053768), 'results/qint/qd/qb/rp-T30-s84-unrot-dh.json'),
        }},
        ('TQ3.5', 'dense-a'): {"hot": 'k4', "pts": {
            19: ((0.018749156063441307, 0.01681972381685981, 0.015441536736025342, 0.014176793897738075), 'results/qint/qd/qb/rp-T35-s19.json'),
            53: ((0.01541496095240093, 0.014570945647395203, 0.014085318124042973, 0.012964847337846851), 'results/qint/qd/qb/rp-T35-s53.json'),
            84: ((0.013441043695901192, 0.013883413396676115, 0.014080586235451938, 0.013092989652944205), 'results/qint/qd/qb/rp-T35-s84.json'),
        }},
        ('TQ3.5', 'dense-hi'): {"hot": 'H4', "pts": {
            19: ((0.01616604498801352, 0.01281996743741931, 0.01322685954522837, 0.011473330401521986), 'results/qint/qd/qb/rp-T35-s19-unrot-dh.json'),
            53: ((0.012790555515009034, 0.011520533983156672, 0.011495229915269835, 0.010395442572463767), 'results/qint/qd/qb/rp-T35-s53-unrot-dh.json'),
            84: ((0.01244353515204526, 0.010988868029218063, 0.011370637016693065, 0.010568799869403966), 'results/qint/qd/qb/rp-T35-s84-unrot-dh.json'),
        }},
        ('TQ4.0', 'dense-a'): {"hot": 'k4', "pts": {
            19: ((0.01521760778957679, 0.014356578112597121, 0.013745128418456688, 0.013321581864436039), 'results/qint/qd/qb/rp-T40-s19.json'),
            53: ((0.014328139566801013, 0.014393379869485764, 0.013517800919821011, 0.012495476655881667), 'results/qint/qd/qb/rp-T40-s53.json'),
            84: ((0.014131498996205836, 0.013401194725918619, 0.01399619920463514, 0.012682580299732559), 'results/qint/qd/qb/rp-T40-s84.json'),
        }},
        ('TQ4.0', 'dense-hi'): {"hot": 'H4', "pts": {
            19: ((0.011951101466255942, 0.011429022127876578, 0.011188575639158944, 0.010421063450293718), 'results/qint/qd/qb/rp-T40-s19-unrot-dh.json'),
            53: ((0.011648138419642216, 0.01048335478855827, 0.011081398020290102, 0.009984126589795089), 'results/qint/qd/qb/rp-T40-s53-unrot-dh.json'),
            84: ((0.011138901456341409, 0.010434059152364807, 0.011349524802653095, 0.010648664387833504), 'results/qint/qd/qb/rp-T40-s84-unrot-dh.json'),
        }},
    },
}


def _mean(kls) -> float:   # the reference's arithmetic, in its order (sum of the four sets / 4)
    return sum(kls) / 4


def _score_of(kls) -> float:  # mean of DEV and OOD-DEV
    return sum(kls[SETS.index(s)] for s in SCORE_SETS) / len(SCORE_SETS)


def one_tier(variant: str, dense: str):
    """(4-set mean, tuning-set score, the four set KLs, source) of variant V + recipe d one tier, or None."""
    e = EVIDENCE["all_hot"].get(dense) if variant == "H4" else EVIDENCE["one"].get((variant, dense))
    if e is None:
        return None
    return _mean(e[0]), _score_of(e[0]), e[0], e[1]


def _interp(pts: dict, x: float) -> float:
    xs = sorted(pts)
    if x <= xs[0]:
        return pts[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if x <= b:
            return pts[a] + (pts[b] - pts[a]) * (x - a) / (b - a)
    return pts[xs[-1]]


def two_tier(rung: str, dense: str, frac: float):
    """(4-set mean, score, source) of rung pinned + 4-bit copies at cache fraction frac (the reference's
    interpolation between the measured replays), or None (not measured)."""
    t = EVIDENCE["two"].get((rung, dense))
    one = EVIDENCE["one"].get((rung, dense))
    end = (EVIDENCE["all_hot"] if t and t["hot"] == "H4" else EVIDENCE["all_k4"]).get(dense)
    if t is None or one is None or end is None:
        return None
    out = []
    for f in (_mean, _score_of):
        pts = {0.0: f(one[0]), 1.0: f(end[0])}
        pts.update({s / 128: f(v[0]) for s, v in t["pts"].items()})
        out.append(_interp(pts, frac))
    exact = [s for s in t["pts"] if s / 128 == frac]
    src = t["pts"][exact[0]][1] if exact else f"interpolated in the cache fraction ({t['hot']} replays)"
    return out[0], out[1], src


# ---------------------------------------------------------------------------------------------------------------------
# facts


@dataclass
class DenseFacts:
    """One dense recipe of the model as the engine holds it."""
    recipe: str
    vram_bytes: int           # VRAM the engine's sizing counts as "dense" (attach: memory_allocated before the cache)
    embed_bytes: int = 0      # bytes embed_host moves from VRAM to pinned host memory
    head_bytes: int = 0       # bytes head_host would move (only where head_host_ok)
    head_host_ok: bool = False  # the engine has a host variant of this recipe's lm_head (rtn QuantLinear: yes,
    #                             KLinear: no, so False for every dense-q recipe today)
    gib: float | None = None  # sizing="qd": the reference's GiB figure (sizes.json)


@dataclass
class Variant:
    """One variant present: a store-v2 directory (or a grid entry)."""
    name: str
    nbytes: tuple             # cold blob bytes of every MoE layer (hot from cold: the cache holds the same blobs)
    store: str | None = None
    dense: tuple = DENSES     # recipes this store ships (meta.json "dense_q")
    counts: bool = True       # counts.safetensors present (a split needs it)
    gib: float | None = None  # sizing="qd": the reference's experts GiB (sizes.json, rounded)
    dense_facts: dict | None = None  # {recipe: DenseFacts} of this store's own dense parts (default: Facts.dense)

    def experts_bytes(self, E: int) -> int:
        return E * sum(self.nbytes)


@dataclass
class Facts:
    model: str
    E: int
    dense: dict               # {recipe: DenseFacts}
    two_tier: bool = False    # a WP18 hot tier (H4 copies over a pinned rung) can be built: the decision grid only
    sizing: str = "engine"    # "engine" (loader.cache_slots on bytes) | "qd" (the reference's GiB arithmetic)
    qd_dense_c5: float | None = None  # sizing="qd": dense at 5 bpw (the reference's refusal test)


@dataclass
class Config:
    fit: bool
    variant: str | None = None
    dense: str | None = None
    mode: str | None = None          # "all-hot" (H4 resident) | "resident" | "cached" (pinned) | "split"
    slots: int = 0                   # cache slots per layer (capped at E)
    frozen: int = 0                  # F experts per layer frozen in VRAM (E for resident)
    dynamic: int = 0
    hot_placement: str | None = None  # two tiers (WP18, not built): "pinned" | "nvme"
    hot_slots: int = 0
    embed_host: bool = False
    head_host: bool = False
    store: str | None = None
    pinned_bytes: int = 0
    pinned_budget_bytes: int = 0
    dense_bytes: int = 0             # the dense part in VRAM after any host fallback
    cache_bytes: int = 0             # VRAM left for the cache (cache_slots "free")
    slot_cost: int = 0
    experts_bytes: int = 0
    kl: float | None = None          # 30B 4-set mean KL of this configuration (reported)
    score: float | None = None       # 30B DEV + OOD-DEV mean (chosen on)
    sets: tuple | None = None
    kl_source: str | None = None
    why: str | None = None
    alternative: dict | None = None
    inputs: dict = field(default_factory=dict)
    qd: dict | None = None           # sizing="qd": the reference's own GiB figures (golden comparison)

    @property
    def placement(self) -> str | None:
        return {"all-hot": "resident", "resident": "resident", "cached": "pinned", "split": "split"}.get(self.mode)

    def build_kwargs(self, E: int) -> dict:
        """moe_tiering.loader.build_engine keywords for this configuration (one tier, hot from cold)."""
        if not self.fit:
            raise ValueError(self.why)
        if self.hot_placement:
            raise NotImplementedError("two tiers (design-quality WP18) are not built")
        frozen = E if self.placement == "resident" else (self.frozen if self.mode == "split" else None)
        return {"hot_from_cold": True, "dense_bits": 0, "dense_q": self.dense, "frozen": frozen,
                "embed_host": self.embed_host, "head_host": self.head_host}

    def as_dict(self) -> dict:
        d = asdict(self)
        d["placement"] = self.placement
        return d

    def describe(self) -> str:
        if not self.fit:
            return f"refused: {self.why}"
        host = (" + embedding" if self.embed_host else "") + (" + lm_head" if self.head_host else "")
        s = (f"{self.variant} + {self.dense}, {self.mode}: {self.slots} slots per layer"
             + (f" ({self.frozen} frozen + {self.dynamic} dynamic)" if self.mode == "split" else "")
             + f", {self.pinned_bytes / GiB:.2f} GiB of experts pinned (budget {self.pinned_budget_bytes / GiB:.2f})"
             + (f", {host[3:]} on host" if host else "")
             + f"; 30B KL {self.kl:.4f} (tuning-set score {self.score:.4f})")
        return s


# ---------------------------------------------------------------------------------------------------------------------
# budgets


def reserves(sizing: str = "engine", ram_reserve: float | None = None) -> tuple:
    """(RAM reserve, engine RSS) in GiB: the engine's refitted ones, or the reference's for sizing "qd"; an explicit
    ram_reserve (--ram-reserve) replaces the first."""
    rr, er = (QD_RAM_RESERVE, QD_ENGINE_RSS_GIB) if sizing == "qd" else (RAM_RESERVE, ENGINE_RSS_GIB)
    return (rr if ram_reserve is None else ram_reserve), er


def pinned_budget(ram_gib: float, mem_available_gib=None, ram_reserve: float | None = None,
                  sizing: str = "engine") -> float:
    """GiB of expert blobs the rule may page-lock: RAM - reserve, capped at MemAvailable - ENGINE_RSS_GIB when
    known (reserves(sizing, ram_reserve))."""
    rr, er = reserves(sizing, ram_reserve)
    p = ram_gib - rr
    if mem_available_gib is not None:
        p = min(p, mem_available_gib - er)
    return p


def _meminfo() -> dict:
    """{"total": GiB, "available": GiB} of this machine: Linux /proc/meminfo (MemTotal, MemAvailable), Windows
    GlobalMemoryStatusEx (ullTotalPhys, ullAvailPhys: free + standby pages, as MemAvailable). {} if unknown."""
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return {}
        return {"total": m.ullTotalPhys / GiB, "available": m.ullAvailPhys / GiB}
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                if k in ("MemTotal", "MemAvailable"):
                    out["total" if k == "MemTotal" else "available"] = int(v.split()[0]) * 1024 / GiB
    except OSError:
        return {}
    return out


def mem_available_gib():
    """MemAvailable (Linux) / ullAvailPhys (Windows) in GiB, None if this OS does not say."""
    return _meminfo().get("available")


def mem_total_gib():
    return _meminfo().get("total")


def _slots_engine(vram_gib: float, dense_bytes: int, hot_nb, cold_nb) -> dict:
    from .loader import cache_slots  # the engine's own sizing (amendment 5)

    return cache_slots(vram_gib, dense_bytes, hot_nb, cold_nb)


# ---------------------------------------------------------------------------------------------------------------------
# the rule


def _place(facts: Facts, v: Variant, d: DenseFacts, vram: float, pinned: float, moved: int):
    """One-tier placement of variant v with recipe d: (mode, F, pinned bytes|GiB, slots, info) or None.
    moved: bytes of the dense part moved to host (host fallback). pinned: budget in GiB (already less `moved`)."""
    E = facts.E
    if facts.sizing == "qd":
        g = v.gib
        cache = vram - d.gib - max(0.75, 0.08 * vram)
        if cache <= 0:
            return None
        slot = g / E
        S = min(E, int(cache // slot))
        info = {"cache": cache, "dense": d.gib}
        if g <= cache:
            return "resident", E, 0.0, E, info
        if g <= pinned:
            return "cached", 0, g, S, info
        F = math.ceil((g - pinned) / slot - 1e-9)
        if F > S:
            return None
        return "split", F, (E - F) * slot, S, info
    dense = d.vram_bytes - moved
    sz = _slots_engine(vram, dense, v.nbytes, v.nbytes)
    if sz["free"] <= 0:
        return None
    slot = sz["slot_cost"]
    g = E * slot
    S = min(E, sz["slots"])
    info = {"cache": sz["free"], "dense": dense, "slot_cost": slot, "experts": g}
    if S >= E:
        return "resident", E, 0, E, info
    pb = int(pinned * GiB)
    if g <= pb:
        return "cached", 0, g, S, info
    if not v.counts:
        return None  # a split needs counts.safetensors to pick its frozen experts
    F = -(-(g - pb) // slot)
    if F > S:
        return None
    return "split", F, (E - F) * slot, S, info


def _two_slots(facts: Facts, h4: Variant, v: Variant, d: DenseFacts, vram: float, moved: int):
    E = facts.E
    if facts.sizing == "qd":
        cache = vram - d.gib - max(0.75, 0.08 * vram)
        return min(E, int(cache // (h4.gib / E)))
    return min(E, _slots_engine(vram, d.vram_bytes - moved, h4.nbytes, v.nbytes)["slots"])


def candidates(facts: Facts, variants: list, vram: float, ram: float, mem_available=None, *, nvme=None,
               hot: str = "auto", ram_reserve: float | None = None, host: bool = False, only=None) -> list:
    """Every candidate (dicts the tie-break reads) of this machine. host: the fallback placement (embedding, and the
    head where possible, on host). only: restrict to one variant name."""
    pinned0 = pinned_budget(ram, mem_available, ram_reserve, facts.sizing)
    E = facts.E
    h4 = next((v for v in variants if v.name == "H4"), None)
    out = []
    for dn in DENSES:
        order = [v for v in variants if v.name == "H4"] + [v for r in RUNGS for v in variants if v.name == r]
        for v in order:
            if only and v.name != only:
                continue
            d = (v.dense_facts or facts.dense).get(dn)
            if dn not in v.dense or d is None:
                continue
            head = host and d.head_host_ok
            moved = (d.embed_bytes + (d.head_bytes if head else 0)) if host else 0
            pinned = pinned0 - moved / GiB  # the moved bytes are page-locked too
            base = {"dense": dn, "embed_host": host, "head_host": head, "moved": moved, "pinned_budget": pinned}
            ev = one_tier(v.name, dn)
            if ev is None:
                continue
            p = _place(facts, v, d, vram, pinned, moved)
            if p is None:
                continue
            mode, F, pin, S, info = p
            if v.name == "H4" and mode == "resident":
                mode = "all-hot"
            one = {**base, "tiers": 1, "rung": v.name, "variant": v, "mode": mode, "frozen": F, "pinned": pin,
                   "slots": S, "dynamic": 0 if mode in ("resident", "all-hot") else S - F, "hot": None, "info": info,
                   "kl": ev[0], "score": ev[1], "sets": ev[2], "src": ev[3]}
            out.append(one)
            # two tiers (WP18, not built): only where the facts say a hot tier exists (the decision grid)
            g = v.gib if facts.sizing == "qd" else v.experts_bytes(E)
            gp = pinned if facts.sizing == "qd" else int(pinned * GiB)
            if (facts.two_tier and hot != "off" and h4 is not None and v.name != "H4" and g <= gp):
                hs = _two_slots(facts, h4, v, d, vram, moved)
                frac = hs / E
                if frac >= HOT_MIN_FRACTION and hs < E:
                    gh = h4.gib if facts.sizing == "qd" else h4.experts_bytes(E)
                    place2 = "pinned" if g + gh <= gp else ("nvme" if nvme else None)
                    tt = two_tier(v.name, dn, frac)
                    if place2 and tt is not None:
                        out.append({**one, "tiers": 2, "mode": "cached", "frozen": 0, "slots": hs, "dynamic": hs,
                                    "pinned": g + (gh if place2 == "pinned" else 0),
                                    "hot": {"placement": place2, "slots": hs, "slot_fraction": round(frac, 3)},
                                    "kl": tt[0], "score": tt[1], "sets": None, "src": tt[2]})
    return out


def _pick(cs: list):
    """The reference's tie-break: lowest score; within NOISE_BAND of it the simplest placement, then pinned H4
    before H4 from disk, then the highest variant."""
    if not cs:
        return None
    best = min(c["score"] for c in cs)
    near = [c for c in cs if c["score"] <= best * (1 + NOISE_BAND)]
    order = list(VARIANTS)
    return min(near, key=lambda c: (_SIMPLE[c["mode"]],
                                    0 if not c.get("hot") or c["hot"]["placement"] == "pinned" else 1,
                                    -order.index(c["rung"]), c["score"]))


def _to_config(c: dict, facts: Facts, pinned_budget_gib: float, alt) -> Config:
    v = c["variant"]
    qd = facts.sizing == "qd"
    info = c["info"]
    cfg = Config(fit=True, variant=c["rung"], dense=c["dense"], mode=c["mode"], slots=c["slots"], frozen=c["frozen"],
                 dynamic=c["dynamic"], hot_placement=(c["hot"] or {}).get("placement"),
                 hot_slots=(c["hot"] or {}).get("slots", 0), embed_host=c["embed_host"], head_host=c["head_host"],
                 store=v.store, pinned_bytes=0 if qd else int(c["pinned"]),
                 pinned_budget_bytes=int(c["pinned_budget"] * GiB), dense_bytes=0 if qd else int(info["dense"]),
                 cache_bytes=0 if qd else int(info["cache"]), slot_cost=0 if qd else int(info["slot_cost"]),
                 experts_bytes=0 if qd else int(info["experts"]), kl=c["kl"], score=c["score"], sets=c["sets"],
                 kl_source=c["src"])
    if qd:
        cfg.qd = {"pinned_GiB": round(c["pinned"], 2), "cache_GiB": round(info["cache"], 2),
                  "pinned_budget_GiB": round(pinned_budget_gib, 2), "dense_GiB": info["dense"]}
    if alt is not None:
        cfg.alternative = {"variant": alt["rung"], "mode": alt["mode"], "dense": alt["dense"],
                           "hot_placement": (alt["hot"] or {}).get("placement"), "kl": alt["kl"],
                           "score": alt["score"], "kl_source": alt["src"]}
    return cfg


def _decide(cs: list, hot: str):
    one = [c for c in cs if c["tiers"] == 1]
    two = [c for c in cs if c["tiers"] == 2]
    best1 = _pick(one)
    passing = two if (best1 is None or hot == "on") else [c for c in two
                                                          if c["score"] <= (1 - HOT_MARGIN) * best1["score"]]
    best2 = _pick(passing)
    return (best2, best1) if best2 is not None else (best1, _pick(two))


def choose(facts: Facts, variants: list, vram_gib: float, ram_gib: float, mem_available_gib=None, nvme=None, *,
           ram_reserve: float | None = None, hot: str = "auto", only: str | None = None,
           host_fallback: bool | None = None) -> Config:
    """Rule v3 (module docstring). variants: the Variant entries present (stores, or a grid's). only: a variant name
    (--variant NAME): the rule restricted to it. host_fallback: amendment 4 (default: on for sizing "engine", off for
    the reference's "qd")."""
    if hot not in ("off", "auto", "on"):
        raise ValueError(f"hot={hot!r}: off, auto or on")
    if host_fallback is None:
        host_fallback = facts.sizing == "engine"
    variants = [v for v in variants if v.name in VARIANTS]
    if only is not None and only not in {v.name for v in variants}:
        raise ValueError(f"variant {only!r} is not present ({sorted(v.name for v in variants)})")
    inputs = {"vram_gib": vram_gib, "ram_gib": ram_gib, "mem_available_gib": mem_available_gib, "nvme": nvme,
              "ram_reserve": ram_reserve, "hot": hot, "only": only, "sizing": facts.sizing,
              "variants": [v.name for v in variants]}
    ram_reserve = reserves(facts.sizing, ram_reserve)[0]
    inputs["ram_reserve"] = ram_reserve
    pinned = pinned_budget(ram_gib, mem_available_gib, ram_reserve, facts.sizing)
    if facts.sizing == "qd" and vram_gib - facts.qd_dense_c5 - max(0.75, 0.08 * vram_gib) <= 0:
        return Config(fit=False, why=f"VRAM {vram_gib} GiB does not hold the dense part ({facts.qd_dense_c5:.2f} GiB) "
                                     "+ reserve", inputs=inputs)
    kw = dict(nvme=nvme, hot=hot, ram_reserve=ram_reserve, only=only)
    cs = candidates(facts, variants, vram_gib, ram_gib, mem_available_gib, **kw)
    if not cs and host_fallback:
        cs = candidates(facts, variants, vram_gib, ram_gib, mem_available_gib, host=True, **kw)
    if not cs:
        return Config(fit=False, why=_refusal(facts, variants, vram_gib, ram_gib, mem_available_gib, ram_reserve,
                                              only, host_fallback), inputs=inputs)
    ch, alt = _decide(cs, hot)
    cfg = _to_config(ch, facts, pinned, alt)
    cfg.inputs = inputs
    return cfg


def _refusal(facts, variants, vram, ram, avail, ram_reserve, only, host_fallback) -> str:
    """Why nothing fits, naming the RAM (or VRAM) that would fit the smallest variant present."""
    pinned = pinned_budget(ram, avail, ram_reserve, facts.sizing)
    E = facts.E
    vs = [v for v in variants if (not only or v.name == only)]
    if facts.sizing == "qd":  # the reference's _fail
        dense = facts.qd_dense_c5
        cache = max(0.0, vram - dense - max(0.75, 0.08 * vram))
        name, low = min(((v.name, v.gib) for v in vs), key=lambda x: x[1]) if vs else (RUNGS[0], 0.0)
        need = max(0.0, low - cache)
        return (f"smallest variant {name} ({low:.1f} GiB) does not fit cache {cache:.1f} + pinned "
                f"{max(pinned, 0):.1f} GiB: needs RAM >= {math.ceil(need + QD_RAM_RESERVE)} GB at {vram} GiB VRAM, or a "
                f"disk-backed cold tier (not in store v2)")
    if not vs:
        return "no variant with evidence is present"
    v = min(vs, key=lambda x: x.experts_bytes(E))
    g = v.experts_bytes(E)
    best = None  # the most VRAM-only experts any recipe (in VRAM or with the host fallback) leaves room for
    for dn in DENSES:
        d = (v.dense_facts or facts.dense).get(dn)
        if d is None or dn not in v.dense:
            continue
        for host in ((False, True) if host_fallback else (False,)):
            moved = (d.embed_bytes + (d.head_bytes if d.head_host_ok else 0)) if host else 0
            sz = _slots_engine(vram, d.vram_bytes - moved, v.nbytes, v.nbytes)
            if sz["free"] <= 0:
                continue
            S = min(E, sz["slots"])
            need = g - S * sz["slot_cost"] + moved  # pinned bytes needed: the rest of the experts + what moved
            if best is None or need < best[0]:
                best = (need, dn, host, S)
    if best is None:
        d = min((facts.dense[dn] for dn in DENSES if dn in facts.dense), key=lambda x: x.vram_bytes)
        moved = d.embed_bytes + (d.head_bytes if d.head_host_ok else 0) if host_fallback else 0
        need_v = (d.vram_bytes - moved + max(v.nbytes)) / GiB
        need_v = max(need_v + 0.75, need_v / 0.92)
        return (f"VRAM {vram:g} GiB does not hold the smallest dense part ({d.recipe}, {(d.vram_bytes - moved) / GiB:.2f}"
                f" GiB{' with the embedding on host' if host_fallback else ''}) + staging + reserve: needs VRAM >= "
                f"{math.ceil(need_v * 10) / 10:g} GiB")
    need, dn, host, S = best
    ram_need = math.ceil(need / GiB + ram_reserve)
    s = (f"smallest variant {v.name} ({g / GiB:.2f} GiB) does not fit {S} cache slots ({dn}"
         f"{', embedding on host' if host else ''}) + pinned {max(pinned, 0):.2f} GiB: needs RAM >= {ram_need} GB "
         f"at {vram:g} GiB VRAM")
    if avail is not None:
        s += f" and MemAvailable >= {need / GiB + ENGINE_RSS_GIB:.1f} GiB"
    return s + " (a disk-backed cold tier is not in store v2)"


# ---------------------------------------------------------------------------------------------------------------------
# facts from store-v2 directories


_DT = {"F64": 8, "F32": 4, "I64": 8, "I32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
       "F8_E4M3": 1, "F8_E5M2": 1}


def safetensors_header(path: str) -> dict:
    """{name: (dtype, shape)} from a safetensors file's JSON header (no tensor data read)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in h.items() if k != "__metadata__"}


def _alloc(nbytes: int) -> int:
    """Bytes the caching allocator counts for one tensor (memory_allocated: rounded up to 512)."""
    return -(-nbytes // 512) * 512


def _numel(shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def dense_facts(root: str, meta: dict, recipe: str) -> DenseFacts:
    """The dense part of a store with packed recipe `recipe` as the engine holds it (moe_tiering.loader.load_dense):
    the packed arrays of every packed module (routers and non-input embeddings stay as dense.safetensors holds them),
    every other tensor of dense.safetensors in its load dtype (bf16; float32 kept float32), each rounded as the
    allocator counts it, plus the expert-bias tables attach allocates before it sizes the cache (float32 [E, 2I] and
    [E, H] per layer). Non-persistent buffers (rotary tables) are not counted (KiB). embed_bytes: what embed_host
    moves to pinned host memory (the packed or bf16 input embedding)."""
    e = next((x for x in meta.get("dense_q") or [] if recipe in (x.get("recipe"), x.get("dir"))), None)
    if e is None:
        raise ValueError(f"{root}: no packed dense part {recipe!r}")
    ddir = os.path.join(root, e["dir"])
    dm = json.load(open(os.path.join(ddir, "meta.json")))
    hdr = {}
    for fn in sorted({t["file"] for t in dm["tensors"].values()}):
        hdr.update(safetensors_header(os.path.join(ddir, fn)))
    packed, emb_name, emb_bytes, head_bytes = 0, None, 0, 0
    replaced = set()
    for name, t in dm["tensors"].items():
        role = t["role"]
        if role == "router":
            continue
        if role == "embed":
            if not name.endswith("embed_tokens"):
                continue  # another embedding (Gemma-4's per-layer table): kept bf16
            # moe_tiering.qembed.QEmbedding: one uint8 record per row [V, R] (fp16 d, dmin, codes, group bytes, padded
            # to 4) + its int32 `_want` flag; the uploaded arrays are freed once the records are packed
            V, H = (int(x) for x in t["shape"])
            R = -(-(4 + H * int(t["bits"]) // 8 + H // int(t["group"])) // 4) * 4
            emb_name, emb_bytes = name, _alloc(V * R)
            packed += emb_bytes + 512
        else:
            # moe_tiering.klinear.KLinear: the packed arrays (+ bias) and two int32 [8] index buffers
            nb = sum(_alloc(_numel(hdr[k][1]) * _DT[hdr[k][0]]) for k in hdr if k.startswith(name + ".")
                     and k[len(name) + 1:] in ("qweight", "sb", "d", "dmin", "bias"))
            if role == "lm_head":
                head_bytes = nb
            packed += nb + 2 * 512
        replaced.update({f"{name}.weight", f"{name}.bias"})
    dense = safetensors_header(os.path.join(root, "dense.safetensors"))
    rest = 0
    for k, (dt, shape) in dense.items():
        if k in replaced:
            continue
        nb = _alloc(_numel(shape) * (4 if dt in ("F32", "F64") else 2))
        if emb_name is None and k.endswith("embed_tokens.weight"):
            emb_bytes = nb  # a bf16 input embedding (gpt-oss): embed_host moves the table
        rest += nb
    ex = meta.get("experts") or {}
    bias = 0
    if ex.get("bias"):
        E, (n2i, _h) = int(ex["E"]), ex["gu_shape"]
        H = ex["dn_shape"][0]
        bias = len(meta["moe_layers"]) * (_alloc(E * n2i * 4) + _alloc(E * H * 4))
    return DenseFacts(recipe=e.get("recipe", recipe), vram_bytes=packed + rest + bias, embed_bytes=emb_bytes,
                      head_bytes=head_bytes, head_host_ok=False)


def cold_nbytes(meta: dict) -> tuple:
    """Cold blob bytes of every MoE layer (k4row / rtn: one size; trellis: cold.layers[L].nbytes)."""
    c = meta["cold"]
    moe = [int(x) for x in meta["moe_layers"]]
    if c.get("format") == "tcq":
        return tuple(int(c["layers"][str(li)]["nbytes"]) for li in moe)
    return tuple(int(c["nbytes"]) for _ in moe)


def read_stores(stores) -> tuple:
    """(Facts, [Variant]) from store-v2 directories: `stores` is a directory of store directories, one store
    directory, a comma-separated list of either, or a list. Stores must hold the same model. A store is a variant
    when meta.json names one with evidence (H4, TQx) and ships at least one dense recipe the rule knows."""
    if isinstance(stores, str):
        stores = [s for s in stores.split(",") if s]
    dirs = []
    for s in stores:
        s = os.path.abspath(s)
        if os.path.exists(os.path.join(s, "meta.json")):
            dirs.append(s)
        else:
            dirs += sorted(os.path.join(s, x) for x in os.listdir(s) if os.path.exists(os.path.join(s, x, "meta.json")))
    if not dirs:
        raise ValueError(f"no store-v2 directory (meta.json) under {stores}")
    ref, facts, variants, skipped = None, None, [], {}
    for root in dirs:
        meta = json.load(open(os.path.join(root, "meta.json")))
        name = meta.get("variant")
        recipes = tuple(x.get("recipe") for x in meta.get("dense_q") or [] if x.get("recipe") in DENSES)
        if name not in VARIANTS or not recipes:
            skipped[root] = f"variant {name!r}, dense recipes {recipes}"
            continue
        ex = meta.get("experts") or {}
        key = (meta.get("arch"), tuple(meta["moe_layers"]), int(meta["num_experts"]),
               tuple(meta.get("gu_shape") or ex.get("gu_shape") or ()), tuple(meta.get("dn_shape") or ex.get("dn_shape") or ()))
        if ref is None:
            ref = key
            facts = Facts(model=meta["config"].get("_name_or_path") or os.path.basename(os.path.dirname(root)),
                          E=int(meta["num_experts"]), dense={})
        elif key != ref:
            raise ValueError(f"{root} holds another model (config, MoE layers or E differ) than {variants[0].store}")
        if any(v.name == name for v in variants):
            skipped[root] = f"a second {name} store"
            continue
        own = {}
        for r in recipes:
            own[r] = df = dense_facts(root, meta, r)
            old = facts.dense.get(r)
            if old is None or df.vram_bytes > old.vram_bytes:
                facts.dense[r] = df  # the model's figure (refusal messages): the largest of the stores'
        variants.append(Variant(name=name, nbytes=cold_nbytes(meta), store=root, dense=recipes,
                                counts=bool(meta.get("counts")), dense_facts=own))
    if not variants:
        raise ValueError(f"no usable variant among {dirs}: {skipped}")
    facts.skipped = skipped
    return facts, variants


def plan(stores, vram_gib: float, ram_gib: float | None = None, *, mem_available_gib="read",
         ram_reserve: float | None = None, variant: str = "auto", hot: str = "auto", nvme=None, log=print) -> Config:
    """choose() over store-v2 directories. ram_gib None: this machine's MemTotal. mem_available_gib "read": read it
    now (Linux MemAvailable / Windows ullAvailPhys), None: not used. variant "auto" or a name (--variant NAME)."""
    facts, variants = read_stores(stores)
    meminfo = _meminfo()
    if ram_gib is None:
        ram_gib = meminfo.get("total")
        if ram_gib is None:
            raise ValueError("this OS does not report its RAM: pass --ram-gb")
    avail = meminfo.get("available") if mem_available_gib == "read" else mem_available_gib
    cfg = choose(facts, variants, vram_gib, ram_gib, avail, nvme, ram_reserve=ram_reserve, hot=hot,
                 only=None if variant in (None, "auto") else variant)
    cfg.inputs["mem_total_gib"] = meminfo.get("total")
    cfg.inputs["dense_facts"] = {k: asdict(v) for k, v in facts.dense.items()}
    cfg.inputs["E"] = facts.E
    log(f"auto-config (rule v3, design-quality WP13): VRAM {vram_gib:g} GiB, RAM {ram_gib:.1f} GiB, MemAvailable "
        f"{'-' if avail is None else f'{avail:.1f} GiB'} -> pinned budget "
        f"{pinned_budget(ram_gib, avail, ram_reserve):.2f} GiB (RAM reserve {reserves('engine', ram_reserve)[0]:g}, "
        f"engine {ENGINE_RSS_GIB:g}); variants {[v.name for v in variants]}"
        + (f" (skipped {facts.skipped})" if facts.skipped else ""))
    log(f"auto-config: {cfg.describe()}" + (f" [{cfg.store}]" if cfg.fit else ""))
    return cfg


def memory_report() -> dict:
    """Process memory now (logged at start so the RAM reserve can be refitted, design-quality WP13): RSS and its
    peak, page-locked expert bytes, the part outside them, and MemAvailable."""
    from . import hostmem

    out = {"pinned_bytes": hostmem.registered_bytes(), "peak_rss_bytes": hostmem.peak_rss()}
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                if k in ("VmRSS", "VmHWM", "VmPin", "VmLck", "RssAnon"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    rss = out.get("VmRSS", out["peak_rss_bytes"])
    out["rss_bytes"] = rss
    out["outside_pinned_bytes"] = rss - out["pinned_bytes"]
    out["mem_available_gib"] = mem_available_gib()
    return out


def build(cfg: Config, device="cuda:0", vram_gib: float | None = None, log=print):
    """moe_tiering.loader.build_engine for an auto-config choice; checks that the engine realised the rule's slots and
    frozen counts, and logs RSS and pinned bytes. Returns (model, info); info["autoconfig"] holds the choice, the
    realised numbers and the memory report."""
    from . import loader

    if not cfg.fit:
        raise SystemExit(f"auto-config refused: {cfg.why}")
    vram = cfg.inputs.get("vram_gib") if vram_gib is None else vram_gib
    E = cfg.inputs.get("E") or loader.read_store_meta(cfg.store).num_experts
    model, info = loader.build_engine(cfg.store, device, vram, log=log, **cfg.build_kwargs(E))
    fz = info.get("frozen") or {}
    got_f = fz.get("F", 0) if cfg.mode == "split" else (E if info.get("placement") == "resident-vram" else 0)
    want_f = cfg.frozen if cfg.mode in ("split", "resident", "all-hot") else 0
    real = {"slots": info["slots"], "frozen": got_f, "dense_vram_bytes": info.get("dense_vram_total"),
            "slots_equal": info["slots"] == cfg.slots, "frozen_equal": got_f == want_f}
    mem = memory_report()
    info["autoconfig"] = {"config": cfg.as_dict(), "realised": real, "memory": mem}
    log(f"auto-config realised: {info['slots']} slots (rule {cfg.slots}), frozen {got_f} (rule {want_f}), dense "
        f"{(info.get('dense_vram_total') or 0) / GiB:.3f} GiB (rule {cfg.dense_bytes / GiB:.3f}); memory at start: RSS "
        f"{mem['rss_bytes'] / GiB:.2f} GiB (peak {mem['peak_rss_bytes'] / GiB:.2f}), pinned {mem['pinned_bytes'] / GiB:.2f}"
        f" GiB, outside the pinned blobs {mem['outside_pinned_bytes'] / GiB:.2f} GiB, MemAvailable "
        f"{'-' if mem['mem_available_gib'] is None else format(mem['mem_available_gib'], '.1f')} GiB")
    if not (real["slots_equal"] and real["frozen_equal"]):
        log(f"warning: the engine realised {info['slots']} slots / F {got_f}, the rule planned {cfg.slots} / {want_f}")
    return model, info


def add_args(ap) -> None:
    """The auto-config flags of eval_engine.py, bench_graph.py and tools/engine_qbench.py."""
    g = ap.add_argument_group("auto-config (design-quality WP13)")
    g.add_argument("--variant", help="auto (rule v3) or a variant name (H4, TQ4.0, ...): build from --stores")
    g.add_argument("--stores", help="store-v2 directories: a directory of them, one, or a comma-separated list")
    g.add_argument("--ram-gb", type=float, help="the machine's RAM in GiB for the pinned budget (default: MemTotal)")
    g.add_argument("--ram-reserve", type=float, default=None,
                   help=f"GiB kept back from RAM (default {RAM_RESERVE:g} = {OS_RESERVE_GIB:g} OS + {ENGINE_RSS_GIB:g} engine)")
    g.add_argument("--hot", default="auto", choices=("off", "auto", "on"),
                   help="two tiers (design-quality WP18, not built): off never, auto when >= 10%% better, on always")


def from_args(a, vram_gib: float, log=print) -> Config:
    if not a.stores:
        raise SystemExit("--variant needs --stores")
    cfg = plan(a.stores, vram_gib, a.ram_gb, ram_reserve=a.ram_reserve, variant=a.variant, hot=a.hot, log=log)
    if not cfg.fit:
        raise SystemExit(f"auto-config refused: {cfg.why}")
    return cfg
