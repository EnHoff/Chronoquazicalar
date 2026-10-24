"""
data/ou/stats.py
Smogon chaos usage stats -> usage_marginals: per-species probabilities of moves, items, abilities,
Tera types, spreads and teammates, recency-weighted across months. For each month a species uses the
1825 cut-off when it has enough data there, otherwise 1695.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from data.ou.teams import recency_weight, to_id

PREFERRED_CUTOFF = 1825
FALLBACK_CUTOFF = 1695
MIN_RAW_COUNT = 200  # below this many appearances at 1825, fall back to 1695 for that species/month
MIN_PROB = 0.001
TOP_K = {"moves": 40, "items": 25, "abilities": 10, "tera": 19, "spreads": 30, "teammates": 40}
CHAOS_FIELDS = {
    "Moves": "moves",
    "Items": "items",
    "Abilities": "abilities",
    "Tera Types": "tera",
    "Spreads": "spreads",
    "Teammates": "teammates",
}


def species_marginals(entry: Dict) -> Dict:
    """One chaos species entry -> normalised per-category probabilities (moves sum to ~4)."""
    total = sum(entry.get("Abilities", {}).values()) or 1.0
    out = {}
    for chaos_key, key in CHAOS_FIELDS.items():
        dist = {}
        for name, count in (entry.get(chaos_key) or {}).items():
            if not name:
                continue
            k = name if key == "spreads" else to_id(name)
            dist[k] = dist.get(k, 0.0) + count / total
        out[key] = dist
    return out


def load_chaos(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_usage_marginals(chaos_files: Dict[str, Dict[int, Path]], ref_month: Optional[str] = None) -> Dict:
    """chaos_files: {month: {cutoff: path}}. Returns {"months", "species": {id: {...}}}."""
    months = sorted(chaos_files, reverse=True)
    ref_month = ref_month or months[0]
    acc: Dict[str, Dict] = defaultdict(lambda: {"w": 0.0, "usage": 0.0, "raw": 0, "cutoffs": {}, "dists": {}})
    for month in months:
        w = recency_weight(month, ref_month)
        pref = load_chaos(chaos_files[month][PREFERRED_CUTOFF])["data"]
        fall = load_chaos(chaos_files[month][FALLBACK_CUTOFF])["data"]
        for name in set(pref) | set(fall):
            entry = pref.get(name)
            cutoff = PREFERRED_CUTOFF
            if entry is None or entry.get("Raw count", 0) < MIN_RAW_COUNT:
                if name in fall:
                    entry, cutoff = fall[name], FALLBACK_CUTOFF
            if entry is None:
                continue
            sid = to_id(name)
            a = acc[sid]
            a["w"] += w
            a["usage"] += w * float(entry.get("usage", 0.0))
            a["raw"] += int(entry.get("Raw count", 0))
            a["cutoffs"][month] = cutoff
            for key, dist in species_marginals(entry).items():
                d = a["dists"].setdefault(key, defaultdict(float))
                for k, p in dist.items():
                    d[k] += w * p

    species = {}
    for sid, a in acc.items():
        rec = {"usage": a["usage"] / a["w"], "raw_count": a["raw"], "cutoff_by_month": a["cutoffs"]}
        for key, dist in a["dists"].items():
            norm = {k: v / a["w"] for k, v in dist.items() if v / a["w"] >= MIN_PROB}
            top = sorted(norm.items(), key=lambda kv: -kv[1])[: TOP_K[key]]
            rec[key] = {k: round(v, 5) for k, v in top}
        species[sid] = rec
    species = dict(sorted(species.items(), key=lambda kv: -kv[1]["usage"]))
    return {
        "months": months,
        "ref_month": ref_month,
        "preferred_cutoff": PREFERRED_CUTOFF,
        "fallback_cutoff": FALLBACK_CUTOFF,
        "min_raw_count": MIN_RAW_COUNT,
        "species": species,
    }


def chaos_paths(cache_dir: Path, months: List[str]) -> Dict[str, Dict[int, Path]]:
    return {
        m: {c: cache_dir / "chaos" / f"{m}-gen9ou-{c}.json" for c in (PREFERRED_CUTOFF, FALLBACK_CUTOFF)}
        for m in months
    }
