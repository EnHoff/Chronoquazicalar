"""
engine/oujax/coverage.py
What OU-JAX implements (the M2 set) and a builder for "M2-covered" test teams: teams assembled from M1
pool sets whose every move, item and ability is implemented, validated by Showdown.

A move is implemented when all of its data fields map to mechanics the engine models (table-driven) or it is
one of the hand-written moves in engine.SPECIAL / damage.MOVE. Anything else is M3.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from engine.oujax.effects import _ITEM_EFFECTS as ITEM_EFFECTS
from engine.oujax.effects import ABILITY, ITEM, T
from engine.oujax.tables import MC, PSEUDO_WEATHERS, SIDE_CONDITIONS, VOLATILES

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "ou" / "generated" / "team_pool.jsonl"

# Abilities with no effect on singles battle mechanics the engine models (information-only or out-of-battle)
INERT_ABILITIES = {"frisk", "illuminate", "runaway", "pickup", "honeygather", "ballfetch", "anticipation", "forewarn"}

HANDWRITTEN_MOVES = {
    "rapidspin", "mortalspin", "defog", "courtchange", "rest", "roost", "moonlight", "morningsun", "synthesis",
    "shoreup", "protect", "detect", "substitute", "partingshot", "teleport", "chillyreception", "haze",
    "auroraveil", "toxic", "thunder", "blizzard", "knockoff", "facade", "hex", "acrobatics", "weatherball",
    "hydrosteam",
    # field-effect moves whose conditions (hazards, screens, Tailwind, Trick Room, terrains, weather) the engine models
    "stealthrock", "spikes", "toxicspikes", "stickyweb", "reflect", "lightscreen", "tailwind", "trickroom",
    "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain", "sunnyday", "raindance", "sandstorm",
    "snowscape",
    # M3 batch 1: base-power callbacks, type changes, effectiveness changes
    "lowkick", "grassknot", "heavyslam", "heatcrash", "eruption", "waterspout", "dragonenergy", "storedpower",
    "powertrip", "ruination", "superfang", "freezedry", "collisioncourse", "electrodrift", "ivycudgel", "poltergeist",
}  # fmt: skip
SUPPORTED_SIDE_CONDITIONS = {
    "",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
    "reflect",
    "lightscreen",
    "auroraveil",
    "tailwind",
}
SUPPORTED_VOLATILES = {"", "protect", "substitute"}
SUPPORTED_SECONDARY_VOLATILES = {"", "flinch"}
SUPPORTED_SELF_VOLATILES = {"", "roost"}
SUPPORTED_PSEUDO_WEATHERS = {"", "trickroom"}

SUPPORTED_ITEMS = {"", *(k for k in vars(ITEM)), *ITEM_EFFECTS["type_boost_items"], *ITEM_EFFECTS["resist_berries"]}
SUPPORTED_ABILITIES = {*(k for k in vars(ABILITY)), *INERT_ABILITIES}


# Raw Showdown move fields with mechanics the table-driven path doesn't model
UNSUPPORTED_FIELDS = {
    "selfdestruct", "mindBlownRecoil", "hasCrashDamage", "sleepUsable", "slotCondition", "multiaccuracy",
    "breaksProtect", "tracksTarget", "smartTarget", "nonGhostTarget", "struggleRecoil", "stallingMove",
}  # fmt: skip


def supported_move(move_id: str) -> bool:
    mid = T.move_ids.get(move_id)
    if mid is None:
        return False
    if move_id in HANDWRITTEN_MOVES:
        return True
    raw = T.dex["moves"][move_id]
    if UNSUPPORTED_FIELDS & raw.keys():
        return False
    if raw.get("ignoreImmunity") not in (None, False) and raw["category"] != "Status":
        return False  # Thousand Arrows and friends hit through specific immunities
    row = T.moves[mid]
    if row[MC["has_handlers"]]:
        return False
    if row[MC["ohko"]] or row[MC["force_switch"]] or row[MC["self_switch"]] > 1:
        return False
    if row[MC["fixed_damage"]] > 0:  # fixed numbers (Dragon Rage, ...); level-based damage is supported
        return False
    return (
        VOLATILES[row[MC["volatile"]]] in SUPPORTED_VOLATILES
        and VOLATILES[row[MC["sec_volatile"]]] in SUPPORTED_SECONDARY_VOLATILES
        and VOLATILES[row[MC["self_volatile"]]] in SUPPORTED_SELF_VOLATILES
        and SIDE_CONDITIONS[row[MC["side_condition"]]] in SUPPORTED_SIDE_CONDITIONS
        and PSEUDO_WEATHERS[row[MC["pseudo_weather"]]] in SUPPORTED_PSEUDO_WEATHERS
    )


def covered_set(s: Dict) -> bool:
    """s: a normalised pool set (data/ou/teams.normalize_set)."""
    return (
        s["item"] in SUPPORTED_ITEMS
        and s["ability"] in SUPPORTED_ABILITIES
        and all(supported_move(m) for m in s["moves"])
    )


def coverage_report(pool_path: Path = POOL) -> Dict:
    """Share of pool sets and teams fully covered by the M2 set, plus the most common uncovered elements."""
    from collections import Counter

    n_sets = n_cov_sets = n_teams = n_cov_teams = 0
    w_all = w_cov = 0.0
    missing: Counter = Counter()
    with open(pool_path) as f:
        for line in f:
            t = json.loads(line)
            n_teams += 1
            ok_team = True
            for s in t["sets"]:
                n_sets += 1
                ok = covered_set(s)
                n_cov_sets += ok
                ok_team &= ok
                if not ok:
                    if s["item"] not in SUPPORTED_ITEMS:
                        missing[("item", s["item"])] += 1
                    if s["ability"] not in SUPPORTED_ABILITIES:
                        missing[("ability", s["ability"])] += 1
                    for m in s["moves"]:
                        if not supported_move(m):
                            missing[("move", m)] += 1
            n_cov_teams += ok_team
            w_all += t["weight"]
            w_cov += t["weight"] if ok_team else 0.0
    return {
        "sets": n_sets,
        "covered_sets": n_cov_sets,
        "teams": n_teams,
        "covered_teams": n_cov_teams,
        "weighted_team_coverage": round(w_cov / max(w_all, 1e-9), 4),
        "top_missing": missing.most_common(40),
    }


def build_covered_teams(n: int, seed: int = 0, pool_path: Path = POOL, validate: bool = True) -> List[str]:
    """n teams of 6 distinct species built from covered pool sets, as Showdown export text. Teams that fail
    Showdown's gen9ou validator are discarded and redrawn."""
    sets: Dict[str, List[Dict]] = {}
    exports: Dict[str, List[str]] = {}
    with open(pool_path) as f:
        for line in f:
            t = json.loads(line)
            blocks = [b for b in t["export"].strip().split("\n\n") if b.strip()]
            for s, block in zip(t["sets"], blocks):
                if covered_set(s):
                    sets.setdefault(s["species"], []).append(s)
                    exports.setdefault(s["species"], []).append(block.strip())
    species = sorted(exports)
    if len(species) < 6:
        raise ValueError(f"only {len(species)} species have covered sets")
    rng = random.Random(seed)
    teams: List[str] = []
    attempts = 0
    while len(teams) < n and attempts < 20 * n:
        attempts += 1
        batch = []
        for _ in range(n - len(teams)):
            picks = rng.sample(species, 6)
            batch.append("\n\n".join(rng.choice(exports[sp]) for sp in picks) + "\n")
        if validate:
            from data.ou.validator import validate_teams
            from engine.oujax.ref import DEFAULT_SHOWDOWN_DIR

            res = validate_teams({str(i): t for i, t in enumerate(batch)}, DEFAULT_SHOWDOWN_DIR, workers=2)
            batch = [t for i, t in enumerate(batch) if res[str(i)]["valid"]]
        teams += batch
    return teams[:n]


def main(argv: Optional[List[str]] = None) -> int:
    rep = coverage_report()
    print(json.dumps({k: v for k, v in rep.items() if k != "top_missing"}, indent=1))
    for (kind, name), count in rep["top_missing"]:
        print(f"  {kind:8s} {name:24s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
