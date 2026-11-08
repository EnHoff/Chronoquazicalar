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
    # M3 damage_moves family
    "tripleaxel", "triplekick", "populationbomb", "dragondarts", "payback", "avalanche", "stompingtantrum",
    "temperflare", "ficklebeam", "counter", "mirrorcoat", "metalburst", "comeuppance", "endeavor", "brickbreak",
    "psychicfangs", "clearsmog", "highjumpkick", "supercellslam", "axekick", "steelbeam", "grassyglide", "growth",
    "bellydrum", "clangoroussoul", "takeheart", "tidyup", "healbell", "strengthsap", "bleakwindstorm",
    "wildboltstorm", "sandsearstorm",
    # M3 volatiles family
    "hurricane", "strangesteam", "waterpulse", "psychicnoise", "saltcure", "swagger", "taunt", "encore", "throatchop", "curse", "leechseed", "destinybond",
    # M3 move_misc family
    "suckerpunch", "thunderclap", "upperhand", "fakeout", "firstimpression", "icespinner", "spikyshield",
    "banefulbunker", "endure",
    # M3 charge_moves family
    "outrage", "meteorbeam", "solarbeam", "solarblade", "phantomforce", "hyperbeam",
    # M3 call_moves family
    "sleeptalk",
    # M3 trapping family
    "magmastorm", "whirlpool", "firespin", "infestation", "meanlook", "block", "jawlock", "noretreat",
    # M3 power / type / category moves (damage.py, families/power_moves.py)
    "electroball", "gyroball", "risingvoltage", "expandingforce", "psyblade", "barbbarrage", "infernalparade",
    "revelationdance", "ragingbull", "photongeyser", "flyingpress", "watershuriken", "charge",
    # M3 field_moves family
    "smackdown", "magnetrise", "gravity", "perishsong",
    # M3 status_moves family, move_misc / charge_moves / power_moves additions
    "focusenergy", "imprison", "stockpile", "spite", "speedswap", "junglehealing", "lunarblessing",
    "hyperspacefury", "doubleshock", "fellstinger", "stoneaxe", "sparklingaria", "burningbulwark", "beakblast",
    "focuspunch", "skyattack", "rockwrecker", "beatup",
    "reversal", "flail", "shellsidearm",
    "sandtomb", "thundercage", "gigaimpact",
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
SUPPORTED_SECONDARY_VOLATILES = {"", "flinch", "confusion", "healblock", "saltcure"}
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


def missing_elements(s: Dict) -> List[tuple]:
    """The (kind, id) elements of a pool set the engine doesn't implement."""
    out = []
    if s["item"] not in SUPPORTED_ITEMS:
        out.append(("item", s["item"]))
    if s["ability"] not in SUPPORTED_ABILITIES:
        out.append(("ability", s["ability"]))
    out += [("move", m) for m in s["moves"] if not supported_move(m)]
    return out


def worklist(pool_path: Path = POOL) -> List[tuple]:
    """Unsupported elements ranked by usage-weighted share of teams that need them, with the strict
    usage-weighted team coverage reached once every element up to and including it is implemented:
    [(kind, id, team_share, cumulative_coverage), ...]."""
    from collections import defaultdict

    need = defaultdict(float)
    teams = []
    w_all = w_cov = 0.0
    with open(pool_path) as f:
        for line in f:
            t = json.loads(line)
            miss = {e for s in t["sets"] for e in missing_elements(s)}
            w_all += t["weight"]
            if not miss:
                w_cov += t["weight"]
                continue
            teams.append((t["weight"], miss))
            for e in miss:
                need[e] += t["weight"]
    order = sorted(need, key=lambda e: -need[e])
    rank = {e: i for i, e in enumerate(order)}
    done_at = [0.0] * len(order)
    for w, miss in teams:
        done_at[max(rank[e] for e in miss)] += w
    out, cum = [], w_cov
    for i, e in enumerate(order):
        cum += done_at[i]
        out.append((e[0], e[1], need[e] / w_all, cum / w_all))
    return out


def covered_pool_teams(n: int, seed: int = 0, pool_path: Path = POOL) -> List[str]:
    """n real pool teams (Showdown export text) whose every set is covered, drawn without replacement with
    probability proportional to their recency weight."""
    exports, weights = [], []
    with open(pool_path) as f:
        for line in f:
            t = json.loads(line)
            if all(covered_set(s) for s in t["sets"]):
                exports.append(t["export"].strip() + "\n")
                weights.append(t["weight"])
    if len(exports) < n:
        raise ValueError(f"only {len(exports)} fully covered pool teams, {n} requested")
    rng = random.Random(seed)
    # Weighted sampling without replacement (Efraimidis-Spirakis keys)
    keys = [rng.random() ** (1.0 / max(w, 1e-12)) for w in weights]
    order = sorted(range(len(exports)), key=lambda i: -keys[i])
    return [exports[i] for i in order[:n]]


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
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worklist", action="store_true", help="every missing element, in usage order, with the "
                    "strict team coverage reached after implementing it and everything above it")  # fmt: skip
    args = ap.parse_args(argv)
    if args.worklist:
        for i, (kind, name, share, cum) in enumerate(worklist(), 1):
            print(f"{i:4d}  {kind:8s} {name:24s} teams {share:6.1%}  cumulative {cum:6.1%}")
        return 0
    rep = coverage_report()
    print(json.dumps({k: v for k, v in rep.items() if k != "top_missing"}, indent=1))
    for (kind, name), count in rep["top_missing"]:
        print(f"  {kind:8s} {name:24s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
