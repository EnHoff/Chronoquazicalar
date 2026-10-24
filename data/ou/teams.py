"""
data/ou/teams.py
Team-level logic for the OU pool: set normalisation, exact and near-duplicate removal,
recency weighting and rule-based archetype tags.

Sets here are Showdown PokemonSet dicts as returned by validate_teams.js (canonical names),
normalised to Showdown IDs so they compare cleanly.
"""

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NEAR_DUP_THRESHOLD = 0.9
RECENCY_HALF_LIFE_MONTHS = 3.0
UNDATED_RECENCY_WEIGHT = 0.5


def to_id(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def normalize_set(s: Dict) -> Dict:
    """Showdown PokemonSet -> compact ID form used throughout the pipeline."""
    evs = s.get("evs") or {}
    ivs = s.get("ivs") or {}
    return {
        "species": to_id(s.get("species") or s.get("name")),
        "item": to_id(s.get("item")),
        "ability": to_id(s.get("ability")),
        "tera": to_id(s.get("teraType")),
        "nature": to_id(s.get("nature")),
        "evs": [int(evs.get(k, 0) or 0) for k in STATS],
        "ivs": [int(ivs.get(k, 31) if ivs.get(k) is not None else 31) for k in STATS],
        "moves": sorted(to_id(m) for m in (s.get("moves") or []) if m),
    }


def team_key(sets: List[Dict]) -> Tuple:
    """Order-independent identity of a team (for exact duplicate removal)."""
    return tuple(
        sorted(
            (s["species"], s["item"], s["ability"], s["tera"], s["nature"], tuple(s["evs"]), tuple(s["moves"]))
            for s in sets
        )
    )


def species_key(sets: List[Dict]) -> Tuple[str, ...]:
    return tuple(sorted(s["species"] for s in sets))


def set_similarity(a: Dict, b: Dict) -> float:
    """Share of identical fields between two sets of the same species: item, ability, tera,
    nature, EV spread, and each of the 4 move slots (9 fields)."""
    same = sum(a[f] == b[f] for f in ("item", "ability", "tera", "nature", "evs"))
    same += len(set(a["moves"]) & set(b["moves"]))
    total = 5 + max(len(a["moves"]), len(b["moves"]), 4)
    return same / total


def team_similarity(a: List[Dict], b: List[Dict]) -> float:
    """Mean set similarity for two teams with the same six species (0 if species differ)."""
    if species_key(a) != species_key(b):
        return 0.0
    by_species = {s["species"]: s for s in b}
    return sum(set_similarity(s, by_species[s["species"]]) for s in a) / len(a)


def months_between(month: str, ref: str) -> int:
    y1, m1 = map(int, month.split("-"))
    y2, m2 = map(int, ref.split("-"))
    return (y2 - y1) * 12 + (m2 - m1)


def recency_weight(month: Optional[str], ref_month: str) -> float:
    """Exponential decay with a 3-month half-life; the reference (latest) month weighs 1."""
    if not month:
        return UNDATED_RECENCY_WEIGHT
    return 0.5 ** (max(months_between(month, ref_month), 0) / RECENCY_HALF_LIFE_MONTHS)


def dedupe(teams: Iterable[Dict], threshold: float = NEAR_DUP_THRESHOLD) -> Tuple[List[Dict], Dict[str, int]]:
    """Removes exact duplicates and near-duplicates (same six species, mean set similarity >= threshold).

    Each team dict needs "sets" (normalised) and "recency_weight". The kept team absorbs the
    removed ones: "copies" counts them and its recency weight becomes the highest among them.
    Teams are visited newest/heaviest first so the kept representative is the most recent one.
    """
    ordered = sorted(teams, key=lambda t: -t["recency_weight"])
    stats = {"exact": 0, "near": 0}
    seen_exact: Dict[Tuple, Dict] = {}
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    kept: List[Dict] = []
    for t in ordered:
        key = team_key(t["sets"])
        if key in seen_exact:
            seen_exact[key]["copies"] += t.get("copies", 1)
            stats["exact"] += 1
            continue
        group = groups[species_key(t["sets"])]
        match = next((k for k in group if team_similarity(t["sets"], k["sets"]) >= threshold), None)
        if match is not None:
            match["copies"] += t.get("copies", 1)
            seen_exact[key] = match
            stats["near"] += 1
            continue
        t = dict(t, copies=t.get("copies", 1))
        seen_exact[key] = t
        group.append(t)
        kept.append(t)
    return kept, stats


# --- archetype tags -------------------------------------------------------------------------

WEATHER_ABILITIES = {"drizzle": "rain", "drought": "sun", "sandstream": "sand", "snowwarning": "snow"}
WEATHER_MOVES = {
    "raindance": "rain",
    "sunnyday": "sun",
    "sandstorm": "sand",
    "snowscape": "snow",
    "chillyreception": "snow",
}
TERRAIN_ABILITIES = {
    "electricsurge": "electric_terrain",
    "hadronengine": "electric_terrain",
    "grassysurge": "grassy_terrain",
    "psychicsurge": "psychic_terrain",
    "mistysurge": "misty_terrain",
}
HAZARD_MOVES = {
    "stealthrock": "rocks",
    "stoneaxe": "rocks",
    "spikes": "spikes",
    "ceaselessedge": "spikes",
    "toxicspikes": "tspikes",
    "stickyweb": "webs",
}
HAZARD_REMOVAL = {"defog", "rapidspin", "mortalspin", "tidyup", "courtchange"}
PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "shedtail"}
RECOVERY_MOVES = {
    "recover", "roost", "slackoff", "softboiled", "moonlight", "morningsun", "synthesis", "wish", "rest",
    "strengthsap", "shoreup", "milkdrink", "junglehealing", "lunarblessing", "healorder", "matchagotcha",
}  # fmt: skip
OFFENSIVE_ITEMS = {"choiceband", "choicespecs", "choicescarf", "lifeorb", "boosterenergy", "expertbelt"}


def set_role(s: Dict) -> str:
    """'offensive', 'defensive' or 'mixed' from EV investment, items and recovery."""
    hp, atk, de, spa, spd, spe = s["evs"]
    offence = max(atk, spa) + spe
    bulk = hp + max(de, spd)
    has_recovery = bool(RECOVERY_MOVES & set(s["moves"])) or s["item"] == "leftovers" and s["ability"] == "regenerator"
    if s["item"] in OFFENSIVE_ITEMS or (offence >= 400 and bulk < 300):
        return "offensive"
    if bulk >= 400 and offence < 300 or (has_recovery and bulk >= 300):
        return "defensive"
    return "mixed"


def archetype_tags(sets: List[Dict]) -> List[str]:
    tags = set()
    for s in sets:
        if s["ability"] in WEATHER_ABILITIES:
            tags.add(WEATHER_ABILITIES[s["ability"]])
        if s["ability"] in TERRAIN_ABILITIES:
            tags.add(TERRAIN_ABILITIES[s["ability"]])
        for m in s["moves"]:
            if m in WEATHER_MOVES:
                tags.add(WEATHER_MOVES[m])
    hazards = {HAZARD_MOVES[m] for s in sets for m in s["moves"] if m in HAZARD_MOVES}
    if "webs" in hazards:
        tags.add("webs")
    if len(hazards - {"webs"}) >= 2:
        tags.add("hazard_stack")
    if any(HAZARD_REMOVAL & set(s["moves"]) for s in sets):
        tags.add("hazard_removal")
    if sum(bool(PIVOT_MOVES & set(s["moves"])) for s in sets) >= 3:
        tags.add("volt_turn")
    if sum("trickroom" in s["moves"] for s in sets) >= 2:
        tags.add("trick_room")

    roles = [set_role(s) for s in sets]
    n_off, n_def = roles.count("offensive"), roles.count("defensive")
    if n_def >= 5:
        tags.add("stall")
    elif n_off >= 5:
        tags.add("hyper_offence")
    elif n_off >= 3 and n_def <= 1:
        tags.add("offence")
    else:
        tags.add("balance")
    return sorted(tags)


STYLE_TAGS = ("stall", "balance", "offence", "hyper_offence")
