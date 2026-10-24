"""
engine/oujax/tables.py
Compact integer game-data tables for OU-JAX, built from engine/oujax/data/dex_gen9.json (exported from the
pinned Showdown commit by tools/ref/export_dex.js), plus their id maps.

Every table row 0 is a "none" entry, so id 0 always means "nothing" (no item, no ability, empty move slot).
Tables are plain numpy here; the engine moves the ones it needs onto the device.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

DEX_PATH = Path(__file__).resolve().parent / "data" / "dex_gen9.json"

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

# Types: the 18 real types in Showdown's order, then Stellar (Tera only) and a "no type" filler
TYPES = (
    "bug", "dark", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost",
    "grass", "ground", "ice", "normal", "poison", "psychic", "rock", "steel", "water",
)  # fmt: skip
TYPE_STELLAR = len(TYPES)  # 18
TYPE_NONE = len(TYPES) + 1  # 19
N_TYPES = len(TYPES) + 2

CATEGORY = {"Physical": 0, "Special": 1, "Status": 2}
ACCURACY_ALWAYS = 101

STATUS = ("", "brn", "par", "psn", "tox", "slp", "frz")
STATUS_ID = {s: i for i, s in enumerate(STATUS)}

TARGETS = (
    "normal", "self", "allAdjacentFoes", "any", "all", "allAdjacent", "allySide", "randomNormal",
    "adjacentAlly", "scripted", "allies", "foeSide", "adjacentAllyOrSelf", "adjacentFoe", "allyTeam",
)  # fmt: skip
TARGET_ID = {t: i for i, t in enumerate(TARGETS)}

# Move flags kept as a bitmask (the ones that matter for singles mechanics)
FLAGS = (
    "contact", "protect", "sound", "punch", "bite", "slicing", "pulse", "bullet", "wind", "powder",
    "heal", "bypasssub", "reflectable", "defrost", "charge", "recharge", "dance", "mirror",
)  # fmt: skip
FLAG_BIT = {f: 1 << i for i, f in enumerate(FLAGS)}

WEATHERS = ("", "sunnyday", "raindance", "sandstorm", "snowscape")
TERRAINS = ("", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")
SIDE_CONDITIONS = (
    "", "stealthrock", "spikes", "toxicspikes", "stickyweb", "reflect", "lightscreen", "auroraveil",
    "tailwind", "safeguard", "mist", "quickguard", "wideguard",
)  # fmt: skip
PSEUDO_WEATHERS = ("", "trickroom", "gravity", "magicroom", "wonderroom", "fairylock")
VOLATILES = (
    "", "flinch", "confusion", "protect", "substitute", "partiallytrapped", "leechseed", "taunt",
    "encore", "yawn", "focusenergy", "destinybond", "endure", "curse", "disable", "torment",
    "mustrecharge", "lockedmove", "roost", "saltcure", "healblock", "charge", "aquaring", "ingrain",
    "magnetrise", "smackdown", "tarshot", "noretreat", "glaiverush", "stockpile", "minimize",
    "defensecurl", "banefulbunker", "burningbulwark", "spikyshield", "silktrap", "syrupbomb",
    "sparklingaria", "dragoncheer", "powertrick", "gastroacid", "imprison", "attract", "uproar",
    "followme", "ragepowder", "helpinghand",
)  # fmt: skip

# Move table columns
MOVE_COLUMNS = (
    "type", "category", "bp", "accuracy", "priority", "pp", "target", "flags", "crit_ratio",
    "drain_num", "drain_den", "recoil_num", "recoil_den", "heal_num", "heal_den",
    "multihit_min", "multihit_max", "status", "volatile", "side_condition", "weather", "terrain",
    "pseudo_weather", "self_switch", "force_switch", "fixed_damage", "ohko",
    "sec_chance", "sec_status", "sec_volatile", "self_volatile",
    *(f"boost_{b}" for b in BOOSTS),  # status-move boosts on the target (self for self-target moves)
    *(f"self_boost_{b}" for b in BOOSTS),  # move.self.boosts / selfBoost.boosts (e.g. Close Combat)
    *(f"sec_boost_{b}" for b in BOOSTS),  # secondary boosts on the target (e.g. Moonblast -1 SpA)
    *(f"sec_self_boost_{b}" for b in BOOSTS),  # secondary self boosts (e.g. Flame Charge +1 Spe)
    "has_handlers",  # Showdown runs custom code for this move (onHit, onBasePower, ...)
)  # fmt: skip
MC = {c: i for i, c in enumerate(MOVE_COLUMNS)}

SPECIES_COLUMNS = (
    *(f"base_{s}" for s in STATS), "type1", "type2", "weight_hg", "ability0", "ability1", "ability_h", "num",
)  # fmt: skip
SC = {c: i for i, c in enumerate(SPECIES_COLUMNS)}


def to_id(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _index(names) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names)}


@dataclass(frozen=True)
class Tables:
    species_ids: Dict[str, int]
    move_ids: Dict[str, int]
    item_ids: Dict[str, int]
    ability_ids: Dict[str, int]
    nature_ids: Dict[str, int]
    type_ids: Dict[str, int]
    species: np.ndarray  # (n_species, len(SPECIES_COLUMNS)) int16
    moves: np.ndarray  # (n_moves, len(MOVE_COLUMNS)) int32 (the flags bitmask needs 18 bits)
    natures: np.ndarray  # (25, 6) int16: x10 multiplier per stat (10 neutral, 11 boosted, 9 lowered)
    type_chart: np.ndarray  # (N_TYPES, N_TYPES) float32: [attacking type, defending type] -> multiplier
    showdown_commit: str
    dex: dict  # the raw export, for lookups the arrays don't cover


def _boosts(d) -> List[int]:
    d = d or {}
    return [int(d.get(b, 0)) for b in BOOSTS]


def _frac(x) -> List[int]:
    return [int(x[0]), int(x[1])] if x else [0, 1]


def _move_row(m: dict, type_ids: Dict[str, int], vol_ids: Dict[str, int]) -> List[int]:
    sec = (m.get("secondaries") or [None])[0] or {}
    self_eff = m.get("self") or {}
    self_boosts = (m.get("selfBoost") or {}).get("boosts") or self_eff.get("boosts")
    mh = m.get("multihit")
    mh_min, mh_max = (mh, mh) if isinstance(mh, int) else (mh or [0, 0])
    acc = m.get("accuracy", True)
    self_switch = m.get("selfSwitch")
    row = {
        "type": type_ids[to_id(m["type"])],
        "category": CATEGORY[m["category"]],
        "bp": m.get("basePower", 0),
        "accuracy": ACCURACY_ALWAYS if acc is True else int(acc),
        "priority": m.get("priority", 0),
        "pp": m.get("pp", 0) * 8 // 5 if m.get("pp", 0) > 1 else m.get("pp", 0),  # with PP Ups, as Showdown
        "target": TARGET_ID.get(m.get("target", "normal"), 0),
        "flags": sum(FLAG_BIT[f] for f in m.get("flags", {}) if f in FLAG_BIT),
        "crit_ratio": m.get("critRatio", 1) + (5 if m.get("willCrit") else 0),
        "drain_num": _frac(m.get("drain"))[0], "drain_den": _frac(m.get("drain"))[1],
        "recoil_num": _frac(m.get("recoil"))[0], "recoil_den": _frac(m.get("recoil"))[1],
        "heal_num": _frac(m.get("heal"))[0], "heal_den": _frac(m.get("heal"))[1],
        "multihit_min": mh_min, "multihit_max": mh_max,
        "status": STATUS_ID.get(m.get("status") or "", 0),
        "volatile": vol_ids.get(to_id(m.get("volatileStatus")), 0),
        "side_condition": _index(SIDE_CONDITIONS).get(to_id(m.get("sideCondition")), 0),
        "weather": _index(WEATHERS).get(to_id(m.get("weather")), 0),
        "terrain": _index(TERRAINS).get(to_id(m.get("terrain")), 0),
        "pseudo_weather": _index(PSEUDO_WEATHERS).get(to_id(m.get("pseudoWeather")), 0),
        "self_switch": {None: 0, True: 1, "copyvolatile": 2, "shedtail": 3}.get(self_switch, 1 if self_switch else 0),
        "force_switch": int(bool(m.get("forceSwitch"))),
        "fixed_damage": -1 if m.get("damage") == "level" else int(m.get("damage") or 0),
        "ohko": int(bool(m.get("ohko"))),
        "sec_chance": int(sec.get("chance", 0)) if sec else 0,
        "sec_status": STATUS_ID.get(sec.get("status") or "", 0),
        "sec_volatile": vol_ids.get(to_id(sec.get("volatileStatus")), 0),
        "self_volatile": vol_ids.get(to_id(self_eff.get("volatileStatus")), 0),
        "has_handlers": int(bool(m.get("handlers") or m.get("conditionHandlers"))),
    }  # fmt: skip
    for b, v in zip(BOOSTS, _boosts(m.get("boosts"))):
        row[f"boost_{b}"] = v
    for b, v in zip(BOOSTS, _boosts(self_boosts)):
        row[f"self_boost_{b}"] = v
    for b, v in zip(BOOSTS, _boosts(sec.get("boosts"))):
        row[f"sec_boost_{b}"] = v
    for b, v in zip(BOOSTS, _boosts((sec.get("self") or {}).get("boosts"))):
        row[f"sec_self_boost_{b}"] = v
    return [row[c] for c in MOVE_COLUMNS]


@lru_cache(maxsize=1)
def load_tables(path: Path = DEX_PATH) -> Tables:
    dex = json.loads(Path(path).read_text(encoding="utf-8"))
    type_ids = {t: i for i, t in enumerate(TYPES)}
    type_ids["stellar"] = TYPE_STELLAR
    vol_ids = _index(VOLATILES)

    item_ids = _index([""] + sorted(dex["items"]))
    ability_ids = _index([""] + sorted(dex["abilities"]))
    move_ids = _index([""] + sorted(dex["moves"]))
    species_ids = _index([""] + sorted(dex["species"]))
    nature_ids = _index(sorted(dex["natures"]))

    species = np.zeros((len(species_ids), len(SPECIES_COLUMNS)), np.int16)
    species[0, SC["type1"]] = species[0, SC["type2"]] = TYPE_NONE
    for sid, s in dex["species"].items():
        row = species[species_ids[sid]]
        for i, st in enumerate(STATS):
            row[SC[f"base_{st}"]] = s["baseStats"][st]
        types = [type_ids[to_id(t)] for t in s["types"]]
        row[SC["type1"]] = types[0]
        row[SC["type2"]] = types[1] if len(types) > 1 else TYPE_NONE
        row[SC["weight_hg"]] = round(s["weightkg"] * 10)
        for key, col in (("0", "ability0"), ("1", "ability1"), ("H", "ability_h")):
            row[SC[col]] = ability_ids.get(to_id(s["abilities"].get(key)), 0)
        row[SC["num"]] = s["num"]

    moves = np.zeros((len(move_ids), len(MOVE_COLUMNS)), np.int32)
    for mid, m in dex["moves"].items():
        moves[move_ids[mid]] = _move_row(m, type_ids, vol_ids)

    natures = np.full((len(nature_ids), len(STATS)), 10, np.int16)
    for nid, n in dex["natures"].items():
        if n["plus"]:
            natures[nature_ids[nid], STATS.index(n["plus"])] = 11
        if n["minus"]:
            natures[nature_ids[nid], STATS.index(n["minus"])] = 9

    chart = np.ones((N_TYPES, N_TYPES), np.float32)
    code_mult = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}
    for def_id, t in dex["types"].items():
        if def_id not in type_ids or type_ids[def_id] >= len(TYPES):
            continue
        for atk_name, code in t["damageTaken"].items():
            atk = to_id(atk_name)
            if atk in type_ids and type_ids[atk] < len(TYPES):
                chart[type_ids[atk], type_ids[def_id]] = code_mult[code]

    return Tables(
        species_ids=species_ids,
        move_ids=move_ids,
        item_ids=item_ids,
        ability_ids=ability_ids,
        nature_ids=nature_ids,
        type_ids=type_ids,
        species=species,
        moves=moves,
        natures=natures,
        type_chart=chart,
        showdown_commit=dex["showdownCommit"],
        dex=dex,
    )
