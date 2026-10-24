"""
engine/oujax/effects.py
Ids of the moves, items and abilities OU-JAX implements, grouped by what they do, plus static per-id lookup
arrays derived from them. Everything not listed here is outside the implemented set (see coverage.py).
"""

from types import SimpleNamespace

import numpy as np

from engine.oujax.tables import SC, load_tables

T = load_tables()


def _ns(ids: dict, names: str) -> SimpleNamespace:
    return SimpleNamespace(**{n: ids[n] for n in names.split()})


ITEM = _ns(
    T.item_ids,
    "leftovers blacksludge lifeorb choiceband choicespecs choicescarf boosterenergy heavydutyboots "
    "rockyhelmet assaultvest sitrusberry airballoon expertbelt lightclay damprock heatrock smoothrock "
    "icyrock terrainextender",
)
ABILITY = _ns(
    T.ability_ids,
    "protosynthesis quarkdrive drizzle drought sandstream snowwarning electricsurge grassysurge "
    "psychicsurge mistysurge levitate swiftswim chlorophyll sandrush slushrush guts",
)
MOVE = _ns(T.move_ids, "knockoff facade hex acrobatics weatherball hydrosteam")

# Modifier constants in Showdown's 4096 fixed point
M_1_5 = 6144
M_1_3 = 5325
M_0_5 = 2048
M_2 = 8192
M_LIFE_ORB = 5324
M_EXPERT_BELT = 4915


def _item_removable() -> np.ndarray:
    """(n_items, n_species) bool: can Knock Off remove this item from this species? Mirrors the items'
    onTakeItem handlers: species-bound items (Ogerpon masks, ...) stay on their user and Booster Energy
    stays on Paradox Pokemon."""
    n_items, n_species = len(T.item_ids), len(T.species_ids)
    out = np.ones((n_items, n_species), bool)
    out[0] = False  # no item
    paradox = T.species[:, SC["paradox"]].astype(bool)
    out[ITEM.boosterenergy, paradox] = False
    for iid, item in T.dex["items"].items():
        users = item.get("itemUser") or []
        if not users and "onTakeItem" not in item.get("handlers", []):
            continue
        idx = T.item_ids[iid]
        if users:
            for sid, sp in T.dex["species"].items():
                if sp["name"] in users or sp["baseSpecies"] in users:
                    out[idx, T.species_ids[sid]] = False
    return out


ITEM_REMOVABLE = _item_removable()
