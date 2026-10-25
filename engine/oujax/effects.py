"""
engine/oujax/effects.py
Ids of the moves, items and abilities OU-JAX implements, grouped by what they do, plus static per-id lookup
arrays derived from them. Everything not listed here is outside the implemented set (see coverage.py).
"""

import json
from pathlib import Path
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
    "icyrock terrainextender wellspringmask hearthflamemask cornerstonemask souldew wiseglasses muscleband "
    "punchingglove eviolite",
)
ABILITY = _ns(
    T.ability_ids,
    "protosynthesis quarkdrive drizzle drought sandstream snowwarning electricsurge grassysurge "
    "psychicsurge mistysurge levitate swiftswim chlorophyll sandrush slushrush guts regenerator naturalcure "
    "pressure technician ironfist reckless toughclaws sharpness strongjaw megalauncher supremeoverlord sandforce "
    "sheerforce punkrock hugepower purepower torrent blaze overgrow swarm transistor dragonsmaw steelworker "
    "rockypayload waterbubble solarpower adaptability tintedlens sniper liquidvoice multiscale shadowshield filter "
    "solidrock prismarmor fluffy icescales thickfat heatproof purifyingsalt furcoat vesselofruin tabletsofruin "
    "swordofruin beadsofruin waterveil",
)
MOVE = _ns(
    T.move_ids,
    "knockoff facade hex acrobatics weatherball hydrosteam struggle lowkick grassknot heavyslam heatcrash eruption "
    "waterspout dragonenergy storedpower powertrip hardpress ruination superfang freezedry collisioncourse "
    "electrodrift ivycudgel poltergeist",
)

# Modifier constants in Showdown's 4096 fixed point
M_1_5 = 6144
M_1_3 = 5325
M_0_5 = 2048
M_2 = 8192
M_LIFE_ORB = 5324
M_EXPERT_BELT = 4915
M_STELLAR = 4915


def _item_removable() -> np.ndarray:
    """(n_items, n_species) bool: can Knock Off remove this item from this species? Mirrors the items'
    onTakeItem handlers: species-bound items with onTakeItem (Ogerpon masks, ...) stay on their user and
    Booster Energy stays on Paradox Pokemon."""
    n_items, n_species = len(T.item_ids), len(T.species_ids)
    out = np.ones((n_items, n_species), bool)
    out[0] = False  # no item
    paradox = T.species[:, SC["paradox"]].astype(bool)
    out[ITEM.boosterenergy, paradox] = False
    for iid, item in T.dex["items"].items():
        # Only items whose onTakeItem refuses removal stay on their user (Soul Dew has an itemUser list but
        # no onTakeItem, so Knock Off removes it)
        users = item.get("itemUser") or []
        if "onTakeItem" not in item.get("handlers", []):
            continue
        idx = T.item_ids[iid]
        if users:
            for sid, sp in T.dex["species"].items():
                if sp["name"] in users or sp["baseSpecies"] in users:
                    out[idx, T.species_ids[sid]] = False
    return out


ITEM_REMOVABLE = _item_removable()

_ITEM_EFFECTS = json.loads((Path(__file__).resolve().parent / "data" / "item_effects.json").read_text())


def _item_type_table(mapping) -> np.ndarray:
    out = np.full(len(T.item_ids), -1, np.int32)
    for item, t in mapping.items():
        out[T.item_ids[item]] = T.type_ids[t if isinstance(t, str) else t["type"]]
    return out


# Type-boosting held items (Charcoal, Mystic Water, plates, ...): 4915/4096 base power for their type
TYPE_BOOST_ITEM = _item_type_table(_ITEM_EFFECTS["type_boost_items"])
# Resist berries: halve one super-effective hit of their type (Chilan: any Normal hit), then are eaten
RESIST_BERRY = _item_type_table(_ITEM_EFFECTS["resist_berries"])
RESIST_BERRY_ANY_HIT = np.zeros(len(T.item_ids), bool)
for _item, _e in _ITEM_EFFECTS["resist_berries"].items():
    RESIST_BERRY_ANY_HIT[T.item_ids[_item]] = not _e["super_effective_only"]

# Ogerpon forms: Ivy Cudgel's type and the mask that boosts it
OGERPON_TYPE = np.full(len(T.species_ids), -1, np.int32)
for _sid, _sp in T.dex["species"].items():
    if _sp["baseSpecies"] == "Ogerpon":
        name = _sp["name"]
        t = (
            "water"
            if "Wellspring" in name
            else "fire"
            if "Hearthflame" in name
            else "rock"
            if "Cornerstone" in name
            else "grass"
        )
        OGERPON_TYPE[T.species_ids[_sid]] = T.type_ids[t]
MASK_FOR_TYPE = {
    T.type_ids["water"]: ITEM.wellspringmask,
    T.type_ids["fire"]: ITEM.hearthflamemask,
    T.type_ids["rock"]: ITEM.cornerstonemask,
}
SOUL_DEW_USERS = np.zeros(len(T.species_ids), bool)
for _name in ("latios", "latias"):
    if _name in T.species_ids:
        SOUL_DEW_USERS[T.species_ids[_name]] = True
