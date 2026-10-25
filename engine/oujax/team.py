"""
engine/oujax/team.py
Team loading for OU-JAX: parses Showdown's packed team format into fixed-shape integer arrays and
precomputes final stats from base stats, EVs, IVs, level and nature exactly as Showdown does.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from engine.oujax.tables import MC, SC, TYPE_NONE, Tables, load_tables, to_id

TEAM_SIZE = 6
N_MOVES = 4


@dataclass
class TeamArrays:
    """One side's team, every field shaped (6, ...) and padded with zeros past the team's size."""

    species: np.ndarray  # (6,) int16
    level: np.ndarray  # (6,) int16
    nature: np.ndarray  # (6,) int16
    item: np.ndarray  # (6,) int16
    ability: np.ndarray  # (6,) int16
    tera_type: np.ndarray  # (6,) int16
    moves: np.ndarray  # (6, 4) int16
    max_pp: np.ndarray  # (6, 4) int16
    evs: np.ndarray  # (6, 6) int16
    ivs: np.ndarray  # (6, 6) int16
    stats: np.ndarray  # (6, 6) int16: [maxhp, atk, def, spa, spd, spe]
    types: np.ndarray  # (6, 2) int16
    size: int


def _split_packed(packed: str) -> List[List[str]]:
    mons = []
    for buf in packed.split("]"):
        if not buf:
            continue
        fields = buf.split("|")
        fields += [""] * (12 - len(fields))
        mons.append(fields)
    return mons


def _spread(text: str, default: int) -> List[int]:
    if not text:
        return [default] * 6
    parts = text.split(",")
    parts += [""] * (6 - len(parts))
    return [int(p) if p != "" else default for p in parts[:6]]


def calc_stats(base: np.ndarray, evs: np.ndarray, ivs: np.ndarray, level: np.ndarray, nature_x10: np.ndarray):
    """Showdown's spreadModify + natureModify, vectorised over leading dims.

    base/evs/ivs/nature_x10: (..., 6); level: (...,). HP of 1-base-HP species (Shedinja) is 1."""
    base = base.astype(np.int64)
    lvl = level.astype(np.int64)[..., None]
    raw = (2 * base + ivs + evs // 4) * lvl // 100
    hp = raw[..., 0] + lvl[..., 0] + 10
    hp = np.where(base[..., 0] == 1, 1, hp)
    other = raw[..., 1:] + 5
    other = other * nature_x10[..., 1:] // 10  # tr(tr(stat * 110, 16) / 100) == floor(stat * 1.1) here
    return np.concatenate([hp[..., None], other], axis=-1).astype(np.int16)


def load_packed_team(packed: str, tables: Optional[Tables] = None) -> TeamArrays:
    t = tables or load_tables()
    mons = _split_packed(packed)
    if not 1 <= len(mons) <= TEAM_SIZE:
        raise ValueError(f"team has {len(mons)} Pokemon")
    z = lambda *shape: np.zeros((TEAM_SIZE, *shape), np.int16)  # noqa: E731
    out = dict(
        species=z(), level=z(), nature=z(), item=z(), ability=z(), tera_type=z(), moves=z(N_MOVES),
        max_pp=z(N_MOVES), evs=z(6), ivs=z(6), types=np.full((TEAM_SIZE, 2), TYPE_NONE, np.int16),
    )  # fmt: skip
    for i, f in enumerate(mons):
        nickname, species, item, ability, moves, nature, evs, _gender, ivs, _shiny, level, misc = f[:12]
        sid = t.species_ids[to_id(species or nickname)]
        out["species"][i] = sid
        out["item"][i] = t.item_ids[to_id(item)]
        spec = t.dex["species"][to_id(species or nickname)]
        if ability in ("", "0", "1", "H"):
            ability = spec["abilities"].get(ability or "0", "")
        out["ability"][i] = t.ability_ids[to_id(ability)]
        for j, m in enumerate([m for m in moves.split(",") if m][:N_MOVES]):
            mid = t.move_ids[to_id(m)]
            out["moves"][i, j] = mid
            out["max_pp"][i, j] = t.moves[mid, MC["pp"]]
        out["nature"][i] = t.nature_ids[to_id(nature) or "serious"]
        out["evs"][i] = _spread(evs, 0)
        out["ivs"][i] = _spread(ivs, 31)
        out["level"][i] = int(level) if level else 100
        misc_fields = misc.split(",")
        tera = to_id(misc_fields[5]) if len(misc_fields) > 5 and misc_fields[5] else to_id(spec["types"][0])
        out["tera_type"][i] = t.type_ids[tera]
        out["types"][i] = t.species[sid, [SC["type1"], SC["type2"]]]
    base = t.species[out["species"], SC["base_hp"] : SC["base_spe"] + 1]
    stats = calc_stats(base, out["evs"], out["ivs"], out["level"], t.natures[out["nature"]])
    stats[len(mons) :] = 0
    return TeamArrays(stats=stats, size=len(mons), **out)
