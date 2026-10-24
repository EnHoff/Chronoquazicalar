"""
engine/oujax/damage.py
Showdown's Gen 9 damage formula (BattleActions.getDamage + modifyDamage) for one attacker, one defender and
one move, in integer arithmetic with Showdown's 4096-based modifier rounding. Single instance; vmap it.

Implemented modifiers: base power (Hex, Acrobatics, Weather Ball, Facade, Knock Off, terrain boosts, Tera's
60 BP floor), attack/defense (boosts, crit boost rules, Body Press / Psyshock / Foul Play overrides, Choice
Band/Specs, Assault Vest, Protosynthesis/Quark Drive, snow Def / sand SpD), weather, crit, random roll,
STAB (incl. Tera STAB), type effectiveness and immunities (incl. Levitate / Air Balloon vs Ground), burn,
Life Orb, Expert Belt and screens.
"""

from typing import NamedTuple

import jax.numpy as jnp

from engine.oujax.effects import (
    ABILITY,
    ITEM,
    ITEM_REMOVABLE,
    M_0_5,
    M_1_3,
    M_1_5,
    M_2,
    M_EXPERT_BELT,
    M_LIFE_ORB,
    MOVE,
    T,
)
from engine.oujax.tables import MC, STATUS_ID, TYPE_NONE, TYPE_STELLAR

TYPE_CHART = jnp.asarray(T.type_chart)
MOVES = jnp.asarray(T.moves)
ITEM_REMOVABLE_J = jnp.asarray(ITEM_REMOVABLE)
TY = T.type_ids
BRN, SLP = STATUS_ID["brn"], STATUS_ID["slp"]
WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_SNOW = 1, 2, 3, 4
TERRAIN_ELECTRIC, TERRAIN_GRASSY, TERRAIN_MISTY, TERRAIN_PSYCHIC = 1, 2, 3, 4
# Stat stage multipliers as Showdown's boostTable [1, 1.5, 2, 2.5, 3, 3.5, 4], in halves
_BOOST_NUM = jnp.array([2, 3, 4, 5, 6, 7, 8], jnp.int32)


class Combatant(NamedTuple):
    """What the damage formula needs about one Pokemon (int32 scalars unless noted)."""

    species: jnp.ndarray
    level: jnp.ndarray
    stats: jnp.ndarray  # (6,) [maxhp, atk, def, spa, spd, spe]
    boosts: jnp.ndarray  # (7,) [atk, def, spa, spd, spe, accuracy, evasion]
    base_types: jnp.ndarray  # (2,) species types (TYPE_NONE padded)
    tera_type: jnp.ndarray
    terastallized: jnp.ndarray  # bool
    status: jnp.ndarray
    item: jnp.ndarray
    ability: jnp.ndarray
    paradox_stat: jnp.ndarray  # 0 = inactive, else 1..5 = boosted stat (atk..spe)


class FieldInfo(NamedTuple):
    weather: jnp.ndarray
    terrain: jnp.ndarray
    reflect: jnp.ndarray  # bool, on the defender's side
    light_screen: jnp.ndarray
    aurora_veil: jnp.ndarray


def current_types(c: Combatant) -> jnp.ndarray:
    """Types after Terastallization (Stellar keeps the original types defensively)."""
    tera = c.terastallized & (c.tera_type != TYPE_STELLAR)
    return jnp.where(tera, jnp.array([c.tera_type, TYPE_NONE]), c.base_types)


def has_type(types: jnp.ndarray, t) -> jnp.ndarray:
    return jnp.any(types == t)


def is_grounded(c: Combatant) -> jnp.ndarray:
    return ~has_type(current_types(c), TY["flying"]) & (c.ability != ABILITY.levitate) & (c.item != ITEM.airballoon)


def modify(value, mod4096):
    """Battle.modify(value, mod / 4096): round half down."""
    return (value * mod4096 + 2047) // 4096


def chain(mod4096, nxt):
    """Battle.chainModify: accumulate another 4096-based modifier."""
    return (mod4096 * nxt + 2048) >> 12


def chain_if(mod4096, cond, nxt):
    return jnp.where(cond, chain(mod4096, nxt), mod4096)


def boosted_stat(stat, boost):
    """Pokemon.calculateStat's stage multiplier: floor(stat * table[b]) or floor(stat / table[-b])."""
    b = jnp.clip(boost, -6, 6)
    num = _BOOST_NUM[jnp.abs(b)]
    return jnp.where(b >= 0, stat * num // 2, stat * 2 // num)


def type_effectiveness(move_type, def_types):
    """(immune, typeMod) as Pokemon.runImmunity / runEffectiveness for a typed move."""
    mults = TYPE_CHART[move_type, def_types]  # (2,)
    immune = jnp.any(mults == 0.0)
    type_mod = jnp.sum(jnp.where(mults == 2.0, 1, jnp.where(mults == 0.5, -1, 0)))
    return immune, type_mod


def move_type_and_bp(move_id, att: Combatant, dfn: Combatant, weather):
    """Move type and base power after onModifyType / onModifyMove / basePowerCallback hooks."""
    row = MOVES[move_id]
    mtype, bp = row[MC["type"]], row[MC["bp"]]
    is_wb = move_id == MOVE.weatherball
    wb_type = jnp.select(
        [weather == WEATHER_SUN, weather == WEATHER_RAIN, weather == WEATHER_SAND, weather == WEATHER_SNOW],
        [TY["fire"], TY["water"], TY["rock"], TY["ice"]],
        mtype,
    )
    mtype = jnp.where(is_wb, wb_type, mtype)
    bp = jnp.where(is_wb & (weather != 0), bp * 2, bp)
    bp = jnp.where((move_id == MOVE.hex) & (dfn.status != 0), bp * 2, bp)
    bp = jnp.where((move_id == MOVE.acrobatics) & (att.item == 0), bp * 2, bp)
    mtype = jnp.where(move_id == MOVE.struggle, TYPE_NONE, mtype)  # Struggle is typeless ("???")
    return mtype, bp


def damage(move_id, att: Combatant, dfn: Combatant, field: FieldInfo, crit, roll):
    """Damage dealt (int32, 0 if immune or not a damaging move). roll: 0..15, 0 = max damage."""
    row = MOVES[move_id]
    category = row[MC["category"]]
    physical = category == 0
    crit = crit | (row[MC["crit_ratio"]] >= 5)  # willCrit moves (Flower Trick, Wicked Blow, ...)
    mtype, bp = move_type_and_bp(move_id, att, dfn, field.weather)
    att_types, def_types = current_types(att), current_types(dfn)

    immune, type_mod = type_effectiveness(mtype, def_types)
    immune |= (mtype == TY["ground"]) & ~is_grounded(dfn)
    damaging = (category != 2) & (bp > 0)

    # --- BasePower event -------------------------------------------------------------------------
    bp_mod = jnp.int32(4096)
    knock_boost = (move_id == MOVE.knockoff) & ITEM_REMOVABLE_J[dfn.item, dfn.species]
    bp_mod = chain_if(bp_mod, knock_boost, M_1_5)
    bp_mod = chain_if(bp_mod, (move_id == MOVE.facade) & (att.status != 0) & (att.status != SLP), M_2)
    att_grounded, def_grounded = is_grounded(att), is_grounded(dfn)
    terrain_boost = att_grounded & (
        ((field.terrain == TERRAIN_ELECTRIC) & (mtype == TY["electric"]))
        | ((field.terrain == TERRAIN_GRASSY) & (mtype == TY["grass"]))
        | ((field.terrain == TERRAIN_PSYCHIC) & (mtype == TY["psychic"]))
    )
    bp_mod = chain_if(bp_mod, terrain_boost, M_1_3)
    misty_dragon = (field.terrain == TERRAIN_MISTY) & def_grounded & (mtype == TY["dragon"])
    grassy_quake = (
        (field.terrain == TERRAIN_GRASSY)
        & def_grounded
        & ((move_id == T.move_ids["earthquake"]) | (move_id == T.move_ids["bulldoze"]))
    )
    bp_mod = chain_if(bp_mod, misty_dragon | grassy_quake, M_0_5)
    bp = jnp.maximum(modify(bp, bp_mod), 1)
    tera_floor = (
        att.terastallized
        & (att.tera_type != TYPE_STELLAR)
        & has_type(att_types, mtype)
        & (bp < 60)
        & (row[MC["priority"]] <= 0)
        & (row[MC["multihit_max"]] == 0)
    )
    bp = jnp.where(tera_floor, 60, bp)

    # --- attack and defense ----------------------------------------------------------------------
    off_stat = jnp.where(row[MC["off_stat"]] > 0, row[MC["off_stat"]], jnp.where(physical, 1, 3))
    def_stat = jnp.where(row[MC["def_stat"]] > 0, row[MC["def_stat"]], jnp.where(physical, 2, 4))
    off_target = row[MC["off_target"]] == 1
    off_stats = jnp.where(off_target, dfn.stats, att.stats)
    off_boosts = jnp.where(off_target, dfn.boosts, att.boosts)
    atk_boost = off_boosts[off_stat - 1]
    def_boost = dfn.boosts[def_stat - 1]
    atk_boost = jnp.where(crit & (atk_boost < 0), 0, atk_boost)
    def_boost = jnp.where((crit & (def_boost > 0)) | (row[MC["ignore_defensive"]] == 1), 0, def_boost)
    attack = boosted_stat(off_stats[off_stat], atk_boost)
    defense = boosted_stat(dfn.stats[def_stat], def_boost)

    # ModifyAtk / ModifySpA run on the attacker for its move category's stat
    event_stat = jnp.where(physical, 1, 3)
    atk_mod = jnp.int32(4096)
    atk_mod = chain_if(atk_mod, physical & (att.item == ITEM.choiceband), M_1_5)
    atk_mod = chain_if(atk_mod, ~physical & (att.item == ITEM.choicespecs), M_1_5)
    atk_mod = chain_if(atk_mod, att.paradox_stat == event_stat, M_1_3)
    atk_mod = chain_if(atk_mod, physical & (att.ability == ABILITY.guts) & (att.status != 0), M_1_5)
    attack = modify(attack, atk_mod)

    # ModifyDef / ModifySpD run on the defender for the stat actually used
    defense = jnp.where(
        (def_stat == 2) & (field.weather == WEATHER_SNOW) & has_type(def_types, TY["ice"]),
        modify(defense, M_1_5),
        defense,
    )
    defense = jnp.where(
        (def_stat == 4) & (field.weather == WEATHER_SAND) & has_type(def_types, TY["rock"]),
        modify(defense, M_1_5),
        defense,
    )
    def_mod = jnp.int32(4096)
    def_mod = chain_if(def_mod, (def_stat == 4) & (dfn.item == ITEM.assaultvest), M_1_5)
    def_mod = chain_if(def_mod, dfn.paradox_stat == def_stat, M_1_3)
    defense = jnp.maximum(modify(defense, def_mod), 1)

    base = (((2 * att.level // 5 + 2) * bp * attack) // defense) // 50

    # --- modifyDamage ----------------------------------------------------------------------------
    base = base + 2
    w_mod = jnp.int32(4096)
    sun, rain = field.weather == WEATHER_SUN, field.weather == WEATHER_RAIN
    hydro_steam = move_id == MOVE.hydrosteam  # boosted rather than weakened by sun
    w_mod = chain_if(w_mod, (sun & ((mtype == TY["fire"]) | hydro_steam)) | (rain & (mtype == TY["water"])), M_1_5)
    w_mod = chain_if(w_mod, (sun & (mtype == TY["water"]) & ~hydro_steam) | (rain & (mtype == TY["fire"])), M_0_5)
    base = modify(base, w_mod)
    base = jnp.where(crit, base * 3 // 2, base)
    base = base * (100 - roll) // 100

    orig_types = att.base_types
    stab_any = has_type(att_types, mtype) | has_type(orig_types, mtype)
    tera_match = att.terastallized & (att.tera_type == mtype) & has_type(orig_types, mtype)
    stab = jnp.where(tera_match, M_2, jnp.where(stab_any, M_1_5, 4096))
    base = modify(base, stab)

    base = jnp.where(type_mod > 0, base << jnp.maximum(type_mod, 0), base)
    base = jnp.where(type_mod < 0, base >> jnp.maximum(-type_mod, 0), base)

    burned = (att.status == BRN) & physical & (att.ability != ABILITY.guts) & (move_id != MOVE.facade)
    base = jnp.where(burned, modify(base, M_0_5), base)

    d_mod = jnp.int32(4096)
    screen = ~crit & (
        (physical & (field.reflect | field.aurora_veil)) | (~physical & (field.light_screen | field.aurora_veil))
    )
    d_mod = chain_if(d_mod, screen, M_0_5)
    d_mod = chain_if(d_mod, att.item == ITEM.lifeorb, M_LIFE_ORB)
    d_mod = chain_if(d_mod, (att.item == ITEM.expertbelt) & (type_mod > 0), M_EXPERT_BELT)
    base = modify(base, d_mod)
    base = jnp.where(base == 0, 1, base) & 0xFFFF

    return jnp.where(damaging & ~immune, base, 0).astype(jnp.int32)


def blocked_by_psychic_terrain(move_id, dfn: Combatant, field: FieldInfo, priority=None):
    """Psychic Terrain makes priority moves fail against grounded targets (checked before damage)."""
    priority = MOVES[move_id, MC["priority"]] if priority is None else priority
    return (field.terrain == TERRAIN_PSYCHIC) & (priority > 0) & is_grounded(dfn)
