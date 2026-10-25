"""
engine/oujax/damage.py
Showdown's Gen 9 damage formula (BattleActions.getDamage + modifyDamage) for one attacker, one defender and
one move, in integer arithmetic with Showdown's 4096-based modifier rounding. Single instance; vmap it.

Each event's modifiers are chained in Showdown's handler priority order (higher first), because every
chainModify rounds:
  BasePower  Technician (30) > Iron Fist, Reckless, Punching Glove (23) > Supreme Overlord, Tough Claws,
             Sand Force, Sheer Force (21) > Sharpness, Strong Jaw, Mega Launcher (19) > Dry Skin (17) >
             Wise Glasses, Muscle Band (16) > type items, masks, Soul Dew (15) > Punk Rock (7) > terrain (6) >
             the move's own hook (Knock Off, Facade, Collision Course; 0)
  Atk/SpA    defender's Thick Fat, Heatproof, Purifying Salt (6) > abilities incl. Protosynthesis (5) >
             Choice items (1) > Ruin abilities (0)
  Def/SpD    sand/snow (10, direct) > Fur Coat, Protosynthesis (6) > Eviolite (2) > Assault Vest (1) > Ruin (0)
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
    M_STELLAR,
    MASK_FOR_TYPE,
    MOVE,
    OGERPON_TYPE,
    RESIST_BERRY,
    RESIST_BERRY_ANY_HIT,
    SOUL_DEW_USERS,
    TYPE_BOOST_ITEM,
    T,
)
from engine.oujax.tables import FLAG_BIT, MC, SC, STATUS_ID, TYPE_NONE, TYPE_STELLAR

TYPE_CHART = jnp.asarray(T.type_chart)
MOVES = jnp.asarray(T.moves)
SPECIES = jnp.asarray(T.species)
ITEM_REMOVABLE_J = jnp.asarray(ITEM_REMOVABLE)
TYPE_BOOST_ITEM_J = jnp.asarray(TYPE_BOOST_ITEM)
RESIST_BERRY_J = jnp.asarray(RESIST_BERRY)
RESIST_BERRY_ANY_HIT_J = jnp.asarray(RESIST_BERRY_ANY_HIT)
OGERPON_TYPE_J = jnp.asarray(OGERPON_TYPE)
SOUL_DEW_USERS_J = jnp.asarray(SOUL_DEW_USERS)
TY = T.type_ids
BRN, SLP = STATUS_ID["brn"], STATUS_ID["slp"]
WEATHER_SUN, WEATHER_RAIN, WEATHER_SAND, WEATHER_SNOW = 1, 2, 3, 4
TERRAIN_ELECTRIC, TERRAIN_GRASSY, TERRAIN_MISTY, TERRAIN_PSYCHIC = 1, 2, 3, 4
# Stat stage multipliers as Showdown's boostTable [1, 1.5, 2, 2.5, 3, 3.5, 4], in halves
_BOOST_NUM = jnp.array([2, 3, 4, 5, 6, 7, 8], jnp.int32)
M_1_2 = 4915
M_1_1 = 4505
M_PUNCHING_GLOVE = 4506
M_0_75 = 3072
M_SUPER_EFFECTIVE_4_3 = 5461
OVERLORD = jnp.array([4096, 4506, 4915, 5325, 5734, 6144], jnp.int32)
LOW_KICK_BP = jnp.array([20, 40, 60, 80, 100, 120], jnp.int32)
LOW_KICK_LIMITS_HG = jnp.array([100, 250, 500, 1000, 2000], jnp.int32)
HALF_HP_MOVES = (MOVE.ruination, MOVE.superfang)


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
    stellar_used: jnp.ndarray = 0  # bitmask of move types that already got their Stellar Tera boost
    hp: jnp.ndarray = -1  # current HP; -1 = full
    fallen: jnp.ndarray = 0  # Supreme Overlord: fainted teammates counted on switch-in (0..5)


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


def current_hp(c: Combatant):
    return jnp.where(jnp.asarray(c.hp) < 0, c.stats[0], c.hp)


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


def type_effectiveness(move_type, def_types, move_id=None):
    """(immune, typeMod) as Pokemon.runImmunity / runEffectiveness for a typed move (Freeze-Dry treats Water
    as weak)."""
    mults = TYPE_CHART[move_type, def_types]  # (2,)
    immune = jnp.any(mults == 0.0)
    per_type = jnp.where(mults == 2.0, 1, jnp.where(mults == 0.5, -1, 0))
    if move_id is not None:
        per_type = jnp.where((move_id == MOVE.freezedry) & (def_types == TY["water"]), 1, per_type)
    return immune, jnp.sum(per_type)


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
    ogerpon = OGERPON_TYPE_J[att.species]
    mtype = jnp.where((move_id == MOVE.ivycudgel) & (ogerpon >= 0), ogerpon, mtype)
    sound = (row[MC["flags"]] & FLAG_BIT["sound"]) > 0
    mtype = jnp.where((att.ability == ABILITY.liquidvoice) & sound, TY["water"], mtype)
    mtype = jnp.where(move_id == MOVE.struggle, TYPE_NONE, mtype)  # Struggle is typeless ("???")
    # basePowerCallback moves
    bp = jnp.where((move_id == MOVE.hex) & (dfn.status != 0), bp * 2, bp)
    bp = jnp.where((move_id == MOVE.acrobatics) & (att.item == 0), bp * 2, bp)
    tw = SPECIES[dfn.species, SC["weight_hg"]]
    uw = SPECIES[att.species, SC["weight_hg"]]
    low_kick = LOW_KICK_BP[jnp.sum(tw >= LOW_KICK_LIMITS_HG)]
    bp = jnp.where((move_id == MOVE.lowkick) | (move_id == MOVE.grassknot), low_kick, bp)
    slam = jnp.select([uw >= tw * 5, uw >= tw * 4, uw >= tw * 3, uw >= tw * 2], [120, 100, 80, 60], 40)
    bp = jnp.where((move_id == MOVE.heavyslam) | (move_id == MOVE.heatcrash), slam, bp)
    hp_scaled = jnp.maximum(150 * current_hp(att) // att.stats[0], 1)  # clampIntRange(bp, 1)
    bp = jnp.where(
        (move_id == MOVE.eruption) | (move_id == MOVE.waterspout) | (move_id == MOVE.dragonenergy), hp_scaled, bp
    )
    positive = jnp.sum(jnp.maximum(att.boosts, 0))
    bp = jnp.where((move_id == MOVE.storedpower) | (move_id == MOVE.powertrip), bp + 20 * positive, bp)
    return mtype, bp


class Parts(NamedTuple):
    """The per-move part of the damage calculation (everything before the crit multiplier), plus the flags
    the per-hit part needs. Computed once per move for crit and non-crit; each hit then only runs
    damage_finish (crit multiplier, roll, STAB, effectiveness, burn, final modifiers)."""

    base: jnp.ndarray  # after +2 and weather
    stab: jnp.ndarray  # 4096-based
    type_mod: jnp.ndarray
    burned: jnp.ndarray
    screen: jnp.ndarray  # screens apply when not a crit
    multiscale: jnp.ndarray  # halves damage while the defender is at full HP
    fluffy: jnp.ndarray  # 0 none, 1 halve, 2 double
    punkrock: jnp.ndarray
    icescales: jnp.ndarray
    solid: jnp.ndarray
    sniper: jnp.ndarray
    tinted: jnp.ndarray
    expert_belt: jnp.ndarray
    life_orb: jnp.ndarray
    berry: jnp.ndarray  # resist berry applies (unless the hit lands on a Substitute)
    half: jnp.ndarray  # Super Fang / Ruination
    ok: jnp.ndarray  # damaging and not immune


def damage(move_id, att: Combatant, dfn: Combatant, field: FieldInfo, crit, roll, hits_sub=False, with_berry=False):
    """Damage dealt (int32, 0 if immune or not a damaging move). roll: 0..15, 0 = max damage.
    hits_sub: the hit lands on a Substitute (resist berries don't activate). with_berry: also return whether
    the defender's resist berry activated (and so is eaten)."""
    crit = jnp.asarray(crit) | (MOVES[move_id, MC["crit_ratio"]] >= 5)
    parts = damage_parts(move_id, att, dfn, field, crit)
    return damage_finish(parts, crit, roll, current_hp(dfn), dfn.stats[0], hits_sub, with_berry)


def damage_parts(move_id, att: Combatant, dfn: Combatant, field: FieldInfo, crit) -> Parts:
    """Per-move part of the formula for a given crit outcome (see Parts)."""
    row = MOVES[move_id]
    category = row[MC["category"]]
    physical = category == 0
    flags = row[MC["flags"]]
    crit = crit | (row[MC["crit_ratio"]] >= 5)  # willCrit moves (Flower Trick, Wicked Blow, ...)
    mtype, bp = move_type_and_bp(move_id, att, dfn, field.weather)
    att_types, def_types = current_types(att), current_types(dfn)
    att_hp = current_hp(att)

    immune, type_mod = type_effectiveness(mtype, def_types, move_id)
    immune |= (mtype == TY["ground"]) & ~is_grounded(dfn)
    half = (move_id == HALF_HP_MOVES[0]) | (move_id == HALF_HP_MOVES[1])  # damageCallback moves with 0 base power
    damaging = ((category != 2) & (bp > 0)) | half
    bp = jnp.maximum(bp, 1)

    punch = (flags & FLAG_BIT["punch"]) > 0
    contact = ((flags & FLAG_BIT["contact"]) > 0) & ~((att.item == ITEM.punchingglove) & punch)
    has_secondary = (row[MC["sec_chance"]] > 0) | (row[MC["sec_status"]] > 0)
    sheer_force = (att.ability == ABILITY.sheerforce) & has_secondary

    # --- BasePower event, in handler priority order ----------------------------------------------
    m = jnp.int32(4096)
    m = chain_if(m, (att.ability == ABILITY.technician) & (bp <= 60), M_1_5)  # 30
    m = chain_if(m, (att.ability == ABILITY.ironfist) & punch, M_1_2)  # 23
    m = chain_if(m, (att.ability == ABILITY.reckless) & (row[MC["recoil_num"]] > 0), M_1_2)
    m = chain_if(m, (att.item == ITEM.punchingglove) & punch, M_PUNCHING_GLOVE)
    m = chain_if(m, att.ability == ABILITY.supremeoverlord, OVERLORD[jnp.clip(att.fallen, 0, 5)])  # 21
    m = chain_if(m, (att.ability == ABILITY.toughclaws) & contact, M_1_3)
    sandy = (mtype == TY["rock"]) | (mtype == TY["ground"]) | (mtype == TY["steel"])
    m = chain_if(m, (att.ability == ABILITY.sandforce) & (field.weather == WEATHER_SAND) & sandy, M_1_3)
    m = chain_if(m, sheer_force, M_1_3)
    m = chain_if(m, (att.ability == ABILITY.sharpness) & ((flags & FLAG_BIT["slicing"]) > 0), M_1_5)  # 19
    m = chain_if(m, (att.ability == ABILITY.strongjaw) & ((flags & FLAG_BIT["bite"]) > 0), M_1_5)
    m = chain_if(m, (att.ability == ABILITY.megalauncher) & ((flags & FLAG_BIT["pulse"]) > 0), M_1_5)
    m = chain_if(m, (dfn.ability == T.ability_ids["dryskin"]) & (mtype == TY["fire"]), 5120)  # 17
    m = chain_if(m, (att.item == ITEM.wiseglasses) & (category == 1), M_1_1)  # 16
    m = chain_if(m, (att.item == ITEM.muscleband) & physical, M_1_1)
    m = chain_if(m, TYPE_BOOST_ITEM_J[att.item] == mtype, M_1_2)  # 15
    ogerpon = OGERPON_TYPE_J[att.species]
    mask_ok = jnp.select([ogerpon == t for t in MASK_FOR_TYPE], [att.item == i for i in MASK_FOR_TYPE.values()], False)
    m = chain_if(m, mask_ok, M_1_2)
    m = chain_if(
        m,
        (att.item == ITEM.souldew)
        & SOUL_DEW_USERS_J[att.species]
        & ((mtype == TY["psychic"]) | (mtype == TY["dragon"])),
        M_1_2,
    )
    m = chain_if(m, (att.ability == ABILITY.punkrock) & ((flags & FLAG_BIT["sound"]) > 0), M_1_3)  # 7
    att_grounded, def_grounded = is_grounded(att), is_grounded(dfn)
    terrain_boost = att_grounded & (
        ((field.terrain == TERRAIN_ELECTRIC) & (mtype == TY["electric"]))
        | ((field.terrain == TERRAIN_GRASSY) & (mtype == TY["grass"]))
        | ((field.terrain == TERRAIN_PSYCHIC) & (mtype == TY["psychic"]))
    )
    m = chain_if(m, terrain_boost, M_1_3)  # 6
    misty_dragon = (field.terrain == TERRAIN_MISTY) & def_grounded & (mtype == TY["dragon"])
    grassy_quake = (
        (field.terrain == TERRAIN_GRASSY)
        & def_grounded
        & ((move_id == T.move_ids["earthquake"]) | (move_id == T.move_ids["bulldoze"]))
    )
    m = chain_if(m, misty_dragon | grassy_quake, M_0_5)
    knock_boost = (move_id == MOVE.knockoff) & ITEM_REMOVABLE_J[dfn.item, dfn.species]  # 0: the move itself
    m = chain_if(m, knock_boost, M_1_5)
    m = chain_if(m, (move_id == MOVE.facade) & (att.status != 0) & (att.status != SLP), M_2)
    m = chain_if(
        m, ((move_id == MOVE.collisioncourse) | (move_id == MOVE.electrodrift)) & (type_mod > 0), M_SUPER_EFFECTIVE_4_3
    )
    bp = jnp.maximum(modify(bp, m), 1)
    stellar = att.terastallized & (att.tera_type == TYPE_STELLAR)
    stellar_fresh = stellar & ((jnp.asarray(att.stellar_used, jnp.int32) & (1 << mtype)) == 0)
    tera_floor = (
        att.terastallized
        & jnp.where(stellar, stellar_fresh, has_type(att_types, mtype))
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
    pinch_type = jnp.select(
        [
            att.ability == ABILITY.torrent,
            att.ability == ABILITY.blaze,
            att.ability == ABILITY.overgrow,
            att.ability == ABILITY.swarm,
        ],
        [TY["water"], TY["fire"], TY["grass"], TY["bug"]],
        -1,
    )
    a = jnp.int32(4096)
    a = chain_if(a, (dfn.ability == ABILITY.thickfat) & ((mtype == TY["fire"]) | (mtype == TY["ice"])), M_0_5)  # 6
    a = chain_if(a, (dfn.ability == ABILITY.heatproof) & (mtype == TY["fire"]), M_0_5)
    a = chain_if(a, (dfn.ability == ABILITY.purifyingsalt) & (mtype == TY["ghost"]), M_0_5)
    a = chain_if(a, (dfn.ability == ABILITY.waterbubble) & (mtype == TY["fire"]), M_0_5)  # 5
    a = chain_if(a, physical & ((att.ability == ABILITY.hugepower) | (att.ability == ABILITY.purepower)), M_2)
    a = chain_if(a, physical & (att.ability == ABILITY.guts) & (att.status != 0), M_1_5)
    a = chain_if(a, (pinch_type == mtype) & (att_hp * 3 <= att.stats[0]), M_1_5)
    a = chain_if(a, (att.ability == ABILITY.transistor) & (mtype == TY["electric"]), M_1_3)
    a = chain_if(a, (att.ability == ABILITY.dragonsmaw) & (mtype == TY["dragon"]), M_1_5)
    a = chain_if(a, (att.ability == ABILITY.steelworker) & (mtype == TY["steel"]), M_1_5)
    a = chain_if(a, (att.ability == ABILITY.rockypayload) & (mtype == TY["rock"]), M_1_5)
    a = chain_if(a, (att.ability == ABILITY.waterbubble) & (mtype == TY["water"]), M_2)
    a = chain_if(a, ~physical & (att.ability == ABILITY.solarpower) & (field.weather == WEATHER_SUN), M_1_5)
    a = chain_if(a, att.paradox_stat == event_stat, M_1_3)
    a = chain_if(a, physical & (att.item == ITEM.choiceband), M_1_5)  # 1
    a = chain_if(a, ~physical & (att.item == ITEM.choicespecs), M_1_5)
    a = chain_if(
        a, physical & (dfn.ability == ABILITY.tabletsofruin) & (att.ability != ABILITY.tabletsofruin), M_0_75
    )  # 0
    a = chain_if(a, ~physical & (dfn.ability == ABILITY.vesselofruin) & (att.ability != ABILITY.vesselofruin), M_0_75)
    attack = modify(attack, a)

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
    d = jnp.int32(4096)
    d = chain_if(d, (def_stat == 2) & (dfn.ability == ABILITY.furcoat), M_2)  # 6
    d = chain_if(d, dfn.paradox_stat == def_stat, M_1_3)
    d = chain_if(d, (dfn.item == ITEM.eviolite) & (SPECIES[dfn.species, SC["nfe"]] == 1), M_1_5)  # 2
    d = chain_if(d, (def_stat == 4) & (dfn.item == ITEM.assaultvest), M_1_5)  # 1
    d = chain_if(
        d, (def_stat == 2) & (att.ability == ABILITY.swordofruin) & (dfn.ability != ABILITY.swordofruin), M_0_75
    )  # 0
    d = chain_if(
        d, (def_stat == 4) & (att.ability == ABILITY.beadsofruin) & (dfn.ability != ABILITY.beadsofruin), M_0_75
    )
    defense = jnp.maximum(modify(defense, d), 1)

    base = (((2 * att.level // 5 + 2) * bp * attack) // defense) // 50

    # --- modifyDamage ----------------------------------------------------------------------------
    base = base + 2
    w_mod = jnp.int32(4096)
    sun, rain = field.weather == WEATHER_SUN, field.weather == WEATHER_RAIN
    hydro_steam = move_id == MOVE.hydrosteam  # boosted rather than weakened by sun
    w_mod = chain_if(w_mod, (sun & ((mtype == TY["fire"]) | hydro_steam)) | (rain & (mtype == TY["water"])), M_1_5)
    w_mod = chain_if(w_mod, (sun & (mtype == TY["water"]) & ~hydro_steam) | (rain & (mtype == TY["fire"])), M_0_5)
    base = modify(base, w_mod)

    orig_types = att.base_types
    stab_any = (mtype != TYPE_NONE) & (has_type(att_types, mtype) | has_type(orig_types, mtype))
    tera_match = att.terastallized & (att.tera_type == mtype) & has_type(orig_types, mtype)
    stab = jnp.where(tera_match, M_2, jnp.where(stab_any, M_1_5, 4096))
    adapt = (att.ability == ABILITY.adaptability) & (mtype != TYPE_NONE) & has_type(att_types, mtype)
    stab = jnp.where(adapt, jnp.where(stab == M_2, 9216, M_2), stab)
    # Stellar Tera: the first move of each type gets 2x (original STAB types) or 4915/4096
    stab = jnp.where(stellar_fresh, jnp.where(stab_any, M_2, M_STELLAR), stab)
    burned = (att.status == BRN) & physical & (att.ability != ABILITY.guts) & (move_id != MOVE.facade)
    screen = (physical & (field.reflect | field.aurora_veil)) | (~physical & (field.light_screen | field.aurora_veil))
    fluffy_x = jnp.where(mtype == TY["fire"], 2.0, 1.0) * jnp.where(contact, 0.5, 1.0)
    berry_type = RESIST_BERRY_J[dfn.item]
    return Parts(
        base=base,
        stab=stab,
        type_mod=type_mod,
        burned=burned,
        screen=screen,
        multiscale=(dfn.ability == ABILITY.multiscale) | (dfn.ability == ABILITY.shadowshield),
        fluffy=jnp.where(dfn.ability == ABILITY.fluffy, jnp.where(fluffy_x > 1, 2, jnp.where(fluffy_x < 1, 1, 0)), 0),
        punkrock=(dfn.ability == ABILITY.punkrock) & ((flags & FLAG_BIT["sound"]) > 0),
        icescales=(dfn.ability == ABILITY.icescales) & (category == 1),
        solid=(
            (dfn.ability == ABILITY.filter) | (dfn.ability == ABILITY.solidrock) | (dfn.ability == ABILITY.prismarmor)
        )
        & (type_mod > 0),
        sniper=att.ability == ABILITY.sniper,
        tinted=(att.ability == ABILITY.tintedlens) & (type_mod < 0),
        expert_belt=(att.item == ITEM.expertbelt) & (type_mod > 0),
        life_orb=att.item == ITEM.lifeorb,
        berry=(berry_type == mtype) & ((type_mod > 0) | RESIST_BERRY_ANY_HIT_J[dfn.item]),
        half=half,
        ok=damaging & ~immune,
    )


def damage_finish(p: Parts, crit, roll, def_hp, def_maxhp, hits_sub=False, with_berry=False):
    """Per-hit part: crit multiplier, roll, STAB, effectiveness, burn and the final modifier chain."""
    base = jnp.where(crit, p.base * 3 // 2, p.base)
    base = base * (100 - roll) // 100
    base = modify(base, p.stab)
    base = jnp.where(p.type_mod > 0, base << jnp.maximum(p.type_mod, 0), base)
    base = jnp.where(p.type_mod < 0, base >> jnp.maximum(-p.type_mod, 0), base)
    base = jnp.where(p.burned, modify(base, M_0_5), base)

    f = jnp.int32(4096)
    f = chain_if(f, p.screen & ~crit, M_0_5)
    f = chain_if(f, p.multiscale & (def_hp >= def_maxhp), M_0_5)
    f = chain_if(f, p.fluffy > 0, jnp.where(p.fluffy == 2, M_2, M_0_5))
    f = chain_if(f, p.punkrock, M_0_5)
    f = chain_if(f, p.icescales, M_0_5)
    f = chain_if(f, p.solid, M_0_75)
    f = chain_if(f, p.sniper & crit, M_1_5)
    f = chain_if(f, p.tinted, M_2)
    f = chain_if(f, p.expert_belt, M_EXPERT_BELT)
    f = chain_if(f, p.life_orb, M_LIFE_ORB)
    berry_hit = p.berry & ~hits_sub
    f = chain_if(f, berry_hit, M_0_5)
    base = modify(base, f)
    base = jnp.where(base == 0, 1, base) & 0xFFFF

    # Super Fang / Ruination: half the target's current HP (rounded down, at least 1)
    base = jnp.where(p.half, jnp.maximum(def_hp // 2, 1), base)
    out = jnp.where(p.ok, base, 0).astype(jnp.int32)
    if with_berry:
        return out, berry_hit & p.ok & ~p.half
    return out


def blocked_by_psychic_terrain(move_id, dfn: Combatant, field: FieldInfo, priority=None):
    """Psychic Terrain makes priority moves fail against grounded targets (checked before damage)."""
    priority = MOVES[move_id, MC["priority"]] if priority is None else priority
    return (field.terrain == TERRAIN_PSYCHIC) & (priority > 0) & is_grounded(dfn)
