"""
engine/oujax/engine.py
OU-JAX battle step: Showdown-style request flow, switching, move execution and end of turn for one battle,
written branch-free (jnp.where masks) so it vmaps cleanly. Each step consumes one action per side:

  PHASE_PREVIEW  action = lead slot (0..5); both leads switch in, then turn 1 starts
  PHASE_MOVE     action 0..3 = move slot, 4..7 = move slot + Terastallize, 8..12 = switch to the n-th benched
                 team slot (team order, skipping the active one)
  PHASE_SWITCH   only sides in need_switch act (switch actions); the other side's action is ignored. Used for
                 faint replacements at end of turn and for the mid-turn switch after a pivot move.

Randomness: every random check is an independent draw from the step key. With Config(forced=True) each kind
of draw is replaced by a fixed outcome matching tools/ref/ref.js's forced-outcome mode, so battles can be
compared with Showdown turn by turn.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from engine.oujax.damage import (
    M_0_5,
    M_1_5,
    M_2,
    MOVES,
    Combatant,
    FieldInfo,
    blocked_by_psychic_terrain,
    boosted_stat,
    chain_if,
    damage,
    has_type,
    modify,
    type_effectiveness,
)
from engine.oujax.effects import ABILITY, ITEM, ITEM_REMOVABLE, T
from engine.oujax.state import (
    HAZ_ROCKS,
    HAZ_SPIKES,
    HAZ_TSPIKES,
    HAZ_WEBS,
    NO_ACTION,
    PHASE_MOVE,
    PHASE_PREVIEW,
    PHASE_SWITCH,
    RESUME_END_OF_TURN,
    RESUME_NONE,
    RESUME_SECOND_MOVE,
    SCR_LIGHT_SCREEN,
    SCR_REFLECT,
    SCR_VEIL,
    WINNER_NONE,
    WINNER_TIE,
    Battle,
)
from engine.oujax.tables import (
    FLAG_BIT,
    MC,
    PSEUDO_WEATHERS,
    SIDE_CONDITIONS,
    STATUS_ID,
    TARGET_ID,
    TYPE_NONE,
    VOLATILES,
)

N_ACTIONS = 13
A_MOVE, A_TERA, A_SWITCH = 0, 4, 8
BRN, PAR, PSN, TOX, SLP, FRZ = (STATUS_ID[s] for s in ("brn", "par", "psn", "tox", "slp", "frz"))
SUN, RAIN, SAND, SNOW = 1, 2, 3, 4
E_TERRAIN, G_TERRAIN, M_TERRAIN, P_TERRAIN = 1, 2, 3, 4
TY = T.type_ids
MID = T.move_ids
ITEM_REMOVABLE_J = jnp.asarray(ITEM_REMOVABLE)
VOL = {v: i for i, v in enumerate(VOLATILES)}
SIDE = {v: i for i, v in enumerate(SIDE_CONDITIONS)}
PSEUDO = {v: i for i, v in enumerate(PSEUDO_WEATHERS)}
STRUGGLE = MID["struggle"]
CRIT_DEN = jnp.array([0, 24, 8, 2, 1], jnp.int32)
MULTIHIT_2_5 = jnp.array([2, 2, 2, 3, 3, 3, 4, 5], jnp.int32)
ACC_NUM = jnp.array([3, 4, 5, 6, 7, 8, 9], jnp.int32)  # accuracy stages x3: [1, 4/3, 5/3, 2, 7/3, 8/3, 3]
SPIKES_24THS = jnp.array([0, 3, 4, 6], jnp.int32)
MAX_HITS = 5
FOE_TARGETS = jnp.array(
    [TARGET_ID[t] for t in ("normal", "any", "allAdjacentFoes", "allAdjacent", "randomNormal", "adjacentFoe")]
)
WEATHER_ROCK = {SUN: ITEM.heatrock, RAIN: ITEM.damprock, SAND: ITEM.smoothrock, SNOW: ITEM.icyrock}
WEATHER_ABILITY = jnp.zeros(len(T.ability_ids), jnp.int32)
for _ab, _w in (
    (ABILITY.drought, SUN),
    (ABILITY.drizzle, RAIN),
    (ABILITY.sandstream, SAND),
    (ABILITY.snowwarning, SNOW),
):
    WEATHER_ABILITY = WEATHER_ABILITY.at[_ab].set(_w)
TERRAIN_ABILITY = jnp.zeros(len(T.ability_ids), jnp.int32)
for _ab, _t in (
    (ABILITY.electricsurge, E_TERRAIN),
    (ABILITY.grassysurge, G_TERRAIN),
    (ABILITY.mistysurge, M_TERRAIN),
    (ABILITY.psychicsurge, P_TERRAIN),
):
    TERRAIN_ABILITY = TERRAIN_ABILITY.at[_ab].set(_t)
SPEED_ABILITY_WEATHER = jnp.zeros(len(T.ability_ids), jnp.int32)
for _ab, _w in (
    (ABILITY.chlorophyll, SUN),
    (ABILITY.swiftswim, RAIN),
    (ABILITY.sandrush, SAND),
    (ABILITY.slushrush, SNOW),
):
    SPEED_ABILITY_WEATHER = SPEED_ABILITY_WEATHER.at[_ab].set(_w)

# Moves with hand-written behaviour (everything else is table-driven)
SPECIAL = {
    name: MID[name]
    for name in (
        "rapidspin mortalspin defog courtchange rest roost moonlight morningsun synthesis shoreup protect detect "
        "substitute partingshot teleport chillyreception haze auroraveil toxic thunder blizzard knockoff"
    ).split()
}


class Config(NamedTuple):
    """Static engine options. forced=True replaces every random draw by the fixed outcome below."""

    forced: bool = False
    hit: bool = True
    crit: bool = False
    roll: int = 0
    secondaries: bool = False
    full_para: bool = False
    thaw: bool = False
    stall: bool = True


# --- small helpers ---------------------------------------------------------------------------------


def _i(x):
    return jnp.asarray(x, jnp.int32)


def js_round_div(num, den):
    """Math.round(num / den) for non-negative integers (JavaScript rounds halves up)."""
    num, den = _i(num), jnp.maximum(_i(den), 1)
    return (2 * num + den) // (2 * den)


def chance(key, num, den, forced_value, cfg: Config):
    if cfg.forced:
        return jnp.bool_(forced_value)
    return jax.random.randint(key, (), 0, jnp.maximum(den, 1)) < num


def mon(arr, s, slot):
    return arr[s, slot]


def foe(s):
    return 1 - s


def act(st: Battle, s):
    return _i(st.active[s])


def alive_mask(st: Battle):
    return (st.hp > 0) & (st.species > 0)


def bench_available(st: Battle, s):
    slots = jnp.arange(6)
    return jnp.any(alive_mask(st)[s] & (slots != act(st, s)))


def types_now(st: Battle, s, slot):
    """Current types of a Pokemon: Tera type once Terastallized, minus Flying while roosting (active only)."""
    base = st.base_types[s, slot].astype(jnp.int32)
    is_active = slot == act(st, s)
    roost = is_active & st.roosted[s]
    base = jnp.where(roost & (base == TY["flying"]), TYPE_NONE, base)
    tera = st.terastallized[s, slot]
    return jnp.where(tera, jnp.array([_i(st.tera_type[s, slot]), TYPE_NONE]), base)


def grounded(st: Battle, s, slot):
    t = types_now(st, s, slot)
    return (
        ~has_type(t, TY["flying"]) & (st.ability[s, slot] != ABILITY.levitate) & (st.item[s, slot] != ITEM.airballoon)
    )


def combatant(st: Battle, s) -> Combatant:
    slot = act(st, s)
    base = st.base_types[s, slot].astype(jnp.int32)
    base = jnp.where(st.roosted[s] & (base == TY["flying"]), TYPE_NONE, base)
    return Combatant(
        species=_i(st.species[s, slot]),
        level=_i(st.level[s, slot]),
        stats=st.stats[s, slot].astype(jnp.int32),
        boosts=st.boosts[s].astype(jnp.int32),
        base_types=base,
        tera_type=_i(st.tera_type[s, slot]),
        terastallized=st.terastallized[s, slot],
        status=_i(st.status[s, slot]),
        item=_i(st.item[s, slot]),
        ability=_i(st.ability[s, slot]),
        paradox_stat=_i(st.paradox_stat[s]),
    )


def field_info(st: Battle, target_side) -> FieldInfo:
    return FieldInfo(
        weather=_i(st.weather),
        terrain=_i(st.terrain),
        reflect=st.screens[target_side, SCR_REFLECT] > 0,
        light_screen=st.screens[target_side, SCR_LIGHT_SCREEN] > 0,
        aurora_veil=st.screens[target_side, SCR_VEIL] > 0,
    )


def set_hp(st: Battle, s, slot, hp):
    return st.replace(hp=st.hp.at[s, slot].set(jnp.clip(hp, 0, st.stats[s, slot, 0]).astype(st.hp.dtype)))


def lose_hp(st: Battle, s, slot, amount, cond=True):
    """Battle.damage for a non-move source: truncate, at least 1, never below 0. Ignored when cond is False."""
    amount = jnp.maximum(_i(amount), 1)
    new = jnp.where(cond & (st.hp[s, slot] > 0), _i(st.hp[s, slot]) - amount, _i(st.hp[s, slot]))
    return set_hp(st, s, slot, new)


def gain_hp(st: Battle, s, slot, amount, cond=True):
    """Battle.heal: truncate, at least 1, only for a living Pokemon below max HP."""
    amount = jnp.maximum(_i(amount), 1)
    hp = _i(st.hp[s, slot])
    new = jnp.where(cond & (hp > 0), hp + amount, hp)
    return set_hp(st, s, slot, new)


def apply_boosts(st: Battle, s, deltas, cond=True):
    """Add stat stage changes to side s's active Pokemon. Returns (state, whether any stage changed)."""
    old = st.boosts[s].astype(jnp.int32)
    new = jnp.clip(old + jnp.where(cond, deltas, 0), -6, 6)
    return st.replace(boosts=st.boosts.at[s].set(new.astype(st.boosts.dtype))), jnp.any(new != old)


def eat_sitrus(st: Battle, s):
    """Sitrus Berry: at or below half HP, heal 1/4 max HP once (checked after every HP loss)."""
    slot = act(st, s)
    hp, maxhp = _i(st.hp[s, slot]), _i(st.stats[s, slot, 0])
    eat = (st.item[s, slot] == ITEM.sitrusberry) & (hp > 0) & (hp * 2 <= maxhp)
    st = gain_hp(st, s, slot, maxhp // 4, eat)
    return st.replace(item=st.item.at[s, slot].set(jnp.where(eat, 0, st.item[s, slot])))


def status_immune(st: Battle, s, slot, status):
    t = types_now(st, s, slot)
    return (
        ((status == BRN) & has_type(t, TY["fire"]))
        | ((status == PAR) & has_type(t, TY["electric"]))
        | (((status == PSN) | (status == TOX)) & (has_type(t, TY["poison"]) | has_type(t, TY["steel"])))
        | ((status == FRZ) & has_type(t, TY["ice"]))
    )


def try_set_status(st: Battle, s, slot, status, key, cfg: Config, from_foe=True, cond=True):
    """Pokemon.trySetStatus with Showdown's status immunities, terrain blocks and Sleep Clause.
    Returns (state, applied)."""
    is_grounded_ = grounded(st, s, slot)
    misty = (st.terrain == M_TERRAIN) & is_grounded_
    electric_sleep = (st.terrain == E_TERRAIN) & is_grounded_ & (status == SLP)
    asleep_from_foe = alive_mask(st)[s] & (st.status[s] == SLP) & ~st.self_sleep[s]
    sleep_clause = (status == SLP) & from_foe & jnp.any(asleep_from_foe)
    ok = (
        cond
        & (status > 0)
        & (st.hp[s, slot] > 0)
        & (st.status[s, slot] == 0)
        & ~status_immune(st, s, slot, status)
        & ~misty
        & ~electric_sleep
        & ~sleep_clause
    )
    sleep_turns = 2 if cfg.forced else jax.random.randint(key, (), 2, 5)  # Showdown: random(2, 5)
    turns = jnp.where(status == SLP, sleep_turns, 0)
    st = st.replace(
        status=st.status.at[s, slot].set(jnp.where(ok, status, st.status[s, slot]).astype(st.status.dtype)),
        status_turns=st.status_turns.at[s, slot].set(jnp.where(ok, turns, st.status_turns[s, slot]).astype(jnp.int8)),
        self_sleep=st.self_sleep.at[s, slot].set(jnp.where(ok, False, st.self_sleep[s, slot])),
    )
    return st, ok


# --- speed and order -------------------------------------------------------------------------------


def action_speed(st: Battle, s):
    """Pokemon.getActionSpeed: boosted Speed with ModifySpe effects, Trick Room inversion, 13-bit truncation."""
    slot = act(st, s)
    spe = boosted_stat(_i(st.stats[s, slot, 5]), _i(st.boosts[s, 4]))
    mod = jnp.int32(4096)
    mod = chain_if(mod, (st.status[s, slot] == PAR), M_0_5)
    mod = chain_if(mod, st.item[s, slot] == ITEM.choicescarf, M_1_5)
    mod = chain_if(mod, st.paradox_stat[s] == 5, M_1_5)
    w = SPEED_ABILITY_WEATHER[st.ability[s, slot]]
    mod = chain_if(mod, (w > 0) & (w == st.weather), M_2)
    mod = chain_if(mod, st.tailwind[s] > 0, M_2)
    spe = modify(spe, mod)
    spe = jnp.where(st.trick_room > 0, 10000 - spe, spe)
    return spe & 8191


def p1_first(key, prio, spe, cfg: Config):
    """Whether side 0 acts before side 1: priority, then speed, then a coin flip (p1 when forced)."""
    tie = (prio[0] == prio[1]) & (spe[0] == spe[1])
    coin = True if cfg.forced else jax.random.bernoulli(key)
    return jnp.where(prio[0] != prio[1], prio[0] > prio[1], jnp.where(tie, coin, spe[0] > spe[1]))


# --- field effects ---------------------------------------------------------------------------------


def refresh_paradox(st: Battle):
    """Protosynthesis / Quark Drive: field-granted boosts end with sun / Electric Terrain; then an inactive
    ability activates from the field, or by consuming Booster Energy when the field doesn't provide it."""
    for s in (0, 1):
        slot = act(st, s)
        on_field = (slot >= 0) & (st.hp[s, jnp.maximum(slot, 0)] > 0)
        slot = jnp.maximum(slot, 0)
        ab = st.ability[s, slot]
        proto, quark = ab == ABILITY.protosynthesis, ab == ABILITY.quarkdrive
        field_on = (proto & (st.weather == SUN)) | (quark & (st.terrain == E_TERRAIN))
        ends = (st.paradox_stat[s] > 0) & ~field_on & ~st.paradox_from_item[s]
        current = jnp.where(ends, 0, st.paradox_stat[s])
        item = st.item[s, slot]
        use_booster = on_field & (proto | quark) & ~field_on & (item == ITEM.boosterenergy) & (current == 0)
        newly = on_field & (current == 0) & (field_on | use_booster)
        stat = jnp.where(newly, best_stat(st, s, slot), current)
        st = st.replace(
            paradox_stat=st.paradox_stat.at[s].set(stat.astype(jnp.int8)),
            paradox_from_item=st.paradox_from_item.at[s].set(
                jnp.where(newly, use_booster, jnp.where(ends, False, st.paradox_from_item[s]))
            ),
            item=st.item.at[s, slot].set(jnp.where(use_booster, 0, item)),
        )
    return st


def best_stat(st: Battle, s, slot):
    """Pokemon.getBestStat(unboosted=false, unmodified=true): highest boosted stat, ties to the earlier one."""
    vals = jnp.stack([boosted_stat(_i(st.stats[s, slot, i]), _i(st.boosts[s, i - 1])) for i in range(1, 6)])
    return jnp.argmax(vals) + 1


def set_weather(st: Battle, weather, s, slot, cond):
    """Field.setWeather from side s's Pokemon: fails if that weather is already up; 8 turns with its rock."""
    rock = jnp.select([weather == w for w in WEATHER_ROCK], [r for r in WEATHER_ROCK.values()], -1)
    turns = jnp.where(st.item[s, slot] == rock, 8, 5)
    ok = cond & (weather > 0) & (st.weather != weather)
    st = st.replace(
        weather=jnp.where(ok, weather, st.weather).astype(jnp.int8),
        weather_turns=jnp.where(ok, turns, st.weather_turns).astype(jnp.int8),
    )
    return refresh_paradox(st), ok


def set_terrain(st: Battle, terrain, s, slot, cond):
    turns = jnp.where(st.item[s, slot] == ITEM.terrainextender, 8, 5)
    ok = cond & (terrain > 0) & (st.terrain != terrain)
    st = st.replace(
        terrain=jnp.where(ok, terrain, st.terrain).astype(jnp.int8),
        terrain_turns=jnp.where(ok, turns, st.terrain_turns).astype(jnp.int8),
    )
    return refresh_paradox(st), ok


# --- switching -------------------------------------------------------------------------------------


def switch_out(st: Battle, s, cond):
    """Leaving the field: Regenerator, Natural Cure, and every volatile resets."""
    slot = jnp.maximum(act(st, s), 0)
    alive = st.hp[s, slot] > 0
    regen = cond & alive & (st.ability[s, slot] == ABILITY.regenerator)
    st = gain_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 3, regen)
    cure = cond & alive & (st.ability[s, slot] == ABILITY.naturalcure)
    st = st.replace(status=st.status.at[s, slot].set(jnp.where(cure, 0, st.status[s, slot])))
    return reset_volatiles(st, s, cond)


def reset_volatiles(st: Battle, s, cond):
    def w(arr, value):
        return arr.at[s].set(jnp.where(cond, value, arr[s]).astype(arr.dtype))

    return st.replace(
        boosts=st.boosts.at[s].set(jnp.where(cond, 0, st.boosts[s])),
        paradox_stat=w(st.paradox_stat, 0),
        paradox_from_item=w(st.paradox_from_item, False),
        choice_lock=w(st.choice_lock, -1),
        protect_count=w(st.protect_count, 0),
        protect_used=w(st.protect_used, False),
        protected=w(st.protected, False),
        sub_hp=w(st.sub_hp, 0),
        flinch=w(st.flinch, False),
        roosted=w(st.roosted, False),
    )


def enter_field(st: Battle, s, slot, cond):
    """Put team slot `slot` in side s's active position (no entry effects yet)."""
    st = st.replace(active=st.active.at[s].set(jnp.where(cond, slot, st.active[s]).astype(jnp.int8)))
    # Toxic counter restarts on switch-in
    tox = cond & (st.status[s, slot] == TOX)
    return st.replace(status_turns=st.status_turns.at[s, slot].set(jnp.where(tox, 0, st.status_turns[s, slot])))


def run_switch_in(st: Battle, s, key, cfg: Config, cond):
    """Entry effects for side s's new active Pokemon (Battle.runSwitch): hazards, then ability, then item."""
    slot = act(st, s)
    boots = st.item[s, slot] == ITEM.heavydutyboots
    gr = grounded(st, s, slot)
    maxhp = _i(st.stats[s, slot, 0])
    hz = st.hazards[s].astype(jnp.int32)
    # Stealth Rock
    _, rock_mod = type_effectiveness(TY["rock"], types_now(st, s, slot))
    rock_dmg = jnp.where(rock_mod >= 0, maxhp << jnp.maximum(rock_mod, 0), maxhp >> jnp.maximum(-rock_mod, 0)) // 8
    st = lose_hp(st, s, slot, rock_dmg, cond & (hz[HAZ_ROCKS] > 0) & ~boots)
    # Spikes
    st = lose_hp(
        st,
        s,
        slot,
        SPIKES_24THS[jnp.minimum(hz[HAZ_SPIKES], 3)] * maxhp // 24,
        cond & (hz[HAZ_SPIKES] > 0) & gr & ~boots,
    )
    # Toxic Spikes: absorbed by grounded Poison types, ignored by Steel types and Boots
    t = types_now(st, s, slot)
    tsp = cond & (hz[HAZ_TSPIKES] > 0) & gr & (st.hp[s, slot] > 0)
    absorb = tsp & has_type(t, TY["poison"])
    st = st.replace(hazards=st.hazards.at[s, HAZ_TSPIKES].set(jnp.where(absorb, 0, st.hazards[s, HAZ_TSPIKES])))
    poison = tsp & ~absorb & ~has_type(t, TY["steel"]) & ~boots
    st, _ = try_set_status(st, s, slot, jnp.where(hz[HAZ_TSPIKES] >= 2, TOX, PSN), key, cfg, from_foe=True, cond=poison)
    # Sticky Web
    webs = cond & (hz[HAZ_WEBS] > 0) & gr & ~boots & (st.hp[s, slot] > 0)
    st, _ = apply_boosts(st, s, jnp.array([0, 0, 0, 0, -1, 0, 0]), webs)
    st = eat_sitrus(st, s)
    # Ability: weather / terrain setters
    alive = cond & (st.hp[s, slot] > 0)
    st, _ = set_weather(st, WEATHER_ABILITY[st.ability[s, slot]], s, slot, alive)
    st, _ = set_terrain(st, TERRAIN_ABILITY[st.ability[s, slot]], s, slot, alive)
    # Protosynthesis / Quark Drive and Booster Energy
    return refresh_paradox(st)


def switch_target(st: Battle, s, action):
    """Team slot for switch action 8..12: the n-th non-active team slot, in team order."""
    n = action - A_SWITCH
    slots = jnp.arange(6)
    benched = slots != act(st, s)
    rank = jnp.cumsum(benched) - 1
    return jnp.argmax(benched & (rank == n))


def do_switches(st: Battle, wants, targets, key, cfg: Config):
    """Switch both sides' Pokemon simultaneously-chosen switches: the faster side switches (and runs its entry
    effects) first."""
    k_order, k0, k1 = jax.random.split(key, 3)
    spe = jnp.stack([action_speed(st, 0), action_speed(st, 1)])
    first0 = p1_first(k_order, jnp.zeros(2, jnp.int32), spe, cfg)
    order = jnp.where(first0, jnp.array([0, 1]), jnp.array([1, 0]))
    keys = (k0, k1)
    for i in range(2):
        s = order[i]
        cond = wants[s]
        st = switch_out(st, s, cond & (st.active[s] >= 0))
        st = enter_field(st, s, targets[s], cond)
        st = run_switch_in(st, s, keys[i], cfg, cond)
    return st


# --- moves -----------------------------------------------------------------------------------------


def move_row(move_id, col):
    return MOVES[move_id, MC[col]]


def boosts_of(move_id, prefix):
    return jnp.stack(
        [MOVES[move_id, MC[f"{prefix}{b}"]] for b in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")]
    )


def run_move(st: Battle, s, move_slot, struggle, enabled, last_to_act, key, cfg: Config):
    """Side s's active Pokemon uses the move in `move_slot` (or Struggle). Returns (state, pivot), where
    pivot asks for a mid-turn switch from side s (U-turn, Volt Switch, Flip Turn, Parting Shot, Teleport,
    Chilly Reception)."""
    keys = jax.random.split(key, 16)
    o = foe(s)
    slot = act(st, s)
    tslot = act(st, o)
    user_alive = enabled & (st.hp[s, slot] > 0)
    mid = jnp.where(struggle, STRUGGLE, _i(st.moves[s, slot, move_slot]))
    row = MOVES[mid]
    category = row[MC["category"]]
    is_status = category == 2
    targets_foe = jnp.any(row[MC["target"]] == FOE_TARGETS)
    flags = row[MC["flags"]]

    # BeforeMove: sleep and freeze, flinch, full paralysis
    status = st.status[s, slot]
    asleep = status == SLP
    turns_left = _i(st.status_turns[s, slot]) - 1
    wakes = asleep & (turns_left <= 0)
    st = st.replace(
        status_turns=st.status_turns.at[s, slot].set(
            jnp.where(asleep & user_alive, jnp.maximum(turns_left, 0), st.status_turns[s, slot]).astype(jnp.int8)
        ),
        status=st.status.at[s, slot].set(jnp.where(wakes & user_alive, 0, st.status[s, slot])),
    )
    thaws = (status == FRZ) & (((flags & FLAG_BIT["defrost"]) > 0) | chance(keys[0], 1, 5, cfg.thaw, cfg))
    st = st.replace(
        status=st.status.at[s, slot].set(jnp.where((status == FRZ) & thaws & user_alive, 0, st.status[s, slot]))
    )
    blocked = (asleep & ~wakes) | ((status == FRZ) & ~thaws)
    blocked |= st.flinch[s]
    blocked |= (status == PAR) & chance(keys[1], 1, 4, cfg.full_para, cfg)
    moves = user_alive & ~blocked

    # Use the move: PP (Pressure costs one more), Choice lock
    target_alive = st.hp[o, tslot] > 0
    # Pokemon.getMoveTargets' pressureTargets: targeted foes and whole-field moves, not foe-side moves unless
    # flagged mustpressure (Stealth Rock, Spikes, Toxic Spikes)
    aims_at_foe = targets_foe | (row[MC["target"]] == TARGET_ID["all"]) | ((flags & FLAG_BIT["mustpressure"]) > 0)
    pressure = aims_at_foe & (st.ability[o, tslot] == ABILITY.pressure) & target_alive
    pp_cost = jnp.where(struggle, 0, 1 + pressure.astype(jnp.int32))
    st = st.replace(
        pp=st.pp.at[s, slot, move_slot].set(
            jnp.where(moves, jnp.maximum(_i(st.pp[s, slot, move_slot]) - pp_cost, 0), st.pp[s, slot, move_slot]).astype(
                jnp.int8
            )
        )
    )
    is_choice = (
        (st.item[s, slot] == ITEM.choiceband)
        | (st.item[s, slot] == ITEM.choicespecs)
        | (st.item[s, slot] == ITEM.choicescarf)
    )
    st = st.replace(
        choice_lock=st.choice_lock.at[s].set(
            jnp.where(moves & is_choice & ~struggle & (st.choice_lock[s] < 0), move_slot, st.choice_lock[s]).astype(
                jnp.int8
            )
        )
    )

    # Targeting, then the move's own failure conditions
    no_target = targets_foe & ~target_alive
    fails = no_target
    fails |= targets_foe & blocked_by_psychic_terrain(mid, combatant(st, o), field_info(st, o))
    fails |= (mid == MID["auroraveil"]) & (st.weather != SNOW)
    is_protect = (mid == SPECIAL["protect"]) | (mid == SPECIAL["detect"])
    stall_den = jnp.array([1, 3, 9, 27, 81, 243, 729])[jnp.minimum(st.protect_count[s], 6)]
    protect_ok = ~last_to_act & chance(keys[2], 1, stall_den, cfg.stall, cfg)
    fails |= is_protect & ~protect_ok
    maxhp = _i(st.stats[s, slot, 0])
    fails |= (mid == MID["substitute"]) & ((st.sub_hp[s] > 0) | (_i(st.hp[s, slot]) <= maxhp // 4))
    fails |= (mid == MID["rest"]) & (
        (st.hp[s, slot] >= maxhp) | (st.status[s, slot] == SLP) | ~try_rest_allowed(st, s, slot)
    )
    heal_frac = (
        (row[MC["heal_num"]] > 0)
        | (mid == SPECIAL["moonlight"])
        | (mid == SPECIAL["morningsun"])
        | (mid == SPECIAL["synthesis"])
        | (mid == SPECIAL["shoreup"])
    )
    fails |= heal_frac & (st.hp[s, slot] >= maxhp)
    fails |= (mid == MID["teleport"]) & ~bench_available(st, s)
    st = st.replace(protect_used=st.protect_used.at[s].set(st.protect_used[s] | (moves & is_protect)))
    proceeds = moves & ~fails

    # Protect blocks moves with the protect flag aimed at the target
    blocked_by_protect = targets_foe & st.protected[o] & ((flags & FLAG_BIT["protect"]) > 0)
    proceeds_hit = proceeds & ~blocked_by_protect

    # Accuracy
    acc = _i(row[MC["accuracy"]])
    acc = jnp.where(
        ((mid == SPECIAL["thunder"]) & (st.weather == RAIN)) | ((mid == SPECIAL["blizzard"]) & (st.weather == SNOW)),
        101,
        acc,
    )
    acc = jnp.where((mid == SPECIAL["thunder"]) & (st.weather == SUN), 50, acc)
    acc = jnp.where((mid == SPECIAL["toxic"]) & has_type(types_now(st, s, slot), TY["poison"]), 101, acc)
    acc = jnp.where(struggle, 101, acc)
    stage = jnp.clip(_i(st.boosts[s, 5]) - _i(st.boosts[o, 6]), -6, 6)
    acc_mod = jnp.where(stage >= 0, acc * ACC_NUM[jnp.abs(stage)] // 3, acc * 3 // ACC_NUM[jnp.abs(stage)])
    needs_roll = targets_foe & (acc < 101)
    hits = ~needs_roll | chance(keys[3], acc_mod, 100, cfg.hit, cfg)
    landed = proceeds_hit & hits

    # Type immunity (damaging moves, and status moves that respect it like Thunder Wave)
    t_types = types_now(st, o, tslot)
    mtype = jnp.where(mid == MID["weatherball"], weather_ball_type(st), row[MC["type"]])
    immune, _ = type_effectiveness(mtype, t_types)
    immune |= (mtype == TY["ground"]) & ~grounded(st, o, tslot)
    immune &= targets_foe & (row[MC["type_immunity"]] == 1) & ~struggle
    immune |= targets_foe & ((flags & FLAG_BIT["powder"]) > 0) & has_type(t_types, TY["grass"])
    landed &= ~immune

    st, dealt, hit_sub = damage_phase(st, s, mid, landed & ~is_status, keys[4:10], cfg)
    st = after_damage(st, s, mid, landed & ~is_status, dealt, hit_sub, keys[10], cfg)
    st, pivot_ok = status_phase(st, s, mid, landed & is_status, keys[11], cfg)
    st = st.replace(protected=st.protected.at[s].set(st.protected[s] | (proceeds & is_protect)))

    # Pivot: damaging pivots switch if they connected; status pivots per status_phase
    self_switch = row[MC["self_switch"]] > 0
    pivot = self_switch & jnp.where(is_status, pivot_ok, landed) & (st.hp[s, slot] > 0) & bench_available(st, s)
    return st, pivot


def try_rest_allowed(st: Battle, s, slot):
    """Rest fails under Electric / Misty Terrain while grounded, and for Pokemon that can't fall asleep."""
    gr = grounded(st, s, slot)
    return ~(((st.terrain == E_TERRAIN) | (st.terrain == M_TERRAIN)) & gr)


def weather_ball_type(st: Battle):
    w = st.weather
    return jnp.select(
        [w == SUN, w == RAIN, w == SAND, w == SNOW], [TY["fire"], TY["water"], TY["rock"], TY["ice"]], TY["normal"]
    )


def damage_phase(st: Battle, s, mid, active, keys, cfg: Config):
    """All hits of a damaging move: per-hit crit and roll, Substitute absorption, Air Balloon, Sitrus.
    Returns (state, total damage dealt, whether a substitute took the hits)."""
    o = foe(s)
    tslot = act(st, o)
    row = MOVES[mid]
    mh_min, mh_max = row[MC["multihit_min"]], row[MC["multihit_max"]]
    if cfg.forced:
        n_hits = jnp.where(mh_max == 0, 1, jnp.where(mh_min == mh_max, mh_min, 2))
    else:
        sampled = MULTIHIT_2_5[jax.random.randint(keys[0], (), 0, 8)]
        n_hits = jnp.where(mh_max == 0, 1, jnp.where(mh_min == mh_max, mh_min, sampled))
    crit_stage = jnp.clip(row[MC["crit_ratio"]], 0, 4)
    bypass_sub = (row[MC["flags"]] & (FLAG_BIT["sound"] | FLAG_BIT["bypasssub"])) > 0
    had_sub = active & (st.sub_hp[o] > 0) & ~bypass_sub
    total = jnp.int32(0)
    for h in range(MAX_HITS):
        hk = jax.random.fold_in(keys[1], h)
        kc, kr = jax.random.split(hk)
        go = active & (h < n_hits) & (st.hp[o, tslot] > 0) & (st.hp[s, act(st, s)] > 0)
        if cfg.forced:
            crit = jnp.bool_(cfg.crit)
            roll = jnp.int32(cfg.roll)
        else:
            crit = jax.random.randint(kc, (), 0, CRIT_DEN[jnp.maximum(crit_stage, 1)]) == 0
            crit = crit & (crit_stage > 0)
            roll = jax.random.randint(kr, (), 0, 16)
        d = damage(mid, combatant(st, s), combatant(st, o), field_info(st, o), crit, roll)
        # Seismic Toss / Night Shade: damage equals the user's level (type immunity already checked)
        d = jnp.where(row[MC["fixed_damage"]] == -1, _i(st.level[s, act(st, s)]), d)
        sub_now = go & (st.sub_hp[o] > 0) & ~bypass_sub
        to_sub = jnp.minimum(d, _i(st.sub_hp[o]))
        st = st.replace(
            sub_hp=st.sub_hp.at[o].set(
                jnp.where(sub_now, _i(st.sub_hp[o]) - to_sub, st.sub_hp[o]).astype(st.sub_hp.dtype)
            )
        )
        to_mon = jnp.minimum(d, _i(st.hp[o, tslot]))
        st = set_hp(st, o, tslot, jnp.where(go & ~sub_now, _i(st.hp[o, tslot]) - to_mon, st.hp[o, tslot]))
        # Air Balloon pops when the holder itself is hit
        pop = go & ~sub_now & (d > 0) & (st.item[o, tslot] == ITEM.airballoon)
        st = st.replace(item=st.item.at[o, tslot].set(jnp.where(pop, 0, st.item[o, tslot])))
        # Berries activate between hits of a multi-hit move; after the last hit they wait for the move's
        # after-hit effects (so Knock Off can remove a Sitrus Berry before it is eaten)
        st = _eat_if(st, o, go & (h + 1 < n_hits))
        total += jnp.where(go, jnp.where(sub_now, to_sub, to_mon), 0)
    return st, total, had_sub


def _eat_if(st: Battle, s, cond):
    eaten = eat_sitrus(st, s)
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), eaten, st)


def after_damage(st: Battle, s, mid, landed, dealt, hit_sub, key, cfg: Config):
    """Effects after a damaging move connects: drain, Rocky Helmet, secondaries and self drops, Knock Off,
    hazard removal, recoil and Life Orb."""
    o = foe(s)
    slot, tslot = act(st, s), act(st, o)
    row = MOVES[mid]
    maxhp = _i(st.stats[s, slot, 0])
    connected = landed & (dealt > 0)
    # Drain
    drain = (row[MC["drain_num"]] > 0) & connected
    st = gain_hp(st, s, slot, js_round_div(dealt * row[MC["drain_num"]], row[MC["drain_den"]]), drain)
    # Rocky Helmet on contact (not through a substitute)
    contact = (row[MC["flags"]] & FLAG_BIT["contact"]) > 0
    helmet = connected & contact & ~hit_sub & (st.item[o, tslot] == ITEM.rockyhelmet)
    st = lose_hp(st, s, slot, maxhp // 6, helmet)
    # Self stat drops / raises from the move (Close Combat, Draco Meteor, ...)
    st, _ = apply_boosts(st, s, boosts_of(mid, "self_boost_"), landed & (st.hp[s, slot] > 0))
    # Secondary effect: one roll against the move's chance (100% effects always apply)
    if cfg.forced:
        sec_roll = jnp.int32(0 if cfg.secondaries else 99)
    else:
        sec_roll = jax.random.randint(key, (), 0, 100)
    k_status = jax.random.fold_in(key, 1)
    sec = landed & (row[MC["sec_chance"]] > 0) & (sec_roll < row[MC["sec_chance"]])
    on_target = sec & ~hit_sub & (st.hp[o, tslot] > 0)
    st, _ = try_set_status(
        st, o, tslot, row[MC["sec_status"]], k_status, cfg, cond=on_target & (row[MC["sec_status"]] > 0)
    )
    st, _ = apply_boosts(st, o, boosts_of(mid, "sec_boost_"), on_target)
    st = st.replace(flinch=st.flinch.at[o].set(st.flinch[o] | (on_target & (row[MC["sec_volatile"]] == VOL["flinch"]))))
    st, _ = apply_boosts(st, s, boosts_of(mid, "sec_self_boost_"), sec & (st.hp[s, slot] > 0))
    # Knock Off removes a removable item (the user must still be standing)
    knock = landed & (mid == MID["knockoff"]) & (st.hp[s, slot] > 0) & ~hit_sub
    knock &= ITEM_REMOVABLE_J[st.item[o, tslot], st.species[o, tslot]]
    st = st.replace(item=st.item.at[o, tslot].set(jnp.where(knock, 0, st.item[o, tslot])))
    # Hazard removal from the user's side
    spin = landed & ((mid == MID["rapidspin"]) | (mid == MID["mortalspin"])) & (st.hp[s, slot] > 0)
    st = st.replace(hazards=st.hazards.at[s].set(jnp.where(spin, 0, st.hazards[s])))
    # Recoil (from damage dealt) and Life Orb
    recoil = (row[MC["recoil_num"]] > 0) & connected
    st = lose_hp(st, s, slot, js_round_div(dealt * row[MC["recoil_num"]], row[MC["recoil_den"]]), recoil)
    struggle_recoil = landed & (mid == STRUGGLE)
    st = lose_hp(st, s, slot, js_round_div(maxhp, 4), struggle_recoil)
    st = lose_hp(st, s, slot, maxhp // 10, landed & (st.item[s, slot] == ITEM.lifeorb))
    return eat_sitrus(eat_sitrus(st, o), s)


def status_phase(st: Battle, s, mid, landed, key, cfg: Config):
    """Status moves: stat changes, status conditions, healing, Protect, Substitute, hazards, screens,
    weather, terrain, Trick Room, Haze, Defog, Court Change and status pivots. Returns (state, pivot_ok)."""
    o = foe(s)
    slot, tslot = act(st, s), act(st, o)
    row = MOVES[mid]
    targets_self = (
        (row[MC["target"]] == TARGET_ID["self"])
        | (row[MC["target"]] == TARGET_ID["allySide"])
        | (row[MC["target"]] == TARGET_ID["all"])
    )
    on_foe = landed & ~targets_self
    bypass_sub = (row[MC["flags"]] & (FLAG_BIT["sound"] | FLAG_BIT["bypasssub"])) > 0
    blocked_sub = on_foe & (st.sub_hp[o] > 0) & ~bypass_sub
    hits_foe = on_foe & ~blocked_sub
    maxhp = _i(st.stats[s, slot, 0])

    # Stat changes
    deltas = boosts_of(mid, "boost_")
    st, _ = apply_boosts(st, s, deltas, landed & targets_self)
    st, foe_boosted = apply_boosts(st, o, deltas, hits_foe & (st.hp[o, tslot] > 0))
    # Status conditions
    st, _ = try_set_status(st, o, tslot, row[MC["status"]], key, cfg, cond=hits_foe & (row[MC["status"]] > 0))
    # Healing: fixed fractions, weather-dependent heals, Roost, Rest
    frac_heal = js_round_div(maxhp * row[MC["heal_num"]], row[MC["heal_den"]])
    st = gain_hp(st, s, slot, frac_heal, landed & (row[MC["heal_num"]] > 0))
    w = st.weather
    sun_heal = (mid == MID["moonlight"]) | (mid == MID["morningsun"]) | (mid == MID["synthesis"])
    factor = jnp.where(w == SUN, 2732, jnp.where((w == RAIN) | (w == SAND) | (w == SNOW), 1024, 2048))
    st = gain_hp(st, s, slot, modify(maxhp, factor), landed & sun_heal)
    shore = jnp.where(w == SAND, 2732, 2048)
    st = gain_hp(st, s, slot, modify(maxhp, shore), landed & (mid == MID["shoreup"]))
    st = st.replace(roosted=st.roosted.at[s].set(st.roosted[s] | (landed & (mid == MID["roost"]))))
    rest = landed & (mid == MID["rest"])
    st = set_hp(st, s, slot, jnp.where(rest, maxhp, st.hp[s, slot]))
    st = st.replace(
        status=st.status.at[s, slot].set(jnp.where(rest, SLP, st.status[s, slot]).astype(st.status.dtype)),
        status_turns=st.status_turns.at[s, slot].set(jnp.where(rest, 3, st.status_turns[s, slot]).astype(jnp.int8)),
        self_sleep=st.self_sleep.at[s, slot].set(jnp.where(rest, True, st.self_sleep[s, slot])),
    )
    # Protect counter, Substitute
    is_protect = (mid == MID["protect"]) | (mid == MID["detect"])
    st = st.replace(
        protect_count=st.protect_count.at[s].set(
            jnp.where(landed & is_protect, st.protect_count[s] + 1, st.protect_count[s]).astype(jnp.int8)
        )
    )
    sub = landed & (mid == MID["substitute"])
    st = lose_hp(st, s, slot, maxhp // 4, sub)
    st = st.replace(sub_hp=st.sub_hp.at[s].set(jnp.where(sub, maxhp // 4, st.sub_hp[s]).astype(st.sub_hp.dtype)))
    # Entry hazards on the foe's side
    sc = row[MC["side_condition"]]
    hz = st.hazards[o].astype(jnp.int32)
    hz = jnp.where(landed & (sc == SIDE["stealthrock"]), hz.at[HAZ_ROCKS].set(1), hz)
    hz = jnp.where(landed & (sc == SIDE["spikes"]), hz.at[HAZ_SPIKES].set(jnp.minimum(hz[HAZ_SPIKES] + 1, 3)), hz)
    hz = jnp.where(
        landed & (sc == SIDE["toxicspikes"]), hz.at[HAZ_TSPIKES].set(jnp.minimum(hz[HAZ_TSPIKES] + 1, 2)), hz
    )
    hz = jnp.where(landed & (sc == SIDE["stickyweb"]), hz.at[HAZ_WEBS].set(1), hz)
    st = st.replace(hazards=st.hazards.at[o].set(hz.astype(jnp.int8)))
    # Screens and Tailwind on the user's side (fail while already up)
    clay = st.item[s, slot] == ITEM.lightclay
    scr = st.screens[s].astype(jnp.int32)
    for cond_id, idx in (
        (SIDE["reflect"], SCR_REFLECT),
        (SIDE["lightscreen"], SCR_LIGHT_SCREEN),
        (SIDE["auroraveil"], SCR_VEIL),
    ):
        put = landed & (sc == cond_id) & (scr[idx] == 0)
        scr = jnp.where(put, scr.at[idx].set(jnp.where(clay, 8, 5)), scr)
    st = st.replace(screens=st.screens.at[s].set(scr.astype(jnp.int8)))
    tail = landed & (sc == SIDE["tailwind"]) & (st.tailwind[s] == 0)
    st = st.replace(tailwind=st.tailwind.at[s].set(jnp.where(tail, 4, st.tailwind[s]).astype(jnp.int8)))
    # Weather, terrain, Trick Room
    st, _ = set_weather(st, row[MC["weather"]], s, slot, landed)
    st, _ = set_terrain(st, row[MC["terrain"]], s, slot, landed)
    tr = landed & (row[MC["pseudo_weather"]] == PSEUDO["trickroom"])
    st = st.replace(trick_room=jnp.where(tr, jnp.where(st.trick_room > 0, 0, 5), st.trick_room).astype(jnp.int8))
    # Haze: clear both sides' stat stages
    haze = landed & (mid == MID["haze"])
    st = st.replace(boosts=jnp.where(haze, 0, st.boosts))
    # Defog: -1 evasion to the target (its onHit; a substitute blocks it), then clear hazards on both sides,
    # screens on the target's side, and terrain
    defog = landed & (mid == MID["defog"])
    st, _ = apply_boosts(st, o, jnp.array([0, 0, 0, 0, 0, 0, -1]), defog & ~blocked_sub & (st.hp[o, tslot] > 0))
    st = st.replace(
        hazards=jnp.where(defog, 0, st.hazards),
        screens=st.screens.at[o].set(jnp.where(defog, 0, st.screens[o])),
        terrain=jnp.where(defog, 0, st.terrain).astype(jnp.int8),
        terrain_turns=jnp.where(defog, 0, st.terrain_turns).astype(jnp.int8),
    )
    # Court Change: swap every side condition
    court = landed & (mid == MID["courtchange"])
    st = st.replace(
        hazards=jnp.where(court, st.hazards[::-1], st.hazards),
        screens=jnp.where(court, st.screens[::-1], st.screens),
        tailwind=jnp.where(court, st.tailwind[::-1], st.tailwind),
    )
    st = refresh_paradox(st)
    # Status pivots: Parting Shot only switches if it lowered a stat; Teleport / Chilly Reception always
    pivot_ok = landed & jnp.where(mid == MID["partingshot"], foe_boosted, True)
    return st, pivot_ok


# --- end of turn -----------------------------------------------------------------------------------


def end_of_turn(st: Battle, key, cfg: Config, cond):
    """Residual effects in Showdown's onResidualOrder, faster Pokemon first within each effect."""
    spe = jnp.stack([action_speed(st, 0), action_speed(st, 1)])
    first0 = p1_first(key, jnp.zeros(2, jnp.int32), spe, cfg)
    order = jnp.where(first0, jnp.array([0, 1]), jnp.array([1, 0]))

    def each(fn, st):
        for i in range(2):
            st = fn(st, order[i])
        return st

    # 1. Weather: count down, then sandstorm damage
    w_turns = jnp.where(cond & (st.weather > 0), st.weather_turns - 1, st.weather_turns)
    ends = cond & (st.weather > 0) & (w_turns <= 0)
    st = st.replace(
        weather=jnp.where(ends, 0, st.weather).astype(jnp.int8),
        weather_turns=jnp.where(ends, 0, w_turns).astype(jnp.int8),
    )
    st = refresh_paradox(st)

    def sand(st, s):
        slot = act(st, s)
        t = types_now(st, s, slot)
        immune = has_type(t, TY["rock"]) | has_type(t, TY["ground"]) | has_type(t, TY["steel"])
        st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 16, cond & (st.weather == SAND) & ~immune)
        return eat_sitrus(st, s)

    st = each(sand, st)

    # 2. Grassy Terrain heal, then Leftovers / Black Sludge (onResidualOrder 5)
    def grassy(st, s):
        slot = act(st, s)
        return gain_hp(
            st, s, slot, _i(st.stats[s, slot, 0]) // 16, cond & (st.terrain == G_TERRAIN) & grounded(st, s, slot)
        )

    st = each(grassy, st)

    def leftovers(st, s):
        slot = act(st, s)
        maxhp = _i(st.stats[s, slot, 0])
        item = st.item[s, slot]
        poison = has_type(types_now(st, s, slot), TY["poison"])
        st = gain_hp(
            st, s, slot, maxhp // 16, cond & ((item == ITEM.leftovers) | ((item == ITEM.blacksludge) & poison))
        )
        st = lose_hp(st, s, slot, maxhp // 8, cond & (item == ITEM.blacksludge) & ~poison)
        return eat_sitrus(st, s)

    st = each(leftovers, st)

    # 3. Poison (order 9), then burn (order 10)
    def poison(st, s):
        slot = act(st, s)
        maxhp = _i(st.stats[s, slot, 0])
        status = st.status[s, slot]
        st = lose_hp(st, s, slot, maxhp // 8, cond & (status == PSN))
        stage = jnp.minimum(_i(st.status_turns[s, slot]) + 1, 15)
        is_tox = cond & (status == TOX) & (st.hp[s, slot] > 0)
        st = st.replace(
            status_turns=st.status_turns.at[s, slot].set(
                jnp.where(is_tox, stage, st.status_turns[s, slot]).astype(jnp.int8)
            )
        )
        st = lose_hp(st, s, slot, jnp.maximum(maxhp // 16, 1) * stage, is_tox)
        return eat_sitrus(st, s)

    st = each(poison, st)

    def burn(st, s):
        slot = act(st, s)
        st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 16, cond & (st.status[s, slot] == BRN))
        return eat_sitrus(st, s)

    st = each(burn, st)

    # 4. Roost ends (25); screens and Tailwind count down (26); Trick Room and terrain count down (27)
    st = st.replace(roosted=jnp.where(cond, False, st.roosted))
    st = st.replace(
        screens=jnp.where(cond, jnp.maximum(st.screens - 1, 0), st.screens).astype(jnp.int8),
        tailwind=jnp.where(cond, jnp.maximum(st.tailwind - 1, 0), st.tailwind).astype(jnp.int8),
        trick_room=jnp.where(cond, jnp.maximum(st.trick_room - 1, 0), st.trick_room).astype(jnp.int8),
    )
    t_turns = jnp.where(cond & (st.terrain > 0), st.terrain_turns - 1, st.terrain_turns)
    t_ends = cond & (st.terrain > 0) & (t_turns <= 0)
    st = st.replace(
        terrain=jnp.where(t_ends, 0, st.terrain).astype(jnp.int8),
        terrain_turns=jnp.where(t_ends, 0, t_turns).astype(jnp.int8),
    )
    st = refresh_paradox(st)

    # Turn-scoped volatiles: flinch and Protect; the Protect chain breaks if Protect wasn't used this turn
    st = st.replace(
        flinch=jnp.where(cond, False, st.flinch),
        protected=jnp.where(cond, False, st.protected),
        protect_count=jnp.where(cond & ~st.protect_used, 0, st.protect_count).astype(jnp.int8),
        protect_used=jnp.where(cond, False, st.protect_used),
    )
    return st


# --- legality --------------------------------------------------------------------------------------


def legal_actions(st: Battle):
    """(2, 13) bool mask of legal actions per side for the current request (leads use 0..5)."""

    def side(s):
        slot = jnp.maximum(act(st, s), 0)
        alive = alive_mask(st)[s]
        # Team preview: any team member may lead
        preview = jnp.concatenate([alive, jnp.zeros(N_ACTIONS - 6, bool)])
        # Switches: the n-th benched slot must be alive
        slots = jnp.arange(6)
        benched = slots != slot
        rank = jnp.cumsum(benched) - 1
        sw = jnp.stack([jnp.any(benched & (rank == n) & alive) for n in range(5)])
        # Moves: PP left, Choice lock, Assault Vest bans status moves
        mids = _i(st.moves[s, slot])
        has_pp = (st.pp[s, slot] > 0) & (mids > 0)
        locked = choice_locked(st, s)
        lock_ok = ~locked | (jnp.arange(4) == st.choice_lock[s])
        av = (st.item[s, slot] == ITEM.assaultvest) & (MOVES[mids, MC["category"]] == 2)
        mv = has_pp & lock_ok & ~av
        struggle = ~jnp.any(mv)
        mv = jnp.where(struggle, jnp.arange(4) == 0, mv)  # move action 0 = Struggle
        tera = mv & ~st.tera_used[s] & ~struggle
        move_req = jnp.concatenate([mv, tera, sw])
        switch_req = jnp.concatenate([jnp.zeros(8, bool), sw])
        # Sides without a decision this step get a single no-op action (index 0)
        noop = jnp.zeros(N_ACTIONS, bool).at[0].set(True)
        mask = jnp.where(
            st.phase == PHASE_PREVIEW,
            preview,
            jnp.where(st.phase == PHASE_MOVE, move_req, jnp.where(st.need_switch[s], switch_req, noop)),
        )
        return jnp.where(st.winner >= 0, noop, mask)

    return jnp.stack([side(0), side(1)])


def choice_locked(st: Battle, s):
    """The Choice lock only holds while the Pokemon still has its Choice item (Knock Off frees it)."""
    item = st.item[s, jnp.maximum(act(st, s), 0)]
    holding = (item == ITEM.choiceband) | (item == ITEM.choicespecs) | (item == ITEM.choicescarf)
    return (st.choice_lock[s] >= 0) & holding


def struggling(st: Battle, s):
    slot = jnp.maximum(act(st, s), 0)
    mids = _i(st.moves[s, slot])
    locked = choice_locked(st, s)
    lock_ok = ~locked | (jnp.arange(4) == st.choice_lock[s])
    av = (st.item[s, slot] == ITEM.assaultvest) & (MOVES[mids, MC["category"]] == 2)
    return ~jnp.any((st.pp[s, slot] > 0) & (mids > 0) & lock_ok & ~av)


# --- step ------------------------------------------------------------------------------------------


def step(st: Battle, actions, key, cfg: Config = Config()):
    """Apply both sides' actions (int (2,)) for the current request and advance to the next request."""
    actions = _i(actions)
    keys = jax.random.split(key, 8)
    phase, resume = st.phase, st.resume
    live = st.winner == WINNER_NONE
    in_preview, in_move, in_switch = (
        live & (phase == PHASE_PREVIEW),
        live & (phase == PHASE_MOVE),
        live & (phase == PHASE_SWITCH),
    )

    # 1. Switches: leads, voluntary switches, or requested replacements / pivot switches
    is_switch_action = actions >= A_SWITCH
    wants = jnp.where(in_preview, True, jnp.where(in_move, is_switch_action, in_switch & st.need_switch))
    target_slot = jnp.stack([jnp.where(in_preview, actions[s], switch_target(st, s, actions[s])) for s in (0, 1)])
    st = do_switches(st, wants, target_slot, keys[0], cfg)
    st = st.replace(need_switch=jnp.where(in_switch, False, st.need_switch))

    # 2. Terastallization (after switches, before moves)
    tera = in_move & (actions >= A_TERA) & (actions < A_SWITCH)
    for s in (0, 1):
        slot = act(st, s)
        st = st.replace(
            terastallized=st.terastallized.at[s, slot].set(st.terastallized[s, slot] | tera[s]),
            tera_used=st.tera_used.at[s].set(st.tera_used[s] | tera[s]),
        )

    # 3. Moves: decide who moves and in what order
    move_action = jnp.where(in_move, jnp.where(is_switch_action, NO_ACTION, actions), st.queued_action)
    resuming = in_switch & (resume == RESUME_SECOND_MOVE)
    uses_move = jnp.where(in_move, ~is_switch_action, resuming & (st.queued_action >= 0))
    move_slot = jnp.where(move_action >= A_TERA, move_action - A_TERA, move_action)
    move_slot = jnp.clip(move_slot, 0, 3)
    struggle = jnp.stack([struggling(st, 0), struggling(st, 1)])
    prio = jnp.stack(
        [jnp.where(struggle[s], 0, _i(MOVES[st.moves[s, act(st, s), move_slot[s]], MC["priority"]])) for s in (0, 1)]
    )
    spe = jnp.stack([action_speed(st, 0), action_speed(st, 1)])
    first0 = p1_first(keys[1], prio, spe, cfg)
    order = jnp.where(first0, jnp.array([0, 1]), jnp.array([1, 0]))

    pivot_side = jnp.int32(-1)
    resume_after = jnp.int8(RESUME_NONE)
    queued = jnp.full((2,), NO_ACTION, jnp.int32)
    for i in range(2):
        s = order[i]
        o = 1 - s
        go = uses_move[s] & (pivot_side < 0)
        last = ~(uses_move[o] & (i == 0))
        st, pivot = run_move(st, s, move_slot[s], struggle[s], go, last, keys[2 + i], cfg)
        pivot = go & pivot
        # A pivot suspends the turn: remember the other side's move if it still has to act
        queued = jnp.where(pivot & (i == 0) & uses_move[o], queued.at[o].set(move_action[o]), queued)
        resume_after = jnp.where(pivot, jnp.int8(RESUME_SECOND_MOVE if i == 0 else RESUME_END_OF_TURN), resume_after)
        pivot_side = jnp.where(pivot, s, pivot_side)
        if i == 0:
            # The second mover's move only runs if no pivot happened
            uses_move = uses_move.at[o].set(uses_move[o] & ~pivot)
    suspended = pivot_side >= 0

    # 4. End of turn
    decided = jnp.any(jnp.sum(alive_mask(st), axis=1) == 0)  # Showdown stops as soon as a side is out
    eot = ~suspended & ~decided & (in_move | (in_switch & (resume != RESUME_NONE)))
    st = end_of_turn(st, keys[6], cfg, eot)

    # 5. Next request
    return next_request(st, suspended, pivot_side, resume_after, queued, in_preview | eot, live)


def next_request(st: Battle, suspended, pivot_side, resume_after, queued, turn_over, live):
    alive = alive_mask(st)
    left = jnp.sum(alive, axis=1)
    winner = jnp.where(
        (left[0] == 0) & (left[1] == 0), WINNER_TIE, jnp.where(left[1] == 0, 0, jnp.where(left[0] == 0, 1, WINNER_NONE))
    )
    winner = jnp.where(live, winner, st.winner)
    over = winner != WINNER_NONE
    active_fainted = jnp.stack([st.hp[s, act(st, s)] <= 0 for s in (0, 1)])
    has_bench = jnp.stack([jnp.any(alive[s]) for s in (0, 1)])
    faint_switch = active_fainted & has_bench
    pivot_need = jnp.stack([pivot_side == 0, pivot_side == 1])
    need = jnp.where(suspended, pivot_need, faint_switch)
    to_switch = ~over & jnp.any(need)
    phase = jnp.where(over, st.phase, jnp.where(to_switch, PHASE_SWITCH, PHASE_MOVE))
    resume = jnp.where(suspended, resume_after, RESUME_NONE)
    new_turn = ~over & ~to_switch & (turn_over | (st.phase == PHASE_SWITCH))
    return st.replace(
        winner=winner.astype(jnp.int8),
        phase=jnp.where(live, phase, st.phase).astype(jnp.int8),
        need_switch=jnp.where(live, need & ~over, st.need_switch),
        resume=jnp.where(live, resume, st.resume).astype(jnp.int8),
        queued_action=jnp.where(live, jnp.where(suspended, queued, NO_ACTION), st.queued_action).astype(jnp.int8),
        turn=jnp.where(live & new_turn, st.turn + 1, st.turn).astype(jnp.int16),
    )
