"""
engine/oujax/families/entry_abilities.py
Ability infrastructure shared by the ability families, plus the abilities that act when a Pokemon enters or
leaves the field.

Infrastructure:
- st.ability holds each active Pokemon's working ability. On entering the field a Pokemon gets back its own
  ability (MON "ab_base"; Trace / Skill Swap copies last until it leaves) and its species types (Libero /
  Protean changes last until it leaves).
- Neutralizing Gas suppresses the other active Pokemon's ability (unless it can't be suppressed: Disguise, ...)
  by writing 0 to its st.ability while the gas holder is on the field, so every ability check in the engine
  respects it; VOL "ab_cur" keeps the real ability. When the holder leaves or faints the ability comes back and
  starts again (weather and terrain setters, Dauntless Shield, Download, Protosynthesis / Quark Drive).
- Mold Breaker / Teravolt / Turboblaze ignore the target's breakable abilities while their user's move runs:
  FIELD "mold" holds that user's side + 1 during the move, and eff_ability() applies it, so the families' checks
  (and their boost and status hooks, which run inside the move) honour it.
- FIELD "boost_src" tags the stat change being applied: BOOST_INTIMIDATE while Intimidate lowers Attack (Guard
  Dog, Rattled, and Inner Focus / Oblivious / Own Tempo / Scrappy react to it), BOOST_MIRROR for a Mirror Armor
  reflection.

Abilities: Neutralizing Gas, Intimidate, Dauntless Shield (once per battle), Download, Trace, Wind Rider's
Tailwind boost on entry, Libero / Protean (type change once per switch-in), and the move Skill Swap.
Illusion is inert: there is no observation layer yet, so disguising the user changes nothing mechanical (it
only ends when hit). Hospitality heals an ally, so it does nothing in singles.
Simplifications: a Trace holder that found no target on entry copies the next foe's ability when it comes in,
without starting it; on Neutralizing Gas ending, Intimidate doesn't restart (its only target is the leaving or
fainted gas holder).
"""

import jax.numpy as jnp
import numpy as np

from engine.oujax import hooks
from engine.oujax.damage import boosted_stat, move_type_and_bp
from engine.oujax.effects import BREAKABLE, T
from engine.oujax.engine import (
    MID,
    TERRAIN_ABILITY,
    WEATHER_ABILITY,
    _i,
    act,
    apply_boosts,
    combatant,
    field_ex,
    foe,
    ignores_abilities_ab,
    mon_ex,
    set_field_ex,
    set_mon_ex,
    set_terrain,
    set_vol,
    set_weather,
    vol,
)
from engine.oujax.tables import SC, TYPE_NONE

AB = T.ability_ids
NG, TRACE, INTIMIDATE = AB["neutralizinggas"], AB["trace"], AB["intimidate"]
DAUNTLESS, DOWNLOAD, WINDRIDER = AB["dauntlessshield"], AB["download"], AB["windrider"]
LIBERO, PROTEAN, FLASHFIRE = AB["libero"], AB["protean"], AB["flashfire"]
PROTO, QUARK = AB["protosynthesis"], AB["quarkdrive"]
BOOST_INTIMIDATE, BOOST_MIRROR = 1, 2
SPECIES_TYPES = jnp.asarray(T.species[:, [SC["type1"], SC["type2"]]])
BREAKABLE_J = jnp.asarray(BREAKABLE)


def _flag_table(flag):
    out = np.zeros(len(AB), bool)
    for aid, a in T.dex["abilities"].items():
        out[AB[aid]] = bool(a.get("flags", {}).get(flag))
    return jnp.asarray(out)


CANT_SUPPRESS, NO_TRACE, NO_SKILLSWAP = (
    _flag_table("cantsuppress"),
    _flag_table("notrace"),
    _flag_table("failskillswap"),
)

hooks.MON.add("ab_base")  # own ability + 1, recorded the first time it enters
hooks.MON.add("shield_used")  # Dauntless Shield already boosted this battle
hooks.VOL.add("ab_cur")  # working ability + 1 (own, Traced or Skill Swapped), even while suppressed
hooks.VOL.add("ab_off")  # suppressed by Neutralizing Gas
hooks.VOL.add("gas_end")  # this Neutralizing Gas holder is leaving the field
hooks.VOL.add("entry_turn")  # st.turn when it entered (Speed Boost skips that turn)
hooks.VOL.add("type_changed")  # Libero / Protean already changed its type this switch-in
hooks.VOL.add("trace_seek")  # Trace found no target yet
hooks.VOL.add("flashfire")  # Flash Fire activated (read by engine.combatant)
hooks.FIELD.add("mold")  # side + 1 of a Mold Breaker-type Pokemon whose move is running
hooks.FIELD.add("boost_src")  # BOOST_* tag of the stat change being applied


# --- helpers for every ability family ---------------------------------------------------------------


def on_field(st, s):
    slot = act(st, s)
    return (slot >= 0) & (st.hp[s, jnp.maximum(slot, 0)] > 0)


def ability(st, s):
    """Side s's active ability (0 while suppressed)."""
    return _i(st.ability[s, jnp.maximum(act(st, s), 0)])


def eff_ability(st, s):
    """Side s's active ability as the current move sees it: a breakable ability doesn't count while the other
    side's Mold Breaker-type Pokemon is using a move."""
    ab = ability(st, s)
    return jnp.where(BREAKABLE_J[ab] & (field_ex(st, "mold") == foe(s) + 1), 0, ab)


def is_ab(ab, *names):
    out = jnp.bool_(False)
    for n in names:
        out = out | (ab == AB[n])
    return out


# --- Neutralizing Gas --------------------------------------------------------------------------------


def gas_holder(st, s):
    slot = jnp.maximum(act(st, s), 0)
    return on_field(st, s) & (st.ability[s, slot] == NG) & (vol(st, s, "gas_end") == 0)


def apply_gas(st):
    """Suppress or restore each active Pokemon's ability for the Neutralizing Gas now on the field.
    Returns (state, (2,) bool: the ability just came back)."""
    gas = [gas_holder(st, 0), gas_holder(st, 1)]
    back = []
    for s in (0, 1):
        slot = jnp.maximum(act(st, s), 0)
        cur = vol(st, s, "ab_cur") - 1
        known = (act(st, s) >= 0) & (cur >= 0)
        cur = jnp.maximum(cur, 0)
        off = gas[1 - s] & known & ~CANT_SUPPRESS[cur] & (cur != NG)
        was = vol(st, s, "ab_off") > 0
        ab = jnp.where(known, jnp.where(off, 0, cur), st.ability[s, slot])
        st = st.replace(ability=st.ability.at[s, slot].set(ab.astype(st.ability.dtype)))
        st = set_vol(st, s, "ab_off", off.astype(jnp.int32), known)
        back.append(known & was & ~off)
    return st, jnp.stack(back)


# --- ability start (switch-in, Trace, Skill Swap, coming back from Neutralizing Gas) -------------------


def _trace(st, s, cond):
    """Trace copies the foe's ability (if it can be traced); without a target it keeps looking."""
    o = foe(s)
    slot = jnp.maximum(act(st, s), 0)
    seek = cond & ((ability(st, s) == TRACE) | (vol(st, s, "trace_seek") > 0))
    target_ab = jnp.maximum(vol(st, o, "ab_cur") - 1, 0)
    ok = seek & on_field(st, o) & ~NO_TRACE[target_ab] & (target_ab > 0)
    st = st.replace(
        ability=st.ability.at[s, slot].set(jnp.where(ok, target_ab, st.ability[s, slot]).astype(st.ability.dtype))
    )
    st = set_vol(st, s, "ab_cur", target_ab + 1, ok)
    st = set_vol(st, s, "trace_seek", (seek & ~ok).astype(jnp.int32), seek)
    return st, ok


def ability_start(st, s, cond, entry=True, weather=False):
    """The active ability's Start effects for side s. entry: Trace and Intimidate run too (on switching in or
    getting a new ability; not when it comes back after Neutralizing Gas). weather: the weather / terrain setters
    and Protosynthesis run (the engine's run_switch_in already ran them on switch-in, except for a Traced
    ability)."""
    o = foe(s)
    slot = jnp.maximum(act(st, s), 0)
    alive = cond & on_field(st, s)
    weather_cond = alive if weather else jnp.bool_(False)
    if entry:
        st, traced = _trace(st, s, alive)
        weather_cond = weather_cond | traced
    ab = ability(st, s)
    st, _ = set_weather(st, WEATHER_ABILITY[ab], s, slot, weather_cond)
    st, _ = set_terrain(st, TERRAIN_ABILITY[ab], s, slot, weather_cond)
    # Self boosts: Dauntless Shield (once per battle), Download, Wind Rider under its side's Tailwind
    oslot = jnp.maximum(act(st, o), 0)
    foe_on = on_field(st, o)
    foe_def = boosted_stat(_i(st.stats[o, oslot, 2]), _i(st.boosts[o, 1]))
    foe_spd = boosted_stat(_i(st.stats[o, oslot, 4]), _i(st.boosts[o, 3]))
    shield = (ab == DAUNTLESS) & (mon_ex(st, s, slot, "shield_used") == 0)
    dl = (ab == DOWNLOAD) & foe_on
    wind = (ab == WINDRIDER) & (st.tailwind[s] > 0)
    deltas = jnp.stack([dl & (foe_def < foe_spd) | wind, shield, dl & (foe_def >= foe_spd), *([jnp.bool_(False)] * 4)])
    st, _ = apply_boosts(st, s, deltas.astype(jnp.int32), alive & (shield | dl | wind))
    st = set_mon_ex(st, s, slot, "shield_used", 1, alive & shield)
    if entry:
        # Intimidate: -1 Attack to the foe, unless it is behind a Substitute
        intim = alive & (ab == INTIMIDATE) & foe_on & (st.sub_hp[o] == 0)
        st = set_field_ex(st, "boost_src", BOOST_INTIMIDATE)
        st, _ = apply_boosts(st, o, jnp.array([-1, 0, 0, 0, 0, 0, 0]), intim, from_foe=True)
        st = set_field_ex(st, "boost_src", 0)
    return st


def _restart_back(st, back):
    for s in (0, 1):
        st = ability_start(st, s, back[s], entry=False, weather=True)
    return st


# --- hooks -------------------------------------------------------------------------------------------


@hooks.hook("enter", priority=100)
def _enter(st, s, slot, cond):
    base = mon_ex(st, s, slot, "ab_base")
    base = jnp.where(base == 0, _i(st.ability[s, slot]) + 1, base)
    st = set_mon_ex(st, s, slot, "ab_base", base, cond)
    st = set_vol(st, s, "ab_cur", base, cond)
    st = set_vol(st, s, "ab_off", 0, cond)
    st = set_vol(st, s, "entry_turn", st.turn, cond)
    types = SPECIES_TYPES[st.species[s, slot]]
    st = st.replace(
        ability=st.ability.at[s, slot].set(jnp.where(cond, base - 1, st.ability[s, slot]).astype(st.ability.dtype)),
        base_types=st.base_types.at[s, slot].set(jnp.where(cond, types, st.base_types[s, slot]).astype(jnp.int8)),
    )
    st, _ = apply_gas(st)
    return st


@hooks.hook("switch_in", priority=10)
def _switch_in(st, s, key, cfg, cond):
    st = ability_start(st, s, cond, entry=True)
    # A Trace holder still looking for a target copies the newcomer's ability
    st, _ = _trace(st, foe(s), cond & (vol(st, foe(s), "trace_seek") > 0))
    return st


@hooks.hook("switch_out", priority=10)
def _switch_out(st, s, cond):
    # Neutralizing Gas leaving: the foe's ability comes back (before this Pokemon is gone, as in Showdown)
    st = set_vol(st, s, "gas_end", 1, cond & gas_holder(st, s))
    st, back = apply_gas(st)
    return _restart_back(st, back & cond)


@hooks.hook("before_move", priority=100)
def _mold_on(st, ctx):
    mold = ignores_abilities_ab(st.ability[ctx.s, ctx.slot])
    return set_field_ex(st, "mold", jnp.where(mold, ctx.s + 1, 0)), jnp.bool_(False)


@hooks.hook("after_move", priority=-100)
def _after_move(st, ctx):
    st = set_field_ex(st, "mold", 0)
    # A fainted Neutralizing Gas holder stops suppressing at once
    st, back = apply_gas(st)
    return _restart_back(st, back)


@hooks.residual(10.5, "gas_faint")
def _gas_faint(st, key, cfg, cond):
    st, back = apply_gas(st)
    return _restart_back(st, back & cond)


@hooks.hook("try_hit", priority=50)
def _libero(st, ctx):
    # Libero / Protean (onPrepareHit): become the move's type, once per switch-in, unless Terastallized
    s, slot = ctx.s, ctx.slot
    ab = st.ability[s, slot]
    mtype, _ = move_type_and_bp(ctx.mid, combatant(st, s), combatant(st, ctx.o), _i(st.weather))
    cur = _i(st.base_types[s, slot])
    already = (cur[0] == mtype) & (cur[1] == TYPE_NONE)
    go = (
        ctx.proceeds
        & ((ab == LIBERO) | (ab == PROTEAN))
        & (vol(st, s, "type_changed") == 0)
        & ~st.terastallized[s, slot]
        & (mtype != TYPE_NONE)
        & ~already
    )
    new = jnp.stack([mtype, jnp.int32(TYPE_NONE)])
    st = st.replace(base_types=st.base_types.at[s, slot].set(jnp.where(go, new, cur).astype(jnp.int8)))
    return set_vol(st, s, "type_changed", 1, go), jnp.bool_(False)


def _end_ability(st, s, cond):
    """An ability being swapped away ends its own state (Protosynthesis / Quark Drive, Flash Fire, Libero)."""
    ab = ability(st, s)
    paradox = cond & ((ab == PROTO) | (ab == QUARK))
    st = st.replace(
        paradox_stat=st.paradox_stat.at[s].set(jnp.where(paradox, 0, st.paradox_stat[s]).astype(st.paradox_stat.dtype)),
        paradox_from_item=st.paradox_from_item.at[s].set(jnp.where(paradox, False, st.paradox_from_item[s])),
    )
    st = set_vol(st, s, "flashfire", 0, cond)
    return set_vol(st, s, "type_changed", 0, cond)


@hooks.hook("status_move")
def _skill_swap(st, ctx):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    a, b = jnp.maximum(vol(st, s, "ab_cur") - 1, 0), jnp.maximum(vol(st, o, "ab_cur") - 1, 0)
    ok = ctx.landed & (ctx.mid == MID["skillswap"]) & on_field(st, s) & on_field(st, o)
    ok &= ~NO_SKILLSWAP[a] & ~NO_SKILLSWAP[b]
    st = _end_ability(_end_ability(st, s, ok), o, ok)
    st = set_vol(set_vol(st, s, "ab_cur", b + 1, ok), o, "ab_cur", a + 1, ok)
    ability_ = st.ability.at[s, slot].set(jnp.where(ok, b, st.ability[s, slot]).astype(st.ability.dtype))
    st = st.replace(ability=ability_.at[o, tslot].set(jnp.where(ok, a, ability_[o, tslot]).astype(st.ability.dtype)))
    st, _ = apply_gas(st)
    # The target's new ability starts first, then the user's
    st = ability_start(st, o, ok, weather=True)
    return ability_start(st, s, ok, weather=True)
