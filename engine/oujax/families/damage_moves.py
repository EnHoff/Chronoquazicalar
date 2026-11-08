"""
engine/oujax/families/damage_moves.py
Damage-side move effects that depend on the battle so far or on the hit: Triple Axel / Triple Kick (rising power),
Payback, Avalanche, Stomping Tantrum / Temper Flare, Fickle Beam, Counter / Mirror Coat / Metal Burst /
Comeuppance, Endeavor, screen-breaking moves, Clear Smog, crash damage (High Jump Kick, Supercell Slam, Axe
Kick), Steel Beam's recoil, Grassy Glide's priority, and a few status moves (Growth, Belly Drum, Clangorous Soul,
Take Heart, Tidy Up, Heal Bell, Strength Sap).

Per-turn records (engine core VOL slots): "moved", "dmg_phys" / "dmg_spec", "dmg_last", "move_result",
"last_failed".
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import boosted_stat
from engine.oujax.engine import (
    G_TERRAIN,
    MID,
    SCR_LIGHT_SCREEN,
    SCR_REFLECT,
    SCR_VEIL,
    SUN,
    _i,
    act,
    apply_boosts,
    gain_hp,
    grounded,
    js_round_div,
    lose_hp,
    set_vol,
    vol,
)


def _is(mid, *names):
    out = jnp.bool_(False)
    for n in names:
        out = out | (mid == MID[n])
    return out


# --- base power ---------------------------------------------------------------------------------------


@hooks.hook("hit_power")
def _power(st, ctx, hit):
    mid, s, o = ctx.mid, ctx.s, ctx.o
    mult = jnp.int32(1)
    # Triple Axel / Triple Kick: 20/40/60 (10/20/30) by hit number
    mult = jnp.where(_is(mid, "tripleaxel", "triplekick"), hit + 1, mult)
    # Payback doubles if the target already acted this turn (and didn't just switch in)
    mult = jnp.where(_is(mid, "payback") & (vol(st, o, "moved") > 0), mult * 2, mult)
    # Avalanche doubles if the target damaged the user this turn
    hurt = (vol(st, s, "dmg_phys") > 0) | (vol(st, s, "dmg_spec") > 0)
    mult = jnp.where(_is(mid, "avalanche") & hurt, mult * 2, mult)
    # Stomping Tantrum / Temper Flare double if the user's previous move failed
    mult = jnp.where(_is(mid, "stompingtantrum", "temperflare") & (vol(st, s, "last_failed") > 0), mult * 2, mult)
    # Fickle Beam: 30% chance of double power. The roll happens inside getDamage, so the harness's forced mode
    # answers it like a crit roll
    if ctx.cfg.forced:
        fickle = jnp.bool_(ctx.cfg.crit)
    else:
        fickle = jax.random.randint(jax.random.fold_in(ctx.key, 31), (), 0, 10) < 3
    chain = jnp.where(_is(mid, "ficklebeam") & fickle & (hit == 0), 8192, 4096)
    return mult, jnp.int32(chain)


# --- damage overrides ---------------------------------------------------------------------------------


def _counter_damage(st, ctx):
    """Damage of Counter / Mirror Coat / Metal Burst / Comeuppance, 0 if it fails, -1 for other moves."""
    s = ctx.s
    phys, spec = vol(st, s, "dmg_phys"), vol(st, s, "dmg_spec")
    # Metal Burst also works after a hit on the user's Substitute (Showdown records it as 0 damage: 1 HP)
    last = vol(st, s, "dmg_last")
    burst = jnp.where(last > 0, jnp.maximum((last - 1) * 3 // 2, 1), 0)
    return jnp.select(
        [_is(ctx.mid, "counter"), _is(ctx.mid, "mirrorcoat"), _is(ctx.mid, "metalburst", "comeuppance")],
        [2 * phys, 2 * spec, burst],
        -1,
    )


@hooks.hook("damage_override")
def _override(st, ctx, dmg):
    counter = _counter_damage(st, ctx)
    dmg = jnp.where(counter >= 0, jnp.maximum(counter, 1), dmg)
    # Endeavor: the target's HP minus the user's
    endeavor = _i(st.hp[ctx.o, ctx.tslot]) - _i(st.hp[ctx.s, ctx.slot])
    return jnp.where(_is(ctx.mid, "endeavor"), jnp.maximum(endeavor, 0), dmg)


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, o = ctx.s, ctx.o
    counter = _counter_damage(st, ctx)
    fails = counter == 0  # nothing of the right kind hit the user this turn
    fails |= _is(ctx.mid, "endeavor") & (_i(st.hp[s, ctx.slot]) >= _i(st.hp[o, ctx.tslot]))
    maxhp = _i(st.stats[s, ctx.slot, 0])
    hp = _i(st.hp[s, ctx.slot])
    fails |= _is(ctx.mid, "bellydrum") & ((hp * 2 <= maxhp) | (st.boosts[s, 0] >= 6))
    fails |= _is(ctx.mid, "clangoroussoul") & (hp * 100 <= maxhp * 33)
    fails |= _is(ctx.mid, "strengthsap") & (st.boosts[o, 0] <= -6)
    return fails


# --- moves that act on the target before or after the hit ---------------------------------------------


@hooks.hook("move_try_hit")
def _break_screens(st, ctx):
    # Brick Break / Psychic Fangs / Raging Bull shatter the target side's screens once they connect, even
    # into a Substitute (not on a miss or into a type immunity)
    brk = ctx.landed & _is(ctx.mid, "brickbreak", "psychicfangs", "ragingbull")
    o = ctx.o
    scr = st.screens[o]
    scr = jnp.where(brk, scr.at[SCR_REFLECT].set(0).at[SCR_LIGHT_SCREEN].set(0).at[SCR_VEIL].set(0), scr)
    return st.replace(screens=st.screens.at[o].set(scr))


@hooks.hook("after_hit")
def _after_hit(st, ctx, dealt, hit_sub):
    # Clear Smog resets the target's stat stages (not through a Substitute)
    smog = ctx.landed & _is(ctx.mid, "clearsmog") & ~hit_sub & (dealt > 0)
    return st.replace(boosts=st.boosts.at[ctx.o].set(jnp.where(smog, 0, st.boosts[ctx.o])))


@hooks.hook("after_move")
def _after_move(st, ctx):
    s, slot = ctx.s, ctx.slot
    maxhp = _i(st.stats[s, slot, 0])
    # Crash damage when the move reaches the target but misses or doesn't affect it
    crash = ctx.proceeds & ~ctx.landed & _is(ctx.mid, "highjumpkick", "supercellslam", "axekick")
    st = lose_hp(st, s, slot, maxhp // 2, crash)
    # Steel Beam costs half the user's max HP whether it hits or not (Math.round)
    beam = ctx.proceeds & _is(ctx.mid, "steelbeam")
    return lose_hp(st, s, slot, js_round_div(maxhp, 2), beam)


@hooks.hook("status_move")
def _status(st, ctx):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    ok = ctx.landed
    maxhp = _i(st.stats[s, slot, 0])
    # Growth: +2/+2 in sun (the table already applied +1/+1)
    st, _ = apply_boosts(st, s, jnp.array([1, 0, 1, 0, 0, 0, 0]), ok & _is(ctx.mid, "growth") & (st.weather == SUN))
    # Belly Drum: half max HP for +12 Attack
    drum = ok & _is(ctx.mid, "bellydrum")
    st = lose_hp(st, s, slot, maxhp // 2, drum, direct=True)
    st, _ = apply_boosts(st, s, jnp.array([12, 0, 0, 0, 0, 0, 0]), drum)
    # Clangorous Soul: a third of max HP (the table applied the +1s)
    st = lose_hp(st, s, slot, maxhp * 33 // 100, ok & _is(ctx.mid, "clangoroussoul"), direct=True)
    # Take Heart: cure status, +1 SpA / SpD
    heart = ok & _is(ctx.mid, "takeheart")
    st = st.replace(status=st.status.at[s, slot].set(jnp.where(heart, 0, st.status[s, slot])))
    st, _ = apply_boosts(st, s, jnp.array([0, 0, 1, 1, 0, 0, 0]), heart)
    # Tidy Up: clear hazards on both sides and both Substitutes, +1 Atk / Spe
    tidy = ok & _is(ctx.mid, "tidyup")
    st = st.replace(hazards=jnp.where(tidy, 0, st.hazards), sub_hp=jnp.where(tidy, 0, st.sub_hp))
    st, _ = apply_boosts(st, s, jnp.array([1, 0, 0, 0, 1, 0, 0]), tidy)
    # Heal Bell: cure the user's whole team
    bell = ok & _is(ctx.mid, "healbell")
    st = st.replace(status=st.status.at[s].set(jnp.where(bell, 0, st.status[s])))
    # Strength Sap: heal by the target's boosted Attack, then -1 Attack (blocked by a Substitute)
    sap = ok & _is(ctx.mid, "strengthsap") & (st.sub_hp[o] == 0)
    target_atk = boosted_stat(_i(st.stats[o, tslot, 1]), _i(st.boosts[o, 0]))
    st, _ = apply_boosts(st, o, jnp.array([-1, 0, 0, 0, 0, 0, 0]), sap, from_foe=True)
    return gain_hp(st, s, slot, target_atk, sap)


@hooks.hook("priority")
def _priority(st, side, mid, prio):
    # Grassy Glide: +1 priority in Grassy Terrain for a grounded user
    glide = (mid == MID["grassyglide"]) & (st.terrain == G_TERRAIN) & grounded(st, side, act(st, side))
    return jnp.where(glide, prio + 1, prio)


@hooks.residual(31, "turn_records")
def _turn_records(st, key, cfg, cond):
    # End of turn: this turn's move result becomes "last turn's", and the per-turn records clear
    for s in (0, 1):
        result = vol(st, s, "move_result")
        st = set_vol(st, s, "last_failed", jnp.where(result == 2, 1, 0), cond)
        for name in ("moved", "dmg_phys", "dmg_spec", "dmg_last", "move_result"):
            st = set_vol(st, s, name, 0, cond)
    return st
