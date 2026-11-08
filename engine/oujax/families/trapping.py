"""
engine/oujax/families/trapping.py
Moves that keep the target in: partial trapping (Magma Storm, Whirlpool, Fire Spin, Infestation:
partiallytrapped, 4-5 turns of 1/8 max HP), Mean Look / Block / Jaw Lock (trapped while the trapper stays in),
No Retreat (trapped for good), plus Rapid Spin / Mortal Spin freeing the user from partial trapping and Leech
Seed.

Ghost types can't be trapped (Showdown's 'trapped' status immunity), but partial trapping still damages them.
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import has_type
from engine.oujax.engine import MID, TY, _i, act, eat_sitrus, lose_hp, set_vol, types_now, vol
from engine.oujax.families.volatiles import move_table, speed_order

for _name in (
    "ptrap_turns",  # partiallytrapped's duration left, 0 = none
    "ptrap_src",  # 1 + the team slot of the Pokemon that set it (it ends once that one leaves)
    "trapped",  # 1: trapped by the foe's active Pokemon (Mean Look, Block, Jaw Lock); 2: by No Retreat
    "noretreat",
):
    hooks.VOL.add(_name)

PARTIAL_TRAP = move_table("magmastorm whirlpool firespin infestation")
SPINS = move_table("rapidspin mortalspin")


def _ghost(st, s):
    return has_type(types_now(st, s, act(st, s)), TY["ghost"])


def _ptrap_on(st, s):
    """partiallytrapped holds side s: set, and its source is still the foe's active Pokemon."""
    o = 1 - s
    return (
        (vol(st, s, "ptrap_turns") > 0) & (vol(st, s, "ptrap_src") == _i(st.active[o]) + 1) & (st.hp[o, act(st, o)] > 0)
    )


@hooks.hook("trapped")
def _trapped(st, s):
    return ~_ghost(st, s) & (_ptrap_on(st, s) | (vol(st, s, "trapped") > 0))


@hooks.hook("after_hit")
def _after_hit(st, ctx, dealt, hit_sub):
    s, o, mid = ctx.s, ctx.o, ctx.mid
    hit_mon = ctx.landed & ~hit_sub & (st.hp[o, ctx.tslot] > 0)
    # Partial trapping: 5 or 6 turns (forced: 5), not renewed while it lasts
    turns = jnp.int32(5) if ctx.cfg.forced else jax.random.randint(jax.random.fold_in(ctx.key, 601), (), 5, 7)
    ptrap = hit_mon & PARTIAL_TRAP[mid] & (vol(st, o, "ptrap_turns") == 0)
    st = set_vol(st, o, "ptrap_turns", turns, ptrap)
    st = set_vol(st, o, "ptrap_src", ctx.slot + 1, ptrap)
    # Jaw Lock: both Pokemon are trapped while both stay in
    jaw = hit_mon & (mid == MID["jawlock"]) & (st.hp[s, ctx.slot] > 0)
    st = set_vol(st, s, "trapped", 1, jaw & ~_ghost(st, s) & (vol(st, s, "trapped") == 0))
    st = set_vol(st, o, "trapped", 1, jaw & ~_ghost(st, o) & (vol(st, o, "trapped") == 0))
    # Rapid Spin / Mortal Spin free the user (also when they hit a Substitute)
    spin = ctx.landed & SPINS[mid] & (st.hp[s, ctx.slot] > 0)
    st = set_vol(st, s, "ptrap_turns", 0, spin)
    return set_vol(st, s, "leechseed", 0, spin)


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, o, mid = ctx.s, ctx.o, ctx.mid
    # Mean Look / Block: addVolatile('trapped') fails if already trapped or for a Ghost type
    look = (mid == MID["meanlook"]) | (mid == MID["block"])
    fails = look & ((vol(st, o, "trapped") > 0) | _ghost(st, o))
    return fails | ((mid == MID["noretreat"]) & (vol(st, s, "noretreat") > 0))


@hooks.hook("status_move")
def _status(st, ctx):
    s, o, mid, ok = ctx.s, ctx.o, ctx.mid, ctx.landed
    look = ok & ((mid == MID["meanlook"]) | (mid == MID["block"])) & (st.sub_hp[o] == 0)
    st = set_vol(st, o, "trapped", 1, look)
    # No Retreat (after its +1s): the volatile only starts if the user can be trapped
    retreat = ok & (mid == MID["noretreat"]) & ~_ghost(st, s) & (vol(st, s, "trapped") == 0)
    st = set_vol(st, s, "noretreat", 1, retreat)
    return set_vol(st, s, "trapped", 2, retreat)


@hooks.hook("switch_out")
def _leave(st, s, cond):
    # The trapper leaving frees the Pokemon it trapped (linked volatiles)
    o = 1 - s
    return set_vol(st, o, "trapped", 0, cond & (vol(st, o, "trapped") == 1))


@hooks.residual(13, "partial_trap")
def _partial_trap(st, key, cfg, cond):
    order = speed_order(st, key, cfg)
    for i in range(2):
        s = order[i]
        slot = act(st, s)
        on = cond & (vol(st, s, "ptrap_turns") > 0) & (st.hp[s, slot] > 0)
        left = vol(st, s, "ptrap_turns") - 1
        st = set_vol(st, s, "ptrap_turns", left, on)
        # Ends silently once its source is gone; otherwise 1/8 max HP while turns remain
        gone = on & ~_ptrap_on(st, s)
        st = set_vol(st, s, "ptrap_turns", 0, gone)
        hurt = on & ~gone & (left > 0)
        st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 8, hurt)
        st = eat_sitrus(st, s)
    return st


hooks.showdown_volatile("partiallytrapped")(lambda st, s: vol(st, s, "ptrap_turns") > 0)
hooks.showdown_volatile("trapped")(lambda st, s: vol(st, s, "trapped") > 0)
hooks.showdown_volatile("noretreat")(lambda st, s: vol(st, s, "noretreat") > 0)
