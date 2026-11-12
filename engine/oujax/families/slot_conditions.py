"""
engine/oujax/families/slot_conditions.py
Showdown's slot conditions (effects that stay on a side's active position, whoever is in it):
- Wish: heals half the user's max HP to whoever is in the slot at the end of the next turn (residual order 4);
- Healing Wish / Lunar Dance: the user faints; the next Pokemon to come in that needs it is fully healed and
  cured (Lunar Dance also restores PP), before entry hazards.
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.engine import MID, MOVES, _i, act, bench_available, gain_hp, set_hp, set_side_ex, side_ex
from engine.oujax.tables import MC

hooks.SIDE.add("wish_at")  # turn at whose end the Wish heals, 0 = none
hooks.SIDE.add("wish_hp")
hooks.SIDE.add("heal_wish")  # 1 Healing Wish, 2 Lunar Dance waiting for a switch-in


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, mid = ctx.s, ctx.mid
    fails = (mid == MID["wish"]) & (side_ex(st, s, "wish_at") > 0)
    return fails | (((mid == MID["healingwish"]) | (mid == MID["lunardance"])) & ~bench_available(st, s))


@hooks.hook("status_move")
def _status(st, ctx):
    s, slot, mid, ok = ctx.s, ctx.slot, ctx.mid, ctx.landed
    wish = ok & (mid == MID["wish"])
    st = set_side_ex(st, s, "wish_at", _i(st.turn) + 1, wish)
    st = set_side_ex(st, s, "wish_hp", _i(st.stats[s, slot, 0]) // 2, wish)
    hw, ld = ok & (mid == MID["healingwish"]), ok & (mid == MID["lunardance"])
    st = set_side_ex(st, s, "heal_wish", jnp.where(ld, 2, 1), hw | ld)
    return set_hp(st, s, slot, jnp.where(hw | ld, 0, st.hp[s, slot]))


@hooks.hook("switch_in_slot")
def _heal_wish(st, side, key, cfg, cond):
    slot = act(st, side)
    kind = side_ex(st, side, "heal_wish")
    maxhp = _i(st.stats[side, slot, 0])
    max_pp = MOVES[_i(st.moves[side, slot]), MC["pp"]]
    lunar = kind == 2
    needs = (st.hp[side, slot] < maxhp) | (st.status[side, slot] > 0) | (lunar & jnp.any(st.pp[side, slot] < max_pp))
    heal = cond & (kind > 0) & (st.hp[side, slot] > 0) & needs
    st = gain_hp(st, side, slot, maxhp, heal)
    st = st.replace(
        status=st.status.at[side, slot].set(jnp.where(heal, 0, st.status[side, slot])),
        pp=st.pp.at[side, slot].set(jnp.where(heal & lunar, max_pp, st.pp[side, slot]).astype(st.pp.dtype)),
    )
    return set_side_ex(st, side, "heal_wish", 0, heal)


@hooks.residual(4, "wish")
def _wish(st, key, cfg, cond):
    for s in (0, 1):
        due = cond & (side_ex(st, s, "wish_at") > 0) & (_i(st.turn) >= side_ex(st, s, "wish_at"))
        slot = act(st, s)
        st = gain_hp(st, s, slot, side_ex(st, s, "wish_hp"), due & (st.hp[s, slot] > 0))
        st = set_side_ex(st, s, "wish_at", 0, due)
    return st
