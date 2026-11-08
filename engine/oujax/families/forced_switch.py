"""
engine/oujax/families/forced_switch.py
Moves and items that force a Pokemon out mid-turn, on top of the engine's switch flags (engine.process_flags):
- phazing (Roar, Whirlwind, Dragon Tail, Circle Throw): the target is dragged out for a random benched Pokemon
  (Showdown's forceSwitchFlag); status phazers fail when the target has nothing to switch to;
- Red Card: the attacker is dragged out after a damaging hit (and skips its Life Orb recoil);
- Eject Button: the holder asks for a switch after a damaging hit, cancelling the attacker's own pivot;
- Eject Pack: the holder asks for a switch once a stat stage drop has happened, at the end of the move (any
  Pokemon's), after a switch-in, or at the end of the turn; not from Parting Shot.

Not modelled here: drag-out immunities (Suction Cups, Guard Dog, Ingrain) belong to their owners.
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.effects import ITEM
from engine.oujax.engine import (
    MID,
    MOVES,
    SW_TRUE,
    bench_available,
    set_side_ex,
    set_vol,
    side_ex,
    vol,
)
from engine.oujax.tables import MC

hooks.VOL.add("eject_pack")  # a stat drop is waiting to trigger the Eject Pack


def _drags(mid):
    return MOVES[mid, MC["force_switch"]] > 0


def _no_true_flag(st):
    """No active Pokemon already has switchFlag === true (Eject Button / Eject Pack), which blocks another."""
    return (side_ex(st, 0, "switch_flag") != SW_TRUE) & (side_ex(st, 1, "switch_flag") != SW_TRUE)


@hooks.hook("move_fails")
def _phaze_fails(st, ctx):
    # Roar / Whirlwind fail if the target can't switch (runMoveEffects' canSwitch check)
    return ctx.is_status & _drags(ctx.mid) & ~bench_available(st, ctx.o)


@hooks.hook("status_move")
def _status_phaze(st, ctx):
    o = ctx.o
    drag = ctx.landed & _drags(ctx.mid) & (st.hp[o, ctx.tslot] > 0) & (st.hp[ctx.s, ctx.slot] > 0)
    st = set_side_ex(st, o, "drag_flag", 1, drag & bench_available(st, o))
    # Parting Shot's drops don't arm the target's Eject Pack
    return set_vol(st, o, "eject_pack", 0, ctx.landed & (ctx.mid == MID["partingshot"]))


@hooks.hook("after_hit", priority=10)
def _after_hit(st, ctx, dealt, hit_sub):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    hit = ctx.landed & ~hit_sub & (dealt > 0)
    user_up, target_up = st.hp[s, slot] > 0, st.hp[o, tslot] > 0
    # Dragon Tail / Circle Throw (spreadMoveHit step 6, before the target's items react)
    drag = hit & _drags(ctx.mid) & user_up & target_up & bench_available(st, o)
    st = set_side_ex(st, o, "drag_flag", 1, drag)
    # AfterMoveSecondary: Eject Button (priority 2) before Red Card
    dragged = (side_ex(st, o, "drag_flag") > 0) | (side_ex(st, s, "drag_flag") > 0)
    eject = hit & target_up & (st.item[o, tslot] == ITEM.ejectbutton) & bench_available(st, o)
    eject &= (side_ex(st, o, "drag_flag") == 0) & _no_true_flag(st)
    st = set_side_ex(st, o, "switch_flag", SW_TRUE, eject)
    st = set_side_ex(st, s, "switch_flag", 0, eject)
    red = hit & user_up & target_up & (st.item[o, tslot] == ITEM.redcard) & bench_available(st, s) & ~dragged
    st = set_side_ex(st, s, "drag_flag", 1, red)
    return st.replace(item=st.item.at[o, tslot].set(jnp.where(eject | red, 0, st.item[o, tslot])))


@hooks.hook("after_boost")
def _arm_eject_pack(st, side, applied, cond):
    arm = cond & jnp.any(applied < 0) & (st.item[side, jnp.maximum(st.active[side], 0)] == ITEM.ejectpack)
    return set_vol(st, side, "eject_pack", 1, arm)


def _fire_eject_pack(st, cond):
    for s in (0, 1):
        slot = jnp.maximum(st.active[s], 0)
        armed = cond & (vol(st, s, "eject_pack") > 0)
        holds = (st.item[s, slot] == ITEM.ejectpack) & (st.hp[s, slot] > 0)
        # useItem fails without a Pokemon to switch to or while another switchFlag === true is pending (the
        # pack stays); a pivot's flag (move id) doesn't block it
        fire = armed & holds & bench_available(st, s) & _no_true_flag(st)
        st = set_side_ex(st, s, "switch_flag", SW_TRUE, fire)
        st = st.replace(item=st.item.at[s, slot].set(jnp.where(fire, 0, st.item[s, slot])))
        st = set_vol(st, s, "eject_pack", 0, armed)
    return st


@hooks.hook("after_move", priority=-10)
def _eject_pack_after_move(st, ctx):
    return _fire_eject_pack(st, jnp.bool_(True))


@hooks.hook("switch_in", priority=-4)
def _eject_pack_switch_in(st, side, key, cfg, cond):
    return _fire_eject_pack(st, cond)


@hooks.residual(29, "eject_pack")
def _eject_pack_residual(st, key, cfg, cond):
    return _fire_eject_pack(st, cond)
