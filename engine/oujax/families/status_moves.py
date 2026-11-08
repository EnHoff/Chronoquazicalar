"""
engine/oujax/families/status_moves.py
Status moves with their own rules: Focus Energy (+2 crit stage), Imprison (the foe can't use moves the user
knows), Stockpile (up to 3 layers of +1 Def / SpD), Spite (-4 PP on the target's last move), Speed Swap (swaps
Speed stats until switch-out), Jungle Healing / Lunar Blessing (1/4 max HP and a status cure).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import modify
from engine.oujax.engine import MID, STRUGGLE, _i, act, apply_boosts, gain_hp, mon_ex, set_mon_ex, set_vol, vol
from engine.oujax.families.volatiles import _last_move_slot

for _name in ("focusenergy", "imprison", "stockpile", "spe_swapped"):
    hooks.VOL.add(_name)
hooks.MON.add("spe_orig")  # Speed stat before Speed Swap (restored on switch-out, as Showdown's setSpecies)

HEAL_CURE = jnp.zeros(len(MID), bool).at[jnp.array([MID["junglehealing"], MID["lunarblessing"]])].set(True)


def _imprisoned(st, s, mids):
    """(4,) which of these moves side s can't use because the foe's active Pokemon has Imprison and knows them."""
    o = 1 - s
    foe_moves = _i(st.moves[o, jnp.maximum(act(st, o), 0)])
    shared = jnp.any((mids[:, None] == foe_moves[None, :]) & (mids[:, None] > 0), axis=1) & (mids != STRUGGLE)
    return shared & (vol(st, o, "imprison") > 0)


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, o, slot, mid = ctx.s, ctx.o, ctx.slot, ctx.mid
    fails = (mid == MID["focusenergy"]) & (vol(st, s, "focusenergy") > 0)
    fails |= (mid == MID["imprison"]) & (vol(st, s, "imprison") > 0)
    fails |= (mid == MID["stockpile"]) & (vol(st, s, "stockpile") >= 3)
    # Spite: the target's last move must be one of its moves with PP left
    lslot = _last_move_slot(st, o)
    spite_pp = _i(st.pp[o, ctx.tslot, jnp.maximum(lslot, 0)])
    fails |= (mid == MID["spite"]) & ((lslot < 0) | (spite_pp <= 0))
    # Jungle Healing / Lunar Blessing fail with nothing to heal or cure
    hp, maxhp = _i(st.hp[s, slot]), _i(st.stats[s, slot, 0])
    fails |= HEAL_CURE[mid] & (hp >= maxhp) & (st.status[s, slot] == 0)
    return fails


@hooks.hook("status_move")
def _status(st, ctx):
    s, o, slot, tslot, mid, ok = ctx.s, ctx.o, ctx.slot, ctx.tslot, ctx.mid, ctx.landed
    st = set_vol(st, s, "focusenergy", 1, ok & (mid == MID["focusenergy"]))
    st = set_vol(st, s, "imprison", 1, ok & (mid == MID["imprison"]))
    pile = ok & (mid == MID["stockpile"])
    st = set_vol(st, s, "stockpile", vol(st, s, "stockpile") + 1, pile)
    st, _ = apply_boosts(st, s, jnp.array([0, 1, 0, 1, 0, 0, 0]), pile)
    # Spite (bypasses Substitute)
    spite = ok & (mid == MID["spite"])
    lslot = jnp.maximum(_last_move_slot(st, o), 0)
    pp = st.pp[o, tslot, lslot]
    st = st.replace(pp=st.pp.at[o, tslot, lslot].set(jnp.where(spite, jnp.maximum(pp - 4, 0), pp).astype(pp.dtype)))
    # Speed Swap: swap the stored Speed stats, remembering each one's own for when it leaves
    swap = ok & (mid == MID["speedswap"])
    for side, sl in ((s, slot), (o, tslot)):
        first = swap & (vol(st, side, "spe_swapped") == 0)
        st = set_mon_ex(st, side, sl, "spe_orig", st.stats[side, sl, 5], first)
        st = set_vol(st, side, "spe_swapped", 1, swap)
    mine, theirs = st.stats[s, slot, 5], st.stats[o, tslot, 5]
    stats = st.stats.at[s, slot, 5].set(jnp.where(swap, theirs, mine))
    st = st.replace(stats=stats.at[o, tslot, 5].set(jnp.where(swap, mine, theirs)))
    # Jungle Healing / Lunar Blessing: heal this.modify(maxhp, 0.25), then cure status
    heal = ok & HEAL_CURE[mid]
    st = gain_hp(st, s, slot, modify(_i(st.stats[s, slot, 0]), 1024), heal)
    return st.replace(status=st.status.at[s, slot].set(jnp.where(heal, 0, st.status[s, slot])))


@hooks.hook("switch_out")
def _leave(st, s, cond):
    slot = jnp.maximum(act(st, s), 0)
    restore = cond & (vol(st, s, "spe_swapped") > 0)
    spe = jnp.where(restore, mon_ex(st, s, slot, "spe_orig"), st.stats[s, slot, 5])
    return st.replace(stats=st.stats.at[s, slot, 5].set(spe.astype(st.stats.dtype)))


@hooks.hook("crit_stage")
def _focus_energy(st, ctx, stage):
    return stage + 2 * (vol(st, ctx.s, "focusenergy") > 0)


@hooks.hook("legal_moves")
def _imprison_disable(st, s, mask):
    mids = _i(st.moves[s, jnp.maximum(act(st, s), 0)])
    return mask & ~_imprisoned(st, s, mids)


@hooks.hook("before_move", priority=4)
def _imprison_block(st, ctx):
    mids = jnp.full((4,), ctx.mid)
    return st, ctx.moves & _imprisoned(st, ctx.s, mids)[0]


for _v in ("focusenergy", "imprison", "stockpile"):
    hooks.showdown_volatile(_v)((lambda name: lambda st, s: vol(st, s, name) > 0)(_v))
