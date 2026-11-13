"""
engine/oujax/families/self_ko.py
Moves that set HP directly:
- Explosion / Self-Destruct / Misty Explosion (selfdestruct 'always'): the user faints as the move starts, even
  into Protect, a miss or an immunity, and the hit still lands (engine.SELF_KO_HITS); Misty Explosion is 1.5x on
  Misty Terrain for a grounded user;
- Memento (-2 Atk / -2 SpA from the table) faints the user only if it hits (not into a Substitute or Protect);
- Final Gambit deals the user's HP and faints it once it connects (before the damage lands);
- Pain Split averages the user's and the target's HP.
The user faints first, so when both last Pokemon go down the target's side wins (Battle.checkWin).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.effects import ABILITY
from engine.oujax.engine import M_TERRAIN, MID, MOVES, _i, grounded, set_hp, set_vol, vol
from engine.oujax.tables import FLAG_BIT, MC

hooks.VOL.add("gambit_hp")  # Final Gambit: the user's HP when it fainted using it

ALWAYS = tuple(MID[m] for m in ("explosion", "selfdestruct", "mistyexplosion"))


def _always(mid):
    return (mid == ALWAYS[0]) | (mid == ALWAYS[1]) | (mid == ALWAYS[2])


def blocked_by_sub(st, ctx):
    """A non-sound, non-bypasssub move from a user without Infiltrator stops at the target's Substitute."""
    bypass = ((MOVES[ctx.mid, MC["flags"]] & (FLAG_BIT["sound"] | FLAG_BIT["bypasssub"])) > 0) | (
        st.ability[ctx.s, ctx.slot] == ABILITY.infiltrator
    )
    return (st.sub_hp[ctx.o] > 0) & ~bypass


@hooks.hook("try_hit", priority=10)
def _explode(st, ctx):
    # useMoveInner faints the user right after TryMove, before Protect, accuracy and immunity
    st = set_hp(st, ctx.s, ctx.slot, jnp.where(ctx.proceeds & _always(ctx.mid), 0, st.hp[ctx.s, ctx.slot]))
    return st, jnp.bool_(False)


@hooks.hook("move_try_hit")
def _gambit(st, ctx):
    gambit = ctx.landed & (ctx.mid == MID["finalgambit"])
    st = set_vol(st, ctx.s, "gambit_hp", _i(st.hp[ctx.s, ctx.slot]), gambit)
    return set_hp(st, ctx.s, ctx.slot, jnp.where(gambit, 0, st.hp[ctx.s, ctx.slot]))


@hooks.hook("damage_override")
def _gambit_damage(st, ctx, dmg):
    return jnp.where(ctx.mid == MID["finalgambit"], vol(st, ctx.s, "gambit_hp"), dmg)


@hooks.hook("hit_power")
def _misty_explosion(st, ctx, hit):
    boost = (ctx.mid == MID["mistyexplosion"]) & (st.terrain == M_TERRAIN) & grounded(st, ctx.s, ctx.slot)
    return jnp.int32(1), jnp.where(boost, 6144, 4096).astype(jnp.int32)


@hooks.hook("status_move")
def _status(st, ctx):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    reaches = ctx.landed & ~blocked_by_sub(st, ctx)
    st = set_hp(st, s, slot, jnp.where(reaches & (ctx.mid == MID["memento"]), 0, st.hp[s, slot]))
    # Pain Split: both take the floored average (at least 1), capped at their max HP
    split = reaches & (ctx.mid == MID["painsplit"]) & (st.hp[o, tslot] > 0)
    avg = jnp.maximum((_i(st.hp[o, tslot]) + _i(st.hp[s, slot])) // 2, 1)
    st = set_hp(st, o, tslot, jnp.where(split, avg, st.hp[o, tslot]))
    return set_hp(st, s, slot, jnp.where(split, avg, st.hp[s, slot]))
