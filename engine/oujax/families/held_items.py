"""
engine/oujax/families/held_items.py
Held items with in-battle triggers:
- Focus Sash: survives a move's hit at 1 HP from full HP (each hit of a multi-hit move checks again).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.effects import T
from engine.oujax.engine import _i, act

FOCUS_SASH = T.item_ids["focussash"]


def take_item(st, s, slot, cond):
    """The item is used up (useItem / eatItem)."""
    return st.replace(item=st.item.at[s, slot].set(jnp.where(cond, 0, st.item[s, slot])))


@hooks.hook("damage_taken", priority=-40)
def _focus_sash(st, ctx, side, dmg):
    slot = act(st, side)
    hp = _i(st.hp[side, slot])
    full = hp == _i(st.stats[side, slot, 0])
    use = ctx.landed & (st.item[side, slot] == FOCUS_SASH) & full & (hp > 0) & (dmg >= hp)
    return take_item(st, side, slot, use), jnp.where(use, hp - 1, dmg)
