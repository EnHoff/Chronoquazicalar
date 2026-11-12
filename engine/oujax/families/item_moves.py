"""
engine/oujax/families/item_moves.py
Moves that move items:
- Trick / Switcheroo: swap items with the target (fails if neither holds one, if either item can't leave its holder
  or go to the other Pokemon (Booster Energy and Paradox Pokemon, Ogerpon's masks), or into a Substitute); a
  received Choice item clears the Choice lock.
Not modelled here: Sticky Hold (abilities).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.effects import ABILITY, ITEM_REMOVABLE
from engine.oujax.engine import MID, MOVES, eat_sitrus, refresh_paradox
from engine.oujax.tables import FLAG_BIT, MC

REMOVABLE = jnp.asarray(ITEM_REMOVABLE)
TRICKS = (MID["trick"], MID["switcheroo"])


def blocked_by_sub(st, ctx):
    """A non-sound, non-bypasssub move from a user without Infiltrator stops at the target's Substitute."""
    bypass = ((MOVES[ctx.mid, MC["flags"]] & (FLAG_BIT["sound"] | FLAG_BIT["bypasssub"])) > 0) | (
        st.ability[ctx.s, ctx.slot] == ABILITY.infiltrator
    )
    return (st.sub_hp[ctx.o] > 0) & ~bypass


def can_move(item, species):
    """The item may leave or join a Pokemon of this species (onTakeItem); no item is always fine."""
    return (item == 0) | REMOVABLE[item, species]


@hooks.hook("status_move")
def _trick(st, ctx):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    mine, yours = st.item[s, slot], st.item[o, tslot]
    sp_s, sp_o = st.species[s, slot], st.species[o, tslot]
    is_trick = (ctx.mid == TRICKS[0]) | (ctx.mid == TRICKS[1])
    ok = ctx.landed & is_trick & ~blocked_by_sub(st, ctx) & (st.hp[o, tslot] > 0) & ((mine > 0) | (yours > 0))
    ok &= can_move(mine, sp_s) & can_move(yours, sp_o) & can_move(mine, sp_o) & can_move(yours, sp_s)
    item = st.item.at[s, slot].set(jnp.where(ok, yours, mine))
    item = item.at[o, tslot].set(jnp.where(ok, mine, yours))
    # Choice items' onStart removes choicelock; a lock without a Choice item is dropped anyway
    st = st.replace(item=item, choice_lock=jnp.where(ok, -1, st.choice_lock).astype(st.choice_lock.dtype))
    st = refresh_paradox(st)
    return eat_sitrus(eat_sitrus(st, s), o)
