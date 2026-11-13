"""
engine/oujax/families/item_moves.py
Moves that move items or leave hazards behind:
- Trick / Switcheroo: swap items with the target (fails if neither holds one, if either item can't leave its holder
  or go to the other Pokemon (Booster Energy and Paradox Pokemon, Ogerpon's masks), or into a Substitute); a
  received Choice item clears the Choice lock;
- Thief: an item-less user that is still standing takes the target's item after the hit (not through a
  Substitute);
- Ceaseless Edge: after hitting (or hitting the Substitute) with the user still standing, add a layer of Spikes
  to the target's side (Stone Axe is in move_misc).
Not modelled here: Sticky Hold (abilities).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.effects import ITEM_REMOVABLE
from engine.oujax.engine import MID, eat_sitrus, refresh_paradox
from engine.oujax.families.self_ko import blocked_by_sub
from engine.oujax.state import HAZ_SPIKES

REMOVABLE = jnp.asarray(ITEM_REMOVABLE)
TRICKS = (MID["trick"], MID["switcheroo"])


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


@hooks.hook("after_hit", priority=20)
def _thief_and_hazards(st, ctx, dealt, hit_sub):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    user_up = st.hp[s, slot] > 0
    # Thief (onAfterHit: the target itself was hit)
    yours = st.item[o, tslot]
    steal = ctx.landed & ~hit_sub & (dealt > 0) & (ctx.mid == MID["thief"]) & user_up & (st.item[s, slot] == 0)
    steal &= (yours > 0) & can_move(yours, st.species[o, tslot]) & can_move(yours, st.species[s, slot])
    item = st.item.at[s, slot].set(jnp.where(steal, yours, st.item[s, slot]))
    st = st.replace(item=item.at[o, tslot].set(jnp.where(steal, 0, yours)))
    # Ceaseless Edge (onAfterHit and onAfterSubDamage, user still standing); Stone Axe is in move_misc
    hit = ctx.landed & (hit_sub | (dealt > 0)) & user_up
    hz = st.hazards[o]
    spikes = hit & (ctx.mid == MID["ceaselessedge"])
    hz = hz.at[HAZ_SPIKES].set(jnp.where(spikes, jnp.minimum(hz[HAZ_SPIKES] + 1, 3), hz[HAZ_SPIKES]))
    st = st.replace(hazards=st.hazards.at[o].set(hz))
    return eat_sitrus(st, s)
