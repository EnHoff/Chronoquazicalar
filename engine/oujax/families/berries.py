"""
engine/oujax/families/berries.py
Berries that react to HP, status, PP or hits (Sitrus is built into engine.eat_sitrus; resist berries are in
damage.py; Custap Berry is with Quick Claw in held_items.py):
- Lum Berry (any status), Chesto Berry (sleep), Leppa Berry (a move at 0 PP gets 10 back), Salac Berry (+1 Spe at
  1/4 HP): Showdown's Update, which the engine runs wherever it checks Sitrus Berry, plus once after every action
  (as Battle.runAction does);
- Kee Berry: +1 Def after a physical hit (AfterMoveSecondary, so not through a Substitute or under Sheer Force);
- Bug Bite: the user eats the target's berry and gets its effect.
Not modelled: Lum Berry curing confusion (the volatiles family owns confusion), Unnerve / As One.
"""

import jax.numpy as jnp
import numpy as np

from engine.oujax import hooks
from engine.oujax.effects import ITEM, ITEM_REMOVABLE, T
from engine.oujax.engine import MID, MOVES, SLP, _i, act, apply_boosts, eat_sitrus, gain_hp
from engine.oujax.families.held_items import take_item
from engine.oujax.tables import MC

_ids = T.item_ids
LUM, CHESTO, LEPPA, SALAC, KEE = (_ids[n] for n in ("lumberry", "chestoberry", "leppaberry", "salacberry", "keeberry"))
_berry = np.zeros(len(_ids), bool)
for _iid, _it in T.dex["items"].items():
    _berry[_ids[_iid]] = bool(_it.get("isBerry"))
IS_BERRY = jnp.asarray(_berry)
REMOVABLE = jnp.asarray(ITEM_REMOVABLE)


def berry_effect(st, s, item, cond):
    """A berry's onEat for side s's active Pokemon (eaten from its own item or stolen by Bug Bite)."""
    slot = act(st, s)
    maxhp = _i(st.stats[s, slot, 0])
    st = gain_hp(st, s, slot, maxhp // 4, cond & (item == ITEM.sitrusberry))
    cure = cond & ((item == LUM) | ((item == CHESTO) & (st.status[s, slot] == SLP)))
    st = st.replace(
        status=st.status.at[s, slot].set(jnp.where(cure, 0, st.status[s, slot])),
        status_turns=st.status_turns.at[s, slot].set(jnp.where(cure, 0, st.status_turns[s, slot])),
    )
    st, _ = apply_boosts(st, s, jnp.array([0, 0, 0, 0, 1, 0, 0]), cond & (item == SALAC))
    st, _ = apply_boosts(st, s, jnp.array([0, 1, 0, 0, 0, 0, 0]), cond & (item == KEE))
    # Leppa Berry: the first move at 0 PP, else the first one below max, gets 10 PP back
    mids = _i(st.moves[s, slot])
    pp, max_pp = _i(st.pp[s, slot]), MOVES[mids, MC["pp"]]
    real = mids > 0
    empty, low = real & (pp == 0), real & (pp < max_pp)
    pick = jnp.where(jnp.any(empty), jnp.argmax(empty), jnp.argmax(low))
    leppa = cond & (item == LEPPA) & jnp.any(low)
    new_pp = pp.at[pick].set(jnp.minimum(pp[pick] + 10, max_pp[pick]))
    return st.replace(pp=st.pp.at[s, slot].set(jnp.where(leppa, new_pp, pp).astype(st.pp.dtype)))


@hooks.hook("item_update")
def _update(st, s):
    slot = act(st, s)
    item, hp = st.item[s, slot], _i(st.hp[s, slot])
    status = st.status[s, slot]
    mids = _i(st.moves[s, slot])
    eat = (item == LUM) & (status > 0)
    eat |= (item == CHESTO) & (status == SLP)
    eat |= (item == SALAC) & (hp * 4 <= _i(st.stats[s, slot, 0]))
    eat |= (item == LEPPA) & jnp.any((mids > 0) & (st.pp[s, slot] == 0))
    eat &= (slot >= 0) & (hp > 0)
    st = take_item(st, s, slot, eat)
    return berry_effect(st, s, item, eat)


@hooks.hook("after_move", priority=-20)
def _update_after_action(st, ctx):
    # Battle.runAction runs Update on every active Pokemon (Speed order) once the action is done
    return eat_sitrus(eat_sitrus(st, 0), 1)


@hooks.hook("after_hit", priority=5)
def _kee_berry(st, ctx, dealt, hit_sub):
    from engine.oujax.families.forced_switch import suppressing_secondaries

    o, tslot = ctx.o, ctx.tslot
    physical = MOVES[ctx.mid, MC["category"]] == 0
    eat = ctx.landed & ~hit_sub & (dealt > 0) & physical & ~suppressing_secondaries(st, ctx)
    eat &= (st.item[o, tslot] == KEE) & (st.hp[o, tslot] > 0)
    st = take_item(st, o, tslot, eat)
    return berry_effect(st, o, KEE, eat)


@hooks.hook("damaging_hit", priority=10)
def _bug_bite(st, ctx, hit, dmg):
    # The move's onHit (runMoveEffects, before DamagingHit): steal and eat the target's berry
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    item = _i(st.item[o, tslot])
    steal = ctx.landed & (ctx.mid == MID["bugbite"]) & (st.hp[s, slot] > 0) & IS_BERRY[item]
    steal &= REMOVABLE[item, st.species[o, tslot]]
    st = take_item(st, o, tslot, steal)
    return berry_effect(st, s, item, steal)
