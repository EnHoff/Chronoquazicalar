"""
engine/oujax/families/held_items.py
Held items with in-battle triggers (berries are in berries.py, herbs and seeds in herbs_seeds.py):
- Rocky Helmet (per contact hit; Protective Pads on the attacker ignores it);
- Focus Sash: survives a move's hit at 1 HP from full HP (each hit of a multi-hit move checks again);
- Toxic Orb / Flame Orb (residual 28.3, self-inflicted status) and Sticky Barb (residual 28.3, 1/8 max HP; moves to
  an item-less attacker that makes contact);
- Weakness Policy: +2 Atk / +2 SpA after a super-effective hit (not fixed-damage or damage-callback moves);
- Throat Spray: +1 SpA after the holder's sound move succeeds (AfterMoveSecondarySelf);
- Blunder Policy: +2 Spe when the holder's move misses its accuracy check;
- Covert Cloak: the foe's moves lose their secondary effects on the holder;
- Clear Amulet: stat drops from the other Pokemon are blocked;
- Big Root: drain moves heal 5324/4096 as much;
- Quick Claw (20%) and Custap Berry (at 1/4 HP, eaten) move first within the priority bracket.
"""

import jax
import jax.numpy as jnp
import numpy as np

from engine.oujax import hooks
from engine.oujax.damage import modify, move_type_and_bp, type_effectiveness
from engine.oujax.effects import ITEM, T
from engine.oujax.engine import (
    BRN,
    MOVES,
    TOX,
    _i,
    act,
    action_speed,
    apply_boosts,
    combatant,
    eat_sitrus,
    lose_hp,
    p1_first,
    try_set_status,
    types_now,
)
from engine.oujax.tables import FLAG_BIT, MC

_ids = T.item_ids
FOCUS_SASH, TOXIC_ORB, FLAME_ORB, STICKY_BARB = (_ids[n] for n in ("focussash", "toxicorb", "flameorb", "stickybarb"))
WEAKNESS_POLICY, THROAT_SPRAY, BLUNDER_POLICY = (_ids[n] for n in ("weaknesspolicy", "throatspray", "blunderpolicy"))
COVERT_CLOAK, CLEAR_AMULET, BIG_ROOT, PADS = (
    _ids[n] for n in ("covertcloak", "clearamulet", "bigroot", "protectivepads")
)
QUICK_CLAW, CUSTAP = _ids["quickclaw"], _ids["custapberry"]

# Moves with fixed damage (move.damage) or a damageCallback: Weakness Policy ignores them
_fixed = np.zeros(len(T.move_ids) + 1, bool)
for _m, _raw in T.dex["moves"].items():
    if _raw.get("damage") is not None or "damageCallback" in (_raw.get("handlers") or []):
        _fixed[T.move_ids[_m]] = True
FIXED_DAMAGE = jnp.asarray(_fixed)


def take_item(st, s, slot, cond):
    """The item is used up (useItem / eatItem)."""
    return st.replace(item=st.item.at[s, slot].set(jnp.where(cond, 0, st.item[s, slot])))


def speed_order(st, key, cfg):
    """Sides in Speed order (ties: a coin flip, p1 when forced), as Showdown sorts residual handlers."""
    spe = jnp.stack([action_speed(st, 0), action_speed(st, 1)])
    first0 = p1_first(key, jnp.zeros(2, jnp.int32), spe, cfg)
    return jnp.where(first0, jnp.array([0, 1]), jnp.array([1, 0]))


def makes_contact(st, s, slot, mid):
    """Battle.checkMoveMakesContact: the contact flag, except punches thrown with a Punching Glove (which drops
    the flag) and any move from a Protective Pads holder."""
    flags = MOVES[mid, MC["flags"]]
    punch = (flags & FLAG_BIT["punch"]) > 0
    item = st.item[s, slot]
    return ((flags & FLAG_BIT["contact"]) > 0) & ~(punch & (item == ITEM.punchingglove)) & (item != PADS)


@hooks.hook("damaging_hit")
def _rocky_helmet(st, ctx, hit, dmg):
    s, slot = ctx.s, ctx.slot
    helmet = ctx.landed & (st.item[ctx.o, ctx.tslot] == ITEM.rockyhelmet) & makes_contact(st, s, slot, ctx.mid)
    return lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 6, helmet)


@hooks.hook("damaging_hit", priority=5)
def _sticky_barb_moves(st, ctx, hit, dmg):
    # onHit (runMoveEffects, before DamagingHit): an attacker without an item that makes contact takes the Barb
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    move = ctx.landed & (st.item[o, tslot] == STICKY_BARB) & (st.item[s, slot] == 0) & (st.hp[s, slot] > 0)
    move &= makes_contact(st, s, slot, ctx.mid)
    item = st.item.at[s, slot].set(jnp.where(move, STICKY_BARB, st.item[s, slot]))
    return st.replace(item=item.at[o, tslot].set(jnp.where(move, 0, item[o, tslot])))


@hooks.hook("damaging_hit", priority=-5)
def _weakness_policy(st, ctx, hit, dmg):
    o, tslot = ctx.o, ctx.tslot
    mtype, _ = move_type_and_bp(ctx.mid, combatant(st, ctx.s), combatant(st, o), _i(st.weather))
    _, type_mod = type_effectiveness(mtype, types_now(st, o, tslot), ctx.mid)
    use = ctx.landed & (st.item[o, tslot] == WEAKNESS_POLICY) & (type_mod > 0) & ~FIXED_DAMAGE[ctx.mid]
    use &= st.hp[o, tslot] > 0
    st = take_item(st, o, tslot, use)
    st, _ = apply_boosts(st, o, jnp.array([2, 0, 2, 0, 0, 0, 0]), use)
    return st


@hooks.hook("damage_taken", priority=-40)
def _focus_sash(st, ctx, side, dmg):
    slot = act(st, side)
    hp = _i(st.hp[side, slot])
    full = hp == _i(st.stats[side, slot, 0])
    use = ctx.landed & (st.item[side, slot] == FOCUS_SASH) & full & (hp > 0) & (dmg >= hp)
    return take_item(st, side, slot, use), jnp.where(use, hp - 1, dmg)


@hooks.residual(28.3, "orbs_and_barb")
def _orbs_and_barb(st, key, cfg, cond):
    order = speed_order(st, key, cfg)
    for i in range(2):
        s = order[i]
        slot = act(st, s)
        item, alive = st.item[s, slot], cond & (st.hp[s, slot] > 0)
        k = jax.random.fold_in(key, 10 + i)
        st, _ = try_set_status(st, s, slot, TOX, k, cfg, from_foe=False, cond=alive & (item == TOXIC_ORB))
        st, _ = try_set_status(st, s, slot, BRN, k, cfg, from_foe=False, cond=alive & (item == FLAME_ORB))
        st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 8, alive & (item == STICKY_BARB))
        st = eat_sitrus(st, s)
    return st


def _sound(mid):
    return (MOVES[mid, MC["flags"]] & FLAG_BIT["sound"]) > 0


@hooks.hook("after_move", priority=5)
def _after_move_items(st, ctx):
    from engine.oujax.families.forced_switch import suppressing_secondaries

    s, slot = ctx.s, ctx.slot
    alive = st.hp[s, slot] > 0
    # Throat Spray (AfterMoveSecondarySelf: the move succeeded; skipped under Sheer Force)
    spray = (
        ctx.landed & alive & (st.item[s, slot] == THROAT_SPRAY) & _sound(ctx.mid) & ~suppressing_secondaries(st, ctx)
    )
    st = take_item(st, s, slot, spray)
    st, _ = apply_boosts(st, s, jnp.array([0, 0, 1, 0, 0, 0, 0]), spray)
    # Blunder Policy (hitStepAccuracy: the move missed; OHKO moves excluded)
    blunder = ctx.missed & alive & (st.item[s, slot] == BLUNDER_POLICY) & (MOVES[ctx.mid, MC["ohko"]] == 0)
    st = take_item(st, s, slot, blunder)
    st, _ = apply_boosts(st, s, jnp.array([0, 0, 0, 0, 2, 0, 0]), blunder)
    return st


@hooks.hook("secondaries_blocked")
def _covert_cloak(st, s, o):
    return st.item[o, jnp.maximum(act(st, o), 0)] == COVERT_CLOAK


@hooks.hook("boost_change")
def _clear_amulet(st, s, deltas, from_foe):
    amulet = st.item[s, jnp.maximum(act(st, s), 0)] == CLEAR_AMULET
    return jnp.where(amulet & jnp.asarray(from_foe, bool) & (deltas < 0), 0, deltas)


@hooks.hook("drain_heal")
def _big_root(st, s, amount):
    root = st.item[s, jnp.maximum(act(st, s), 0)] == BIG_ROOT
    return jnp.where(root, modify(amount, 5324), amount)


@hooks.hook("fractional_priority")
def _quick_claw_custap(st, side, mid, prio, cond, key, cfg):
    slot = jnp.maximum(act(st, side), 0)
    item, hp = st.item[side, slot], _i(st.hp[side, slot])
    ok = cond & (prio <= 0) & (hp > 0)
    custap = ok & (item == CUSTAP) & (hp * 4 <= _i(st.stats[side, slot, 0]))
    # The harness's forced mode reads Quick Claw's randomChance(1, 5) as the freeze-thaw draw
    roll = jnp.bool_(cfg.thaw) if cfg.forced else jax.random.randint(key, (), 0, 5) == 0
    claw = ok & (item == QUICK_CLAW) & roll
    return take_item(st, side, slot, custap), custap | claw
