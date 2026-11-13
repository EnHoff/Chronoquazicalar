"""
engine/oujax/families/slot_conditions.py
Showdown's slot conditions (effects that stay on a side's active position, whoever is in it):
- Wish: heals half the user's max HP to whoever is in the slot at the end of the next turn (residual order 4);
- Healing Wish / Lunar Dance: the user faints; the next Pokemon to come in that needs it is fully healed and
  cured (Lunar Dance also restores PP), before entry hazards;
- Future Sight / Doom Desire: hit the slot at the end of the turn after next (residual order 3), with the damage
  calculated then from the user's current state (a user that left the field has no item, ability or boosts);
  simplification: the delayed hit skips its accuracy check (100% accuracy; only evasion boosts could matter);
- Revival Blessing: a switch request that picks a fainted team member to revive at half HP (engine
  process_flags / step).
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import Combatant, damage_finish, damage_parts
from engine.oujax.effects import ABILITY, ITEM
from engine.oujax.engine import (
    CRIT_DEN,
    MID,
    MOVES,
    SW_REVIVE,
    MoveCtx,
    _i,
    act,
    bench_available,
    combatant,
    eat_sitrus,
    field_info,
    gain_hp,
    lose_hp,
    set_hp,
    set_side_ex,
    side_ex,
)
from engine.oujax.tables import MC

hooks.SIDE.add("wish_at")  # turn at whose end the Wish heals, 0 = none
hooks.SIDE.add("wish_hp")
hooks.SIDE.add("heal_wish")  # 1 Healing Wish, 2 Lunar Dance waiting for a switch-in
hooks.SIDE.add("future_at")  # turn at whose end the future attack hits this side's slot, 0 = none
hooks.SIDE.add("future_mid")
hooks.SIDE.add("future_src")  # the attacker's team slot (on the other side)

FUTURE = (MID["futuresight"], MID["doomdesire"])


def _is_future(mid):
    return (mid == FUTURE[0]) | (mid == FUTURE[1])


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, o, mid = ctx.s, ctx.o, ctx.mid
    fails = (mid == MID["wish"]) & (side_ex(st, s, "wish_at") > 0)
    fails |= ((mid == MID["healingwish"]) | (mid == MID["lunardance"])) & ~bench_available(st, s)
    fails |= _is_future(mid) & (side_ex(st, o, "future_at") > 0)
    fainted_mates = jnp.any((st.species[s] > 0) & (st.hp[s] == 0))
    return fails | ((mid == MID["revivalblessing"]) & ~fainted_mates)


@hooks.hook("try_hit")
def _future_set(st, ctx):
    # Future Sight / Doom Desire's onTry: set the slot condition and stop (before Protect, accuracy, immunity)
    o = ctx.o
    put = ctx.proceeds & _is_future(ctx.mid)
    st = set_side_ex(st, o, "future_at", _i(st.turn) + 2, put)
    st = set_side_ex(st, o, "future_mid", ctx.mid, put)
    st = set_side_ex(st, o, "future_src", ctx.slot, put)
    return st, put


@hooks.hook("status_move")
def _status(st, ctx):
    s, slot, mid, ok = ctx.s, ctx.slot, ctx.mid, ctx.landed
    wish = ok & (mid == MID["wish"])
    st = set_side_ex(st, s, "wish_at", _i(st.turn) + 1, wish)
    st = set_side_ex(st, s, "wish_hp", _i(st.stats[s, slot, 0]) // 2, wish)
    hw, ld = ok & (mid == MID["healingwish"]), ok & (mid == MID["lunardance"])
    st = set_side_ex(st, s, "heal_wish", jnp.where(ld, 2, 1), hw | ld)
    st = set_hp(st, s, slot, jnp.where(hw | ld, 0, st.hp[s, slot]))
    return set_side_ex(st, s, "switch_flag", SW_REVIVE, ok & (mid == MID["revivalblessing"]))


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


def _attacker(st, a, slot):
    """The future attack's user as getDamage sees it: off the field it has no item, ability or boosts."""
    on_field = act(st, a) == slot
    c = combatant(st, a)
    bench = Combatant(
        species=_i(st.species[a, slot]),
        level=_i(st.level[a, slot]),
        stats=st.stats[a, slot].astype(jnp.int32),
        boosts=jnp.zeros(7, jnp.int32),
        base_types=st.base_types[a, slot].astype(jnp.int32),
        tera_type=_i(st.tera_type[a, slot]),
        terastallized=st.terastallized[a, slot],
        status=_i(st.status[a, slot]),
        item=jnp.int32(0),
        ability=jnp.int32(0),
        paradox_stat=jnp.int32(0),
        stellar_used=st.stellar_used[a, slot],
        hp=_i(st.hp[a, slot]),
        fallen=jnp.int32(0),
    )
    return jax.tree.map(lambda x, y: jnp.where(on_field, x, y), c, bench), on_field


@hooks.residual(3, "future_attacks")
def _future_hit(st, key, cfg, cond):
    # One traced damage calculation for both sides (fori_loop) keeps the compiled step small
    def hit_side(o, st):
        a = 1 - o
        due = cond & (side_ex(st, o, "future_at") > 0) & (_i(st.turn) >= side_ex(st, o, "future_at"))
        st = set_side_ex(st, o, "future_at", 0, due)
        mid, src, tslot = side_ex(st, o, "future_mid"), side_ex(st, o, "future_src"), act(st, o)
        hit = due & (st.hp[o, tslot] > 0)
        att, on_field = _attacker(st, a, src)
        k = jax.random.fold_in(key, o)
        if cfg.forced:
            crit, roll = jnp.bool_(cfg.crit), jnp.int32(cfg.roll)
        else:
            crit = jax.random.randint(jax.random.fold_in(k, 1), (), 0, CRIT_DEN[1]) == 0
            roll = jax.random.randint(jax.random.fold_in(k, 2), (), 0, 16)
        dfn = combatant(st, o)
        parts = damage_parts(mid, att, dfn, field_info(st, o), crit)
        sub = hit & (st.sub_hp[o] > 0) & (att.ability != ABILITY.infiltrator)
        d, berry = damage_finish(
            parts, crit, roll, _i(st.hp[o, tslot]), _i(st.stats[o, tslot, 0]), sub, with_berry=True
        )
        d = jnp.where(hit & parts.ok, d, 0)
        ctx = MoveCtx(
            s=jnp.int32(a), o=jnp.int32(o), slot=src, tslot=tslot, mid=mid, move_slot=jnp.int32(0),
            key=jax.random.fold_in(k, 3), cfg=cfg, is_status=jnp.bool_(False), targets_foe=jnp.bool_(True),
            foe_pending_move=jnp.int32(0), moves=hit, proceeds=hit, landed=hit & parts.ok,
        )  # fmt: skip
        for h in hooks.handlers("damage_taken"):
            st, d_mon = h(st, ctx._replace(landed=hit & parts.ok & ~sub), o, d)
            d = jnp.where(sub, d, d_mon)
        to_sub = jnp.minimum(d, _i(st.sub_hp[o]))
        st = st.replace(
            sub_hp=st.sub_hp.at[o].set(jnp.where(sub, _i(st.sub_hp[o]) - to_sub, st.sub_hp[o]).astype(st.sub_hp.dtype))
        )
        to_mon = jnp.where(sub, 0, jnp.minimum(d, _i(st.hp[o, tslot])))
        st = set_hp(st, o, tslot, _i(st.hp[o, tslot]) - to_mon)
        pop = hit & (d > 0) & (st.item[o, tslot] == ITEM.airballoon)
        st = st.replace(item=st.item.at[o, tslot].set(jnp.where(pop | (hit & berry & ~sub), 0, st.item[o, tslot])))
        for h in hooks.handlers("damaging_hit"):
            st = h(st, ctx._replace(landed=hit & ~sub & (to_mon > 0)), jnp.int32(0), to_mon)
        st = eat_sitrus(st, o)
        # Life Orb recoil for a user still on the field, even if the target was immune
        orb = hit & on_field & (st.item[a, src] == ITEM.lifeorb) & (st.hp[a, src] > 0)
        st = lose_hp(st, a, src, _i(st.stats[a, src, 0]) // 10, orb)
        st = eat_sitrus(st, a)
        return st

    return jax.lax.fori_loop(0, 2, hit_side, st)


@hooks.residual(4, "wish")
def _wish(st, key, cfg, cond):
    for s in (0, 1):
        due = cond & (side_ex(st, s, "wish_at") > 0) & (_i(st.turn) >= side_ex(st, s, "wish_at"))
        slot = act(st, s)
        st = gain_hp(st, s, slot, side_ex(st, s, "wish_hp"), due & (st.hp[s, slot] > 0))
        st = set_side_ex(st, s, "wish_at", 0, due)
    return st
