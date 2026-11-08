"""
engine/oujax/families/stat_abilities.py
Abilities that change stat stages or move priority:
- ChangeBoost / TryBoost (boost_change): Contrary; Clear Body, White Smoke, Full Metal Body, Big Pecks, Hyper
  Cutter, Keen Eye, Mind's Eye and Mirror Armor block drops from the foe; Guard Dog turns Intimidate into +1 Atk.
- AfterEachBoost / AfterBoost (after_boost): Defiant / Competitive (+2 per stat the foe lowered), Mirror Armor
  (bounces the blocked drops to the foe), Rattled (+1 Spe after Intimidate).
- Moxie, Chilling Neigh, Grim Neigh, Battle Bond (Greninja-Bond, once per battle) after a KO with a move.
- Speed Boost (not on the turn it came in), Unburden (2x Speed once it loses its item on the field).
- Priority: Prankster (status moves), Gale Wings (Flying moves at full HP), Triage (healing moves).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import M_2, MOVES, chain_if
from engine.oujax.effects import T
from engine.oujax.engine import (
    act,
    alive_mask,
    apply_boosts,
    field_ex,
    foe,
    mon_ex,
    set_field_ex,
    set_mon_ex,
    set_vol,
    vol,
)
from engine.oujax.families.entry_abilities import (
    AB,
    BOOST_INTIMIDATE,
    BOOST_MIRROR,
    ability,
    eff_ability,
    is_ab,
    on_field,
)
from engine.oujax.tables import FLAG_BIT, MC

TY = T.type_ids
GRENINJA_BOND = T.species_ids.get("greninjabond", -1)

hooks.MON.add("bond_used")  # Battle Bond already boosted this battle
hooks.VOL.add("koed")  # this move knocked the target out (Moxie, ...)
hooks.VOL.add("had_item")  # held an item while on the field (Unburden)

# Stats each drop-blocking ability protects: [atk, def, spa, spd, spe, accuracy, evasion]
_ALL = jnp.ones(7, bool)
_BLOCKS = {
    "clearbody": _ALL,
    "whitesmoke": _ALL,
    "fullmetalbody": _ALL,
    "mirrorarmor": _ALL,
    "bigpecks": jnp.arange(7) == 1,
    "hypercutter": jnp.arange(7) == 0,
    "keeneye": jnp.arange(7) == 5,
    "mindseye": jnp.arange(7) == 5,
}


@hooks.hook("boost_change", priority=10)
def _contrary(st, s, deltas, from_foe):
    return jnp.where(eff_ability(st, s) == AB["contrary"], -deltas, deltas)


@hooks.hook("boost_change", priority=2)
def _guard_dog(st, s, deltas, from_foe):
    gd = (eff_ability(st, s) == AB["guarddog"]) & (field_ex(st, "boost_src") == BOOST_INTIMIDATE) & (deltas[0] < 0)
    return deltas.at[0].set(jnp.where(gd, 1, deltas[0]))


@hooks.hook("boost_change")
def _block_drops(st, s, deltas, from_foe):
    ab = eff_ability(st, s)
    mask = jnp.zeros(7, bool)
    for name, stats in _BLOCKS.items():
        mask = mask | ((ab == AB[name]) & stats)
    # A drop Mirror Armor already bounced isn't bounced (or blocked) again
    mask &= ~((ab == AB["mirrorarmor"]) & (field_ex(st, "boost_src") == BOOST_MIRROR))
    return jnp.where(jnp.asarray(from_foe) & mask & (deltas < 0), 0, deltas)


@hooks.hook("after_boost")
def _after_boost(st, s, old, new, raw, from_foe, depth):
    from_foe = jnp.asarray(from_foe)
    ab = ability(st, s)
    src = field_ex(st, "boost_src")
    # Mirror Armor: the drops it blocked hit the foe instead (capped at -6)
    reflected = jnp.where((raw < 0) & (old > -6), jnp.maximum(raw, -6 - old), 0)
    mirror = (
        from_foe
        & (eff_ability(st, s) == AB["mirrorarmor"])
        & (src != BOOST_MIRROR)
        & jnp.any(reflected < 0)
        & on_field(st, foe(s))
    )
    st = set_field_ex(st, "boost_src", BOOST_MIRROR, mirror)
    st, _ = apply_boosts(st, foe(s), reflected, mirror, from_foe=True, depth=depth + 1)
    st = set_field_ex(st, "boost_src", src, mirror)
    # Defiant / Competitive: +2 for each stat the foe lowered; Rattled: +1 Spe after Intimidate
    lowered = jnp.sum((new < old).astype(jnp.int32)) * from_foe
    defiant = jnp.where(ab == AB["defiant"], 2 * lowered, 0)
    competitive = jnp.where(ab == AB["competitive"], 2 * lowered, 0)
    rattled = (ab == AB["rattled"]) & (src == BOOST_INTIMIDATE) & (new[0] != old[0])
    self_boost = jnp.stack([defiant, 0, competitive, 0, rattled.astype(jnp.int32), 0, 0])
    st, _ = apply_boosts(st, s, self_boost, jnp.any(self_boost != 0) & on_field(st, s), depth=depth + 1)
    return st


# --- KO boosts ---------------------------------------------------------------------------------------


@hooks.hook("damaging_hit")
def _record_ko(st, ctx, hit, dmg):
    return set_vol(st, ctx.s, "koed", 1, ctx.landed & (st.hp[ctx.o, ctx.tslot] == 0))


@hooks.hook("after_move")
def _ko_boosts(st, ctx):
    s, slot = ctx.s, ctx.slot
    ab = ability(st, s)
    # Showdown's boost fails once the KO'd side has nothing left (the battle is over)
    ko = (vol(st, s, "koed") > 0) & (st.hp[s, slot] > 0) & jnp.any(alive_mask(st)[ctx.o])
    bond = (ab == AB["battlebond"]) & (st.species[s, slot] == GRENINJA_BOND) & (mon_ex(st, s, slot, "bond_used") == 0)
    atk = is_ab(ab, "moxie", "chillingneigh") | bond
    spa = (ab == AB["grimneigh"]) | bond
    deltas = jnp.stack([atk, jnp.bool_(False), spa, jnp.bool_(False), bond, jnp.bool_(False), jnp.bool_(False)])
    st, _ = apply_boosts(st, s, deltas.astype(jnp.int32), ko & jnp.any(deltas))
    st = set_mon_ex(st, s, slot, "bond_used", 1, ko & bond)
    return set_vol(st, s, "koed", 0)


# --- Speed ---------------------------------------------------------------------------------------------


@hooks.residual(28.2, "speed_boost")
def _speed_boost(st, key, cfg, cond):
    for s in (0, 1):
        go = cond & on_field(st, s) & (ability(st, s) == AB["speedboost"]) & (vol(st, s, "entry_turn") < st.turn)
        st, _ = apply_boosts(st, s, jnp.array([0, 0, 0, 0, 1, 0, 0]), go)
    return st


@hooks.hook("enter")
def _note_item(st, s, slot, cond):
    return set_vol(st, s, "had_item", (st.item[s, slot] != 0).astype(jnp.int32), cond)


@hooks.hook("speed")
def _unburden(st, s, mod):
    slot = jnp.maximum(act(st, s), 0)
    lost = (vol(st, s, "had_item") > 0) & (st.item[s, slot] == 0)
    return chain_if(mod, (ability(st, s) == AB["unburden"]) & lost, M_2)


# --- Priority -------------------------------------------------------------------------------------------


@hooks.hook("priority")
def _priority(st, s, mid, prio):
    ab = ability(st, s)
    row = MOVES[mid]
    slot = jnp.maximum(act(st, s), 0)
    status = row[MC["category"]] == 2
    healing = (row[MC["flags"]] & FLAG_BIT["heal"]) > 0
    full_hp = st.hp[s, slot] >= st.stats[s, slot, 0]
    prio = prio + ((ab == AB["prankster"]) & status)
    prio = prio + ((ab == AB["galewings"]) & (row[MC["type"]] == TY["flying"]) & full_hp)
    return prio + 3 * ((ab == AB["triage"]) & healing)
