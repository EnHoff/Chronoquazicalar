"""
engine/oujax/families/passive_abilities.py
Abilities that guard the holder or act at the end of the turn:
- Magic Guard: no damage except from moves (hazards, poison, recoil, Life Orb, Rocky Helmet, ...); Substitute,
  Belly Drum and Struggle recoil are direct damage and still apply.
- Sturdy (survives a hit from full HP with 1 HP), Disguise (Mimikyu's first hit deals 0, then it loses 1/8).
- Leaf Guard (no status in sun), Limber (no paralysis).
- Magnet Pull: the foe's Steel types can't switch out by choice (not Ghosts, not with Shed Shell).
- Bad Dreams (sleeping foes lose 1/8), Hydration (cured in rain), Shed Skin (33% cure; never in forced mode,
  where the harness answers that roll with false).
Healer only affects allies (inert in singles). Gluttony only changes pinch berries other than Sitrus (inert
until those berries exist).
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import has_type
from engine.oujax.effects import T
from engine.oujax.engine import (
    PAR,
    RAIN,
    SLP,
    SUN,
    _i,
    act,
    chance,
    eat_sitrus,
    lose_hp,
    mon_ex,
    set_mon_ex,
    types_now,
)
from engine.oujax.families.entry_abilities import AB, ability, eff_ability, on_field

TY = T.type_ids
MIMIKYU = T.species_ids["mimikyu"]
SHED_SHELL = T.item_ids["shedshell"]
UMBRELLA = T.item_ids["utilityumbrella"]

hooks.MON.add("busted")  # Disguise used up (Mimikyu-Busted)


@hooks.hook("indirect_damage")
def _magic_guard(st, s, slot):
    return st.ability[s, slot] != AB["magicguard"]


@hooks.hook("damage_taken", priority=10)
def _disguise(st, ctx, side, dmg):
    o, tslot = ctx.o, ctx.tslot
    up = (
        ctx.landed
        & (eff_ability(st, o) == AB["disguise"])
        & (st.species[o, tslot] == MIMIKYU)
        & (mon_ex(st, o, tslot, "busted") == 0)
    )
    st = set_mon_ex(st, o, tslot, "busted", 2, up)  # 2: busted by this hit, loses 1/8 in damaging_hit
    return st, jnp.where(up, 0, dmg)


@hooks.hook("damaging_hit", priority=10)
def _disguise_break(st, ctx, hit, dmg):
    o, tslot = ctx.o, ctx.tslot
    broke = mon_ex(st, o, tslot, "busted") == 2
    st = lose_hp(st, o, tslot, _i(st.stats[o, tslot, 0]) // 8, broke)
    return set_mon_ex(st, o, tslot, "busted", 1, broke)


@hooks.hook("damage_taken", priority=-30)
def _sturdy(st, ctx, side, dmg):
    o, tslot = ctx.o, ctx.tslot
    hp = _i(st.hp[o, tslot])
    holds = ctx.landed & (eff_ability(st, o) == AB["sturdy"]) & (hp == _i(st.stats[o, tslot, 0])) & (dmg >= hp)
    return st, jnp.where(holds, hp - 1, dmg)


@hooks.hook("status_immune")
def _status_guard(st, s, slot, status):
    ab = eff_ability(st, s)
    sun = (st.weather == SUN) & (st.item[s, slot] != UMBRELLA)
    return ((ab == AB["leafguard"]) & sun) | ((ab == AB["limber"]) & (status == PAR))


@hooks.hook("trapped")
def _magnet_pull(st, s):
    o = 1 - s
    slot = jnp.maximum(act(st, s), 0)
    types = types_now(st, s, slot)
    steel = has_type(types, TY["steel"]) & ~has_type(types, TY["ghost"])
    return on_field(st, o) & (ability(st, o) == AB["magnetpull"]) & steel & (st.item[s, slot] != SHED_SHELL)


@hooks.residual(5.3, "status_cure_abilities")
def _status_cures(st, key, cfg, cond):
    # Hydration (in rain) and Shed Skin (33%): onResidualOrder 5, subOrder 3
    for s in (0, 1):
        slot = jnp.maximum(act(st, s), 0)
        ab = ability(st, s)
        rain = (st.weather == RAIN) & (st.item[s, slot] != UMBRELLA)
        shed = chance(jax.random.fold_in(key, 50 + s), 33, 100, False, cfg)
        cure = cond & on_field(st, s) & (st.status[s, slot] > 0)
        cure &= ((ab == AB["hydration"]) & rain) | ((ab == AB["shedskin"]) & shed)
        st = st.replace(
            status=st.status.at[s, slot].set(jnp.where(cure, 0, st.status[s, slot]).astype(st.status.dtype)),
            status_turns=st.status_turns.at[s, slot].set(
                jnp.where(cure, 0, st.status_turns[s, slot]).astype(st.status_turns.dtype)
            ),
        )
    return st


@hooks.residual(28.2, "bad_dreams")
def _bad_dreams(st, key, cfg, cond):
    for s in (0, 1):
        o = 1 - s
        oslot = jnp.maximum(act(st, o), 0)
        hits = cond & on_field(st, s) & on_field(st, o) & (ability(st, s) == AB["baddreams"])
        hits &= st.status[o, oslot] == SLP
        st = lose_hp(st, o, oslot, _i(st.stats[o, oslot, 0]) // 8, hits)
        st = eat_sitrus(st, o)
    return st
