"""
engine/oujax/families/contact_abilities.py
Abilities that react to a hit or to the move around it:
- DamagingHit, once per hit on the Pokemon itself (damaging_hit): Static, Flame Body, Effect Spore (30% / 30%
  split status to a contact attacker), Rough Skin (1/8), Gooey / Tangling Hair (-1 Spe), Cursed Body (30%:
  Disable the move for 3 more turns), Weak Armor, Justified, Rattled, Toxic Debris, Gulp Missile; the
  attacker's Poison Touch (30% poison on contact) and Toxic Chain (30% toxic), which Shield Dust blocks.
- After the move: Pickpocket and Magician steal items, Berserk (+1 SpA when a hit drops it below half HP).
- Synchronize passes burn / paralysis / poison back to the foe that caused it (at the end of the move).
- Serene Grace doubles secondary-effect chances; Shield Dust blocks the foe's secondary effects on itself;
  Shell Armor / Battle Armor block critical hits (simplification: not a willCrit move's, like Wicked Blow).
Random 30% rolls are forced to happen in forced mode (the harness answers them with `other: true`).
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import MOVES, has_type
from engine.oujax.effects import ITEM, ITEM_REMOVABLE, T
from engine.oujax.engine import (
    BRN,
    FRZ,
    MID,
    PAR,
    PSN,
    SLP,
    TOX,
    _i,
    apply_boosts,
    bench_available,
    chance,
    eat_sitrus,
    lose_hp,
    set_vol,
    sticky_hold,
    try_set_status,
    types_now,
    vol,
)
from engine.oujax.families.entry_abilities import AB, ability, eff_ability, on_field
from engine.oujax.state import HAZ_TSPIKES
from engine.oujax.tables import FLAG_BIT, MC

TY = T.type_ids
ITEM_REMOVABLE_J = jnp.asarray(ITEM_REMOVABLE)
COVERT_CLOAK = T.item_ids.get("covertcloak", -1)
SAFETY_GOGGLES = T.item_ids.get("safetygoggles", -1)
CRAMORANT = T.species_ids.get("cramorant", -1)

hooks.VOL.add("disabled")  # move slot + 1 of a Disabled move
hooks.VOL.add("disable_turns")
hooks.VOL.add("gulp")  # Gulp Missile form: 1 Gulping, 2 Gorging
hooks.VOL.add("hp_after_hit")  # 1 + HP right after the latest damaging hit (Berserk)
hooks.VOL.add("sync")  # status Synchronize still has to pass back to the foe


def _contact(st, ctx):
    flags = MOVES[ctx.mid, MC["flags"]]
    punch = (flags & FLAG_BIT["punch"]) > 0
    return ((flags & FLAG_BIT["contact"]) > 0) & ~(punch & (st.item[ctx.s, ctx.slot] == ITEM.punchingglove))


@hooks.hook("damaging_hit")
def _damaging_hit(st, ctx, hit, dmg):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    row = MOVES[ctx.mid]
    hit_ = ctx.landed
    contact = hit_ & _contact(st, ctx)
    physical = row[MC["category"]] == 0
    mtype = row[MC["type"]]
    ab, user_ab = ability(st, o), ability(st, s)
    keys = jax.random.split(jax.random.fold_in(ctx.key, 100 + hit), 6)
    c30 = chance(keys[0], 3, 10, True, ctx.cfg)
    st = set_vol(st, o, "hp_after_hit", _i(st.hp[o, tslot]) + 1, hit_)

    # Status to the attacker: Static, Flame Body, Effect Spore, a Gorging Cramorant's Gulp Missile
    user_types = types_now(st, s, slot)
    powder_ok = ~has_type(user_types, TY["grass"]) & (user_ab != AB["overcoat"]) & (st.item[s, slot] != SAFETY_GOGGLES)
    if ctx.cfg.forced:
        spore_roll = jnp.int32(0)  # random(100) answers its minimum
    else:
        spore_roll = jax.random.randint(keys[1], (), 0, 100)
    spore = jnp.select([spore_roll < 11, spore_roll < 21, spore_roll < 30], [SLP, PAR, PSN], 0)
    gulp = vol(st, o, "gulp")
    to_user = jnp.select(
        [
            contact & (ab == AB["static"]) & c30,
            contact & (ab == AB["flamebody"]) & c30,
            contact & (ab == AB["effectspore"]) & powder_ok,
            hit_ & (gulp == 2),
        ],
        [PAR, BRN, spore, PAR],
        0,
    )
    st, _ = try_set_status(st, s, slot, to_user, keys[2], ctx.cfg, cond=to_user > 0)
    # Damage to the attacker: Rough Skin 1/8, Gulp Missile 1/4
    maxhp_s = _i(st.stats[s, slot, 0])
    st = lose_hp(st, s, slot, maxhp_s // 8, contact & (ab == AB["roughskin"]))
    st = lose_hp(st, s, slot, maxhp_s // 4, hit_ & (gulp > 0))
    # Stat drops to the attacker: Gooey / Tangling Hair, a Gulping Cramorant
    slow = contact & ((ab == AB["gooey"]) | (ab == AB["tanglinghair"]))
    drop = -jnp.stack([0, hit_ & (gulp == 1), 0, 0, slow, 0, 0]).astype(jnp.int32)
    st, _ = apply_boosts(st, s, drop, jnp.any(drop != 0) & on_field(st, s), from_foe=True)
    st = set_vol(st, o, "gulp", 0, hit_)
    st = eat_sitrus(st, s)
    # The holder's own boosts: Weak Armor, Justified, Rattled
    weak = physical & (ab == AB["weakarmor"])
    just = (mtype == TY["dark"]) & (ab == AB["justified"])
    rattled = ((mtype == TY["dark"]) | (mtype == TY["bug"]) | (mtype == TY["ghost"])) & (ab == AB["rattled"])
    own = jnp.stack([just, -_i(weak), 0, 0, 2 * _i(weak) + rattled, 0, 0]).astype(jnp.int32)
    st, _ = apply_boosts(st, o, own, hit_ & jnp.any(own != 0) & on_field(st, o))
    # Cursed Body: Disable the move just used (not Struggle, not a move out of PP)
    curse = (
        hit_
        & (ab == AB["cursedbody"])
        & c30
        & (vol(st, s, "disabled") == 0)
        & (ctx.mid != MID["struggle"])
        & (st.pp[s, slot, ctx.move_slot] > 0)
        & on_field(st, s)
    )
    st = set_vol(st, s, "disabled", ctx.move_slot + 1, curse)
    st = set_vol(st, s, "disable_turns", 4, curse)
    # Toxic Debris: a physical hit lays Toxic Spikes on the attacker's side (up to 2 layers)
    debris = hit_ & physical & (ab == AB["toxicdebris"]) & (st.hazards[s, HAZ_TSPIKES] < 2)
    st = st.replace(hazards=st.hazards.at[s, HAZ_TSPIKES].add(debris.astype(st.hazards.dtype)))
    # The attacker's Poison Touch / Toxic Chain (blocked by Shield Dust and Covert Cloak)
    shielded = (ab == AB["shielddust"]) | (st.item[o, tslot] == COVERT_CLOAK)
    c30b = chance(keys[3], 3, 10, True, ctx.cfg)
    to_target = jnp.select(
        [hit_ & (user_ab == AB["toxicchain"]) & c30b, contact & (user_ab == AB["poisontouch"]) & c30b], [TOX, PSN], 0
    )
    st, _ = try_set_status(st, o, tslot, to_target, keys[4], ctx.cfg, cond=(to_target > 0) & ~shielded)
    return st


@hooks.hook("move_try_hit")
def _gulp_missile(st, ctx):
    # Cramorant's Surf picks up a catch: Gorging at half HP or below, else Gulping
    s, slot = ctx.s, ctx.slot
    go = ctx.landed & (ctx.mid == MID["surf"]) & (ability(st, s) == AB["gulpmissile"])
    go &= st.species[s, slot] == CRAMORANT
    form = jnp.where(_i(st.hp[s, slot]) * 2 <= _i(st.stats[s, slot, 0]), 2, 1)
    return set_vol(st, s, "gulp", form, go)


@hooks.hook("after_hit")
def _after_hit(st, ctx, dealt, hit_sub):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    row = MOVES[ctx.mid]
    # Sheer Force suppresses AfterMoveSecondary effects
    sheer = (ability(st, s) == AB["sheerforce"]) & ((row[MC["sec_chance"]] > 0) | (row[MC["sec_status"]] > 0))
    pivoting = (row[MC["self_switch"]] > 0) & (st.hp[s, slot] > 0) & bench_available(st, s)
    hit_mon = ctx.landed & ~hit_sub & ~sheer
    # Berserk: +1 SpA when the move's damage took it from above half HP to half or below
    maxhp_o = _i(st.stats[o, tslot, 0])
    hp_after = vol(st, o, "hp_after_hit") - 1
    berserk = hit_mon & (ability(st, o) == AB["berserk"]) & (st.hp[o, tslot] > 0) & (dealt > 0)
    berserk &= (hp_after * 2 <= maxhp_o) & ((hp_after + dealt) * 2 > maxhp_o)
    st, _ = apply_boosts(st, o, jnp.array([0, 0, 1, 0, 0, 0, 0]), berserk)
    # Pickpocket: an itemless holder takes a contact attacker's item
    user_item, foe_item = st.item[s, slot], st.item[o, tslot]
    takeable = ITEM_REMOVABLE_J[user_item, st.species[s, slot]] & ~sticky_hold(st, s, slot, 0)
    pick = hit_mon & _contact(st, ctx) & (ability(st, o) == AB["pickpocket"]) & (st.hp[o, tslot] > 0)
    pick &= (foe_item == 0) & takeable & ~pivoting
    # Magician: an itemless user takes the target's item after a damaging move
    takeable_o = ITEM_REMOVABLE_J[foe_item, st.species[o, tslot]] & ~sticky_hold(st, o, tslot, st.ability[s, slot])
    magic = ctx.landed & ~sheer & (ability(st, s) == AB["magician"]) & (user_item == 0) & (st.hp[s, slot] > 0)
    magic &= takeable_o & ~pivoting & ~pick
    items = st.item.at[o, tslot].set(jnp.where(pick, user_item, jnp.where(magic, 0, foe_item)).astype(st.item.dtype))
    items = items.at[s, slot].set(jnp.where(pick, 0, jnp.where(magic, foe_item, user_item)).astype(st.item.dtype))
    return st.replace(item=items)


@hooks.hook("after_status")
def _synchronize(st, s, slot, status, from_foe, key, cfg, applied):
    # Synchronize: remember the status to pass back; it is applied once the move is over (_synchronize_back),
    # so the step traces a single extra status attempt rather than one per status source
    sync = applied & from_foe & (st.ability[s, slot] == AB["synchronize"]) & (status != SLP) & (status != FRZ)
    return set_vol(st, s, "sync", status, sync)


@hooks.hook("after_move", priority=-50)
def _synchronize_back(st, ctx):
    for s in (0, 1):
        o = 1 - s
        status = vol(st, s, "sync")
        oslot = jnp.maximum(st.active[o], 0)
        key = jax.random.fold_in(ctx.key, 70 + s)
        st, _ = try_set_status(st, o, oslot, status, key, ctx.cfg, cond=status > 0, hook=False)
        st = set_vol(st, s, "sync", 0)
    return st


@hooks.hook("secondaries")
def _secondaries(st, s, mid, chance_):
    o = 1 - s
    chance_ = jnp.where(ability(st, s) == AB["serenegrace"], chance_ * 2, chance_)
    return chance_, eff_ability(st, o) == AB["shielddust"]


@hooks.hook("crit_stage")
def _shell_armor(st, ctx, stage):
    ab = eff_ability(st, ctx.o)
    return jnp.where((ab == AB["shellarmor"]) | (ab == AB["battlearmor"]), -10, stage)


@hooks.hook("legal_moves")
def _disable_mask(st, s, mask):
    return mask & (jnp.arange(4) != vol(st, s, "disabled") - 1)


@hooks.residual(17, "disable")
def _disable_countdown(st, key, cfg, cond):
    for s in (0, 1):
        turns = vol(st, s, "disable_turns") - 1
        on = cond & (vol(st, s, "disabled") > 0)
        st = set_vol(st, s, "disable_turns", turns, on)
        st = set_vol(st, s, "disabled", 0, on & (turns <= 0))
    return st
