"""
engine/oujax/families/move_misc.py
Moves whose success depends on the turn: Sucker Punch / Thunderclap (the target must be about to attack), Upper
Hand (... with a priority move), Fake Out / First Impression (first turn out), Focus Punch (fails if hit first),
Beak Blast (burns contact attackers while it charges); Ice Spinner (clears terrain), Stone Axe (Stealth Rock),
Sparkling Aria (cures the target's burn), Fell Stinger (+3 Atk on a KO), Double Shock (the user loses its
Electric type), Hyperspace Fury (Hoopa-Unbound only, breaks Protect); the Protect variants Spiky Shield,
Baneful Bunker and Burning Bulwark (contact punishment) and Endure.
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import HAS_SECONDARIES, MOVES, has_type
from engine.oujax.effects import ABILITY, ITEM
from engine.oujax.engine import (
    BRN,
    MID,
    PSN,
    TY,
    _i,
    act,
    apply_boosts,
    eat_sitrus,
    lose_hp,
    refresh_paradox,
    set_vol,
    try_set_status,
    types_now,
    vol,
)
from engine.oujax.families.volatiles import has_flag, move_table
from engine.oujax.state import HAZ_ROCKS
from engine.oujax.tables import MC, TYPE_NONE, load_tables

hooks.VOL.add("protect_kind")  # move id of this turn's successful Protect-family move
hooks.VOL.add("endure")  # Endure is up this turn
hooks.VOL.add("lost_electric")  # Double Shock took the Electric type away

T = load_tables()
HOOPA_UNBOUND = T.species_ids["hoopaunbound"]
BREAKS_PROTECT = move_table("hyperspacefury")

TARGET_MUST_ATTACK = move_table("suckerpunch thunderclap")
FIRST_TURN_ONLY = move_table("fakeout firstimpression")


def makes_contact(st, s, mid):
    """checkMoveMakesContact for side s's active Pokemon (Punching Glove turns punches into non-contact)."""
    glove = has_flag(mid, "punch") & (st.item[s, act(st, s)] == ITEM.punchingglove)
    return has_flag(mid, "contact") & ~glove


@hooks.hook("move_fails")
def _fails(st, ctx):
    o, mid, pend = ctx.o, ctx.mid, ctx.foe_pending_move
    pend_status = MOVES[pend, MC["category"]] == 2
    # Sucker Punch / Thunderclap: the target must have chosen an attack it hasn't used yet (not recharging)
    recharging = vol(st, o, "recharge") == 1
    fails = TARGET_MUST_ATTACK[mid] & ((pend == 0) | pend_status | recharging)
    # Upper Hand: ... an attack with raised priority (after ModifyPriority)
    prio = _i(MOVES[pend, MC["priority"]])
    for h in hooks.handlers("priority"):
        prio = h(st, o, pend, prio)
    fails |= (mid == MID["upperhand"]) & ((pend == 0) | pend_status | (prio <= 0))
    # Fake Out / First Impression only work on the user's first move action since switching in
    fails |= FIRST_TURN_ONLY[mid] & (vol(st, ctx.s, "move_actions") >= 1)
    # Hyperspace Fury: only Hoopa-Unbound can use it
    fails |= (mid == MID["hyperspacefury"]) & (st.species[ctx.s, ctx.slot] != HOOPA_UNBOUND)
    return fails


def sheer_force(st, ctx):
    return (st.ability[ctx.s, ctx.slot] == ABILITY.sheerforce) & HAS_SECONDARIES[ctx.mid]


@hooks.hook("try_move")
def _double_shock_try(st, ctx):
    # Double Shock's onTryMove: only an Electric type can use it (returns null: not a failure)
    stop = ctx.proceeds & (ctx.mid == MID["doubleshock"]) & ~has_type(types_now(st, ctx.s, ctx.slot), TY["electric"])
    return st, stop


@hooks.hook("types")
def _lost_electric(st, s, slot, types):
    return jnp.where((vol(st, s, "lost_electric") > 0) & (types == TY["electric"]), TYPE_NONE, types)


@hooks.hook("move_try_hit")
def _break_protect(st, ctx):
    brk = ctx.landed & BREAKS_PROTECT[ctx.mid]
    return st.replace(protected=st.protected.at[ctx.o].set(st.protected[ctx.o] & ~brk))


@hooks.hook("before_move", priority=0)
def _focus_punch(st, ctx):
    # beforeMoveCallback (after the BeforeMove event, before PP): it lost its focus if a damaging move hit it
    # this turn (the 'focuspunch' volatile's onHit; a Substitute hit doesn't count)
    s = ctx.s
    hurt = (vol(st, s, "dmg_phys") > 0) | (vol(st, s, "dmg_spec") > 0)
    return st, ctx.moves & (ctx.mid == MID["focuspunch"]) & hurt


@hooks.hook("damaging_hit")
def _beak_blast(st, ctx, hit, dmg):
    # A Pokemon about to use Beak Blast this turn burns attackers that make contact with it
    s, slot = ctx.s, ctx.slot
    charging = ctx.foe_pending_move == MID["beakblast"]
    burn = ctx.landed & charging & makes_contact(st, s, ctx.mid) & (st.hp[s, slot] > 0)
    st, _ = try_set_status(st, s, slot, BRN, jax.random.fold_in(ctx.key, 402 + hit), ctx.cfg, cond=burn)
    return st


@hooks.hook("after_hit")
def _ice_spinner(st, ctx, dealt, hit_sub):
    # Ice Spinner clears the terrain after hitting (also a Substitute), if the user is still standing
    clear = ctx.landed & (ctx.mid == MID["icespinner"]) & (dealt > 0) & (st.hp[ctx.s, ctx.slot] > 0)
    st = st.replace(
        terrain=jnp.where(clear, 0, st.terrain).astype(st.terrain.dtype),
        terrain_turns=jnp.where(clear, 0, st.terrain_turns).astype(st.terrain_turns.dtype),
    )
    return refresh_paradox(st)


@hooks.hook("after_hit")
def _after_hit(st, ctx, dealt, hit_sub):
    s, o, slot, tslot, mid, ok = ctx.s, ctx.o, ctx.slot, ctx.tslot, ctx.mid, ctx.landed
    user_up = st.hp[s, slot] > 0
    # Stone Axe: Stealth Rock on the target's side (also after hitting a Substitute), not with Sheer Force
    axe = ok & (mid == MID["stoneaxe"]) & user_up & ~sheer_force(st, ctx)
    st = st.replace(hazards=st.hazards.at[o, HAZ_ROCKS].set(jnp.where(axe, 1, st.hazards[o, HAZ_ROCKS])))
    # Fell Stinger: +3 Atk when the target faints from it
    sting = ok & (mid == MID["fellstinger"]) & user_up & (st.hp[o, tslot] == 0)
    st, _ = apply_boosts(st, s, jnp.array([3, 0, 0, 0, 0, 0, 0]), sting)
    # Double Shock's self effect: the user loses Electric (setType fails once Terastallized)
    shock = ok & (mid == MID["doubleshock"]) & ~st.terastallized[s, slot]
    return set_vol(st, s, "lost_electric", 1, shock)


@hooks.hook("after_move")
def _sparkling_aria(st, ctx):
    # Sparkling Aria's onAfterMove: its secondary volatile cures the target's burn (not with Sheer Force)
    o, tslot = ctx.o, ctx.tslot
    cure = (
        ctx.landed
        & (ctx.mid == MID["sparklingaria"])
        & (st.hp[ctx.s, ctx.slot] > 0)
        & (st.hp[o, tslot] > 0)
        & (st.status[o, tslot] == BRN)
        & ~sheer_force(st, ctx)
    )
    return st.replace(status=st.status.at[o, tslot].set(jnp.where(cure, 0, st.status[o, tslot])))


# --- Protect variants ---------------------------------------------------------------------------------


@hooks.hook("status_move")
def _protect_kind(st, ctx):
    s, mid, ok = ctx.s, ctx.mid, ctx.landed
    kinds = (mid == MID["spikyshield"]) | (mid == MID["banefulbunker"]) | (mid == MID["burningbulwark"])
    st = set_vol(st, s, "protect_kind", mid, ok & kinds)
    return set_vol(st, s, "endure", 1, ok & (mid == MID["endure"]))


@hooks.hook("try_hit")
def _protect_contact(st, ctx):
    # A contact move stopped by Spiky Shield costs the attacker 1/8 of its max HP; Baneful Bunker poisons it
    s, o, slot, mid = ctx.s, ctx.o, ctx.slot, ctx.mid
    stopped = ctx.proceeds & ctx.targets_foe & st.protected[o] & has_flag(mid, "protect")
    hit_back = stopped & makes_contact(st, s, mid)
    kind = vol(st, o, "protect_kind")
    st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 8, hit_back & (kind == MID["spikyshield"]))
    st = eat_sitrus(st, s)
    bunker = hit_back & (kind == MID["banefulbunker"])
    st, _ = try_set_status(st, s, slot, PSN, jax.random.fold_in(ctx.key, 401), ctx.cfg, cond=bunker)
    bulwark = hit_back & (kind == MID["burningbulwark"])
    st, _ = try_set_status(st, s, slot, BRN, jax.random.fold_in(ctx.key, 403), ctx.cfg, cond=bulwark)
    return st, jnp.bool_(False)


@hooks.hook("damage_taken")
def _endure(st, ctx, side, dmg):
    hp = _i(st.hp[side, act(st, side)])
    holds = ctx.landed & (vol(st, side, "endure") > 0) & (dmg >= hp)
    return st, jnp.where(holds, hp - 1, dmg)


@hooks.residual(50, "endure_end")
def _endure_end(st, key, cfg, cond):
    for s in (0, 1):
        st = set_vol(st, s, "endure", 0, cond)
    return st


hooks.showdown_volatile("endure")(lambda st, s: vol(st, s, "endure") > 0)
for _name in ("spikyshield", "banefulbunker", "burningbulwark"):
    hooks.showdown_volatile(_name)(
        (lambda m: lambda st, s: st.protected[s] & (vol(st, s, "protect_kind") == m))(MID[_name])
    )
