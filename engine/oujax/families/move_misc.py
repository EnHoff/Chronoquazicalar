"""
engine/oujax/families/move_misc.py
Moves whose success depends on the turn: Sucker Punch / Thunderclap (the target must be about to attack), Upper
Hand (... with a priority move), Fake Out / First Impression (first turn out); Ice Spinner (clears terrain); the
Protect variants Spiky Shield and Baneful Bunker (contact punishment) and Endure.
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import MOVES
from engine.oujax.effects import ITEM
from engine.oujax.engine import MID, PSN, _i, act, eat_sitrus, lose_hp, refresh_paradox, set_vol, try_set_status, vol
from engine.oujax.families.volatiles import has_flag, move_table
from engine.oujax.tables import MC

hooks.VOL.add("protect_kind")  # move id of this turn's successful Protect-family move
hooks.VOL.add("endure")  # Endure is up this turn

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
    return fails


@hooks.hook("after_hit")
def _ice_spinner(st, ctx, dealt, hit_sub):
    # Ice Spinner clears the terrain after hitting (also a Substitute), if the user is still standing
    clear = ctx.landed & (ctx.mid == MID["icespinner"]) & (dealt > 0) & (st.hp[ctx.s, ctx.slot] > 0)
    st = st.replace(
        terrain=jnp.where(clear, 0, st.terrain).astype(st.terrain.dtype),
        terrain_turns=jnp.where(clear, 0, st.terrain_turns).astype(st.terrain_turns.dtype),
    )
    return refresh_paradox(st)


# --- Protect variants ---------------------------------------------------------------------------------


@hooks.hook("status_move")
def _protect_kind(st, ctx):
    s, mid, ok = ctx.s, ctx.mid, ctx.landed
    st = set_vol(st, s, "protect_kind", mid, ok & ((mid == MID["spikyshield"]) | (mid == MID["banefulbunker"])))
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
for _name in ("spikyshield", "banefulbunker"):
    hooks.showdown_volatile(_name)(
        (lambda m: lambda st, s: st.protected[s] & (vol(st, s, "protect_kind") == m))(MID[_name])
    )
