"""
engine/oujax/families/power_moves.py
Base-power effects that need battle state: Charge (the next Electric move has double power) and Beat Up (one hit
per healthy team member, 5 + its base Attack / 10 power each; OU-JAX takes the members in team order where
Showdown uses its current, switch-reordered team list, which only matters if a berry triggers mid-move).

The moves whose power, type or category only depends on the two Pokemon and the field (Electro Ball, Gyro Ball,
Rising Voltage, Expanding Force, Psyblade, Barb Barrage, Infernal Parade, Revelation Dance, Raging Bull, Photon
Geyser, Flying Press) are in damage.py, where the L1 checks against @smogon/calc reach them.
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import SPECIES, move_type_and_bp
from engine.oujax.engine import MID, TY, _i, alive_mask, combatant, set_vol, vol
from engine.oujax.tables import SC

hooks.VOL.add("charge")

CHARGE = MID["charge"]
BEAT_UP = MID["beatup"]


def _beat_up_members(st, ctx):
    """(6,) team members Beat Up hits with: the user, and every other member not fainted and without status."""
    s = ctx.s
    slots = jnp.arange(6)
    return (slots == ctx.slot) | (alive_mask(st)[s] & (st.status[s] == 0))


@hooks.hook("hit_count")
def _beat_up_hits(st, ctx, n):
    return jnp.where(ctx.mid == BEAT_UP, jnp.sum(_beat_up_members(st, ctx)), n)


@hooks.hook("hit_power")
def _beat_up_power(st, ctx, hit):
    members = _beat_up_members(st, ctx)
    rank = jnp.cumsum(members) - 1
    who = jnp.argmax(members & (rank == hit))
    bp = 5 + SPECIES[st.species[ctx.s, who], SC["base_atk"]].astype(jnp.int32) // 10
    return jnp.where(ctx.mid == BEAT_UP, bp, 1).astype(jnp.int32), jnp.int32(4096)


def _move_type(st, ctx):
    mtype, _ = move_type_and_bp(ctx.mid, combatant(st, ctx.s), combatant(st, ctx.o), _i(st.weather))
    return mtype


@hooks.hook("hit_power")
def _charge_power(st, ctx, hit):
    # Charge's onBasePower (priority 9): Electric moves x2. Chained last: exact for a lone 2x, and Charge users
    # rarely stack other BasePower modifiers
    on = (vol(st, ctx.s, "charge") > 0) & (_move_type(st, ctx) == TY["electric"])
    return jnp.int32(1), jnp.where(on, 8192, 4096).astype(jnp.int32)


@hooks.hook("status_move")
def _charge(st, ctx):
    return set_vol(st, ctx.s, "charge", 1, ctx.landed & (ctx.mid == CHARGE))


@hooks.hook("after_move")
def _charge_used(st, ctx):
    # onAfterMove / onMoveAborted: any Electric move other than Charge uses it up, even one that misses or
    # never gets to run
    used = ctx.acting & (_move_type(st, ctx) == TY["electric"]) & (ctx.mid != CHARGE)
    return set_vol(st, ctx.s, "charge", 0, used)


hooks.showdown_volatile("charge")(lambda st, s: vol(st, s, "charge") > 0)
