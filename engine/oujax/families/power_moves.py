"""
engine/oujax/families/power_moves.py
Base-power effects that need battle state: Charge (the next Electric move has double power). The moves whose
power, type or category only depends on the two Pokemon and the field (Electro Ball, Gyro Ball, Rising Voltage,
Expanding Force, Psyblade, Barb Barrage, Infernal Parade, Revelation Dance, Raging Bull, Photon Geyser, Flying
Press) are in damage.py, where the L1 checks against @smogon/calc reach them.
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import move_type_and_bp
from engine.oujax.engine import MID, TY, _i, combatant, set_vol, vol

hooks.VOL.add("charge")

CHARGE = MID["charge"]


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
