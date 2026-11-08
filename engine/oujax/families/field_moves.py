"""
engine/oujax/families/field_moves.py
Groundedness and whole-field moves: Smack Down (grounds a Flying / Levitate / Magnet Rise target), Magnet Rise
(5 turns airborne), Gravity (5 turns: everyone grounded, accuracy x6840/4096, moves flagged gravity can't be
used) and Perish Song (every active Pokemon faints after 3 more turns unless it switches out).
"""

import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import has_type
from engine.oujax.effects import ABILITY
from engine.oujax.engine import MID, TY, _i, act, field_ex, set_field_ex, set_hp, set_vol, types_now, vol
from engine.oujax.families.charge_moves import SEMI_INVULNERABLE
from engine.oujax.families.volatiles import speed_order
from engine.oujax.tables import load_tables

T = load_tables()

for _name in ("smackdown", "magnetrise", "perishsong"):  # magnetrise / perishsong: turns left
    hooks.VOL.add(_name)
hooks.FIELD.add("gravity")  # turns left

GRAVITY_FLAG = (
    jnp.zeros(len(MID), bool)
    .at[jnp.array([MID[m] for m, d in T.dex["moves"].items() if "gravity" in d.get("flags", {})])]
    .set(True)
)


def gravity_on(st):
    return field_ex(st, "gravity") > 0


@hooks.hook("grounded")
def _grounded(st, s, slot):
    active = slot == act(st, s)
    down = gravity_on(st) | (active & (vol(st, s, "smackdown") > 0))
    up = active & (vol(st, s, "magnetrise") > 0)
    return jnp.where(down, 1, jnp.where(up, 2, 0))


# --- Smack Down -----------------------------------------------------------------------------------------


@hooks.hook("after_hit")
def _smack_down(st, ctx, dealt, hit_sub):
    o, tslot = ctx.o, ctx.tslot
    hit = ctx.landed & (ctx.mid == MID["smackdown"]) & ~hit_sub & (st.hp[o, tslot] > 0)
    # Only applies to a Pokemon that was up: Flying type, Levitate, Magnet Rise (Gravity already grounds it)
    flying = has_type(types_now(st, o, tslot), TY["flying"]) | (st.ability[o, tslot] == ABILITY.levitate)
    applies = hit & ((flying & ~gravity_on(st)) | (vol(st, o, "magnetrise") > 0))
    st = set_vol(st, o, "magnetrise", 0, applies)
    return set_vol(st, o, "smackdown", 1, applies)


# --- status moves -----------------------------------------------------------------------------------------


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, mid = ctx.s, ctx.mid
    rise = (mid == MID["magnetrise"]) & (
        (vol(st, s, "magnetrise") > 0) | (vol(st, s, "smackdown") > 0) | gravity_on(st)
    )
    grav = (mid == MID["gravity"]) & gravity_on(st)
    # Perish Song fails when every active Pokemon already has it
    perish = (mid == MID["perishsong"]) & (vol(st, 0, "perishsong") > 0) & (vol(st, 1, "perishsong") > 0)
    return rise | grav | perish


@hooks.hook("status_move")
def _status(st, ctx):
    s, mid, ok = ctx.s, ctx.mid, ctx.landed
    st = set_vol(st, s, "magnetrise", 5, ok & (mid == MID["magnetrise"]))
    grav = ok & (mid == MID["gravity"])
    st = set_field_ex(st, "gravity", 5, grav)
    for side in (0, 1):
        st = set_vol(st, side, "magnetrise", 0, grav)
    # Perish Song: every active Pokemon without it, except one hidden by Phantom Force (Invulnerability)
    song = ok & (mid == MID["perishsong"])
    for side in (0, 1):
        hidden = (side != s) & SEMI_INVULNERABLE[vol(st, side, "charge_mid")]
        alive = st.hp[side, act(st, side)] > 0
        st = set_vol(st, side, "perishsong", 4, song & ~hidden & alive & (vol(st, side, "perishsong") == 0))
    return st


@hooks.hook("accuracy_mod")
def _gravity_accuracy(st, ctx, mod):
    return jnp.where(gravity_on(st), (mod * 6840 + 2048) >> 12, mod)


@hooks.hook("legal_moves")
def _gravity_disable(st, s, mask):
    mids = _i(st.moves[s, jnp.maximum(act(st, s), 0)])
    return mask & ~(gravity_on(st) & GRAVITY_FLAG[mids])


@hooks.hook("before_move", priority=6)
def _gravity_block(st, ctx):
    return st, ctx.moves & gravity_on(st) & GRAVITY_FLAG[ctx.mid]


# --- end of turn -------------------------------------------------------------------------------------------


@hooks.residual(18, "magnetrise_end")
def _magnet_rise_end(st, key, cfg, cond):
    for s in (0, 1):
        left = vol(st, s, "magnetrise")
        st = set_vol(st, s, "magnetrise", left - 1, cond & (left > 0))
    return st


@hooks.residual(24, "perish_song")
def _perish_song(st, key, cfg, cond):
    # Faster Pokemon first, so on a double faint the slower side's Pokemon faints last (and its side wins)
    order = speed_order(st, key, cfg)
    for i in range(2):
        s = order[i]
        slot = act(st, s)
        left = vol(st, s, "perishsong")
        on = cond & (left > 0)
        st = set_vol(st, s, "perishsong", left - 1, on)
        st = set_hp(st, s, slot, jnp.where(on & (left == 1), 0, st.hp[s, slot]))
    return st


@hooks.residual(27.2, "gravity_end")
def _gravity_end(st, key, cfg, cond):
    left = field_ex(st, "gravity")
    return set_field_ex(st, "gravity", left - 1, cond & (left > 0))


for _v in ("smackdown", "magnetrise", "perishsong"):
    hooks.showdown_volatile(_v)((lambda name: lambda st, s: vol(st, s, name) > 0)(_v))
