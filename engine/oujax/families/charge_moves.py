"""
engine/oujax/families/charge_moves.py
Moves that take over the next turn: Outrage-style rampages (lockedmove: 2-3 turns, then confusion), two-turn
moves (Meteor Beam, Solar Beam / Solar Blade, which skip the charge in sun, and Phantom Force, semi-invulnerable
while charging), Power Herb, and recharge moves (Hyper Beam).

While locked, the Pokemon has a single legal move action (0) and can't switch or Terastallize (Showdown sends a
request with only the locked move, or "Recharge"); the engine's locked_move hook redirects that action.
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import MOVES, has_type
from engine.oujax.effects import ABILITY, ITEM
from engine.oujax.engine import MID, RAIN, SAND, SLP, SNOW, SUN, TY, _i, act, apply_boosts, set_vol, types_now, vol
from engine.oujax.families.volatiles import add_confusion, has_flag, move_table
from engine.oujax.tables import MC, VOLATILES

for _name in (
    "lock_slot",  # 1 + the slot of the rampaging move (lockedmove), 0 if none
    "lock_dur",  # lockedmove's duration
    "lock_true",  # ... and its trueDuration (2-3 turns)
    "charge_mid",  # the two-turn move's own volatile: the move id charged, 0 if none
    "twoturn",  # twoturnmove's duration (the Pokemon is locked into charge_mid while > 0)
    "charge_dur",  # duration of the charged move's volatile when it has one (Phantom Force), 0 otherwise
    "recharge",  # 1: must recharge next turn; 2: recharged this turn
):
    hooks.VOL.add(_name)

RAMPAGE = MOVES[:, MC["self_volatile"]] == VOLATILES.index("lockedmove")
CHARGE = move_table("meteorbeam solarbeam solarblade phantomforce")
SOLAR = move_table("solarbeam solarblade")
SEMI_INVULNERABLE = move_table("phantomforce")
PHANTOM_FORCE = MID["phantomforce"]


def _slot_of(st, s, mid):
    mids = _i(st.moves[s, jnp.maximum(act(st, s), 0)])
    hit = (mids == mid) & (mid > 0)
    return jnp.where(jnp.any(hit), jnp.argmax(hit), -1)


@hooks.hook("locked_move")
def _locked(st, s):
    rampage = vol(st, s, "lock_slot") - 1
    charging = jnp.where(vol(st, s, "twoturn") > 0, _slot_of(st, s, vol(st, s, "charge_mid")), -1)
    recharge = jnp.where(vol(st, s, "recharge") == 1, 0, -1)
    return jnp.maximum(jnp.maximum(rampage, charging), recharge)


@hooks.hook("priority")
def _recharge_priority(st, side, mid, prio):
    return jnp.where(vol(st, side, "recharge") == 1, 0, prio)


@hooks.hook("before_move", priority=11)
def _recharge(st, ctx):
    # mustrecharge (BeforeMove priority 11): the turn is spent recharging (not a failure)
    on = ctx.acting & (vol(st, ctx.s, "recharge") == 1)
    return set_vol(st, ctx.s, "recharge", 2, on), on


@hooks.hook("try_move")
def _charge(st, ctx):
    s, slot, mid = ctx.s, ctx.slot, ctx.mid
    is_charge = ctx.proceeds & CHARGE[mid]
    release = is_charge & (vol(st, s, "charge_mid") == mid)  # attack turn: removeVolatile(move.id)
    st = set_vol(st, s, "charge_mid", 0, release)
    st = set_vol(st, s, "charge_dur", 0, release)
    start = is_charge & ~release
    st, _ = apply_boosts(st, s, jnp.array([0, 0, 1, 0, 0, 0, 0]), start & (mid == MID["meteorbeam"]))
    sunny = SOLAR[mid] & (st.weather == SUN)
    herb = start & ~sunny & (st.item[s, slot] == ITEM.powerherb)
    st = st.replace(item=st.item.at[s, slot].set(jnp.where(herb, 0, st.item[s, slot])))
    stop = start & ~sunny & ~herb
    st = set_vol(st, s, "charge_mid", mid, stop)
    st = set_vol(st, s, "twoturn", 2, stop)
    st = set_vol(st, s, "charge_dur", jnp.where(SEMI_INVULNERABLE[mid], 2, 0), stop)
    return st, stop


@hooks.hook("try_hit")
def _semi_invulnerable(st, ctx):
    # Phantom Force's charge turn: moves aimed at the Pokemon miss (No Guard hits; a Poison type's Toxic hits)
    s, o = ctx.s, ctx.o
    hidden = SEMI_INVULNERABLE[vol(st, o, "charge_mid")]
    no_guard = (st.ability[s, ctx.slot] == ABILITY.noguard) | (st.ability[o, ctx.tslot] == ABILITY.noguard)
    toxic = (ctx.mid == MID["toxic"]) & has_type(types_now(st, s, ctx.slot), TY["poison"])
    return st, ctx.landed & ctx.targets_foe & hidden & ~no_guard & ~toxic


@hooks.hook("move_try_hit")
def _break_protect(st, ctx):
    # Phantom Force lifts the target's Protect
    brk = ctx.landed & (ctx.mid == PHANTOM_FORCE)
    return st.replace(protected=st.protected.at[ctx.o].set(st.protected[ctx.o] & ~brk))


@hooks.hook("hit_power")
def _solar_power(st, ctx, hit):
    weak = SOLAR[ctx.mid] & ((st.weather == RAIN) | (st.weather == SAND) | (st.weather == SNOW))
    return jnp.int32(1), jnp.where(weak, 2048, 4096).astype(jnp.int32)


@hooks.hook("after_hit")
def _self_volatiles(st, ctx, dealt, hit_sub):
    s, mid, ok = ctx.s, ctx.mid, ctx.landed
    # Rampage (lockedmove): starts with a 2-3 turn trueDuration; hitting again on a later turn restarts the
    # 2-turn duration while at least 2 turns remain
    ramp = ok & RAMPAGE[mid]
    first = ramp & (vol(st, s, "lock_slot") == 0)
    true_dur = jnp.int32(2) if ctx.cfg.forced else jax.random.randint(jax.random.fold_in(ctx.key, 501), (), 2, 4)
    st = set_vol(st, s, "lock_slot", ctx.move_slot + 1, first)
    st = set_vol(st, s, "lock_true", true_dur, first)
    st = set_vol(st, s, "lock_dur", 2, first | (ramp & (vol(st, s, "lock_true") >= 2)))
    # Recharge moves: the next turn is spent recharging
    return set_vol(st, s, "recharge", 1, ok & has_flag(mid, "recharge"))


def _end_rampage(st, s, cond, key, cfg):
    """lockedmove ends: confusion from fatigue once its trueDuration has run out."""
    st, _ = add_confusion(st, s, cond & (vol(st, s, "lock_true") <= 1), key, cfg)
    for name in ("lock_slot", "lock_dur", "lock_true"):
        st = set_vol(st, s, name, 0, cond)
    return st


@hooks.hook("after_move")
def _after_move(st, ctx):
    s = ctx.s
    # lockedmove's AfterMove: the last turn of the rampage
    last = ctx.moves & (vol(st, s, "lock_slot") > 0) & (vol(st, s, "lock_dur") == 1)
    st = _end_rampage(st, s, last, jax.random.fold_in(ctx.key, 502), ctx.cfg)
    # twoturnmove's MoveAborted: not getting to move drops the charge lock (the move's own volatile stays)
    st = set_vol(st, s, "twoturn", 0, ctx.acting & ~ctx.moves)
    # A recharge turn counts as neither success nor failure
    recharged = vol(st, s, "recharge") == 2
    st = set_vol(st, s, "move_result", 0, recharged)
    return set_vol(st, s, "recharge", 0, recharged)


@hooks.residual(60, "locked_moves")
def _residual(st, key, cfg, cond):
    for s in (0, 1):
        # lockedmove: duration runs out (end), else falling asleep calms the rampage, and trueDuration ticks
        on = cond & (vol(st, s, "lock_slot") > 0)
        dur = vol(st, s, "lock_dur") - 1
        st = set_vol(st, s, "lock_dur", dur, on)
        ends = on & (dur <= 0)
        st = _end_rampage(st, s, ends, jax.random.fold_in(key, s), cfg)
        calm = on & ~ends & (st.status[s, act(st, s)] == SLP)
        for name in ("lock_slot", "lock_dur", "lock_true"):
            st = set_vol(st, s, name, 0, calm)
        st = set_vol(st, s, "lock_true", vol(st, s, "lock_true") - 1, on & ~ends & ~calm)
        # twoturnmove expires (taking the move's volatile with it); Phantom Force's own volatile expires too
        tt = vol(st, s, "twoturn")
        st = set_vol(st, s, "twoturn", tt - 1, cond & (tt > 0))
        st = set_vol(st, s, "charge_mid", 0, cond & (tt == 1))
        cd = vol(st, s, "charge_dur")
        st = set_vol(st, s, "charge_dur", cd - 1, cond & (cd > 0))
        st = set_vol(st, s, "charge_mid", 0, cond & (cd == 1))
    return st


hooks.showdown_volatile("lockedmove")(lambda st, s: vol(st, s, "lock_slot") > 0)
hooks.showdown_volatile("twoturnmove")(lambda st, s: vol(st, s, "twoturn") > 0)
hooks.showdown_volatile("mustrecharge")(lambda st, s: vol(st, s, "recharge") == 1)
for _name in ("meteorbeam", "solarbeam", "solarblade", "phantomforce"):
    hooks.showdown_volatile(_name)((lambda m: lambda st, s: vol(st, s, "charge_mid") == m)(MID[_name]))
