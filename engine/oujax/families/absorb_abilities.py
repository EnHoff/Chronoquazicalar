"""
engine/oujax/families/absorb_abilities.py
Abilities that stop a move at TryHit (after Protect, before type immunity and accuracy), all breakable:
- Good as Gold (the foe's status moves), Bulletproof, Soundproof, Overcoat (powder moves);
- Water Absorb, Volt Absorb, Dry Skin, Earth Eater (heal 1/4), Flash Fire (1.5x Fire attacks until it leaves),
  Sap Sipper, Storm Drain, Motor Drive, Well-Baked Body, Wind Rider (stat boost);
- Magic Bounce: reflectable status moves (hazards too) are used back at the user. The bounced move re-checks
  accuracy, type immunity and the user's own TryHit abilities; simplification: its accuracy ignores items and
  No Guard, and a bounced Parting Shot doesn't switch the bouncer out;
- Oblivious: Taunt; Sticky Hold: Trick / Switcheroo;
- Prankster's status moves fail against Dark types (Showdown's TryImmunity step; here a TryHit block).
Also Dry Skin's and Rain Dish's weather effects at the end of the turn.
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import MOVES, has_type, move_type_and_bp, type_effectiveness
from engine.oujax.effects import T
from engine.oujax.engine import (
    ACC_NUM,
    RAIN,
    SUN,
    _i,
    act,
    apply_boosts,
    combatant,
    eat_sitrus,
    gain_hp,
    lose_hp,
    set_vol,
    types_now,
)
from engine.oujax.families.entry_abilities import AB, ability, eff_ability, is_ab, on_field
from engine.oujax.tables import FLAG_BIT, MC, TARGET_ID

TY = T.type_ids
MID_TAUNT = T.move_ids["taunt"]
MID_TRICK, MID_SWITCHEROO = T.move_ids["trick"], T.move_ids["switcheroo"]


def _absorb(st, ctx):
    """TryHit abilities of the target (ctx.o) against ctx.s's move. Returns (state, blocked)."""
    s, o, tslot = ctx.s, ctx.o, ctx.tslot
    ab = eff_ability(st, o)
    row = MOVES[ctx.mid]
    flags = row[MC["flags"]]
    reach = ctx.landed & ctx.targets_foe
    mtype, _ = move_type_and_bp(ctx.mid, combatant(st, s), combatant(st, o), _i(st.weather))
    water, elec, fire = mtype == TY["water"], mtype == TY["electric"], mtype == TY["fire"]
    grass, ground = mtype == TY["grass"], mtype == TY["ground"]
    wind = (flags & FLAG_BIT["wind"]) > 0
    heal = (water & is_ab(ab, "waterabsorb", "dryskin")) | (elec & (ab == AB["voltabsorb"]))
    heal |= ground & (ab == AB["eartheater"])
    st = gain_hp(st, o, tslot, _i(st.stats[o, tslot, 0]) // 4, reach & heal)
    flash = fire & (ab == AB["flashfire"])
    st = set_vol(st, o, "flashfire", 1, reach & flash)
    sap, motor = grass & (ab == AB["sapsipper"]), elec & (ab == AB["motordrive"])
    baked, rider = fire & (ab == AB["wellbakedbody"]), wind & (ab == AB["windrider"])
    drain = water & (ab == AB["stormdrain"])
    deltas = jnp.stack([sap | rider, 2 * baked, drain, 0, motor, 0, 0]).astype(jnp.int32)
    boost = sap | motor | baked | rider | drain
    st, _ = apply_boosts(st, o, deltas, reach & boost)
    blocked = heal | flash | boost
    blocked |= ctx.is_status & (ab == AB["goodasgold"])
    blocked |= ((flags & FLAG_BIT["bullet"]) > 0) & (ab == AB["bulletproof"])
    blocked |= ((flags & FLAG_BIT["sound"]) > 0) & (ab == AB["soundproof"])
    blocked |= ((flags & FLAG_BIT["powder"]) > 0) & (ab == AB["overcoat"])
    blocked |= (ctx.mid == MID_TAUNT) & (ab == AB["oblivious"])  # Oblivious: Taunt (and Attract) don't affect it
    # Trick / Switcheroo's onTryImmunity: a Sticky Hold target is immune (hasAbility: Mold Breaker doesn't help)
    blocked |= ((ctx.mid == MID_TRICK) | (ctx.mid == MID_SWITCHEROO)) & (ability(st, o) == AB["stickyhold"])
    return st, reach & blocked


@hooks.hook("try_hit")
def _try_hit(st, ctx):
    return _absorb(st, ctx)


@hooks.hook("try_hit", priority=-10)
def _prankster_vs_dark(st, ctx):
    s, o = ctx.s, ctx.o
    prankster = (ability(st, s) == AB["prankster"]) & ctx.is_status
    dark = has_type(types_now(st, o, ctx.tslot), TY["dark"])
    return st, ctx.landed & ctx.targets_foe & prankster & dark


@hooks.hook("reflect")
def _magic_bounce(st, ctx):
    s, o, slot, tslot = ctx.s, ctx.o, ctx.slot, ctx.tslot
    row = MOVES[ctx.mid]
    side_move = row[MC["target"]] == TARGET_ID["foeSide"]
    reaches = (ctx.targets_foe & ctx.landed) | (side_move & ctx.proceeds)
    reflectable = (row[MC["flags"]] & FLAG_BIT["reflectable"]) > 0
    bounce = reaches & ctx.is_status & reflectable & on_field(st, o) & (eff_ability(st, o) == AB["magicbounce"])
    # The bounced move against the original user: accuracy, type immunity, then its TryHit abilities
    acc = _i(row[MC["accuracy"]])
    stage = jnp.clip(_i(st.boosts[o, 5]) - _i(st.boosts[s, 6]), -6, 6)
    acc_mod = jnp.where(stage >= 0, acc * ACC_NUM[jnp.abs(stage)] // 3, acc * 3 // ACC_NUM[jnp.abs(stage)])
    if ctx.cfg.forced:
        hits = jnp.bool_(ctx.cfg.hit)
    else:
        hits = jax.random.randint(jax.random.fold_in(ctx.key, 41), (), 0, 100) < acc_mod
    hits = hits | (acc >= 101) | ~ctx.targets_foe
    user_types = types_now(st, s, slot)
    immune, _ = type_effectiveness(row[MC["type"]], user_types)
    immune &= row[MC["type_immunity"]] == 1
    immune |= ((row[MC["flags"]] & FLAG_BIT["powder"]) > 0) & has_type(user_types, TY["grass"])
    immune &= ctx.targets_foe
    back = ctx._replace(s=o, o=s, slot=tslot, tslot=slot, landed=bounce & hits & ~immune)
    st, absorbed = _absorb(st, back)
    return st, bounce, bounce & hits & ~immune & ~absorbed


@hooks.residual(1, "weather_abilities")
def _weather_abilities(st, key, cfg, cond):
    # Dry Skin: +1/8 in rain, -1/8 in sun; Rain Dish: +1/16 in rain (onWeather, with the weather's residual)
    for s in (0, 1):
        slot = jnp.maximum(act(st, s), 0)
        ab = ability(st, s)
        maxhp = _i(st.stats[s, slot, 0])
        on = cond & on_field(st, s)
        rain, sun = st.weather == RAIN, st.weather == SUN
        st = gain_hp(st, s, slot, maxhp // 8, on & rain & (ab == AB["dryskin"]))
        st = gain_hp(st, s, slot, maxhp // 16, on & rain & (ab == AB["raindish"]))
        st = lose_hp(st, s, slot, maxhp // 8, on & sun & (ab == AB["dryskin"]))
        st = eat_sitrus(st, s)
    return st
