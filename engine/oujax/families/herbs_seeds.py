"""
engine/oujax/families/herbs_seeds.py
Herbs and terrain seeds:
- White Herb: resets the holder's negative stat stages once a move has finished (any Pokemon's), on any switch-in
  (including its own) and at residual 29;
- Mirror Herb: adds up the positive stat changes the foe gets and copies them at the same points (not changes
  that come from a Mirror Herb itself);
- Grassy / Electric / Psychic / Misty Seed: +1 Def / Def / SpD / SpD when the holder comes in on its terrain or
  the terrain starts while it is in;
- Power Herb: skips a two-turn move's charge turn (the charge_move hook, for the charge-move family to call);
- Mental Herb: on Update, clears Attract / Taunt / Encore / Torment / Disable / Heal Block, using whichever of
  those VOL slots the volatile families claim (each is assumed to read 0 when the condition is off).
"""

import jax.numpy as jnp
import numpy as np

from engine.oujax import hooks
from engine.oujax.effects import T
from engine.oujax.engine import E_TERRAIN, G_TERRAIN, M_TERRAIN, P_TERRAIN, act, apply_boosts, foe, set_vol, vol
from engine.oujax.families.held_items import take_item

_ids = T.item_ids
WHITE_HERB, MIRROR_HERB, POWER_HERB, MENTAL_HERB = (
    _ids[n] for n in ("whiteherb", "mirrorherb", "powerherb", "mentalherb")
)
_terrain = np.zeros(len(_ids), np.int32)
_boost = np.zeros((len(_ids), 7), np.int32)
for _name, _t, _stat in (
    ("grassyseed", G_TERRAIN, 1),
    ("electricseed", E_TERRAIN, 1),
    ("psychicseed", P_TERRAIN, 3),
    ("mistyseed", M_TERRAIN, 3),
):
    _terrain[_ids[_name]] = _t
    _boost[_ids[_name], _stat] = 1
SEED_TERRAIN, SEED_BOOST = jnp.asarray(_terrain), jnp.asarray(_boost)
MENTAL_CONDITIONS = ("attract", "taunt", "encore", "torment", "disable", "healblock")

# Mirror Herb's running total of the foe's stat raises, 3 bits per stat (atk..spe, then accuracy / evasion)
hooks.VOL.add("mirror_a")
hooks.VOL.add("mirror_b")
_copying = [False]  # trace-time flag: the boost being applied comes from a Mirror Herb


def _holder(st, s, item):
    slot = jnp.maximum(act(st, s), 0)
    return slot, (act(st, s) >= 0) & (st.item[s, slot] == item) & (st.hp[s, slot] > 0)


def _mirror_total(st, s):
    a, b = vol(st, s, "mirror_a"), vol(st, s, "mirror_b")
    return jnp.stack([(a >> (3 * i)) & 7 for i in range(5)] + [(b >> (3 * i)) & 7 for i in range(2)])


@hooks.hook("after_boost")
def _mirror_herb_watch(st, side, applied, cond):
    if _copying[0]:
        return st
    h = foe(side)
    _, holds = _holder(st, h, MIRROR_HERB)
    total = jnp.minimum(_mirror_total(st, h) + jnp.maximum(applied, 0), 7)
    a = sum(total[i] << (3 * i) for i in range(5))
    b = total[5] | (total[6] << 3)
    gains = cond & holds & jnp.any(applied > 0)
    st = set_vol(st, h, "mirror_a", a, gains)
    return set_vol(st, h, "mirror_b", b, gains)


def _use_herbs(st, cond):
    for s in (0, 1):
        # White Herb
        slot, holds = _holder(st, s, WHITE_HERB)
        neg = st.boosts[s] < 0
        white = cond & holds & jnp.any(neg)
        st = st.replace(boosts=st.boosts.at[s].set(jnp.where(white & neg, 0, st.boosts[s])))
        st = take_item(st, s, slot, white)
        # Mirror Herb
        slot, holds = _holder(st, s, MIRROR_HERB)
        total = _mirror_total(st, s)
        mirror = cond & holds & jnp.any(total > 0)
        st = take_item(st, s, slot, mirror)
        _copying[0] = True
        try:
            st, _ = apply_boosts(st, s, total, mirror)
        finally:
            _copying[0] = False
        st = set_vol(st, s, "mirror_a", 0, mirror)
        st = set_vol(st, s, "mirror_b", 0, mirror)
    return st


@hooks.hook("after_move", priority=-5)
def _herbs_after_move(st, ctx):
    return _use_herbs(st, ctx.moves)


@hooks.hook("switch_in", priority=-2)
def _herbs_switch_in(st, side, key, cfg, cond):
    return _use_herbs(st, cond)


@hooks.residual(29, "herbs")
def _herbs_residual(st, key, cfg, cond):
    return _use_herbs(st, cond)


def _seed(st, s, cond):
    slot = jnp.maximum(act(st, s), 0)
    item = st.item[s, slot]
    t = SEED_TERRAIN[item]
    use = cond & (act(st, s) >= 0) & (st.hp[s, slot] > 0) & (t > 0) & (st.terrain == t)
    st = take_item(st, s, slot, use)
    st, _ = apply_boosts(st, s, SEED_BOOST[item], use)
    return st


@hooks.hook("switch_in", priority=-1)
def _seed_switch_in(st, side, key, cfg, cond):
    return _seed(st, side, cond)


@hooks.hook("terrain_change")
def _seed_terrain_change(st, cond):
    return _seed(_seed(st, 0, cond), 1, cond)


@hooks.hook("charge_move")
def _power_herb(st, ctx):
    slot, holds = _holder(st, ctx.s, POWER_HERB)
    use = ctx.moves & holds
    return take_item(st, ctx.s, slot, use), use


@hooks.hook("item_update")
def _mental_herb(st, s):
    names = [n for n in MENTAL_CONDITIONS if n in hooks.VOL.index]
    if not names:
        return st
    slot, holds = _holder(st, s, MENTAL_HERB)
    use = holds & jnp.any(jnp.stack([vol(st, s, n) != 0 for n in names]))
    for n in names:
        st = set_vol(st, s, n, 0, use)
    return take_item(st, s, slot, use)
