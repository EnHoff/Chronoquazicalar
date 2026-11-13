"""
engine/oujax/families/volatiles.py
Volatile conditions on the active Pokemon and the moves that set them: confusion (Hurricane, Axe Kick, Strange
Steam, Water Pulse, Psybeam, Dynamic Punch, Swagger, Poison Puppeteer), Taunt, Encore, Heal Block (Psychic Noise),
Throat Chop, Curse, Salt Cure, Leech Seed and Destiny Bond.

Also per-Pokemon records other families read: "last_move" (Pokemon.lastMove), "move_actions"
(activeMoveActions: moves attempted since switching in) and "active_turns" (activeTurns: turns started on the
field), plus `speed_order` for residual effects that run faster Pokemon first.
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.damage import MOVES, boosted_stat, has_type
from engine.oujax.engine import (
    M_TERRAIN,
    MID,
    PSN,
    RAIN,
    SUN,
    TOX,
    TY,
    _i,
    act,
    action_speed,
    apply_boosts,
    chance,
    eat_sitrus,
    foe,
    gain_hp,
    grounded,
    lose_hp,
    p1_first,
    set_vol,
    types_now,
    vol,
)
from engine.oujax.state import PHASE_MOVE, PHASE_SWITCH, RESUME_NONE
from engine.oujax.tables import FLAG_BIT, MC, VOLATILES, load_tables

T = load_tables()
N_MOVES = len(MID)

for _name in (
    "confusion",  # turns left, counted down before each move (Showdown's time)
    "last_move",  # move id of the last move used, 0 if none
    "move_actions",  # moves attempted since switching in
    "active_turns",  # turns started on the field
    "taunt", "encore", "encore_slot", "healblock", "throatchop",  # turns left (encore_slot: the locked slot)
    "curse", "saltcure", "leechseed", "destinybond",
    "puppet_prev",  # 1 + the target's status when this Pokemon started its move (Poison Puppeteer)
):  # fmt: skip
    hooks.VOL.add(_name)


def move_table(names) -> jnp.ndarray:
    """(n_moves,) bool lookup for a set of move ids."""
    out = jnp.zeros(N_MOVES, bool)
    return out.at[jnp.array([MID[n] for n in names.split()])].set(True)


def has_flag(mid, flag: str):
    return (MOVES[mid, MC["flags"]] & FLAG_BIT[flag]) > 0


FAIL_ENCORE = (
    jnp.zeros(N_MOVES, bool)
    .at[jnp.array([MID[m] for m, d in T.dex["moves"].items() if "failencore" in d.get("flags", {})])]
    .set(True)
)
SEC_CONFUSION = MOVES[:, MC["sec_volatile"]] == VOLATILES.index("confusion")
SEC_HEALBLOCK = MOVES[:, MC["sec_volatile"]] == VOLATILES.index("healblock")
SEC_SALTCURE = MOVES[:, MC["sec_volatile"]] == VOLATILES.index("saltcure")
PECHARUNT = T.species_ids["pecharunt"]
POISON_PUPPETEER = T.ability_ids["poisonpuppeteer"]
SHEER_FORCE = T.ability_ids["sheerforce"]


def speed_order(st, key, cfg):
    """[first, second] side for an end-of-turn effect: faster first, speed ties random (p1 when forced)."""
    spe = jnp.stack([action_speed(st, 0), action_speed(st, 1)])
    first0 = p1_first(key, jnp.zeros(2, jnp.int32), spe, cfg)
    return jnp.where(first0, jnp.array([0, 1]), jnp.array([1, 0]))


def secondary_lands(st, ctx, hit_sub, chance_pct, key):
    """A secondary effect of the move in ctx with this chance lands on the target (after_hit hooks): the move
    connected with the Pokemon itself, the target is still standing, no Sheer Force; one independent roll."""
    s, o = ctx.s, ctx.o
    if ctx.cfg.forced:
        roll_ok = jnp.bool_(ctx.cfg.secondaries) | (chance_pct >= 100)
    else:
        roll_ok = jax.random.randint(key, (), 0, 100) < chance_pct
    sheer = (st.ability[s, ctx.slot] == SHEER_FORCE) & (MOVES[ctx.mid, MC["sec_chance"]] > 0)
    return ctx.landed & ~hit_sub & (st.hp[o, ctx.tslot] > 0) & ~sheer & roll_ok


# --- confusion ---------------------------------------------------------------------------------------


def add_confusion(st, s, cond, key, cfg, min_turns=2):
    """addVolatile('confusion') on side s's active Pokemon: 2-5 turns (3-5 from Axe Kick), blocked by Misty
    Terrain for a grounded Pokemon. Returns (state, applied)."""
    slot = act(st, s)
    misty = (st.terrain == M_TERRAIN) & grounded(st, s, slot)
    ok = cond & (vol(st, s, "confusion") == 0) & (st.hp[s, slot] > 0) & ~misty
    for h in hooks.handlers("volatile_immune"):
        ok &= ~h(st, s, "confusion")  # Own Tempo
    turns = jnp.int32(min_turns) if cfg.forced else jax.random.randint(key, (), min_turns, 6)
    return set_vol(st, s, "confusion", turns, ok), ok


def confusion_damage(st, s, slot, roll):
    """BattleActions.getConfusionDamage: a typeless 40 BP physical hit on itself (no crit, STAB or burn)."""
    atk = boosted_stat(_i(st.stats[s, slot, 1]), _i(st.boosts[s, 0]))
    dfn = boosted_stat(_i(st.stats[s, slot, 2]), _i(st.boosts[s, 1]))
    level = _i(st.level[s, slot])
    base = ((2 * level // 5 + 2) * 40 * atk // jnp.maximum(dfn, 1)) // 50 + 2
    base = base & 0xFFFF
    return jnp.maximum(base * (100 - roll) // 100, 1)


@hooks.hook("before_move", priority=3)
def _confusion(st, ctx):
    s, slot, cfg = ctx.s, ctx.slot, ctx.cfg
    left = vol(st, s, "confusion")
    ticks = ctx.moves & (left > 0)
    st = set_vol(st, s, "confusion", left - 1, ticks)
    still = ticks & (left > 1)
    k_hit, k_roll = jax.random.split(jax.random.fold_in(ctx.key, 301))
    hit = still & chance(k_hit, 33, 100, cfg.confusion_hit, cfg)
    roll = jnp.int32(cfg.roll) if cfg.forced else jax.random.randint(k_roll, (), 0, 16)
    dmg = confusion_damage(st, s, slot, roll)
    # The self-hit is move damage to itself (Showdown's 'confused' active move): Endure, Focus Sash, ...
    self_ctx = ctx._replace(o=s, tslot=slot, landed=hit)
    for h in hooks.handlers("damage_taken"):
        st, dmg = h(st, self_ctx, s, dmg)
    st = lose_hp(st, s, slot, dmg, hit)
    return eat_sitrus(st, s), hit


@hooks.hook("accuracy")
def _hurricane(st, ctx, acc):
    # Hurricane never misses in rain and has 50% accuracy in sun
    hurricane = ctx.mid == MID["hurricane"]
    acc = jnp.where(hurricane & (st.weather == RAIN), 101, acc)
    return jnp.where(hurricane & (st.weather == SUN), 50, acc)


@hooks.hook("after_hit")
def _secondary_volatiles(st, ctx, dealt, hit_sub):
    o, mid = ctx.o, ctx.mid
    sec = secondary_lands(st, ctx, hit_sub, MOVES[mid, MC["sec_chance"]], jax.random.fold_in(ctx.key, 302))
    k_conf = jax.random.fold_in(ctx.key, 303)
    st, _ = add_confusion(st, o, sec & SEC_CONFUSION[mid] & (mid != MID["axekick"]), k_conf, ctx.cfg)
    st, _ = add_confusion(st, o, sec & (mid == MID["axekick"]), k_conf, ctx.cfg, min_turns=3)
    # Psychic Noise: Heal Block for 2 turns (a second one doesn't extend it)
    st = set_vol(st, o, "healblock", 2, sec & SEC_HEALBLOCK[mid] & (vol(st, o, "healblock") == 0))
    st = set_vol(st, o, "saltcure", 1, sec & SEC_SALTCURE[mid])
    # Throat Chop: no sound moves for 2 turns
    chop = secondary_lands(st, ctx, hit_sub, 100, ctx.key) & (mid == MID["throatchop"])
    return set_vol(st, o, "throatchop", 2, chop & (vol(st, o, "throatchop") == 0))


# --- status moves -------------------------------------------------------------------------------------


def _last_move_slot(st, s):
    """Slot of side s's last used move among its moves, -1 if it isn't one of them."""
    mids = _i(st.moves[s, act(st, s)])
    last = vol(st, s, "last_move")
    hit = (mids == last) & (last > 0)
    return jnp.where(jnp.any(hit), jnp.argmax(hit), -1)


def _encore_fails(st, o):
    last = vol(st, o, "last_move")
    lslot = _last_move_slot(st, o)
    pp = _i(st.pp[o, act(st, o), jnp.maximum(lslot, 0)])
    return (vol(st, o, "encore") > 0) | (last == 0) | FAIL_ENCORE[last] | (lslot < 0) | (pp <= 0)


def _is_ghost(st, s):
    return has_type(types_now(st, s, act(st, s)), TY["ghost"])


@hooks.hook("move_fails")
def _fails(st, ctx):
    s, o, mid = ctx.s, ctx.o, ctx.mid
    fails = (mid == MID["taunt"]) & (vol(st, o, "taunt") > 0)
    fails |= (mid == MID["encore"]) & _encore_fails(st, o)
    fails |= (mid == MID["curse"]) & _is_ghost(st, s) & (vol(st, o, "curse") > 0)
    seed_immune = has_type(types_now(st, o, ctx.tslot), TY["grass"])
    fails |= (mid == MID["leechseed"]) & ((vol(st, o, "leechseed") > 0) | seed_immune)
    # A second Destiny Bond in a row fails (onPrepareHit removes the first one)
    fails |= (mid == MID["destinybond"]) & (vol(st, s, "destinybond") > 0)
    return fails


@hooks.hook("status_move")
def _status(st, ctx):
    s, o, slot, mid = ctx.s, ctx.o, ctx.slot, ctx.mid
    ok = ctx.landed
    behind_sub = st.sub_hp[o] > 0  # for moves without bypasssub
    # Swagger: the table applied +2 Atk; then confusion (a Substitute blocks both)
    st, _ = add_confusion(st, o, ok & (mid == MID["swagger"]) & ~behind_sub, jax.random.fold_in(ctx.key, 311), ctx.cfg)
    # Taunt: 3 turns, one more if the target already acted this turn (but not on its first turn out)
    target_waits = ctx.foe_pending_move == 0
    taunt_turns = 3 + ((vol(st, o, "active_turns") > 0) & target_waits)
    st = set_vol(st, o, "taunt", taunt_turns, ok & (mid == MID["taunt"]))
    # Encore: 3 turns (4 if the target won't move again this turn), locked into its last move
    enc = ok & (mid == MID["encore"])
    st = set_vol(st, o, "encore_slot", _last_move_slot(st, o), enc)
    st = set_vol(st, o, "encore", 3 + target_waits, enc)
    # Curse: a Ghost type halves its own HP to curse the target; others get +1 Atk / Def, -1 Spe
    curse = ok & (mid == MID["curse"])
    ghost = _is_ghost(st, s)
    st = set_vol(st, o, "curse", 1, curse & ghost)
    st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 2, curse & ghost)
    st, _ = apply_boosts(st, s, jnp.array([1, 1, 0, 0, -1, 0, 0]), curse & ~ghost)
    st = set_vol(st, o, "leechseed", 1, ok & (mid == MID["leechseed"]) & ~behind_sub)
    return set_vol(st, s, "destinybond", 1, ok & (mid == MID["destinybond"]))


# --- move restrictions --------------------------------------------------------------------------------


@hooks.hook("legal_moves")
def _disable(st, s, mask):
    mids = _i(st.moves[s, jnp.maximum(act(st, s), 0)])
    status_move = MOVES[mids, MC["category"]] == 2
    mask &= ~((vol(st, s, "taunt") > 0) & status_move)
    mask &= ~((vol(st, s, "healblock") > 0) & has_flag(mids, "heal"))
    mask &= ~((vol(st, s, "throatchop") > 0) & has_flag(mids, "sound"))
    return mask & ((vol(st, s, "encore") == 0) | (jnp.arange(4) == vol(st, s, "encore_slot")))


@hooks.hook("override_move")
def _encore_override(st, s, move_slot):
    # Encore applied after this Pokemon chose its move this turn replaces the move when its turn comes
    return jnp.where(vol(st, s, "encore") > 0, vol(st, s, "encore_slot"), move_slot)


@hooks.hook("before_move", priority=6)
def _blocked_heal_sound(st, ctx):
    s, mid = ctx.s, ctx.mid
    blocked = ((vol(st, s, "healblock") > 0) & has_flag(mid, "heal")) | (
        (vol(st, s, "throatchop") > 0) & has_flag(mid, "sound")
    )
    return st, ctx.moves & blocked


@hooks.hook("before_move", priority=5)
def _blocked_taunt(st, ctx):
    return st, ctx.moves & (vol(st, ctx.s, "taunt") > 0) & ctx.is_status


@hooks.hook("can_heal")
def _heal_block(st, s, slot):
    return ~((slot == act(st, s)) & (vol(st, s, "healblock") > 0))


@hooks.hook("switch_out")
def _leave(st, s, cond):
    # Volatiles reset after switch-out effects; Regenerator heals through Heal Block (Pokemon.heal)
    return set_vol(st, s, "healblock", 0, cond)


# --- records and move bookkeeping ------------------------------------------------------------------------


@hooks.hook("before_move", priority=100)
def _record_target_status(st, ctx):
    o = ctx.o
    return set_vol(st, ctx.s, "puppet_prev", _i(st.status[o, ctx.tslot]) + 1), jnp.bool_(False)


@hooks.hook("after_move")
def _after_move(st, ctx):
    s, o, slot, tslot, mid = ctx.s, ctx.o, ctx.slot, ctx.tslot, ctx.mid
    acted = ctx.acting
    # Poison Puppeteer: Pecharunt's move poisoning the target also confuses it
    status = st.status[o, tslot]
    puppet = (
        acted
        & (st.species[s, slot] == PECHARUNT)
        & (st.ability[s, slot] == POISON_PUPPETEER)
        & ((status == PSN) | (status == TOX))
        & (vol(st, s, "puppet_prev") == 1)
    )
    st, _ = add_confusion(st, o, puppet, jax.random.fold_in(ctx.key, 321), ctx.cfg)
    # Destiny Bond: the attacker faints when its move makes the bonded target faint
    bonded = ctx.landed & ~ctx.is_status & (st.hp[o, tslot] == 0) & (vol(st, o, "destinybond") > 0)
    st = lose_hp(st, s, slot, _i(st.hp[s, slot]), bonded & (st.hp[s, slot] > 0))
    # Destiny Bond lasts until this Pokemon's next move (or attempt)
    keep = (mid == MID["destinybond"]) & ctx.landed
    st = set_vol(st, s, "destinybond", 0, acted & ~keep)
    st = set_vol(st, s, "move_actions", jnp.minimum(vol(st, s, "move_actions") + 1, 100), acted)
    # Pokemon.lastMove is the move chosen (Sleep Talk, not the move it called)
    return set_vol(st, s, "last_move", ctx.base_mid, acted & ctx.moves)


@hooks.hook("switch_in")
def _switch_in(st, s, key, cfg, cond):
    # activeTurns counts turn starts: a Pokemon entering at team preview or as an end-of-turn replacement
    # starts the next turn with 1; one switching in during a turn gets its first count at end of turn
    mid_turn = (st.phase == PHASE_MOVE) | ((st.phase == PHASE_SWITCH) & (st.resume != RESUME_NONE))
    return set_vol(st, s, "active_turns", jnp.where(mid_turn, 0, 1), cond)


# --- end of turn ----------------------------------------------------------------------------------------


@hooks.residual(8, "leech_seed")
def _leech_seed(st, key, cfg, cond):
    order = speed_order(st, key, cfg)
    for i in range(2):
        s = order[i]
        o = foe(s)
        slot, oslot = act(st, s), act(st, o)
        seeded = cond & (vol(st, s, "leechseed") > 0) & (st.hp[s, slot] > 0) & (st.hp[o, oslot] > 0)
        hp = _i(st.hp[s, slot])
        dmg = jnp.minimum(jnp.maximum(_i(st.stats[s, slot, 0]) // 8, 1), hp)
        st = lose_hp(st, s, slot, dmg, seeded)
        st = gain_hp(st, o, oslot, dmg, seeded)
        st = eat_sitrus(eat_sitrus(st, s), o)
    return st


def _chip(order_, name, frac_fn):
    @hooks.residual(order_, name)
    def _res(st, key, cfg, cond):
        order = speed_order(st, key, cfg)
        for i in range(2):
            s = order[i]
            slot = act(st, s)
            on = cond & (vol(st, s, name) > 0) & (st.hp[s, slot] > 0)
            st = lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // frac_fn(st, s), on)
            st = eat_sitrus(st, s)
        return st

    return _res


_chip(12, "curse", lambda st, s: 4)
_chip(13, "saltcure", lambda st, s: jnp.where(
    has_type(types_now(st, s, act(st, s)), TY["water"]) | has_type(types_now(st, s, act(st, s)), TY["steel"]), 4, 8
))  # fmt: skip


def _countdown(order_, name):
    @hooks.residual(order_, name + "_end")
    def _res(st, key, cfg, cond):
        for s in (0, 1):
            left = vol(st, s, name)
            st = set_vol(st, s, name, left - 1, cond & (left > 0))
        return st

    return _res


_countdown(15, "taunt")


@hooks.residual(16, "encore_end")
def _encore_end(st, key, cfg, cond):
    for s in (0, 1):
        left = vol(st, s, "encore")
        on = cond & (left > 0)
        left = jnp.where(on, left - 1, left)
        # Ends early once the encored move is out of PP
        out_of_pp = st.pp[s, act(st, s), jnp.clip(vol(st, s, "encore_slot"), 0, 3)] <= 0
        st = set_vol(st, s, "encore", jnp.where(out_of_pp, 0, left), on)
    return st


_countdown(20, "healblock")
_countdown(22, "throatchop")


@hooks.residual(21, "healblock_berry")
def _healblock_berry(st, key, cfg, cond):
    # A Sitrus Berry held back by Heal Block is eaten as soon as the block ends
    for s in (0, 1):
        eaten = eat_sitrus(st, s)
        st = jax.tree.map(lambda a, b: jnp.where(cond, a, b), eaten, st)
    return st


@hooks.residual(50, "active_turns")
def _turn_count(st, key, cfg, cond):
    for s in (0, 1):
        st = set_vol(st, s, "active_turns", jnp.minimum(vol(st, s, "active_turns") + 1, 100), cond)
    return st


for _v in ("confusion", "taunt", "encore", "healblock", "throatchop", "curse", "saltcure", "leechseed", "destinybond"):
    hooks.showdown_volatile(_v)((lambda name: lambda st, s: vol(st, s, name) > 0)(_v))
