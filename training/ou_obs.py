"""
training/ou_obs.py
One player's view of an OU-JAX battle, as fixed-shape arrays for the policy (training/ou_ppo.py) and for the live
Showdown bridge, which fills the same fields from the protocol.

What the player knows about the foe: its six species (team preview) and their public types, HP, status, boosts,
Tera once used, and the moves and items it has revealed. Unrevealed moves and items, abilities and the Tera type
are the UNKNOWN id. Revealed moves come from PP being spent; an item counts as revealed once it changes (used up,
knocked off, swapped).
"""

from typing import NamedTuple

import jax.numpy as jnp

from engine.oujax.engine import PHASE_PREVIEW, act
from engine.oujax.tables import load_tables

T = load_tables()
N_SPECIES, N_ITEMS, N_ABILITIES, N_MOVES, N_TYPES = (
    len(T.species_ids), len(T.item_ids), len(T.ability_ids), len(T.move_ids), len(T.type_ids)
)  # fmt: skip
VOCAB = (N_SPECIES, N_ITEMS, N_ABILITIES, N_MOVES, N_TYPES)  # each id < n; id n = UNKNOWN
N_MON_FLOAT = 1 + 7 + 1 + 1 + 1 + 7 + 1 + 6
N_FIELD = 5 + 5 + 1 + 8 + 6 + 2 + 1 + 3 + 2 + 2


class Seen(NamedTuple):
    """What each side has revealed to the other: (owner side, team slot[, move slot])."""

    move: jnp.ndarray  # (2, 6, 4) bool
    item: jnp.ndarray  # (2, 6) bool


class Obs(NamedTuple):
    species: jnp.ndarray  # (12,) int32: own 0..5 then foe 0..5
    item: jnp.ndarray  # (12,)
    ability: jnp.ndarray  # (12,)
    moves: jnp.ndarray  # (12, 4)
    move_float: jnp.ndarray  # (12, 4, 2) [pp fraction (own only), known]
    types: jnp.ndarray  # (12, 2)
    tera: jnp.ndarray  # (12,)
    mon_float: jnp.ndarray  # (12, N_MON_FLOAT)
    side: jnp.ndarray  # (12,) 0 own, 1 foe
    field: jnp.ndarray  # (N_FIELD,)
    own_active: jnp.ndarray  # (6,) one-hot own active slot (zeros at team preview)
    bench_slot: jnp.ndarray  # (5, 6) one-hot: switch action n -> own team slot
    preview: jnp.ndarray  # () bool


def init_seen(st) -> Seen:
    return Seen(move=jnp.zeros((2, 6, 4), bool), item=jnp.zeros((2, 6), bool))


def update_seen(seen: Seen, prev, nxt) -> Seen:
    used = nxt.pp < prev.pp
    changed = (nxt.item != prev.item) & (prev.species > 0)
    return Seen(move=seen.move | used, item=seen.item | changed)


def _i(x):
    return x.astype(jnp.int32)


def observe(st, seen: Seen, p) -> Obs:
    """Side p's view of the battle."""
    o = 1 - p
    order = jnp.stack([p, o])

    def both(x):  # (2, 6, ...) -> (12, ...) own first
        return x[order].reshape((12,) + x.shape[2:])

    foe12 = jnp.arange(12) >= 6
    species = _i(both(st.species))
    item = jnp.where(foe12 & ~both(seen.item), N_ITEMS, _i(both(st.item)))
    ability = jnp.where(foe12, N_ABILITIES, _i(both(st.ability)))
    known_mv = jnp.where(foe12[:, None], both(seen.move), True) & (both(st.moves) > 0)
    moves = jnp.where(known_mv | ~foe12[:, None], _i(both(st.moves)), N_MOVES)
    pp_frac = jnp.where(foe12[:, None], 0.0, jnp.clip(_i(both(st.pp)) / 16.0, 0, 4) / 4.0)
    move_float = jnp.stack([pp_frac, known_mv.astype(jnp.float32)], -1)
    types = _i(both(st.base_types))
    tera_on = both(st.terastallized)
    tera = jnp.where(foe12 & ~tera_on, N_TYPES, _i(both(st.tera_type)))

    hp = both(st.hp).astype(jnp.float32)
    maxhp = jnp.maximum(both(st.stats)[:, 0].astype(jnp.float32), 1.0)
    status = jnp.eye(7)[_i(both(st.status))]
    active_slot = jnp.stack([act(st, p), act(st, o)])
    is_active = (jnp.arange(6)[None] == active_slot[:, None]).reshape(12)
    boosts = jnp.where(is_active[:, None], st.boosts[order].repeat(6, axis=0).astype(jnp.float32) / 6.0, 0.0)
    stats = jnp.where(foe12[:, None], 0.0, both(st.stats)[:, :6].astype(jnp.float32) / 500.0)
    present = species > 0
    mon_float = jnp.concatenate(
        [
            (hp / maxhp)[:, None],
            status,
            is_active[:, None].astype(jnp.float32),
            ((hp <= 0) & present)[:, None].astype(jnp.float32),
            tera_on[:, None].astype(jnp.float32),
            boosts,
            (both(st.level).astype(jnp.float32) / 100.0)[:, None],
            stats,
        ],
        -1,
    )
    mon_float = jnp.where(present[:, None], mon_float, 0.0)

    field = jnp.concatenate(
        [
            jnp.eye(5)[_i(st.weather)],
            jnp.eye(5)[_i(st.terrain)],
            (st.trick_room.astype(jnp.float32) / 5.0)[None],
            st.hazards[order].reshape(8).astype(jnp.float32) / 3.0,
            st.screens[order].reshape(6).astype(jnp.float32) / 8.0,
            st.tailwind[order].astype(jnp.float32) / 4.0,
            (st.turn.astype(jnp.float32) / 100.0)[None],
            jnp.eye(3)[_i(st.phase)],
            st.need_switch[order].astype(jnp.float32),
            (st.sub_hp[order] > 0).astype(jnp.float32),
        ]
    )

    own_slot = act(st, p)
    preview = st.phase == PHASE_PREVIEW
    own_active = jnp.where(preview, 0.0, (jnp.arange(6) == own_slot).astype(jnp.float32))
    benched = jnp.arange(6) != own_slot
    rank = jnp.cumsum(benched) - 1
    bench_slot = ((rank[None] == jnp.arange(5)[:, None]) & benched[None]).astype(jnp.float32)
    return Obs(
        species=species,
        item=_i(item),
        ability=_i(ability),
        moves=_i(moves),
        move_float=move_float,
        types=types,
        tera=_i(tera),
        mon_float=mon_float,
        side=foe12.astype(jnp.int32),
        field=field,
        own_active=own_active,
        bench_slot=bench_slot,
        preview=preview,
    )
