"""
engine/oujax/state.py
OU-JAX battle state: one fixed-shape PyTree per battle, small integer dtypes, side index first (0 = p1).

Per-Pokemon arrays are (2, 6, ...) in team order and never reorder; `active` holds the team slot on the
field. Volatile state only exists for the active Pokemon and is reset on switch-in. Observation masking
(what each side may see) is left to the model input layer (M4); nothing here is hidden.
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from engine.oujax.team import TeamArrays

PHASE_PREVIEW, PHASE_MOVE, PHASE_SWITCH = 0, 1, 2
RESUME_NONE, RESUME_SECOND_MOVE, RESUME_END_OF_TURN = 0, 1, 2
HAZ_ROCKS, HAZ_SPIKES, HAZ_TSPIKES, HAZ_WEBS = range(4)
SCR_REFLECT, SCR_LIGHT_SCREEN, SCR_VEIL = range(3)
NO_ACTION = -1
WINNER_NONE, WINNER_TIE = -1, 2


@struct.dataclass
class Battle:
    # --- Pokemon (2 sides x 6 slots) ---------------------------------------------------------------
    species: jnp.ndarray  # (2, 6) int16, 0 = empty slot
    level: jnp.ndarray  # (2, 6) int16
    stats: jnp.ndarray  # (2, 6, 6) int16 [maxhp, atk, def, spa, spd, spe]
    base_types: jnp.ndarray  # (2, 6, 2) int8
    hp: jnp.ndarray  # (2, 6) int16
    status: jnp.ndarray  # (2, 6) int8 (tables.STATUS)
    status_turns: jnp.ndarray  # (2, 6) int8: sleep turns left / toxic stage
    self_sleep: jnp.ndarray  # (2, 6) bool: asleep from its own Rest (ignored by Sleep Clause)
    moves: jnp.ndarray  # (2, 6, 4) int16
    pp: jnp.ndarray  # (2, 6, 4) int8
    item: jnp.ndarray  # (2, 6) int16, 0 = none (consumed or knocked off)
    ability: jnp.ndarray  # (2, 6) int16
    tera_type: jnp.ndarray  # (2, 6) int8
    terastallized: jnp.ndarray  # (2, 6) bool
    stellar_used: jnp.ndarray  # (2, 6) int32 bitmask of types that used their Stellar Tera boost
    # --- active Pokemon and its volatiles (2 sides) --------------------------------------------------
    active: jnp.ndarray  # (2,) int8 team slot, -1 before team preview
    boosts: jnp.ndarray  # (2, 7) int8 [atk, def, spa, spd, spe, accuracy, evasion]
    paradox_stat: jnp.ndarray  # (2,) int8: Protosynthesis / Quark Drive boosted stat (0 = none)
    paradox_from_item: jnp.ndarray  # (2,) bool: activated by Booster Energy (lasts until switch-out)
    overlord_fallen: jnp.ndarray  # (2,) int8: Supreme Overlord's fainted-teammate count taken on switch-in
    choice_lock: jnp.ndarray  # (2,) int8 move slot, -1 = none
    protect_count: jnp.ndarray  # (2,) int8 consecutive successful Protects
    protect_used: jnp.ndarray  # (2,) bool: used a Protect-family move this turn
    protected: jnp.ndarray  # (2,) bool: protected for the rest of this turn
    sub_hp: jnp.ndarray  # (2,) int16, 0 = no substitute
    flinch: jnp.ndarray  # (2,) bool
    roosted: jnp.ndarray  # (2,) bool: lost Flying type until end of turn
    # --- side conditions ---------------------------------------------------------------------------
    tera_used: jnp.ndarray  # (2,) bool
    hazards: jnp.ndarray  # (2, 4) int8 [rocks, spikes layers, toxic spikes layers, webs]
    screens: jnp.ndarray  # (2, 3) int8 turns left [reflect, light screen, aurora veil]
    tailwind: jnp.ndarray  # (2,) int8 turns left
    # --- field -------------------------------------------------------------------------------------
    weather: jnp.ndarray  # () int8 (tables.WEATHERS index)
    weather_turns: jnp.ndarray  # () int8
    terrain: jnp.ndarray  # () int8 (tables.TERRAINS index)
    terrain_turns: jnp.ndarray  # () int8
    trick_room: jnp.ndarray  # () int8 turns left
    # --- request flow ------------------------------------------------------------------------------
    phase: jnp.ndarray  # () int8 PHASE_*
    need_switch: jnp.ndarray  # (2,) bool: side must pick a switch in PHASE_SWITCH
    resume: jnp.ndarray  # () int8 RESUME_*: what is left of the turn after a mid-turn switch
    queued_action: jnp.ndarray  # (2,) int8: move action still to run after a pivot, -1 = none
    turn: jnp.ndarray  # () int16
    winner: jnp.ndarray  # () int8: -1 ongoing, 0 / 1 side won, 2 tie
    last_faint: jnp.ndarray  # () int8: side of the most recent Pokemon to faint (-1 none); wins a double wipe


def init_battle(p1: TeamArrays, p2: TeamArrays) -> Battle:
    """Battle at team preview from two loaded teams (numpy in, jnp out). Empty team slots have species 0
    and 0 HP, so they behave like fainted Pokemon."""

    def both(name, dtype):
        return jnp.asarray(np.stack([getattr(p1, name), getattr(p2, name)]).astype(dtype))

    stats = both("stats", np.int16)

    def zeros2(dtype):
        return jnp.zeros((2,), dtype)

    return Battle(
        species=both("species", np.int16),
        level=both("level", np.int16),
        stats=stats,
        base_types=both("types", np.int8),
        hp=stats[:, :, 0],
        status=jnp.zeros((2, 6), jnp.int8),
        status_turns=jnp.zeros((2, 6), jnp.int8),
        self_sleep=jnp.zeros((2, 6), bool),
        moves=both("moves", np.int16),
        pp=both("max_pp", np.int8),
        item=both("item", np.int16),
        ability=both("ability", np.int16),
        tera_type=both("tera_type", np.int8),
        terastallized=jnp.zeros((2, 6), bool),
        stellar_used=jnp.zeros((2, 6), jnp.int32),
        active=jnp.full((2,), -1, jnp.int8),
        boosts=jnp.zeros((2, 7), jnp.int8),
        paradox_stat=zeros2(jnp.int8),
        paradox_from_item=zeros2(bool),
        overlord_fallen=zeros2(jnp.int8),
        choice_lock=jnp.full((2,), -1, jnp.int8),
        protect_count=zeros2(jnp.int8),
        protect_used=zeros2(bool),
        protected=zeros2(bool),
        sub_hp=zeros2(jnp.int16),
        flinch=zeros2(bool),
        roosted=zeros2(bool),
        tera_used=zeros2(bool),
        hazards=jnp.zeros((2, 4), jnp.int8),
        screens=jnp.zeros((2, 3), jnp.int8),
        tailwind=zeros2(jnp.int8),
        weather=jnp.int8(0),
        weather_turns=jnp.int8(0),
        terrain=jnp.int8(0),
        terrain_turns=jnp.int8(0),
        trick_room=jnp.int8(0),
        phase=jnp.int8(PHASE_PREVIEW),
        need_switch=zeros2(bool),
        resume=jnp.int8(RESUME_NONE),
        queued_action=jnp.full((2,), NO_ACTION, jnp.int8),
        turn=jnp.int16(0),
        winner=jnp.int8(WINNER_NONE),
        last_faint=jnp.int8(-1),
    )
