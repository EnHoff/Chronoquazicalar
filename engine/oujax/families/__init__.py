"""
OU-JAX mechanic families. Each module registers hooks (engine/oujax/hooks.py) for one family of moves, items or
abilities and has its own test file (engine/oujax/families/test_<name>.py). Add a family by creating its
module and importing it here.
"""

from engine.oujax.families import (  # noqa: F401
    absorb_abilities,
    berries,
    call_moves,
    charge_moves,
    contact_abilities,
    damage_moves,
    entry_abilities,
    field_moves,
    forced_switch,
    held_items,
    herbs_seeds,
    item_moves,
    move_misc,
    passive_abilities,
    power_moves,
    self_ko,
    slot_conditions,
    stat_abilities,
    status_moves,
    trapping,
    volatiles,
)
