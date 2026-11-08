"""
OU-JAX mechanic families. Each module registers hooks (engine/oujax/hooks.py) for one family of moves, items or
abilities and has its own test file (engine/oujax/families/test_<name>.py). Add a family by creating its
module and importing it here.
"""

from engine.oujax.families import (  # noqa: F401
    call_moves,
    charge_moves,
    damage_moves,
    field_moves,
    move_misc,
    power_moves,
    trapping,
    volatiles,
)
