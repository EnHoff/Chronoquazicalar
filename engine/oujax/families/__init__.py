"""
OU-JAX mechanic families. Each module registers hooks (engine/oujax/hooks.py) for one family of moves, items or
abilities and has its own test file (engine/oujax/families/test_<name>.py). Add a family by creating its
module and importing it here.
"""

from engine.oujax.families import (
    absorb_abilities,  # noqa: F401
    contact_abilities,  # noqa: F401
    damage_moves,  # noqa: F401
    entry_abilities,  # noqa: F401  (first: other ability families use its helpers)
    passive_abilities,  # noqa: F401
    stat_abilities,  # noqa: F401
)
