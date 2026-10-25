"""
engine/oujax/hooks.py
Extension points for OU-JAX mechanic families (engine/oujax/families/*.py). A family registers hook functions
for named events and claims named slots in the generic state arrays; engine.py calls the hooks at fixed points,
so adding a mechanic never means editing the engine's central functions.

Hooks are plain Python functions traced into the jitted step: they must be branch-free (jnp.where masks) and
take/return the state. Within an event, hooks run in descending priority, then registration order.

Events (arguments after the state; return value):
  before_move(st, ctx)                  -> (st, blocked)   user can't act (confusion, ...); before PP is spent
  move_fails(st, ctx)                   -> fails           the move fails after PP is spent (Sucker Punch, ...)
  try_hit(st, ctx)                      -> (st, blocked)   target-side blocks after accuracy (absorbing abilities)
  damage_taken(st, ctx, side, dmg)      -> (st, dmg)       adjust a hit's damage to the target (Focus Sash, ...)
  after_hit(st, ctx, dealt, hit_sub)    -> st              after a damaging move's built-in after-hit effects
  status_move(st, ctx)                  -> st              extra effects of a landed status move
  after_move(st, ctx)                   -> st              after any move resolution, whether it landed or not
  switch_in(st, side, key, cfg, cond)   -> st              entry effects after hazards and built-in abilities
  switch_out(st, side, cond)            -> st              leaving the field, before volatiles reset
  residual(order)(st, key, cfg, cond)   -> st              end-of-turn effect at Showdown's onResidualOrder
  legal_moves(st, side, mask)           -> mask            (4,) bool move legality (Encore, Taunt, ...)
  speed(st, side, mod)                  -> mod             4096-based Speed modifier chain
  priority(st, side, mid, prio)         -> prio
  status_immune(st, side, slot, status) -> immune
  can_heal(st, side, slot)              -> allowed
  boost_change(st, side, deltas, ctx)   -> deltas          stat-stage changes before they apply

ctx is a MoveCtx (see engine.py): user side s, target side o, their team slots, move id, move slot, keys, cfg,
and flags about the move (category, landed, ...).

State slots: claim with VOL.add / MON.add / SIDE.add / FIELD.add at import time; read and write through
the helpers below. VOL slots are per active Pokemon and reset on switch-out (unless keep=True); MON slots
persist on each team member; SIDE and FIELD slots are side and field conditions.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Tuple

EVENTS = {
    "before_move", "move_fails", "try_hit", "damage_taken", "after_hit", "status_move", "after_move", "switch_in",
    "switch_out", "residual", "legal_moves", "speed", "priority", "status_immune", "can_heal", "boost_change",
}  # fmt: skip

_hooks: Dict[str, List[Tuple[float, int, Callable]]] = defaultdict(list)
_residual: List[Tuple[float, int, str, Callable]] = []
_counter = [0]


def hook(event: str, priority: float = 0):
    """Decorator: register fn for an event."""
    if event not in EVENTS or event == "residual":
        raise ValueError(f"unknown event {event!r} (use residual(order) for end-of-turn effects)")

    def deco(fn):
        _counter[0] += 1
        _hooks[event].append((-priority, _counter[0], fn))
        _hooks[event].sort(key=lambda x: (x[0], x[1]))
        return fn

    return deco


def residual(order: float, name: str = ""):
    """Decorator: register an end-of-turn effect at Showdown's onResidualOrder (built-ins use 1 weather,
    5.2 Grassy Terrain, 5.4 Leftovers, 9 poison, 10 burn, 25 Roost, 26 screens, 27 terrain/Trick Room)."""

    def deco(fn):
        _counter[0] += 1
        _residual.append((order, _counter[0], name or fn.__name__, fn))
        _residual.sort(key=lambda x: (x[0], x[1]))
        return fn

    return deco


def handlers(event: str) -> List[Callable]:
    return [fn for _, _, fn in _hooks.get(event, [])]


def residual_handlers() -> List[Tuple[float, str, Callable]]:
    return [(order, name, fn) for order, _, name, fn in _residual]


class SlotRegistry:
    """Named integer slots in one generic state array."""

    def __init__(self, label: str):
        self.label = label
        self.index: Dict[str, int] = {}
        self.keep: Dict[str, bool] = {}

    def add(self, name: str, keep: bool = False) -> int:
        if name in self.index:
            raise ValueError(f"{self.label} slot {name!r} claimed twice")
        self.index[name] = len(self.index)
        self.keep[name] = keep
        return self.index[name]

    def __len__(self):
        return len(self.index)


VOL = SlotRegistry("volatile")  # Battle.vol (2, N) int16, per active Pokemon
MON = SlotRegistry("mon")  # Battle.mon_ex (2, 6, N) int16, per team member
SIDE = SlotRegistry("side")  # Battle.side_ex (2, N) int16
FIELD = SlotRegistry("field")  # Battle.field_ex (N,) int16

# Room reserved so states created before every family is imported still have the right shape
VOL_SIZE, MON_SIZE, SIDE_SIZE, FIELD_SIZE = 48, 16, 24, 8


def check_sizes():
    for reg, size in ((VOL, VOL_SIZE), (MON, MON_SIZE), (SIDE, SIDE_SIZE), (FIELD, FIELD_SIZE)):
        if len(reg) > size:
            raise ValueError(f"{reg.label} slots exhausted ({len(reg)} > {size}); raise the size in hooks.py")
