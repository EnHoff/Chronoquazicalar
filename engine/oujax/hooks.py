"""
engine/oujax/hooks.py
Extension points for OU-JAX mechanic families (engine/oujax/families/*.py). A family registers hook functions
for named events and claims named slots in the generic state arrays; engine.py calls the hooks at fixed points,
so adding a mechanic never means editing the engine's central functions.

Hooks are plain Python functions traced into the jitted step: they must be branch-free (jnp.where masks) and
take/return the state. Within an event, hooks run in descending priority, then registration order.

Events (arguments after the state; return value):
  before_move(st, ctx)                  -> (st, blocked)   user can't act (confusion, ...); before PP is spent.
                                                           Showdown's BeforeMove priorities: above 10 run before
                                                           sleep / freeze (recharge), the rest after sleep /
                                                           freeze / flinch, before paralysis
  call_move(st, ctx)                    -> (st, mid, fails)  the move used calls another one (Sleep Talk): after
                                                           PP and Choice lock; mid 0 = no call
  move_fails(st, ctx)                   -> fails           the move fails after PP is spent (Sucker Punch, ...)
  try_move(st, ctx)                     -> (st, stop)      Showdown's TryMove: the move stops without failing
                                                           (charge turn of Solar Beam, ...)
  locked_move(st, side)                 -> slot            move slot the Pokemon must use (Outrage, charge and
                                                           recharge turns), -1 if none: a single legal move
                                                           action, no Tera, no switching, no PP spent
  override_move(st, side, move_slot)    -> move_slot       the move actually used when the turn comes (Encore)
  trapped(st, side)                     -> trapped         can't switch out by choice (partial trapping, ...)
  try_hit(st, ctx)                      -> (st, blocked)   Showdown's TryHit step: after Protect, before type
                                                           immunity and accuracy (absorbing abilities);
                                                           ctx.landed = the move reaches the target
  damaging_hit(st, ctx, hit, dmg)       -> st              once per hit on the target itself (not its Substitute),
                                                           even for 0 damage (Disguise); ctx.landed = that happened
  move_try_hit(st, ctx)                 -> st              the move's own onTryHit: after accuracy and immunity,
                                                           before a Substitute or damage (Brick Break)
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
  boost_change(st, side, deltas, from_foe) -> deltas       stat-stage changes before they apply
  hit_count(st, ctx, n)                 -> n               number of hits (Beat Up), at most engine.MAX_HITS
  hit_power(st, ctx, hit)               -> (mult, chain)   per-hit base power: raw multiplier and an extra
                                                           4096-based BasePower modifier (Triple Axel, Payback)
  damage_override(st, ctx, dmg)         -> dmg             replace a hit's damage (Counter, Endeavor, ...)
  crit_stage(st, ctx, stage)            -> stage           critical-hit stage (Focus Energy, ...)
  accuracy(st, ctx, acc)                -> acc             accuracy before stages; 101 means "always hits"
  enter(st, side, slot, cond)           -> st              a Pokemon was just put on the field, before hazards and
                                                           entry abilities (restore its ability, suppress it, ...)
  after_boost(st, side, old, new, raw, from_foe, depth) -> st  after stat stages changed (old/new (7,) stages, raw =
                                                           the requested deltas before boost_change hooks);
                                                           only for changes from the foe, and not for a boost
                                                           an after_boost hook makes (depth 1)
  reflect(st, ctx)                      -> (st, reflected, lands)  before TryHit: the target bounces the move back
                                                           (Magic Bounce); lands = the bounced move connects
  after_status(st, side, slot, status, from_foe, key, cfg, applied) -> st  after try_set_status (Synchronize)
  secondaries(st, s, mid, chance)       -> (chance, blocked)  the move's secondary-effect chance, and whether the
                                                           target is shielded from it (Serene Grace, Shield Dust)
  indirect_damage(st, side, slot)       -> allowed         non-move damage through lose_hp (Magic Guard)
  volatile_immune(st, side, name)       -> immune          side's active Pokemon can't get the volatile `name` (a
                                                           Python string such as "confusion"; Own Tempo)
  accuracy_mod(st, ctx, mod)            -> mod             4096-based ModifyAccuracy chain, before abilities and
                                                           items (Gravity)
  grounded(st, s, slot)                 -> air             1: grounded regardless (Gravity, Smack Down); 2: airborne
                                                           (Magnet Rise); 0: by type, ability, item (max wins)
  types(st, s, slot, types)             -> types           the (2,) current types of side s's active Pokemon
                                                           before Tera (Double Shock's lost Electric type)

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
    "hit_power", "damage_override", "crit_stage", "accuracy", "move_try_hit", "damaging_hit", "try_move",
    "locked_move", "override_move", "trapped", "call_move", "grounded", "types", "accuracy_mod", "hit_count",
    "enter", "after_boost", "reflect", "after_status", "secondaries", "indirect_damage", "volatile_immune",
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


def handlers(event: str, above=None, at_most=None) -> List[Callable]:
    """Handlers for an event in run order; above / at_most select a priority range (exclusive / inclusive)."""
    return [
        fn for p, _, fn in _hooks.get(event, []) if (above is None or -p > above) and (at_most is None or -p <= at_most)
    ]


# Showdown volatile id -> fn(st, side) -> bool, compared by bridge.Mirror.diff for the active Pokemon
SHOWDOWN_VOLATILES: Dict[str, Callable] = {}


def showdown_volatile(name: str):
    """Decorator: fn(st, side) says whether side's active Pokemon has Showdown's volatile `name`."""

    def deco(fn):
        SHOWDOWN_VOLATILES[name] = fn
        return fn

    return deco


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
VOL_SIZE, MON_SIZE, SIDE_SIZE, FIELD_SIZE = 64, 16, 24, 8


def check_sizes():
    for reg, size in ((VOL, VOL_SIZE), (MON, MON_SIZE), (SIDE, SIDE_SIZE), (FIELD, FIELD_SIZE)):
        if len(reg) > size:
            raise ValueError(f"{reg.label} slots exhausted ({len(reg)} > {size}); raise the size in hooks.py")
