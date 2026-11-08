"""
engine/oujax/families/call_moves.py
Moves that call another move: Sleep Talk (a random one of the user's other moves, while asleep).

Showdown runs the called move through useMove: no BeforeMove, no PP of its own (Pressure's extra PP comes off
Sleep Talk), the caller's priority; the user's lastMove stays Sleep Talk (engine.run_move's call_move hook point).
"""

import jax
import jax.numpy as jnp

from engine.oujax import hooks
from engine.oujax.engine import MID, SLP, _i, act
from engine.oujax.tables import load_tables

T = load_tables()
N_MOVES = len(MID)

# Sleep Talk skips moves flagged nosleeptalk (itself, Focus Punch, Beak Blast, ...) and charge moves
NO_SLEEP_TALK = (
    jnp.zeros(N_MOVES, bool)
    .at[jnp.array([MID[m] for m, d in T.dex["moves"].items() if {"nosleeptalk", "charge"} & set(d.get("flags", {}))])]
    .set(True)
    .at[0]
    .set(True)
)
SLEEP_TALK = MID["sleeptalk"]


@hooks.hook("call_move")
def _sleep_talk(st, ctx):
    s, slot = ctx.s, ctx.slot
    talk = ctx.mid == SLEEP_TALK
    mids = _i(st.moves[s, act(st, s)])
    ok = ~NO_SLEEP_TALK[mids]
    # onTry: the user must (still) be asleep; onHit: this.sample() over the callable moves (any PP), or fail
    asleep = st.status[s, slot] == SLP
    if ctx.cfg.forced:
        pick = jnp.argmax(ok)  # the harness's sample() takes the first
    else:
        u = jax.random.uniform(jax.random.fold_in(ctx.key, 701), (4,))
        pick = jnp.argmax(jnp.where(ok, u, -1.0))
    can = talk & asleep & jnp.any(ok)
    called = jnp.where(can, mids[pick], 0)
    return st, called, talk & ~can
