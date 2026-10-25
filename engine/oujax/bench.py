"""
engine/oujax/bench.py
Throughput benchmark for OU-JAX: B battles in parallel (vmap) driven by a uniformly random legal policy inside
lax.scan, with finished battles reset from a bank of pool teams. Reports JIT time, steady-state steps/s and
peak GPU memory.

    python -m engine.oujax.bench --batch 1024 4096 16384 --steps 200
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from engine.oujax.engine import Config, legal_actions, step
from engine.oujax.state import init_battle
from engine.oujax.team import load_packed_team

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "ou" / "generated" / "team_pool.jsonl"


def team_bank(n: int):
    """n initial battles (pool teams paired at random), stacked along a leading axis."""
    with open(POOL) as f:
        lines = [next(f) for _ in range(2 * n)]
    teams = [load_packed_team(json.loads(line)["packed"]) for line in lines]
    battles = [init_battle(teams[2 * i], teams[2 * i + 1]) for i in range(n)]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *battles)


def random_actions(key, st):
    mask = legal_actions(st)
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(key, logits, axis=-1)


def make_rollout(cfg: Config, bank_size: int):
    def one_step(st, key, bank):
        k_act, k_step, k_reset = jax.random.split(key, 3)
        nxt = step(st, random_actions(k_act, st), k_step, cfg)
        # Finished battles restart from a random bank entry
        idx = jax.random.randint(k_reset, (), 0, bank_size)
        fresh = jax.tree.map(lambda b: b[idx], bank)
        done = nxt.winner >= 0
        return jax.tree.map(lambda a, b: jnp.where(done, b, a), nxt, fresh), done

    def rollout(states, bank, keys):
        # keys: (n_steps, batch, 2)
        def body(st, step_keys):
            st, done = jax.vmap(one_step, in_axes=(0, 0, None))(st, step_keys, bank)
            return st, jnp.sum(done)

        states, finished = jax.lax.scan(body, states, keys)
        return states, jnp.sum(finished)

    return jax.jit(rollout)


def bench(batch: int, n_steps: int, bank_size: int = 256, cfg: Config = Config()):
    bank = team_bank(bank_size)
    reps = -(-batch // bank_size)
    states = jax.tree.map(lambda b: jnp.tile(b, (reps,) + (1,) * (b.ndim - 1))[:batch], bank)
    rollout = make_rollout(cfg, bank_size)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_steps * batch).reshape(n_steps, batch, 2)
    t0 = time.perf_counter()
    out, finished = rollout(states, bank, keys)
    jax.block_until_ready(out)
    jit_and_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    out, finished = rollout(out, bank, keys)
    jax.block_until_ready(out)
    run = time.perf_counter() - t0
    mem = None
    try:
        mem = jax.devices()[0].memory_stats().get("peak_bytes_in_use")
    except Exception:
        pass
    return {
        "batch": batch,
        "steps": n_steps,
        "jit_plus_first_run_s": round(jit_and_first, 1),
        "run_s": round(run, 2),
        "steps_per_s": round(batch * n_steps / run),
        "games_finished": int(finished),
        "peak_gpu_mb": round(mem / 2**20) if mem else None,
        "device": str(jax.devices()[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1024, 4096, 16384])
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    for b in args.batch:
        print(json.dumps(bench(b, args.steps)), flush=True)


if __name__ == "__main__":
    main()
