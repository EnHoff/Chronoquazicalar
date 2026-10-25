"""
engine/oujax/stats_check.py
L3b statistical verification: with real randomness on both sides (OU-JAX keys vs Showdown PRNG seeds), play
the same teams and the same fixed actions many times and compare outcome distributions: HP of each team slot
(two-sample KS test), and status, KO and request outcomes (chi-square). OU-JAX does not reproduce Showdown's
random sequence, only its probabilities, so this is the check for chance-dependent mechanics.

    python -m engine.oujax.stats_check --scenarios 20 --samples 200
"""

import argparse
import json
from collections import Counter
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sps

from engine.oujax.bridge import Mirror
from engine.oujax.coverage import build_covered_teams
from engine.oujax.engine import Config, step
from engine.oujax.ref import RefHarness
from engine.oujax.state import PHASE_SWITCH, init_battle
from engine.oujax.tables import STATUS
from engine.oujax.team import load_packed_team

ALPHA = 0.01


def scenario_actions(legal: np.ndarray, rng: np.random.Generator) -> List[int]:
    """A fixed choice per side for one decision: a random legal move action (no Tera, no switch)."""
    acts = []
    for s in range(2):
        moves = [a for a in np.flatnonzero(legal[s]) if a < 4]
        acts.append(int(rng.choice(moves)) if moves else int(np.flatnonzero(legal[s])[0]))
    return acts


def outcome_from_ours(st, s, slot) -> Dict:
    return {"hp": int(st.hp[s, slot]), "status": STATUS[int(st.status[s, slot])] if st.hp[s, slot] > 0 else "fnt"}


def run_scenario(ref: RefHarness, p1: str, p2: str, n_turns: int, n_samples: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    # Fix the action script with a forced mirror, so both engines play exactly these choices every sample
    m = Mirror.start(ref, p1, p2, Config(forced=True))
    script = [[0, 0]]
    m.play([0, 0])
    for _ in range(n_turns):
        if int(m.ours.phase) == PHASE_SWITCH or m.sd["ended"]:
            break
        acts = scenario_actions(m.legal(), rng)
        script.append(acts)
        m.play(acts)
    names, packed = m.names, m.packed
    m.end()

    # Showdown with real randomness
    sd_out = []
    # Independent 16-bit words for Showdown's Gen 5 PRNG: seeds differing in a single word give correlated
    # draws (e.g. a 10% burn that never procs), so every word is random
    seed_rng = np.random.default_rng([seed, 1])
    for i in range(n_samples):
        battle, _ = ref.start(packed[0], packed[1], seed=[int(x) for x in seed_rng.integers(0, 65536, 4)])
        try:
            state = None
            for acts in script:
                choices = []
                for s, a in enumerate(acts):
                    choices.append(f"team {a + 1}" if state is None else f"move {a % 4 + 1}")
                state = ref.choose(battle, p1=choices[0], p2=choices[1])
                req = state["requests"]
                if state["ended"] or any(r and r.get("forceSwitch") for r in req.values()):
                    break
            sd_out.append(state)
        finally:
            ref.end(battle)

    # OU-JAX with real randomness, all samples at once
    st0 = init_battle(load_packed_team(packed[0]), load_packed_team(packed[1]))
    batch = jax.tree.map(lambda x: jnp.broadcast_to(x, (n_samples,) + x.shape), st0)
    vstep = jax.jit(jax.vmap(step, in_axes=(0, None, 0, None)), static_argnums=3)
    keys = jax.random.split(jax.random.PRNGKey(seed), (len(script), n_samples))
    alive_turns = np.ones(n_samples, bool)
    for t, acts in enumerate(script):
        nxt = vstep(batch, jnp.asarray(acts, jnp.int32), keys[t], Config())
        # Stop advancing samples that reached a switch request or ended, like the Showdown loop
        stop = np.asarray((nxt.phase == PHASE_SWITCH) | (nxt.winner >= 0))
        keep = jnp.asarray(alive_turns)
        batch = jax.tree.map(lambda a, b: jnp.where(keep.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), nxt, batch)
        alive_turns &= ~stop

    # Compare per team slot
    results = []
    for s in range(2):
        for slot, name in enumerate(names[s]):
            sd_hp = [next(p for p in st["sides"][s]["pokemon"] if p["name"] == name)["hp"] for st in sd_out]
            sd_status = [
                (lambda p: "fnt" if p["hp"] == 0 else p["status"])(
                    next(p for p in st["sides"][s]["pokemon"] if p["name"] == name)
                )
                for st in sd_out
            ]
            our_hp = np.asarray(batch.hp[:, s, slot]).tolist()
            our_status = [
                "fnt" if h == 0 else STATUS[int(x)] for h, x in zip(our_hp, np.asarray(batch.status[:, s, slot]))
            ]
            if len(set(sd_hp) | set(our_hp)) > 1:
                p = sps.ks_2samp(sd_hp, our_hp).pvalue
                results.append(
                    {"what": f"p{s + 1}.{name}.hp", "p": p, "sd_mean": np.mean(sd_hp), "ours_mean": np.mean(our_hp)}
                )
            cats = sorted(set(sd_status) | set(our_status))
            if len(cats) > 1:
                table = [[Counter(sd_status)[c] for c in cats], [Counter(our_status)[c] for c in cats]]
                p = sps.chi2_contingency(table).pvalue
                results.append(
                    {
                        "what": f"p{s + 1}.{name}.status",
                        "p": p,
                        "sd": dict(Counter(sd_status)),
                        "ours": dict(Counter(our_status)),
                    }
                )
    return {"script": script, "tests": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)
    teams = build_covered_teams(2 * args.scenarios, seed=args.seed)
    all_tests = []
    with RefHarness() as ref:
        for i in range(args.scenarios):
            res = run_scenario(ref, teams[2 * i], teams[2 * i + 1], args.turns, args.samples, seed=args.seed + i)
            for t in res["tests"]:
                t["scenario"] = i
            all_tests += res["tests"]
            worst = min((t["p"] for t in res["tests"]), default=1.0)
            print(f"scenario {i}: {len(res['tests'])} tests, min p = {worst:.4f}, script {res['script']}", flush=True)
    n = max(len(all_tests), 1)
    threshold = ALPHA / n  # Bonferroni
    failed = [t for t in all_tests if t["p"] < threshold]
    summary = {
        "tests": len(all_tests),
        "bonferroni_threshold": threshold,
        "failed": failed,
        "min_p": min((t["p"] for t in all_tests), default=1.0),
    }
    print(json.dumps(summary, indent=1, default=float))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "tests": all_tests}, f, indent=1, default=float)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
