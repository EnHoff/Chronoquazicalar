"""
engine/oujax/l3_check.py
L3a runner: plays N forced-outcome battles on M2-covered teams in OU-JAX and Showdown with a random legal
policy and reports the first divergence of each battle.

    python -m engine.oujax.l3_check --battles 50 --team-seed 0
"""

import argparse
import json
import time

from engine.oujax.bridge import play_mirrored
from engine.oujax.coverage import build_covered_teams
from engine.oujax.engine import Config
from engine.oujax.ref import RefHarness


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--team-seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)
    teams = build_covered_teams(2 * args.battles, seed=args.team_seed)
    t0 = time.time()
    results = []
    with RefHarness() as ref:
        for g in range(args.battles):
            steps, diffs, log = play_mirrored(ref, teams[2 * g], teams[2 * g + 1], Config(forced=True), seed=g)
            results.append({"battle": g, "steps": steps, "diffs": diffs})
            if diffs:
                print(f"battle {g}: diverged at step {steps}", flush=True)
                for d in diffs[:12]:
                    print("   ", d, flush=True)
                tail = [line for line in log if line.startswith(("|move", "|-", "|switch", "|faint", "|turn"))]
                print("    log:", tail[-25:], flush=True)
            else:
                print(f"battle {g}: OK ({steps} steps)", flush=True)
    matched = sum(not r["diffs"] for r in results)
    summary = {
        "battles": args.battles,
        "matched": matched,
        "steps": sum(r["steps"] for r in results),
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "battles": results}, f, indent=1)
    return 0 if matched == args.battles else 1


if __name__ == "__main__":
    raise SystemExit(main())
