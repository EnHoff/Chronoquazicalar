"""
evaluation/eval_ou_policy.py
Plays a training/ou_ppo.py checkpoint (bot/ou_policy_player.py, no search) against poke-env baselines on the local
Showdown server (scripts/start_showdown.sh). The bot plays covered pool teams; the baseline plays the M1 eval teams.

    python -m evaluation.eval_ou_policy --checkpoint checkpoints/ou_ppo/latest.pkl --opponent heuristic --games 50
"""

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path

from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.teambuilder import Teambuilder

from bot.ou_policy_player import OUPolicyPlayer
from training.ou_ppo import covered_pool

REPO = Path(__file__).resolve().parents[1]
EVAL_TEAMS = REPO / "data" / "ou" / "generated" / "eval_teams"
OPPONENTS = {"random": RandomPlayer, "maxdamage": MaxBasePowerPlayer, "heuristic": SimpleHeuristicsPlayer}


class Rotating(Teambuilder):
    def __init__(self, teams, seed):
        self.teams = teams
        self.rng = random.Random(seed)

    def yield_team(self) -> str:
        t = self.rng.choice(self.teams)
        return t if "|" in t.split("\n")[0] else self.join_team(self.parse_showdown_team(t))


def _account(prefix):
    return AccountConfiguration(f"{prefix}{uuid.uuid4().hex[:6]}", None)


async def run(checkpoint, opponent, games, seed, greedy, game_timeout=600):
    pool = [p for p, _ in covered_pool()]
    eval_teams = [p.read_text() for p in sorted(EVAL_TEAMS.glob("*.txt"))]
    bot = OUPolicyPlayer(
        checkpoint, greedy=greedy, seed=seed, account_configuration=_account("oubot"),
        server_configuration=LocalhostServerConfiguration, battle_format="gen9ou", team=Rotating(pool, seed),
        max_concurrent_battles=1,
    )  # fmt: skip
    opp = OPPONENTS[opponent](
        account_configuration=_account(opponent[:4]), server_configuration=LocalhostServerConfiguration,
        battle_format="gen9ou", team=Rotating(eval_teams, seed + 1), max_concurrent_battles=1,
    )  # fmt: skip
    t0 = time.time()
    timeouts = 0
    for i in range(games):
        try:
            await asyncio.wait_for(bot.battle_against(opp, n_battles=1), timeout=game_timeout)
        except asyncio.TimeoutError:
            timeouts += 1
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{games}: {bot.n_won_battles} wins", flush=True)
    finished = [b for b in bot.battles.values() if b.finished]
    return {
        "checkpoint": checkpoint, "opponent": opponent, "games": games, "finished": len(finished),
        "wins": bot.n_won_battles, "win_rate": round(bot.n_won_battles / max(len(finished), 1), 3),
        "timeouts": timeouts, "mean_turns": round(sum(b.turn for b in finished) / max(len(finished), 1), 1),
        "seconds": round(time.time() - t0),
    }  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "ou_ppo" / "latest.pkl"))
    ap.add_argument("--opponent", choices=sorted(OPPONENTS), default="heuristic")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="sample from the policy instead of taking its argmax")
    ap.add_argument("--out", default=None, help="append the result as a JSON line to this file")
    args = ap.parse_args(argv)
    res = asyncio.run(run(args.checkpoint, args.opponent, args.games, args.seed, not args.sample))
    n, w = res["finished"], res["wins"]
    p = w / max(n, 1)
    # 95% Wilson score interval
    z = 1.96
    den = 1 + z * z / max(n, 1)
    mid = (p + z * z / (2 * max(n, 1))) / den
    half = z * (p * (1 - p) / max(n, 1) + z * z / (4 * max(n, 1) ** 2)) ** 0.5 / den
    res["ci95"] = [round(mid - half, 3), round(mid + half, 3)]
    print(json.dumps(res))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
