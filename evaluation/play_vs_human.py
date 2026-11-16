"""
evaluation/play_vs_human.py
Puts an OU-JAX policy checkpoint on the local Showdown server as a bot that accepts [Gen 9] OU challenges from
anyone, one game at a time, for as long as it runs. Each game it plays a random covered pool team.

    nohup scripts/play_vs_human.sh > logs/play_vs_human.log 2>&1 &
    python -m evaluation.play_vs_human --checkpoint checkpoints/ou_ppo/policy_005032.pkl --name OUBot
"""

import argparse
import asyncio
import random
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from bot.ou_policy_player import OUPolicyPlayer
from evaluation.eval_ou_policy import Rotating
from training.ou_ppo import covered_pool

REPO = Path(__file__).resolve().parents[1]


async def serve(checkpoint: str, name: str, seed: int):
    pool = [p for p, _ in covered_pool()]
    bot = OUPolicyPlayer(
        checkpoint, greedy=True, seed=seed, account_configuration=AccountConfiguration(name, None),
        server_configuration=LocalhostServerConfiguration, battle_format="gen9ou",
        team=Rotating(pool, seed), max_concurrent_battles=1,
    )  # fmt: skip
    print(f"{name} is online with {checkpoint}: challenge it to [Gen 9] OU", flush=True)
    games = 0
    while True:
        await bot.accept_challenges(opponent=None, n_challenges=1)
        games += 1
        last = list(bot.battles.values())[-1]
        print(f"game {games}: {'won' if last.won else 'lost'} in {last.turn} turns "
              f"({bot.n_won_battles}/{games} won)", flush=True)  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "ou_ppo" / "policy_005032.pkl"))
    ap.add_argument("--name", default="OUBot")
    ap.add_argument("--seed", type=int, default=random.randrange(1 << 30))
    args = ap.parse_args(argv)
    asyncio.run(serve(args.checkpoint, args.name, args.seed))


if __name__ == "__main__":
    main()
