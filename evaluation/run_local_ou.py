"""
evaluation/run_local_ou.py
Plays N gen9ou games of ChronosPlayer against poke-env baselines on a local Showdown server
(see docs/SETUP.md / scripts/start_showdown.sh), counting wins, Python exceptions in our
decision code, Showdown error / invalid-choice messages and per-game timeouts.

Usage:
    python -m evaluation.run_local_ou --games 100 --opponent random
    python -m evaluation.run_local_ou --games 100 --opponent heuristic --team teams/sample_ou.txt
"""

import argparse
import asyncio
import json
import logging
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from bot_client import ChronosPlayer, load_team

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEAM = REPO_ROOT / "teams" / "sample_ou.txt"
OPPONENTS = {"random": RandomPlayer, "heuristic": SimpleHeuristicsPlayer}

logger = logging.getLogger("RunLocalOU")


# poke-env logs Showdown |error| messages at this level, between INFO and WARNING
PS_ERROR_LEVEL = 25


class IssueCounter(logging.Handler):
    """Collects Showdown |error| messages and warnings logged by poke-env players."""

    def __init__(self):
        super().__init__(level=PS_ERROR_LEVEL)
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(f"{record.name}: {record.getMessage()}")

    def invalid_choices(self, username: str) -> List[str]:
        return [
            r
            for r in self.records
            if r.startswith(f"{username}:") and ("[Invalid choice]" in r or "[Unavailable choice]" in r)
        ]


class InstrumentedChronosPlayer(ChronosPlayer):
    """ChronosPlayer that records exceptions in its decision code instead of stalling the battle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exceptions: List[str] = []

    def choose_move(self, battle):
        try:
            return super().choose_move(battle)
        except Exception:
            self.exceptions.append(f"{battle.battle_tag} turn {battle.turn}:\n{traceback.format_exc()}")
            return self.choose_random_move(battle)

    def teampreview(self, battle):
        try:
            return super().teampreview(battle)
        except Exception:
            self.exceptions.append(f"{battle.battle_tag} teampreview:\n{traceback.format_exc()}")
            return self.random_teampreview(battle)


def _account(prefix: str) -> AccountConfiguration:
    # Showdown usernames are limited to 18 characters
    return AccountConfiguration(f"{prefix}{uuid.uuid4().hex[:6]}", None)


async def run(
    games: int,
    opponent: str,
    team_path: Path,
    battle_format: str = "gen9ou",
    game_timeout: float = 300.0,
    search_time_budget: float = 1.0,
) -> Dict:
    issues = IssueCounter()
    team = load_team(str(team_path))
    chronos = InstrumentedChronosPlayer(
        account_configuration=_account("chronos"),
        server_configuration=LocalhostServerConfiguration,
        battle_format=battle_format,
        team=team,
        search_time_budget=search_time_budget,
        max_concurrent_battles=1,
        log_level=PS_ERROR_LEVEL,
    )
    opp = OPPONENTS[opponent](
        account_configuration=_account(f"{opponent[:4]}"),
        server_configuration=LocalhostServerConfiguration,
        battle_format=battle_format,
        team=team,
        max_concurrent_battles=1,
        log_level=PS_ERROR_LEVEL,
    )
    # poke-env players log Showdown |error| messages on their own loggers
    for player in (chronos, opp):
        player.logger.addHandler(issues)
        player.logger.propagate = False

    timeouts = 0
    start = time.time()
    for i in range(games):
        try:
            await asyncio.wait_for(chronos.battle_against(opp, n_battles=1), timeout=game_timeout)
        except asyncio.TimeoutError:
            timeouts += 1
            logger.warning(f"game {i + 1} timed out after {game_timeout:.0f}s")
        if (i + 1) % 10 == 0:
            logger.info(
                f"{i + 1}/{games} games: {chronos.n_won_battles} wins, "
                f"{len(chronos.exceptions)} exceptions, {len(issues.records)} warnings, {timeouts} timeouts"
            )

    finished = [b for b in chronos.battles.values() if b.finished]
    turns = [b.turn for b in finished]
    result = {
        "format": battle_format,
        "opponent": opponent,
        "games": games,
        "finished": len(finished),
        "wins": chronos.n_won_battles,
        "losses": chronos.n_lost_battles,
        "ties": chronos.n_tied_battles,
        "win_rate": chronos.n_won_battles / max(len(finished), 1),
        "avg_turns": sum(turns) / max(len(turns), 1),
        "exceptions": len(chronos.exceptions),
        "invalid_choices": len(issues.invalid_choices(chronos.username)),
        "warnings": len(issues.records),
        "timeouts": timeouts,
        "elapsed_sec": round(time.time() - start, 1),
    }
    result["warning_kinds"] = dict(Counter(r.split(": ", 1)[-1][:120] for r in issues.records).most_common(10))
    result["exception_samples"] = chronos.exceptions[:3]
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play local gen9ou games vs poke-env baselines")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="random")
    parser.add_argument("--team", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--format", type=str, default="gen9ou")
    parser.add_argument("--game-timeout", type=float, default=300.0)
    parser.add_argument("--search-time", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="Write the JSON summary here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    result = asyncio.run(run(args.games, args.opponent, args.team, args.format, args.game_timeout, args.search_time))
    summary = json.dumps(result, indent=2)
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary + "\n", encoding="utf-8")

    clean = result["exceptions"] == 0 and result["invalid_choices"] == 0 and result["timeouts"] == 0
    return 0 if clean and result["finished"] == args.games else 1


if __name__ == "__main__":
    raise SystemExit(main())
