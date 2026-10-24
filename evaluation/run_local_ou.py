"""
evaluation/run_local_ou.py
Plays N gen9ou games of ChronosPlayer against poke-env baselines on a local Showdown server
(see docs/SETUP.md / scripts/start_showdown.sh), counting wins, Python exceptions in our
decision code, Showdown error / invalid-choice messages and per-game timeouts.

Both sides use the same team file by default; --rotate-teams instead cycles both sides through a
directory of team exports (by default the eval teams built by `python -m data.ou.build`), each side in
its own shuffled order, and reports results per team.

Usage:
    python -m evaluation.run_local_ou --games 100 --opponent random
    python -m evaluation.run_local_ou --games 100 --opponent heuristic --team teams/sample_ou.txt
    python -m evaluation.run_local_ou --games 20 --opponent heuristic --rotate-teams
"""

import argparse
import asyncio
import json
import logging
import random
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.teambuilder import Teambuilder

from bot_client import ChronosPlayer, load_team

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEAM = REPO_ROOT / "teams" / "sample_ou.txt"
DEFAULT_EVAL_TEAMS = REPO_ROOT / "data" / "ou" / "generated" / "eval_teams"
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


class RotatingTeambuilder(Teambuilder):
    """Cycles through named team exports in a seeded shuffled order, remembering which team each
    battle used (battles run one at a time, so history[i] is the team of the i-th battle)."""

    def __init__(self, teams: List[Tuple[str, str]], seed: int):
        self.teams = [(name, self.join_team(self.parse_showdown_team(text))) for name, text in teams]
        self.order = list(range(len(self.teams)))
        random.Random(seed).shuffle(self.order)
        self.history: List[str] = []

    def yield_team(self) -> str:
        name, packed = self.teams[self.order[len(self.history) % len(self.order)]]
        self.history.append(name)
        return packed


def load_team_dir(team_dir: Path) -> List[Tuple[str, str]]:
    teams = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(team_dir.glob("*.txt"))]
    if not teams:
        raise FileNotFoundError(f"no .txt team exports in {team_dir} (run python -m data.ou.build)")
    return teams


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
    rotate_dir: Optional[Path] = None,
) -> Dict:
    issues = IssueCounter()
    my_team: Union[str, RotatingTeambuilder]
    opp_team: Union[str, RotatingTeambuilder]
    if rotate_dir:
        teams = load_team_dir(rotate_dir)
        my_team, opp_team = RotatingTeambuilder(teams, seed=1), RotatingTeambuilder(teams, seed=2)
    else:
        my_team = opp_team = load_team(str(team_path))
    chronos = InstrumentedChronosPlayer(
        account_configuration=_account("chronos"),
        server_configuration=LocalhostServerConfiguration,
        battle_format=battle_format,
        team=my_team,
        search_time_budget=search_time_budget,
        max_concurrent_battles=1,
        log_level=PS_ERROR_LEVEL,
    )
    opp = OPPONENTS[opponent](
        account_configuration=_account(f"{opponent[:4]}"),
        server_configuration=LocalhostServerConfiguration,
        battle_format=battle_format,
        team=opp_team,
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
    if isinstance(my_team, RotatingTeambuilder):
        per_team: Dict[str, Dict[str, int]] = {}
        for name, battle in zip(my_team.history, chronos.battles.values()):
            rec = per_team.setdefault(name, {"games": 0, "wins": 0})
            rec["games"] += 1
            rec["wins"] += int(bool(battle.won))
        result["team_results"] = dict(sorted(per_team.items()))
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play local gen9ou games vs poke-env baselines")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opponent", choices=sorted(OPPONENTS), default="random")
    parser.add_argument("--team", type=Path, default=DEFAULT_TEAM, help="Team for both sides (single-team mode)")
    parser.add_argument(
        "--rotate-teams",
        type=Path,
        nargs="?",
        const=DEFAULT_EVAL_TEAMS,
        metavar="DIR",
        help=f"Rotate both sides through the .txt teams in DIR (default {DEFAULT_EVAL_TEAMS.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--format", type=str, default="gen9ou")
    parser.add_argument("--game-timeout", type=float, default=300.0)
    parser.add_argument("--search-time", type=float, default=1.0)
    parser.add_argument("--out", type=Path, help="Write the JSON summary here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    result = asyncio.run(
        run(
            args.games,
            args.opponent,
            args.team,
            args.format,
            args.game_timeout,
            args.search_time,
            rotate_dir=args.rotate_teams,
        )
    )
    summary = json.dumps(result, indent=2)
    print(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summary + "\n", encoding="utf-8")

    clean = result["exceptions"] == 0 and result["invalid_choices"] == 0 and result["timeouts"] == 0
    return 0 if clean and result["finished"] == args.games else 1


if __name__ == "__main__":
    raise SystemExit(main())
