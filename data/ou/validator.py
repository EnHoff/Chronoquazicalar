"""
data/ou/validator.py
Python wrapper around validate_teams.js: validates teams with Showdown's TeamValidator in a few
parallel Node processes (each handles a whole batch in-process, not one CLI call per team).
"""

import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT = Path(__file__).resolve().parent / "validate_teams.js"


def find_node() -> str:
    """Native Linux node: PATH first (ignoring the Windows one under /mnt), then nvm installs."""
    node = shutil.which("node")
    if node and not node.startswith("/mnt/"):
        return node
    candidates = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError("native node not found; install Node 22 via nvm (see docs/SETUP.md)")


def showdown_commit(showdown_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(showdown_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _run_batch(node: str, showdown_dir: Path, fmt: str, batch: List[Dict]) -> List[Dict]:
    with tempfile.TemporaryFile("w+", encoding="utf-8") as inp:
        for item in batch:
            inp.write(json.dumps(item) + "\n")
        inp.seek(0)
        proc = subprocess.run(
            [node, str(SCRIPT), str(showdown_dir), fmt], stdin=inp, capture_output=True, text=True, check=True
        )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def validate_teams(
    teams: Dict[str, str],
    showdown_dir: Path,
    fmt: str = "gen9ou",
    workers: int = 4,
    node: Optional[str] = None,
) -> Dict[str, Dict]:
    """Validates {team id: export text}; returns {team id: result dict from validate_teams.js}."""
    node = node or find_node()
    items = [{"id": k, "team": v} for k, v in teams.items()]
    if not items:
        return {}
    size = max(1, -(-len(items) // workers))
    batches = [items[i : i + size] for i in range(0, len(items), size)]
    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch_results in pool.map(lambda b: _run_batch(node, showdown_dir, fmt, b), batches):
            for r in batch_results:
                results[r["id"]] = r
    return results


def problem_kind(problem: str) -> str:
    """Collapses a validator message into a countable reason, e.g. 'X is banned' -> 'banned'."""
    p = problem.lower()
    for key, kind in [
        ("is banned", "banned"),
        ("which is banned", "banned"),
        ("can't learn", "illegal move"),
        ("can only learn", "illegal move"),
        ("is not available in", "not in format"),
        ("does not exist", "unknown name"),
        ("is not a real", "unknown name"),
        ("ability", "illegal ability"),
        ("evs", "illegal EVs"),
        ("tera", "illegal tera"),
        ("species clause", "species clause"),
        ("item clause", "item clause"),
        ("import failed", "unparseable"),
    ]:
        if key in p:
            return kind
    return "other"
