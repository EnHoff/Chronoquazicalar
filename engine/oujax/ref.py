"""
engine/oujax/ref.py
Python bridge to the Node reference harness (tools/ref/ref.js): Showdown's own simulator and
@smogon/calc, used as the test oracle for OU-JAX.

    with RefHarness() as ref:
        ref.stats([packed_team])
        ref.calc(attacker, defender, {"name": "Earthquake"})
        battle, state = ref.start(p1_team, p2_team, forced=True)
        state = ref.choose(battle, p1="move 1", p2="switch 2")
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_JS = REPO_ROOT / "tools" / "ref" / "ref.js"
# Override with SHOWDOWN_DIR (e.g. when working from a git worktree that isn't next to the Showdown clone)
DEFAULT_SHOWDOWN_DIR = Path(os.environ.get("SHOWDOWN_DIR", REPO_ROOT.parent / "pokemon-showdown"))


class RefError(RuntimeError):
    pass


def find_node() -> str:
    node = shutil.which("node")
    if node and not node.startswith("/mnt/"):
        return node
    candidates = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError("native node not found; install Node 22 via nvm (see docs/SETUP.md)")


def harness_available(showdown_dir: Path = DEFAULT_SHOWDOWN_DIR) -> bool:
    """True when node, the built Showdown clone and tools/ref/node_modules are all present."""
    try:
        find_node()
    except FileNotFoundError:
        return False
    return (showdown_dir / "dist" / "sim").exists() and (REF_JS.parent / "node_modules" / "@smogon" / "calc").exists()


class RefHarness:
    def __init__(self, showdown_dir: Path = DEFAULT_SHOWDOWN_DIR):
        self.proc = subprocess.Popen(
            [find_node(), str(REF_JS), str(showdown_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 0

    def request(self, cmd: str, **kwargs) -> Dict[str, Any]:
        self._next_id += 1
        self.proc.stdin.write(json.dumps({"id": self._next_id, "cmd": cmd, **kwargs}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RefError("reference harness exited")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RefError(resp.get("error", "unknown error"))
        return resp

    def pack(self, teams: List[str]) -> List[str]:
        """Showdown export (or packed) teams -> Showdown's packed format."""
        return self.request("pack", teams=teams)["packed"]

    def stats(self, teams: List[str]) -> List[List[Dict]]:
        return self.request("stats", teams=teams)["stats"]

    def calc(self, attacker: Dict, defender: Dict, move: Dict, field: Optional[Dict] = None) -> Dict:
        return self.request("calc", attacker=attacker, defender=defender, move=move, field=field or {})

    def calcs(self, cases: List[Dict]) -> List[Dict]:
        """Many damage calcs in one round trip; each result is {damage, desc} or {error}."""
        return self.request("calcs", cases=cases)["results"]

    def start(
        self, p1: str, p2: str, forced: Any = None, seed: Optional[List[int]] = None, fmt: str = "gen9ou"
    ) -> Tuple[int, Dict]:
        kwargs: Dict[str, Any] = {"p1": p1, "p2": p2, "format": fmt}
        if forced is not None:
            kwargs["forced"] = forced
        if seed is not None:
            kwargs["seed"] = seed
        resp = self.request("start", **kwargs)
        return resp["battle"], resp["state"]

    def choose(self, battle: int, p1: Optional[str] = None, p2: Optional[str] = None) -> Dict:
        return self.request("choose", battle=battle, p1=p1, p2=p2)["state"]

    def end(self, battle: int) -> None:
        self.request("end", battle=battle)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)

    def __enter__(self) -> "RefHarness":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
