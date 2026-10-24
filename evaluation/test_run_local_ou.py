"""Unit tests for the team rotation in evaluation/run_local_ou.py (no server needed)."""

import tempfile
import unittest
from pathlib import Path

from evaluation.run_local_ou import RotatingTeambuilder, load_team_dir

TEAM = """Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Knock Off
"""


class TestRotatingTeambuilder(unittest.TestCase):
    def test_cycles_through_every_team(self):
        teams = [(f"t{i}.txt", TEAM.replace("Knock Off", m)) for i, m in enumerate(["Knock Off", "Close Combat"])]
        tb = RotatingTeambuilder(teams, seed=0)
        packed = [tb.yield_team() for _ in range(4)]
        self.assertEqual(sorted(tb.history[:2]), ["t0.txt", "t1.txt"])
        self.assertEqual(tb.history[:2], tb.history[2:])
        self.assertTrue(all(p.startswith("Great Tusk|") for p in packed))
        self.assertNotEqual(packed[0], packed[1])

    def test_load_team_dir(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text(TEAM)
            (Path(d) / "index.json").write_text("[]")
            self.assertEqual([n for n, _ in load_team_dir(Path(d))], ["a.txt"])
            with self.assertRaises(FileNotFoundError):
                load_team_dir(Path(d) / "missing")


if __name__ == "__main__":
    unittest.main()
