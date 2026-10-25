"""
L3 rollout tests against Showdown on M2-covered teams (teams built from pool sets whose every move, item and
ability OU-JAX implements):

- L3a forced outcomes: full battles with a random legal policy, identical choices on both sides, must match
  Showdown turn by turn (HP, status, boosts, items, PP, Tera, field, side conditions, requests, legal actions).
- L3b statistics: with real randomness, outcome distributions from the same states and actions must agree
  (KS / chi-square, Bonferroni-corrected).

These run a modest number of battles; `python -m engine.oujax.stats_check` and the L3a runner in
docs/OUJAX.md do the full-size runs.
"""

import unittest

from engine.oujax.bridge import play_mirrored
from engine.oujax.coverage import POOL, build_covered_teams
from engine.oujax.engine import Config
from engine.oujax.ref import RefHarness, harness_available
from engine.oujax.stats_check import ALPHA, run_scenario

N_FORCED = 8
N_STAT_SCENARIOS = 4
N_STAT_SAMPLES = 150


@unittest.skipUnless(harness_available() and POOL.exists(), "needs the reference harness and M1 team pool")
class TestL3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = RefHarness()

    @classmethod
    def tearDownClass(cls):
        cls.ref.close()

    def test_forced_battles_match_showdown(self):
        teams = build_covered_teams(2 * N_FORCED, seed=12345)
        for g in range(N_FORCED):
            steps, diffs, log = play_mirrored(self.ref, teams[2 * g], teams[2 * g + 1], Config(forced=True), seed=g)
            tail = "\n".join(line for line in log if line.startswith(("|move", "|-", "|switch", "|faint")))[-3000:]
            self.assertEqual(diffs, [], f"battle {g} diverged at step {steps}:\n{tail}")

    def test_outcome_distributions_match_showdown(self):
        teams = build_covered_teams(2 * N_STAT_SCENARIOS, seed=54321)
        tests = []
        for i in range(N_STAT_SCENARIOS):
            tests += run_scenario(self.ref, teams[2 * i], teams[2 * i + 1], 3, N_STAT_SAMPLES, seed=500 + i)["tests"]
        threshold = ALPHA / max(len(tests), 1)
        failed = [t for t in tests if t["p"] < threshold]
        self.assertEqual(failed, [], f"{len(failed)}/{len(tests)} distribution tests below p={threshold:.2g}")


if __name__ == "__main__":
    unittest.main()
