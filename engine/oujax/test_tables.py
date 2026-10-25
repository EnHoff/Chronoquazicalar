"""
L1 tests for OU-JAX game data tables and team loading: table contents, packed-team parsing, and final
stats checked against Showdown's own calculation for eval teams and a random sample of the M1 team pool.
"""

import json
import random
import unittest
from pathlib import Path

import numpy as np

from engine.oujax.ref import RefHarness, harness_available
from engine.oujax.tables import ACCURACY_ALWAYS, MC, SC, TYPE_NONE, load_tables
from engine.oujax.team import calc_stats, load_packed_team

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "ou" / "generated" / "team_pool.jsonl"
EVAL_TEAMS = REPO / "data" / "ou" / "generated" / "eval_teams"
N_POOL_SAMPLE = 500

T = load_tables()


def move(name, col):
    return int(T.moves[T.move_ids[name], MC[col]])


class TestTables(unittest.TestCase):
    def test_type_chart(self):
        ty = T.type_ids
        self.assertEqual(T.type_chart[ty["ground"], ty["flying"]], 0.0)
        self.assertEqual(T.type_chart[ty["fire"], ty["grass"]], 2.0)
        self.assertEqual(T.type_chart[ty["water"], ty["grass"]], 0.5)
        self.assertEqual(T.type_chart[ty["dragon"], ty["fairy"]], 0.0)
        self.assertEqual(T.type_chart[ty["fighting"], TYPE_NONE], 1.0)

    def test_move_rows(self):
        self.assertEqual(move("closecombat", "bp"), 120)
        self.assertEqual(move("closecombat", "self_boost_def"), -1)
        self.assertEqual(move("knockoff", "has_handlers"), 1)
        self.assertEqual(move("swordsdance", "boost_atk"), 2)
        self.assertEqual(move("swordsdance", "accuracy"), ACCURACY_ALWAYS)
        self.assertEqual(move("scald", "sec_chance"), 30)
        self.assertEqual(move("uturn", "self_switch"), 1)
        self.assertEqual(move("extremespeed", "priority"), 2)
        self.assertEqual(move("recover", "heal_num") / move("recover", "heal_den"), 0.5)
        self.assertEqual(move("bulletseed", "multihit_min"), 2)
        self.assertEqual(move("bulletseed", "multihit_max"), 5)
        self.assertEqual(move("stealthrock", "pp"), 32)  # 20 PP with PP Ups
        self.assertEqual(move("seismictoss", "fixed_damage"), -1)  # damage = user's level

    def test_species_rows(self):
        tusk = T.species[T.species_ids["greattusk"]]
        self.assertEqual(tusk[SC["base_atk"]], 131)
        self.assertEqual({tusk[SC["type1"]], tusk[SC["type2"]]}, {T.type_ids["ground"], T.type_ids["fighting"]})
        self.assertEqual(T.species[T.species_ids["kingambit"], SC["type2"]], T.type_ids["steel"])

    def test_natures(self):
        jolly = T.natures[T.nature_ids["jolly"]]
        self.assertEqual(list(jolly), [10, 10, 10, 9, 10, 11])
        self.assertTrue(np.all(T.natures[T.nature_ids["serious"]] == 10))


class TestPackedTeams(unittest.TestCase):
    def test_parse_defaults_and_codes(self):
        # ability code "H", omitted IVs (31), omitted level (100), Tera in the misc field
        packed = (
            "Great Tusk||BoosterEnergy|0|HeadlongRush,IceSpinner,RapidSpin,KnockOff|Jolly|,252,,,4,252|||||,,,,,Steel"
        )
        t = load_packed_team(packed)
        self.assertEqual(t.size, 1)
        self.assertEqual(t.ability[0], T.ability_ids["protosynthesis"])
        self.assertEqual(list(t.ivs[0]), [31] * 6)
        self.assertEqual(t.level[0], 100)
        self.assertEqual(t.tera_type[0], T.type_ids["steel"])
        self.assertEqual(list(t.stats[0]), [371, 361, 298, 127, 143, 300])
        self.assertEqual(list(t.max_pp[0]), [8, 24, 64, 32])  # base PP 5, 15, 40, 20 with PP Ups
        self.assertTrue(np.all(t.species[1:] == 0))

    def test_one_base_hp_means_one_hp(self):
        # Shedinja (base HP 1) is not in Gen 9, but keep Showdown's rule
        stats = calc_stats(
            np.array([1, 90, 45, 30, 30, 40]), np.zeros(6, int), np.full(6, 31), np.array(100), np.full(6, 10)
        )
        self.assertEqual(stats[0], 1)


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestStatsAgainstShowdown(unittest.TestCase):
    def _teams(self):
        teams = []
        if EVAL_TEAMS.exists():
            teams += [p.read_text() for p in sorted(EVAL_TEAMS.glob("*.txt"))]
        if POOL.exists():
            with open(POOL) as f:
                lines = f.readlines()
            rng = random.Random(0)
            teams += [json.loads(line)["packed"] for line in rng.sample(lines, min(N_POOL_SAMPLE, len(lines)))]
        if not teams:
            self.skipTest("no M1 data (run python -m data.ou.build)")
        return teams

    def test_stats_match_showdown(self):
        teams = self._teams()
        with RefHarness() as ref:
            packed = ref.pack(teams)
            expected = ref.stats(packed)
        n_mons = 0
        for team_packed, want in zip(packed, expected):
            got = load_packed_team(team_packed)
            self.assertEqual(got.size, len(want))
            for i, w in enumerate(want):
                n_mons += 1
                ours = [int(x) for x in got.stats[i]]
                theirs = [w["maxhp"]] + [w["stats"][s] for s in ("atk", "def", "spa", "spd", "spe")]
                self.assertEqual(ours, theirs, f"{w['species']} in {team_packed[:80]}")
        self.assertEqual(n_mons, 6 * len(teams))


if __name__ == "__main__":
    unittest.main()
