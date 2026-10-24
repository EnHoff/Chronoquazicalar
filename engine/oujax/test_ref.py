"""Tests for the reference harness bridge (skipped when node / Showdown / @smogon/calc are missing)."""

import unittest

from engine.oujax.ref import RefError, RefHarness, harness_available

TEAM = """Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Knock Off

Toxapex @ Black Sludge
Ability: Regenerator
Tera Type: Fairy
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 0 Atk
- Surf
- Haze
- Recover
- Toxic Spikes
"""


@unittest.skipUnless(harness_available(), "reference harness not installed (see docs/SETUP.md)")
class TestRefHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = RefHarness()

    @classmethod
    def tearDownClass(cls):
        cls.ref.close()

    def test_stats(self):
        tusk, pex = self.ref.stats([TEAM])[0]
        self.assertEqual(tusk["maxhp"], 371)
        self.assertEqual(tusk["stats"], {"atk": 361, "def": 298, "spa": 127, "spd": 143, "spe": 300})
        self.assertEqual(pex["maxhp"], 304)

    def test_calc_returns_16_rolls(self):
        r = self.ref.calc(
            {"species": "Great Tusk", "nature": "Jolly", "evs": {"atk": 252, "spe": 252}},
            {"species": "Kingambit", "nature": "Adamant", "evs": {"hp": 252}},
            {"name": "Headlong Rush"},
        )
        self.assertEqual(len(r["damage"]), 16)
        self.assertEqual(r["damage"], sorted(r["damage"]))

    def test_forced_battle_is_deterministic(self):
        def play():
            battle, state = self.ref.start(TEAM, TEAM, forced=True)
            self.assertIn("teamPreview", state["requests"]["p1"])
            self.ref.choose(battle, p1="team 12", p2="team 21")
            state = self.ref.choose(battle, p1="move 1", p2="move 1")
            self.ref.end(battle)
            return [(m["species"], m["hp"]) for side in state["sides"] for m in side["pokemon"]]

        first = play()
        self.assertEqual(first, play())
        # Faster Great Tusk's super-effective Headlong Rush KOs p2's Toxapex lead before it moves
        self.assertEqual(dict(first[2:]), {"toxapex": 0, "greattusk": 371})
        self.assertEqual(dict(first[:2]), {"greattusk": 371, "toxapex": 304})

    def test_bad_choice_raises(self):
        battle, _ = self.ref.start(TEAM, TEAM, forced=True)
        with self.assertRaises(RefError):
            self.ref.choose(battle, p1="move 1")  # team preview needs a team choice
        self.ref.end(battle)


if __name__ == "__main__":
    unittest.main()
