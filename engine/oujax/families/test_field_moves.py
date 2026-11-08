"""
L2 scenarios for the field_moves family (Smack Down, Magnet Rise, Gravity, Perish Song), played in OU-JAX and in
Showdown with forced outcomes and compared after every step. Gravity's accuracy boost only shows outside forced
mode (forced accuracy checks ignore the number), so the statistical L3b runs cover it.
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, ZAPDOS, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"

TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Earthquake", "Smack Down", "Gravity", "Spikes"], evs=WALL,
           nature="Impish", tera="Ground")  # fmt: skip
TUSK_HJK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Earthquake", "Gravity", "High Jump Kick", "Spikes"],
               evs=WALL, nature="Impish", tera="Ground")  # fmt: skip
GHOLD = mon("Gholdengo", "Leftovers", "Frisk", ["Magnet Rise", "Shadow Ball", "Recover", "Protect"], evs=WALL,
            nature="Bold", tera="Steel")  # fmt: skip
PRIMARINA = mon("Primarina", "Leftovers", "Torrent", ["Perish Song", "Protect", "Moonblast", "Surf"], evs=SPEC,
                nature="Modest", tera="Water")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Iron Head", "Swords Dance", "Sucker Punch", "Kowtow Cleave"],
             evs=WALL, nature="Impish", tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestFieldMoves(MirrorScenarios, unittest.TestCase):
    def test_smack_down_and_magnet_rise(self):
        # Earthquake misses Corviknight until Smack Down grounds it; the switch-in uses Magnet Rise, which Smack
        # Down ends; Magnet Rise lasts 5 turns and can't be renewed while it lasts
        script = [(0, 0), (0, 1), (1, 3), (0, 1), (0, 8), (0, 0), (0, 0), (0, 2), (1, 2), (0, 2), (2, 0)]
        self.play(team(TUSK, PRIMARINA), team(CORV, GHOLD), script)

    def test_magnet_rise_runs_out(self):
        script = [(0, 0), (0, 0)] + [(0, 2)] * 6
        self.play(team(TUSK, PRIMARINA), team(GHOLD, CORV), script)

    def test_gravity_grounds_everyone_and_disables_moves(self):
        # Gravity grounds Flying types (Earthquake hits) and disables High Jump Kick for 5 turns; afterwards
        # Earthquake and Spikes miss Flying types again
        script = [(0, 0), (3, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 8), (0, 1)]
        self.play(team(TUSK_HJK, PRIMARINA), team(CORV, ZAPDOS), script)

    def test_gravity_stops_magnet_rise(self):
        # Gravity ends Magnet Rise; Smack Down then has nothing to ground
        script = [(0, 0), (0, 0), (2, 2), (0, 2), (1, 2)]
        self.play(team(TUSK, PRIMARINA), team(GHOLD, CORV), script)

    def test_perish_song(self):
        # Both active Pokemon get the count; a second Perish Song fails; switching out clears it; both faint
        # after three more turns (the slower one last)
        script = [(0, 0), (0, 1), (0, 1), (8, 1), (1, 1), (8, 1), (0, 1), (1, 1), (1, 1), (1, 1)]
        self.play(team(PRIMARINA, CORV), team(GAMBIT, ZAPDOS), script)


if __name__ == "__main__":
    unittest.main()
