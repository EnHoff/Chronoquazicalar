"""
L2 scenarios for the slot_conditions family (Wish, Healing Wish, Lunar Dance),
played in OU-JAX and in Showdown with forced outcomes and compared after every step (see
engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

CLEF = mon("Clefable", "Leftovers", "Magic Guard", ["Wish", "Healing Wish", "Protect", "Moonblast"], evs=WALL,
           nature="Bold", tera="Fairy")  # fmt: skip
CRESS = mon("Cresselia", "Leftovers", "Levitate", ["Lunar Dance", "Moonblast", "Thunder Wave", "Psychic"], evs=WALL,
            nature="Bold", tera="Fairy")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
GAMBIT_SD = mon("Kingambit", "Leftovers", "Pressure", ["Kowtow Cleave", "Iron Head", "Swords Dance", "Protect"],
                tera="Dark")  # fmt: skip
CORV = mon("Corviknight", "Leftovers", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestSlotConditions(MirrorScenarios, unittest.TestCase):
    def test_wish_and_healing_wish(self):
        # Wish heals at the end of the next turn (through Protect); Healing Wish waits through a healthy
        # replacement and a healthy switch-in, then heals the damaged Great Tusk on its way in
        script = [(0, 0), (0, 1), (2, 1), (1, 2), (0, 0), (9, 1), (9, 2)]
        self.play(team(CLEF, TUSK, PEX), team(GAMBIT_SD, CORV), script)

    def test_lunar_dance_restores_pp(self):
        # Great Tusk spent PP into Pressure: Lunar Dance's replacement needs it even at full HP
        self.play(team(CRESS, TUSK), team(GAMBIT_SD), [(1, 0), (1, 2), (8, 2), (0, 3), (0, 2)])

    def test_lunar_dance_fails_without_a_bench(self):
        self.play(team(CRESS), team(GAMBIT_SD), [(0, 0), (0, 2), (1, 2)])


if __name__ == "__main__":
    unittest.main()
