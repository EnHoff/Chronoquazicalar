"""
L2 scenarios for the self_ko family (Explosion, Misty Explosion, Memento, Final Gambit, Pain Split), played in
OU-JAX and in Showdown with forced outcomes and compared after every step (see engine/oujax/test_l2.py for the
action encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

WEEZ = mon("Weezing-Galar", "Black Sludge", "Misty Surge", ["Misty Explosion", "Will-O-Wisp", "Protect", "Pain Split"],
           evs=WALL, nature="Bold", tera="Fairy")  # fmt: skip
GLIM = mon("Glimmora", "Focus Sash", "Toxic Debris", ["Explosion", "Memento", "Protect", "Spikes"], tera="Ghost")  # fmt: skip
TUSK_FG = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Final Gambit", "Rapid Spin", "Headlong Rush", "Memento"],
              tera="Ground")  # fmt: skip
GHOLDENGO = mon("Gholdengo", "Leftovers", "Good as Gold", ["Nasty Plot", "Shadow Ball", "Protect", "Recover"],
                evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
GAMBIT_SUB = mon("Kingambit", "Leftovers", "Pressure", ["Substitute", "Kowtow Cleave", "Swords Dance", "Protect"],
                 tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestSelfKO(MirrorScenarios, unittest.TestCase):
    def test_explosions_final_gambit_pain_split(self):
        # Pain Split into Protect, then for real; Misty Explosion is boosted on Misty Terrain; Explosion faints
        # its user even into Protect; Final Gambit into a Ghost fails without fainting the user, then deals the
        # user's HP, and the user faints first (the opponent wins)
        script = [(0, 0), (3, 3), (1, 1), (3, 2), (0, 2), (0, 3), (2, 8), (0, 0), (0, 8)]
        self.play(team(WEEZ, GLIM, TUSK_FG), team(GAMBIT_SUB, GHOLDENGO), script)

    def test_self_destruct(self):
        # Self-Destruct faints its user even though the target survives it
        boom = mon("Glimmora", "Leftovers", "Toxic Debris", ["Self-Destruct", "Spikes", "Protect", "Earth Power"],
                   tera="Ghost")  # fmt: skip
        self.play(team(boom, TUSK_FG), team(GAMBIT_SUB, GHOLDENGO), [(0, 0), (0, 2), (1, 2)])

    def test_memento_and_explosion_into_substitute(self):
        # Memento fails into a Substitute and into Protect (the user stays); Explosion breaks the Substitute;
        # Memento that lands faints the user
        script = [(0, 0), (3, 0), (1, 2), (1, 3), (0, 1), (3, 2)]
        self.play(team(GLIM, TUSK_FG), team(GAMBIT_SUB, GHOLDENGO), script)


if __name__ == "__main__":
    unittest.main()
