"""
L2 scenarios for the held_items family, played in OU-JAX and in Showdown with forced outcomes and compared after
every step (see engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
GLIM_SASH = mon("Glimmora", "Focus Sash", "Corrosion", ["Spikes", "Protect", "Earth Power", "Sludge Bomb"],
                evs="4 HP / 252 SpA / 252 Spe", nature="Timid", tera="Ghost")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestHeldItems(MirrorScenarios, unittest.TestCase):
    def test_focus_sash(self):
        # Great Tusk's Headlong Rush would OHKO Glimmora: the Sash leaves it at 1 HP (consumed); next time it faints.
        # A second Glimmora with a Sash loses it to Stealth Rock damage first, so the Sash can't save it
        script = [(0, 0), (0, 0), (1, 3), (0, 0), (0, 0)]
        self.play(team(GLIM_SASH, GLIM_SASH.replace("Glimmora", "Glim (Glimmora)", 1)), team(TUSK, PEX), script)


if __name__ == "__main__":
    unittest.main()
