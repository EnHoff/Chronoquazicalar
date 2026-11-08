"""
L2 scenarios for the held_items family, played in OU-JAX and in Showdown with forced outcomes and compared after
every step (see engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"

WEAVILE = mon("Weavile", "Heavy-Duty Boots", "Pressure", ["Triple Axel", "Knock Off", "Swords Dance", "Protect"],
              evs=PHYS, nature="Jolly", tera="Ice")  # fmt: skip
MAUSHOLD = mon("Maushold", "Loaded Dice", "Technician", ["Population Bomb", "Tidy Up", "Protect", "U-turn"],
               evs=PHYS, nature="Jolly")  # fmt: skip
CORV = mon("Corviknight", "Rocky Helmet", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestHeldItems(MirrorScenarios, unittest.TestCase):
    def test_rocky_helmet_per_hit(self):
        # Rocky Helmet hurts once per contact hit: Triple Axel takes three chunks; Population Bomb's ten hits stop
        # once the attacker faints from the Helmet
        script = [(0, 0), (0, 1), (8, 1), (0, 1)]
        self.play(team(WEAVILE, MAUSHOLD), team(CORV, PEX), script)


if __name__ == "__main__":
    unittest.main()
