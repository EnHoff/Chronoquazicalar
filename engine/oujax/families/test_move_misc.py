"""
L2 scenarios for the later move_misc additions and the moves they interact with (Hyperspace Fury, Focus Punch,
Beat Up, Stone Axe, Burning Bulwark, Double Shock, Sparkling Aria, Beak Blast, Fell Stinger, Sky Attack, Rock
Wrecker), played in OU-JAX and in Showdown with forced outcomes and compared after every step. The earlier
move_misc moves are in test_volatiles.py.
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, PEX, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

HOOPA = mon("Hoopa-Unbound", "Leftovers", "Frisk", ["Hyperspace Fury", "Focus Punch", "Beat Up", "Stone Axe"],
            evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
HOOPA_CONFINED = mon("Hoopa", "Leftovers", "Frisk", ["Hyperspace Fury", "Psychic", "Protect", "Recover"], evs=WALL,
                     nature="Bold", tera="Psychic")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Tackle", "Protect", "Sucker Punch", "Burning Bulwark"],
             evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
ZAPDOS = mon("Zapdos", "Leftovers", "Pressure", ["Double Shock", "Fell Stinger", "Sky Attack", "Rock Wrecker"],
             evs="248 HP / 8 Def / 252 Spe", nature="Jolly", tera="Flying")  # fmt: skip
PRIMARINA = mon("Primarina", "Leftovers", "Torrent", ["Sparkling Aria", "Will-O-Wisp", "Beak Blast", "Thunder Wave"],
                evs=WALL, nature="Bold", tera="Water")  # fmt: skip
KLEAVOR = mon("Kleavor", "Leftovers", "Sheer Force", ["Stone Axe", "X-Scissor", "Protect", "Swords Dance"], evs=WALL,
              nature="Impish", tera="Rock")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestMoveMiscMore(MirrorScenarios, unittest.TestCase):
    def test_hyperspace_fury_focus_punch_beat_up_stone_axe(self):
        # Hyperspace Fury goes through and breaks Protect; Focus Punch fails after being hit (no PP), works when
        # not hit; Beat Up hits once per healthy team member; Stone Axe into Burning Bulwark burns the user, then
        # lays Stealth Rock
        script = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 3), (3, 0)]
        self.play(team(HOOPA, CORV, PEX), team(GAMBIT, CORV), script)

    def test_hyperspace_fury_needs_hoopa_unbound(self):
        self.play(team(HOOPA_CONFINED, CORV), team(GAMBIT, CORV), [(0, 0), (0, 0), (0, 1)])

    def test_double_shock_sparkling_aria_beak_blast_sky_attack(self):
        # Double Shock takes away Electric (so Thunder Wave works and Double Shock can't be used again);
        # Sparkling Aria cures the burn it hits; contact into a charging Beak Blast burns; Sky Attack charges;
        # Rock Wrecker needs a recharge turn
        script = [(0, 0), (0, 3), (0, 1), (2, 0), (0, 0), (1, 2), (3, 0), (0, 0)]
        self.play(team(ZAPDOS, CORV), team(PRIMARINA, PEX), script)

    def test_stone_axe_with_sheer_force(self):
        # Sheer Force boosts Stone Axe (an empty secondary) and drops its Stealth Rock
        self.play(team(KLEAVOR, CORV), team(PEX, CORV), [(0, 0), (0, 1), (0, 1)])


if __name__ == "__main__":
    unittest.main()
