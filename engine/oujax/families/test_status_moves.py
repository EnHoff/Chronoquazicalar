"""
L2 scenarios for the status_moves family (Focus Energy, Imprison, Stockpile, Spite, Speed Swap, Jungle Healing,
Lunar Blessing), played in OU-JAX and in Showdown with forced outcomes and compared after every step,
including the legal moves (Imprison) and turn order (Speed Swap).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, PEX, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

SLOWKING = mon("Slowking", "Leftovers", "Frisk", ["Imprison", "Spite", "Speed Swap", "Swords Dance"], evs=WALL,
               nature="Bold", tera="Water")  # fmt: skip
LILLIGANT = mon("Lilligant", "Leftovers", "Frisk", ["Jungle Healing", "Lunar Blessing", "Stockpile", "Focus Energy"],
                evs=WALL, nature="Bold", tera="Grass")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Iron Head", "Swords Dance", "Sucker Punch", "Kowtow Cleave"],
             evs=WALL, nature="Impish", tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestStatusMoves(MirrorScenarios, unittest.TestCase):
    def test_imprison_spite_speed_swap(self):
        # Imprison disables Swords Dance for Kingambit until Slowking leaves; Spite takes 4 PP from the last
        # move; Speed Swap flips the turn order until Slowking switches out
        script = [(0, 0), (0, 1), (1, 0), (2, 2), (1, 0), (8, 0), (0, 1)]
        self.play(team(SLOWKING, LILLIGANT), team(GAMBIT, CORV), script)

    def test_heal_cure_stockpile_focus_energy(self):
        # Jungle Healing fails with nothing to do, then heals and cures poison; Lunar Blessing heals; Stockpile
        # stacks to 3 and fails after; Focus Energy can't be used twice
        script = [(0, 0), (0, 1), (8, 0), (8, 2), (0, 2), (1, 2), (2, 2), (2, 2), (2, 2), (2, 2), (3, 2), (3, 2)]
        self.play(team(LILLIGANT, CORV), team(PEX, GAMBIT), script)


if __name__ == "__main__":
    unittest.main()
