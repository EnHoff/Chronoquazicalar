"""
L2 scenarios for the trapping family (partial trapping, Mean Look / Block / Jaw Lock, No Retreat), played in
OU-JAX and in Showdown with forced outcomes (partial trapping lasts 5 turns) and compared after every step,
including the volatiles and which switches each side may make.
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"

HEATRAN = mon("Heatran", "Leftovers", "Frisk", ["Magma Storm", "Earth Power", "Protect", "Stealth Rock"], evs=SPEC,
              nature="Modest", tera="Grass")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Whirlpool", "Infestation", "Recover", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Ice Spinner", "Rapid Spin", "Knock Off", "Bulk Up"],
           evs=WALL, nature="Impish", tera="Ground")  # fmt: skip
DRAGAPULT = mon("Dragapult", "Leftovers", "Infiltrator", ["Shadow Ball", "Draco Meteor", "Fire Spin", "U-turn"],
                evs=SPEC, nature="Modest", tera="Ghost")  # fmt: skip
UMBREON = mon("Umbreon", "Leftovers", "Frisk", ["Mean Look", "Block", "Protect", "Foul Play"], evs=WALL,
              nature="Bold", tera="Poison")  # fmt: skip
TTAR = mon("Tyranitar", "Leftovers", "Frisk", ["Jaw Lock", "No Retreat", "Stone Edge", "Protect"], evs=WALL,
           nature="Impish", tera="Ghost")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Iron Head", "Swords Dance", "Sucker Punch", "Kowtow Cleave"],
             evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
VENUSAUR = mon("Venusaur", "Leftovers", "Chlorophyll", ["Leech Seed", "Giga Drain", "Sludge Bomb", "Synthesis"],
               evs=SPEC, nature="Modest", tera="Grass")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestTrapping(MirrorScenarios, unittest.TestCase):
    def test_magma_storm_traps_until_it_runs_out(self):
        # Magma Storm traps Corviknight (no switching, 1/8 per turn) and isn't renewed; the trap runs out after
        # 4 turns of damage
        script = [(0, 0), (0, 1), (0, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 8)]
        self.play(team(HEATRAN, PEX), team(CORV, TUSK), script)

    def test_partial_trap_ends_when_the_trapper_leaves_and_rapid_spin(self):
        # The trap ends once Toxapex switches out; Great Tusk's Rapid Spin frees itself from Infestation
        script = [(0, 0), (0, 3), (8, 3), (8, 3), (1, 3), (2, 1), (2, 8)]
        self.play(team(PEX, HEATRAN), team(TUSK, CORV), script)

    def test_ghost_types_are_not_trapped(self):
        # Fire Spin damages a Ghost type but doesn't trap it; Mean Look fails on it
        script = [(0, 0), (2, 2), (2, 2), (8, 0), (0, 0), (0, 8)]
        self.play(team(DRAGAPULT, UMBREON), team(DRAGAPULT, CORV), script)

    def test_mean_look_block_jaw_lock_no_retreat(self):
        # Mean Look traps while Umbreon stays in; a second one fails; Jaw Lock traps both; No Retreat traps the
        # user and can't be used twice
        script = [(0, 0), (0, 2), (0, 2), (1, 2), (8, 2), (0, 1), (1, 2), (1, 2), (2, 0)]
        self.play(team(UMBREON, TTAR), team(GAMBIT, CORV), script)

    def test_rapid_spin_removes_leech_seed(self):
        # Great Tusk is faster: its first Rapid Spin has nothing to clear; the second removes Leech Seed
        script = [(0, 0), (0, 1), (2, 1), (2, 1)]
        self.play(team(VENUSAUR, PEX), team(TUSK, CORV), script)


if __name__ == "__main__":
    unittest.main()
