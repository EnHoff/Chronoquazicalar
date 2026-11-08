"""
L2 scenarios for the call_moves family (Sleep Talk), played in OU-JAX and in Showdown with forced outcomes and
compared after every step (see engine/oujax/test_l2.py for the action encoding). Forced mode's sample() picks
the first callable move.
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

PEX_TALK = mon("Toxapex", "Black Sludge", "Regenerator", ["Surf", "Sleep Talk", "Rest", "Haze"], evs=WALL,
               nature="Bold", tera="Fairy")  # fmt: skip
SNORLAX = mon("Snorlax", "Leftovers", "Thick Fat", ["Hyper Beam", "Sleep Talk", "Rest", "Curse"], evs=WALL,
              nature="Impish", tera="Ghost")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Iron Head", "Swords Dance", "Sucker Punch", "Kowtow Cleave"],
             tera="Dark")  # fmt: skip
BRELOOM = mon("Breloom", "Leftovers", "Technician", ["Spore", "Mach Punch", "Bullet Seed", "Swords Dance"],
              evs="252 HP / 252 Def / 4 SpD", nature="Impish", tera="Fire")  # fmt: skip
DRAGAPULT = mon("Dragapult", "Leftovers", "Infiltrator", ["Hyper Beam", "Sleep Talk", "Draco Meteor", "U-turn"],
                evs="252 Atk / 4 SpD / 252 Spe", nature="Jolly", tera="Ghost")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestCallMoves(MirrorScenarios, unittest.TestCase):
    def test_sleep_talk_calls_a_move_with_pressure(self):
        # Rest, then Sleep Talk calls Surf into Pressure (Sleep Talk loses the extra PP); Sucker Punch fails
        # against Sleep Talk; once awake, Sleep Talk fails
        script = [(0, 0), (2, 0), (1, 2), (1, 1), (1, 0), (1, 1)]
        self.play(team(PEX_TALK, SNORLAX), team(GAMBIT, CORV), script)

    def test_sleep_talk_hyper_beam_recharge_does_not_tick_sleep(self):
        # Sleep Talk calls Hyper Beam; the recharge turn comes before sleep (BeforeMove 11 > 10), so it doesn't
        # count down the sleep turns
        script = [(0, 0), (2, 0), (1, 1), (0, 1), (1, 1), (0, 1), (1, 1)]
        self.play(team(SNORLAX, PEX_TALK), team(GAMBIT, CORV), script)

    def test_spore_after_hyper_beam(self):
        # Put to sleep after its Hyper Beam: the recharge turn doesn't use up a sleep turn
        script = [(0, 0), (0, 0), (0, 2), (1, 2), (0, 2), (1, 2), (2, 2)]
        self.play(team(DRAGAPULT, CORV), team(BRELOOM, GAMBIT), script)


if __name__ == "__main__":
    unittest.main()
