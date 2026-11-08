"""
L2 scenarios for the charge_moves family (rampages, two-turn moves, Power Herb, recharge), played in OU-JAX and in
Showdown with forced outcomes and compared after every step. While locked, the only legal action is 0 (Showdown
offers just the locked move or "Recharge"), so the scripts play 0 on those turns.
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"

ARCH = mon("Archaludon", "Leftovers", "Frisk", ["Meteor Beam", "Body Press", "Protect", "Draco Meteor"], evs=SPEC,
           nature="Modest", tera="Steel")  # fmt: skip
CLEF_HERB = mon("Clefable", "Power Herb", "Frisk", ["Meteor Beam", "Moonblast", "Protect", "Moonlight"], evs=SPEC,
                nature="Modest", tera="Water")  # fmt: skip
TORKOAL = mon("Torkoal", "Heat Rock", "Drought", ["Solar Beam", "Heat Wave", "Protect", "Solar Blade"], evs=SPEC,
              nature="Modest", tera="Fire")  # fmt: skip
VENUSAUR = mon("Venusaur", "Leftovers", "Frisk", ["Solar Beam", "Solar Blade", "Sludge Bomb", "Synthesis"], evs=SPEC,
               nature="Modest", tera="Grass")  # fmt: skip
PELIPPER = mon("Pelipper", "Damp Rock", "Drizzle", ["Hurricane", "Surf", "U-turn", "Roost"], evs=WALL, nature="Bold",
               tera="Water")  # fmt: skip
DRAGAPULT = mon("Dragapult", "Leftovers", "Frisk", ["Phantom Force", "Draco Meteor", "Outrage", "Hyper Beam"],
                evs="252 Atk / 4 SpD / 252 Spe", nature="Jolly", tera="Ghost")  # fmt: skip
PEX_BUNKER = mon("Toxapex", "Black Sludge", "Regenerator", ["Baneful Bunker", "Recover", "Surf", "Haze"], evs=WALL,
                 nature="Bold", tera="Fairy")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Supreme Overlord", ["Iron Head", "Swords Dance", "Sucker Punch", "Upper Hand"],
             tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestChargeMoves(MirrorScenarios, unittest.TestCase):
    def test_meteor_beam_and_power_herb(self):
        # Meteor Beam charges (+1 SpA) and fires next turn for one PP; Power Herb skips the charge once
        script = [(0, 0), (0, 1), (0, 1), (8, 1), (0, 1), (0, 1), (0, 1)]
        self.play(team(ARCH, CLEF_HERB), team(CORV, PELIPPER), script)

    def test_solar_beam_in_sun_and_rain(self):
        # Solar Beam fires at once in sun; outside sun it charges, and in rain its power is halved
        script = [(0, 0), (0, 1), (8, 8), (0, 8), (0, 1), (1, 1), (0, 1)]
        self.play(team(TORKOAL, VENUSAUR), team(PELIPPER, CORV), script)

    def test_phantom_force_outrage_hyper_beam(self):
        # Phantom Force dodges Surf while charging and goes through Baneful Bunker; Outrage locks for 2 turns
        # (forced) and confuses; Hyper Beam's recharge turn doesn't tick confusion
        script = [(0, 0), (0, 2), (0, 0), (2, 1), (0, 1), (3, 2), (0, 1), (1, 1)]
        self.play(team(DRAGAPULT, CORV), team(PEX_BUNKER, GAMBIT), script)

    def test_sucker_punch_fails_on_recharge_turn(self):
        script = [(0, 0), (3, 2), (0, 2), (1, 2)]
        self.play(team(DRAGAPULT, CORV), team(GAMBIT, CORV), script)


if __name__ == "__main__":
    unittest.main()
