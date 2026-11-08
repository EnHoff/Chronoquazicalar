"""
L2 scenarios for the herbs_seeds family (White Herb, Mirror Herb, terrain seeds), played in OU-JAX and in Showdown
with forced outcomes and compared after every step (see engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
CORV = mon("Corviknight", "Leftovers", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Parting Shot", "Kowtow Cleave", "Swords Dance", "Protect"],
             tera="Dark")  # fmt: skip


def tusk(item, moves=("Headlong Rush", "Rapid Spin", "Close Combat", "Stealth Rock"), name="Great Tusk"):
    return mon(name, item, "Protosynthesis", list(moves), tera="Ground")


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestHerbsSeeds(MirrorScenarios, unittest.TestCase):
    def test_white_herb(self):
        # Close Combat's drops are reset once the move is over (the herb is used up); Shell Smash's drops later
        # stay; Parting Shot's drops on the second White Herb holder are reset after Parting Shot
        smash = tusk("White Herb", ("Close Combat", "Shell Smash", "Rapid Spin", "Stealth Rock"))
        other = tusk("White Herb", ("Close Combat", "Shell Smash", "Rapid Spin", "Stealth Rock"), "Tusk (Great Tusk)")
        script = [(0, 0), (0, 2), (1, 2), (8, 0), (2, 0)]
        self.play(team(smash, other), team(CORV, GAMBIT), script)

    def test_mirror_herb(self):
        # Swords Dance's +2 Atk is copied after the move (the herb is used up), the next one isn't
        mirror = tusk("Mirror Herb", ("Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"))
        self.play(team(mirror, PEX), team(GAMBIT, CORV), [(0, 0), (3, 2), (3, 2), (1, 0)])

    def test_grassy_seed(self):
        # Rillaboom's Grassy Surge starts the terrain as the leads come in: the Grassy Seed holder gets +1 Def; a
        # second holder that switches in on the terrain gets it too
        rilla = mon("Rillaboom", "Leftovers", "Grassy Surge", ["Grassy Glide", "Wood Hammer", "U-turn", "Knock Off"],
                    tera="Grass")  # fmt: skip
        seed = tusk("Grassy Seed")
        seed2 = tusk("Grassy Seed", name="Tusk (Great Tusk)")
        self.play(team(seed, seed2), team(rilla, CORV), [(0, 0), (3, 0), (8, 0), (1, 0)])

    def test_electric_and_psychic_seeds(self):
        # Electric Terrain set by a move activates the Electric Seed of the Pokemon already in; the Psychic Seed
        # waits for Psychic Terrain
        setter = mon("Raging Bolt", "Leftovers", "Protosynthesis", ["Electric Terrain", "Psychic Terrain", "Thunderbolt",
                     "Draco Meteor"], evs="252 HP / 252 SpA / 4 SpD", nature="Modest", tera="Fairy")  # fmt: skip
        e_seed = mon("Corviknight", "Electric Seed", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
                     nature="Impish", tera="Dragon")  # fmt: skip
        p_seed = PEX.replace("Black Sludge", "Psychic Seed")
        self.play(team(setter, PEX), team(e_seed, p_seed), [(0, 0), (0, 1), (1, 8), (2, 1)])


if __name__ == "__main__":
    unittest.main()
