"""
L2 scenarios for the item_moves family (Trick, Switcheroo, Thief, Ceaseless Edge, Stone Axe), played in OU-JAX and
in Showdown with forced outcomes and compared after every step (see engine/oujax/test_l2.py for the action
encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"

PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
CORV = mon("Corviknight", "Leftovers", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip


def bolt(item, moves=("Trick", "Thunderbolt", "Draco Meteor", "Calm Mind")):
    return mon("Raging Bolt", item, "Protosynthesis", list(moves), evs=SPEC, nature="Modest", tera="Fairy")


def gambit(item, moves=("Kowtow Cleave", "Iron Head", "Swords Dance", "Substitute")):
    return mon("Kingambit", item, "Pressure", list(moves), tera="Dark")


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestItemMoves(MirrorScenarios, unittest.TestCase):
    def test_trick_choice_scarf(self):
        # Trick hands Kingambit the Choice Scarf (it is locked once it moves, and faster from the next turn);
        # tricking it back clears that lock, and the Trick user is locked once it attacks with the Scarf
        script = [(0, 0), (0, 2), (1, 2), (0, 2), (1, 1), (1, 1)]
        self.play(team(bolt("Choice Scarf"), PEX), team(gambit("Leftovers"), CORV), script)

    def test_trick_fails(self):
        # With neither Pokemon holding an item; into a Substitute; Booster Energy can't go to a Paradox Pokemon
        # (Great Tusk), but Raging Bolt (not tagged Paradox) can give it to Kingambit
        script = [(0, 0), (0, 3), (8, 1), (0, 2), (1, 8), (0, 0), (0, 8)]
        no_item = gambit(None, ("Kowtow Cleave", "Iron Head", "Swords Dance", "Substitute"))
        energy = (
            bolt("Booster Energy").replace("Raging Bolt", "Bolt (Raging Bolt)", 1).replace("Protosynthesis", "Pressure")
        )
        p1 = team(bolt(None), energy)
        p2 = team(no_item, TUSK.replace("Leftovers", "Sitrus Berry"))
        self.play(p1, p2, script)

    def test_switcheroo_masks(self):
        # Ogerpon's mask can't leave it, by its own Switcheroo or by the foe's Trick
        ogerpon = mon("Ogerpon-Wellspring", "Wellspring Mask", "Water Absorb",
                      ["Switcheroo", "Ivy Cudgel", "Horn Leech", "U-turn"], tera="Water")  # fmt: skip
        self.play(team(ogerpon, PEX), team(bolt("Leftovers"), CORV), [(0, 0), (0, 3), (1, 0), (0, 3)])

    def test_thief(self):
        # Not through a Substitute; an item-less Thief user takes the Leftovers; with an item it takes nothing
        thief = mon("Weavile", None, "Pressure", ["Thief", "Knock Off", "Swords Dance", "Protect"],
                    evs="252 Atk / 4 SpD / 252 Spe", nature="Jolly", tera="Ice")  # fmt: skip
        self.play(team(thief, PEX), team(gambit("Leftovers"), CORV), [(0, 0), (3, 3), (0, 2), (0, 8), (0, 1)])

    def test_ceaseless_edge_and_stone_axe(self):
        # Each hit adds a Spikes layer (also through a Substitute, up to three); Stone Axe sets Stealth Rock
        samurott = mon("Samurott-Hisui", "Leftovers", "Sharpness", ["Ceaseless Edge", "Razor Shell", "Aqua Jet",
                       "Knock Off"], tera="Dark")  # fmt: skip
        kleavor = mon("Kleavor", "Leftovers", "Sharpness", ["Stone Axe", "X-Scissor", "U-turn", "Close Combat"],
                      tera="Rock")  # fmt: skip
        script = [(0, 0), (0, 3), (0, 1), (0, 1), (0, 1), (8, 1), (0, 1)]
        self.play(team(samurott, kleavor), team(gambit("Leftovers"), CORV), script)


if __name__ == "__main__":
    unittest.main()
