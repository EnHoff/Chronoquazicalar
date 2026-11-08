"""
L2 scenarios for the berries family (Lum, Chesto, Leppa, Salac, Kee, Bug Bite), played in OU-JAX and in Showdown
with forced outcomes and compared after every step (see engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.engine import Config
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
CORV = mon("Corviknight", "Leftovers", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip


def gambit(item, moves):
    return mon("Kingambit", item, "Pressure", list(moves), tera="Dark")


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestBerries(MirrorScenarios, unittest.TestCase):
    def test_chesto_rest(self):
        # Rest, then the Chesto Berry wakes it up at once (Update after the action); the second Rest sleeps
        rester = mon("Corviknight", "Chesto Berry", "Pressure", ["Rest", "Roost", "Defog", "Brave Bird"], evs=WALL,
                     nature="Impish", tera="Dragon")  # fmt: skip
        self.play(team(rester, PEX), team(TUSK, PEX), [(0, 0), (3, 1), (0, 3), (3, 1), (0, 1), (3, 1)])

    def test_lum_berry(self):
        # Toxic is cured right away; Toxic Spikes poison on a later switch-in is not (the berry is gone)
        lum = mon("Great Tusk", "Lum Berry", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
                  tera="Ground")  # fmt: skip
        pex_tox = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Toxic", "Haze"],
                      evs=WALL, nature="Bold", tera="Fairy")  # fmt: skip
        self.play(team(lum, CORV), team(pex_tox, CORV), [(0, 0), (3, 2), (8, 0), (8, 2), (8, 1)])

    def test_leppa_berry(self):
        # Recover (8 PP with PP Ups) runs dry on the eighth use and the Leppa Berry restores it
        leppa = mon("Toxapex", "Leppa Berry", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
                    nature="Bold", tera="Fairy")  # fmt: skip
        self.play(team(leppa, CORV), team(CORV, PEX), [(0, 0)] + [(1, 2)] * 9)

    def test_salac_berry(self):
        # Belly Drum then Substitute leave Kingambit at 1/4 HP: the Salac Berry gives +1 Spe after the action
        salac = gambit("Salac Berry", ("Kowtow Cleave", "Belly Drum", "Substitute", "Protect"))
        self.play(team(salac, PEX), team(TUSK, CORV), [(0, 0), (1, 3), (2, 3), (0, 0)])

    def test_kee_berry(self):
        # A special hit doesn't eat it; a hit on the Substitute doesn't either; a physical hit does (+1 Def)
        kee = mon("Corviknight", "Kee Berry", "Pressure", ["Substitute", "Roost", "Defog", "Brave Bird"],
                  evs="252 HP / 4 Def / 252 Spe", nature="Jolly", tera="Dragon")  # fmt: skip
        bolt = mon("Raging Bolt", "Leftovers", "Protosynthesis", ["Thunderbolt", "Draco Meteor", "Calm Mind", "Body Slam"],
                   evs="252 HP / 4 SpA / 252 Def", nature="Bold", tera="Fairy")  # fmt: skip
        self.play(team(kee, PEX), team(bolt, TUSK), [(0, 0), (0, 3), (1, 1), (1, 1), (1, 3)])

    def test_bug_bite(self):
        # Bug Bite eats the target's Sitrus Berry (healing the damaged user) and its Lum Berry later
        biter = mon("Scizor", "Leftovers", "Technician", ["Bug Bite", "Bullet Punch", "Swords Dance", "Roost"],
                    tera="Steel")  # fmt: skip
        sitrus = mon("Toxapex", "Sitrus Berry", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
                     nature="Bold", tera="Fairy")  # fmt: skip
        lum = sitrus.replace("Sitrus Berry", "Lum Berry").replace("Toxapex", "Pex (Toxapex)", 1)
        self.play(team(biter, CORV), team(sitrus, lum), [(0, 0), (2, 2), (0, 2), (0, 8), (0, 2)])


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestBerriesWithSecondaries(MirrorScenarios, unittest.TestCase):
    def test_lum_cures_secondary_burn(self):
        lum = mon("Great Tusk", "Lum Berry", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
                  evs="252 HP / 252 Def / 4 Atk", nature="Impish", tera="Ground")  # fmt: skip
        burner = mon("Volcarona", "Heavy-Duty Boots", "Swarm", ["Flamethrower", "Quiver Dance", "Giga Drain", "Roost"],
                     evs="252 HP / 252 SpA / 4 Spe", nature="Modest", tera="Fire")  # fmt: skip
        self.play(team(lum, CORV), team(burner, CORV), [(0, 0), (3, 0), (3, 0)],
                  cfg=Config(forced=True, secondaries=True), forced_opts={"secondaries": True})  # fmt: skip


if __name__ == "__main__":
    unittest.main()
