"""
L2 scenarios for the passive_abilities family (Magic Guard, Poison Heal, Sturdy, Disguise, Leaf Guard, Magnet
Pull, Bad Dreams, Hydration) and the engine-level abilities that came with it (Rock Head, Liquid Ooze, Sticky
Hold, Corrosion), played in OU-JAX and in Showdown with forced outcomes and compared after every step.
"""

import unittest

from engine.oujax.families.ability_checks import AbilityScenarios
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 SpA / 4 SpD / 252 Spe"
SWALL = "252 HP / 4 Def / 252 SpD"

PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Toxic", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
CLEFABLE = mon("Clefable", "Life Orb", "Magic Guard", ["Moonblast", "Moonlight", "Knock Off", "Thunder Wave"],
               evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
TINGLU = mon("Ting-Lu", "Leftovers", "Vessel of Ruin", ["Will-O-Wisp", "Stealth Rock", "Earthquake", "Spikes"],
             evs=SWALL, nature="Careful", tera="Poison")  # fmt: skip
CORV = mon("Corviknight", "Rocky Helmet", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
GLISCOR = mon("Gliscor", "Leftovers", "Poison Heal", ["Earthquake", "Knock Off", "Protect", "Swords Dance"],
              evs=WALL, nature="Impish", tera="Water")  # fmt: skip
TUSK_CB = mon("Great Tusk", "Choice Band", "Protosynthesis", ["Headlong Rush", "Close Combat", "Knock Off",
              "Rapid Spin"], evs=PHYS, nature="Jolly", tera="Ground")  # fmt: skip
MAGNEZONE = mon("Magnezone", "Leftovers", "Sturdy", ["Thunderbolt", "Flash Cannon", "Volt Switch", "Body Press"],
                evs=SWALL, nature="Calm", tera="Water")  # fmt: skip
MAGNEZONE_PULL = mon("Magnezone", "Leftovers", "Magnet Pull", ["Thunderbolt", "Flash Cannon", "Volt Switch",
                     "Body Press"], evs=SPEC, nature="Timid", tera="Fire")  # fmt: skip
MIMIKYU = mon("Mimikyu", "Leftovers", "Disguise", ["Play Rough", "Shadow Claw", "Swords Dance", "Shadow Sneak"],
              evs=PHYS, nature="Jolly", tera="Fairy")  # fmt: skip
CLOYSTER = mon("Cloyster", "Leftovers", "Skill Link", ["Icicle Spear", "Rock Blast", "Protect", "Shell Smash"],
               evs=PHYS, nature="Jolly", tera="Ice")  # fmt: skip
LEAFEON = mon("Leafeon", "Leftovers", "Leaf Guard", ["Leaf Blade", "Knock Off", "Swords Dance", "Synthesis"],
              evs=WALL, nature="Impish", tera="Grass")  # fmt: skip
TORKOAL = mon("Torkoal", "Heat Rock", "Drought", ["Will-O-Wisp", "Flamethrower", "Rapid Spin", "Thunder Wave"],
              evs=WALL, nature="Relaxed", tera="Fire")  # fmt: skip
GHOLDENGO = mon("Gholdengo", "Leftovers", "Good as Gold", ["Make It Rain", "Shadow Ball", "Nasty Plot", "Recover"],
                evs=SWALL, nature="Calm", tera="Steel")  # fmt: skip
DARKRAI = mon("Darkrai", "Leftovers", "Bad Dreams", ["Spore", "Dark Pulse", "Nasty Plot", "Sludge Bomb"], evs=SPEC,
              nature="Timid", tera="Poison")  # fmt: skip
MANAPHY = mon("Manaphy", "Leftovers", "Hydration", ["Rain Dance", "Surf", "Calm Mind", "Ice Beam"], evs=WALL,
              nature="Bold", tera="Water")  # fmt: skip
RHYPERIOR = mon("Rhyperior", "Leftovers", "Rock Head", ["Head Smash", "Brave Bird", "Earthquake", "Wave Crash"],
                evs=WALL, nature="Impish", tera="Rock")  # fmt: skip
TENTACRUEL = mon("Tentacruel", "Leftovers", "Liquid Ooze", ["Surf", "Rapid Spin", "Knock Off", "Toxic Spikes"],
                 evs=SWALL, nature="Calm", tera="Water")  # fmt: skip
DRAINER = mon("Venusaur", "Leftovers", "Overgrow", ["Giga Drain", "Strength Sap", "Drain Punch", "Sludge Bomb"],
              evs=WALL, nature="Bold", tera="Grass")  # fmt: skip
MUK = mon("Muk-Alola", "Leftovers", "Sticky Hold", ["Knock Off", "Drain Punch", "Gunk Shot", "Recover"], evs=SWALL,
          nature="Careful", tera="Dark")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Knock Off", "Iron Head", "Swords Dance", "Low Kick"],
             tera="Dark")  # fmt: skip
GLIMMORA = mon("Glimmora", "Leftovers", "Corrosion", ["Toxic", "Power Gem", "Stealth Rock", "Earth Power"], evs=WALL,
               nature="Bold", tera="Grass")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestPassiveAbilities(AbilityScenarios, unittest.TestCase):
    def test_magic_guard(self):
        # No hazard damage on entry, no burn damage, no Life Orb recoil, no Rocky Helmet damage
        script = [(0, 0), (1, 1), (8, 3), (0, 0), (0, 8), (2, 0), (0, 3)]
        self.play(team(PEX, CLEFABLE), team(TINGLU, CORV), script)

    def test_poison_heal(self):
        # Poison Heal: heals 1/8 when badly poisoned instead of taking damage
        script = [(0, 0), (2, 3), (1, 2), (1, 0)]
        self.play(team(PEX, CORV), team(GLISCOR, CORV), script)

    def test_sturdy(self):
        # Sturdy survives a KO from full HP at 1 HP, but not from below full HP
        script = [(0, 0), (0, 0), (0, 0)]
        self.play(team(TUSK_CB, CORV), team(MAGNEZONE, CORV), script)

    def test_disguise(self):
        # Disguise takes the first hit of a multi-hit move for 0 and loses 1/8; later hits land normally
        script = [(0, 0), (0, 0), (0, 0)]
        self.play(team(CLOYSTER, CORV), team(MIMIKYU, CORV), script)

    def test_leaf_guard(self):
        # In sun, Leaf Guard blocks Will-O-Wisp and Thunder Wave
        script = [(0, 0), (0, 0), (3, 0), (8, 0)]
        self.play(team(TORKOAL, PEX), team(LEAFEON, CORV), script)

    def test_magnet_pull(self):
        # Magnet Pull traps Steel types (not Steel / Ghost Gholdengo); a pivot still switches out
        script = [(0, 0), (1, 0), (8, 9), (0, 0)]
        self.play(team(MAGNEZONE_PULL, PEX), team(CORV, GHOLDENGO, TINGLU), script)

    def test_bad_dreams_and_hydration(self):
        # Bad Dreams takes 1/8 from a sleeping foe each turn; Hydration cures status at the end of a rainy turn
        script = [(0, 0), (0, 2), (2, 0), (8, 8), (0, 0), (0, 2)]
        self.play(team(DARKRAI, PEX), team(PEX, MANAPHY), script)

    def test_rock_head(self):
        # Rock Head ignores recoil (Head Smash, Brave Bird, Wave Crash)
        script = [(0, 0), (0, 0), (1, 0), (3, 0)]
        self.play(team(RHYPERIOR, PEX), team(CORV, GLISCOR), script)

    def test_liquid_ooze(self):
        # Draining moves into Liquid Ooze hurt the user instead (Giga Drain, Drain Punch, Strength Sap)
        script = [(0, 0), (0, 0), (2, 0), (1, 0)]
        self.play(team(DRAINER, PEX), team(TENTACRUEL, CORV), script)

    def test_sticky_hold(self):
        # Sticky Hold keeps its item against Knock Off (which still gets its power boost)
        script = [(0, 0), (0, 0), (0, 0)]
        self.play(team(GAMBIT, PEX), team(MUK, CORV), script)

    def test_corrosion(self):
        # Corrosion badly poisons Steel and Poison types
        script = [(0, 0), (0, 0), (0, 8), (0, 0)]
        self.play(team(GLIMMORA, PEX), team(CORV, PEX), script)


if __name__ == "__main__":
    unittest.main()
