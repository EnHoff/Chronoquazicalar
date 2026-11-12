"""
L2 scenarios for the absorb_abilities family (Good as Gold, the absorbing abilities, Magic Bounce, Bulletproof,
Soundproof, Overcoat, Prankster against Dark types, Dry Skin / Rain Dish in weather), played in OU-JAX and in
Showdown with forced outcomes and compared after every step.
"""

import unittest

from engine.oujax.families.ability_checks import AbilityScenarios
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 SpA / 4 SpD / 252 Spe"
SWALL = "252 HP / 4 Def / 252 SpD"

GHOLDENGO = mon("Gholdengo", "Leftovers", "Good as Gold", ["Make It Rain", "Shadow Ball", "Nasty Plot", "Recover"],
                evs=SWALL, nature="Calm", tera="Steel")  # fmt: skip
GRIMMSNARL = mon("Grimmsnarl", "Light Clay", "Prankster", ["Thunder Wave", "Reflect", "Spirit Break", "Will-O-Wisp"],
                 evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Iron Head", "Kowtow Cleave", "Swords Dance", "Low Kick"],
             tera="Dark")  # fmt: skip
MEW = mon("Mew", "Leftovers", "Synchronize", ["Surf", "Flamethrower", "Earthquake", "Energy Ball"],
          evs="252 HP / 252 SpA / 4 Spe", nature="Modest", tera="Psychic")  # fmt: skip
ZAPDOS = mon("Zapdos", "Heavy-Duty Boots", "Pressure", ["Thunderbolt", "Heat Wave", "Roost", "Volt Switch"],
             evs=SPEC, nature="Timid", tera="Electric")  # fmt: skip
CLODSIRE = mon("Clodsire", "Leftovers", "Water Absorb", ["Earthquake", "Recover", "Toxic", "Haze"], evs=SWALL,
               nature="Careful", tera="Ground")  # fmt: skip
HEATRAN = mon("Heatran", "Leftovers", "Flash Fire", ["Fire Blast", "Earth Power", "Stealth Rock", "Protect"],
              evs=SWALL, nature="Calm", tera="Grass")  # fmt: skip
ORTHWORM = mon("Orthworm", "Leftovers", "Earth Eater", ["Iron Head", "Body Press", "Shore Up", "Spikes"], evs=WALL,
               nature="Impish", tera="Electric")  # fmt: skip
TOXICROAK = mon("Toxicroak", "Leftovers", "Dry Skin", ["Drain Punch", "Gunk Shot", "Swords Dance", "Poison Jab"],
                evs=SWALL, tera="Dark")  # fmt: skip
GOODRA = mon("Goodra", "Assault Vest", "Sap Sipper", ["Draco Meteor", "Fire Blast", "Sludge Bomb", "Thunderbolt"],
             evs=SWALL, nature="Calm", tera="Water")  # fmt: skip
LANTURN = mon("Lanturn", "Leftovers", "Volt Absorb", ["Scald", "Volt Switch", "Ice Beam", "Thunder Wave"], evs=SWALL,
              nature="Calm", tera="Flying")  # fmt: skip
HATTERENE = mon("Hatterene", "Leftovers", "Magic Bounce", ["Psychic", "Calm Mind", "Draining Kiss", "Mystical Fire"],
                evs=WALL, nature="Bold", tera="Fairy")  # fmt: skip
CLODSIRE_HAZ = mon("Clodsire", "Leftovers", "Unaware", ["Toxic", "Spikes", "Stealth Rock", "Earthquake"], evs=SWALL,
                   nature="Careful", tera="Ground")  # fmt: skip
TINGLU = mon("Ting-Lu", "Leftovers", "Vessel of Ruin", ["Thunder Wave", "Stealth Rock", "Earthquake", "Spikes"],
             evs=SWALL, nature="Careful", tera="Poison")  # fmt: skip
KOMMO_BP = mon("Chesnaught", "Leftovers", "Bulletproof", ["Close Combat", "Wood Hammer", "Iron Head", "Drain Punch"],
               evs=PHYS, nature="Jolly", tera="Steel")  # fmt: skip
KOMMO_SP = mon("Kommo-o", "Leftovers", "Soundproof", ["Close Combat", "Clangorous Soul", "Iron Head", "Drain Punch"],
               evs=PHYS, nature="Jolly", tera="Steel")  # fmt: skip
FORRETRESS = mon("Forretress", "Leftovers", "Overcoat", ["Spikes", "Rapid Spin", "Body Press", "Iron Head"],
                 evs=WALL, nature="Impish", tera="Water")  # fmt: skip
SPAMMER = mon("Breloom", "Leftovers", "Technician", ["Spore", "Bullet Seed", "Stun Spore", "Mach Punch"], evs=PHYS,
              nature="Jolly", tera="Fire")  # fmt: skip
SINGER = mon("Primarina", "Leftovers", "Liquid Voice", ["Hyper Voice", "Moonblast", "Surf", "Sludge Bomb"],
             evs=SPEC, nature="Modest", tera="Fairy")  # fmt: skip
PELIPPER_DISH = mon("Pelipper", "Damp Rock", "Rain Dish", ["Surf", "U-turn", "Roost", "Rain Dance"], evs=WALL,
                    nature="Bold", tera="Water")  # fmt: skip
TORKOAL = mon("Torkoal", "Heat Rock", "Drought", ["Stealth Rock", "Flamethrower", "Rapid Spin", "Body Press"],
              evs=WALL, nature="Relaxed", tera="Fire")  # fmt: skip
TYRAN = mon("Tyranitar", "Leftovers", "Sand Stream", ["Stone Edge", "Knock Off", "Stealth Rock", "Protect"],
            tera="Ghost")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestAbsorbAbilities(AbilityScenarios, unittest.TestCase):
    def test_good_as_gold_and_prankster(self):
        # Prankster Thunder Wave / Will-O-Wisp into Good as Gold; Reflect still works; a Prankster status move
        # into a Dark type fails
        script = [(0, 0), (0, 2), (3, 3), (1, 3), (0, 8), (3, 0), (8, 0)]
        self.play(team(GRIMMSNARL, ZAPDOS), team(GHOLDENGO, GAMBIT), script)

    def test_absorbing_abilities(self):
        # Water Absorb, Flash Fire (and its boosted Fire Blast), Earth Eater, Dry Skin (and its Fire weakness),
        # Sap Sipper, Volt Absorb
        script = [(0, 0), (0, 0), (1, 8), (1, 0), (2, 9), (0, 10), (1, 0), (3, 11), (8, 12), (0, 1), (0, 3)]
        self.play(team(MEW, ZAPDOS), team(CLODSIRE, HEATRAN, ORTHWORM, TOXICROAK, GOODRA, LANTURN), script)

    def test_magic_bounce(self):
        # Toxic bounced into a Poison type, Spikes and Stealth Rock bounced onto the setter's side, Thunder Wave
        # bounced into a Ground type
        script = [(0, 0), (1, 0), (1, 1), (1, 2), (8, 8), (0, 0), (8, 3)]
        self.play(team(HATTERENE, GHOLDENGO), team(CLODSIRE_HAZ, TINGLU), script)

    def test_bulletproof_soundproof_overcoat(self):
        # Bulletproof stops Shadow Ball / Sludge Bomb; Soundproof stops Hyper Voice; Overcoat stops Spore and
        # sand damage
        script = [(0, 0), (2, 1), (2, 1), (8, 8), (0, 0), (0, 0), (9, 9), (2, 0), (0, 10), (0, 0)]
        self.play(team(KOMMO_BP, KOMMO_SP, FORRETRESS), team(GHOLDENGO, SINGER, SPAMMER, TYRAN), script)

    def test_weather_abilities(self):
        # Dry Skin heals 1/8 in rain and loses 1/8 in sun; Rain Dish heals 1/16 in rain
        script = [(0, 0), (3, 0), (0, 0), (8, 8), (0, 0), (0, 0)]
        self.play(team(PELIPPER_DISH, TOXICROAK), team(TOXICROAK, TORKOAL), script)


if __name__ == "__main__":
    unittest.main()
