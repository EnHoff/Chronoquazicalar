"""
L2 scenarios for the entry_abilities family (Intimidate and its reactions, Dauntless Shield, Download, Trace,
Neutralizing Gas, Libero / Protean, Wind Rider, Skill Swap), played in OU-JAX and in Showdown with forced
outcomes and compared after every step, including each active Pokemon's ability and types
(engine/oujax/families/ability_checks.py).
"""

import unittest

from engine.oujax.families.ability_checks import AbilityScenarios
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 SpA / 4 SpD / 252 Spe"

LANDO = mon("Landorus-Therian", "Leftovers", "Intimidate", ["Earthquake", "U-turn", "Stealth Rock", "Stone Edge"],
            tera="Water")  # fmt: skip
GYARADOS = mon("Gyarados", "Leftovers", "Intimidate", ["Waterfall", "Earthquake", "Dragon Dance", "Stone Edge"],
               tera="Ground")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
GAMBIT_DEFIANT = mon("Kingambit", "Leftovers", "Defiant", ["Iron Head", "Kowtow Cleave", "Swords Dance", "Low Kick"],
                     tera="Dark")  # fmt: skip
OKIDOGI = mon("Okidogi", "Leftovers", "Guard Dog", ["Drain Punch", "Knock Off", "Bulk Up", "Gunk Shot"],
              tera="Dark")  # fmt: skip
CORV_MIRROR = mon("Corviknight", "Leftovers", "Mirror Armor", ["Brave Bird", "Roost", "Defog", "Body Press"],
                  evs=WALL, nature="Impish", tera="Dragon")  # fmt: skip
DRAGAPULT_CB = mon("Dragapult", "Leftovers", "Clear Body", ["Dragon Darts", "U-turn", "Shadow Ball", "Draco Meteor"],
                   evs=PHYS, nature="Jolly", tera="Ghost")  # fmt: skip
VALIANT_COMP = mon("Iron Valiant", "Leftovers", "Competitive", ["Moonblast", "Close Combat", "Calm Mind", "Psychic"],
                   evs=SPEC, nature="Timid", tera="Fairy")  # fmt: skip
ZAMAZENTA = mon("Zamazenta", "Leftovers", "Dauntless Shield", ["Body Press", "Iron Defense", "Crunch", "Close Combat"],
                evs=WALL, nature="Impish", tera="Steel")  # fmt: skip
PORYGONZ = mon("Porygon-Z", "Leftovers", "Download", ["Hyper Voice", "Shadow Ball", "Nasty Plot", "Protect"],
               evs=SPEC, nature="Timid", tera="Normal")  # fmt: skip
PORYGON2 = mon("Porygon2", "Eviolite", "Trace", ["Body Slam", "Recover", "Ice Beam", "Thunder Wave"], evs=WALL,
               nature="Bold", tera="Ghost")  # fmt: skip
WEEZING = mon("Weezing-Galar", "Leftovers", "Neutralizing Gas", ["Sludge Bomb", "Will-O-Wisp", "Protect", "Fire Blast"],
              evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
TYRAN = mon("Tyranitar", "Leftovers", "Sand Stream", ["Stone Edge", "Knock Off", "Stealth Rock", "Protect"],
            tera="Ghost")  # fmt: skip
TUSK_BOOSTER = mon("Great Tusk", "Booster Energy", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off",
                   "Close Combat"], evs=PHYS, nature="Jolly", tera="Ground")  # fmt: skip
PELIPPER = mon("Pelipper", "Damp Rock", "Drizzle", ["Surf", "U-turn", "Roost", "Hydro Pump"], evs=WALL, nature="Bold",
               tera="Water")  # fmt: skip
CINDERACE = mon("Cinderace", "Heavy-Duty Boots", "Libero", ["Pyro Ball", "U-turn", "High Jump Kick", "Court Change"],
                evs=PHYS, nature="Jolly", tera="Fire")  # fmt: skip
MEOWSCARADA = mon("Meowscarada", "Heavy-Duty Boots", "Protean", ["Flower Trick", "Knock Off", "U-turn",
                  "Triple Axel"], evs=PHYS, nature="Jolly", tera="Grass")  # fmt: skip
BRAMBLEGHAST = mon("Brambleghast", "Leftovers", "Wind Rider", ["Power Whip", "Poltergeist", "Rapid Spin", "Spikes"],
                   evs=PHYS, nature="Jolly", tera="Fairy")  # fmt: skip
TORNADUS = mon("Tornadus-Therian", "Heavy-Duty Boots", "Regenerator", ["Bleakwind Storm", "Tailwind", "U-turn",
               "Heat Wave"], evs=SPEC, nature="Timid", tera="Steel")  # fmt: skip
SWAPPER = mon("Hatterene", "Leftovers", "Magic Bounce", ["Skill Swap", "Psychic", "Calm Mind", "Draining Kiss"],
              evs=WALL, nature="Bold", tera="Fairy")  # fmt: skip
GOGOAT_SWAP = mon("Grimmsnarl", "Leftovers", "Prankster", ["Skill Swap", "Spirit Break", "Thunder Wave", "Reflect"],
                  evs=WALL, nature="Impish", tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestEntryAbilities(AbilityScenarios, unittest.TestCase):
    def test_intimidate_defiant_guard_dog_mirror_armor(self):
        # Lead Intimidate into Defiant (net +1 Atk); Intimidate into Guard Dog (+1 Atk instead) with Bulk Up;
        # a U-turn pivot; Intimidate into Mirror Armor (bounced back: Landorus -1 Atk)
        script = [(0, 0), (8, 8), (8, 2), (1, 9), (8, 0), (0, 0)]
        self.play(team(LANDO, PEX), team(GAMBIT_DEFIANT, OKIDOGI, CORV_MIRROR), script)

    def test_intimidate_clear_body_competitive_and_faint_replacement(self):
        # Clear Body blocks Intimidate; Competitive gets +2 SpA from it; a faint replacement comes in together
        # with its foe's replacement, so Intimidate reaches it
        script = [(0, 0), (8, 0), (8, 3), (1, 1), (0, 1), (0, 1), (0, 1)]
        self.play(team(LANDO, PEX, GYARADOS), team(DRAGAPULT_CB, VALIANT_COMP),
                  script)  # fmt: skip

    def test_dauntless_shield_once_and_download(self):
        # Dauntless Shield boosts Defense on its first entry only; Download raises Atk or SpA by the foe's
        # lower defence
        script = [(0, 0), (8, 0), (8, 0), (8, 8), (0, 0), (1, 1)]
        self.play(team(ZAMAZENTA, PEX), team(PORYGONZ, CORV_MIRROR), script)

    def test_trace_copies_intimidate_and_sand_stream(self):
        # Trace copies Intimidate (which then activates) and, on a later entry, Sand Stream (which sets sand)
        script = [(0, 0), (0, 0), (8, 8), (8, 0), (0, 2)]
        self.play(team(PORYGON2, PEX), team(LANDO, TYRAN), script)

    def test_neutralizing_gas(self):
        # Neutralizing Gas stops Intimidate and the Booster Energy / Drizzle starts; when it leaves, the
        # abilities start (Drizzle sets rain, Protosynthesis spends Booster Energy)
        script = [(0, 0), (1, 0), (8, 8), (8, 2), (8, 0), (0, 0)]
        self.play(team(WEEZING, TYRAN), team(PELIPPER, TUSK_BOOSTER, LANDO), script)

    def test_neutralizing_gas_faints(self):
        # A fainting Neutralizing Gas holder gives the foe its ability back at once
        script = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        self.play(team(WEEZING, TYRAN), team(TUSK_BOOSTER, PELIPPER), script)

    def test_libero_and_protean(self):
        # Protean turns Dark for Knock Off and stays Dark for Triple Axel (once per switch-in); Libero turns Bug
        # for U-turn, is Fire again after switching back in, and Tera stops it
        script = [(0, 0), (8, 8), (1, 1), (8, 3), (5, 0)]
        self.play(team(PEX, CINDERACE), team(CORV_MIRROR, MEOWSCARADA), script)

    def test_wind_rider(self):
        # Wind Rider absorbs Bleakwind Storm (+1 Atk) and gets +1 Atk when its side's Tailwind starts / on entry
        script = [(0, 0), (0, 0), (0, 1), (8, 0), (8, 3)]
        self.play(team(BRAMBLEGHAST, PEX), team(TORNADUS, PEX), script)

    def test_skill_swap(self):
        # Skill Swap trades Magic Bounce for Intimidate (which starts on the new holder) and back; the swapped
        # abilities last until the Pokemon switch out
        script = [(0, 0), (0, 0), (1, 0), (8, 0), (8, 0), (0, 3)]
        self.play(team(SWAPPER, PEX), team(LANDO, CORV_MIRROR), script)

    def test_skill_swap_prankster(self):
        script = [(0, 0), (0, 0), (2, 0), (0, 0), (0, 1)]
        self.play(team(GOGOAT_SWAP, PEX), team(TYRAN, ZAMAZENTA), script)


if __name__ == "__main__":
    unittest.main()
