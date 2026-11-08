"""
L2 scenarios for the stat_abilities family (Contrary, drop blockers, Mirror Armor, Defiant / Competitive, KO
boosts, Speed Boost, Unburden, Prankster / Gale Wings / Triage priority), played in OU-JAX and in Showdown with
forced outcomes and compared after every step.
"""

import unittest

from engine.oujax.families.ability_checks import AbilityScenarios
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 SpA / 4 SpD / 252 Spe"
SWALL = "252 HP / 4 Def / 252 SpD"

SERPERIOR = mon("Serperior", "Leftovers", "Contrary", ["Leaf Storm", "Glare", "Dragon Pulse", "Synthesis"], evs=SPEC,
                nature="Timid", tera="Grass")  # fmt: skip
LANDO = mon("Landorus-Therian", "Leftovers", "Intimidate", ["Earthquake", "U-turn", "Stealth Rock", "Stone Edge"],
            tera="Water")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Toxic", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
GRIMM = mon("Grimmsnarl", "Light Clay", "Prankster", ["Parting Shot", "Screech", "Spirit Break", "Thunder Wave"],
            evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
CORV_MIRROR = mon("Corviknight", "Leftovers", "Mirror Armor", ["Brave Bird", "Roost", "Defog", "Body Press"],
                  evs=WALL, nature="Impish", tera="Dragon")  # fmt: skip
PELIPPER = mon("Pelipper", "Leftovers", "Big Pecks", ["Surf", "U-turn", "Roost", "Hydro Pump"], evs=WALL,
               nature="Bold", tera="Water")  # fmt: skip
METAGROSS = mon("Metagross", "Leftovers", "Clear Body", ["Meteor Mash", "Earthquake", "Bullet Punch", "Psychic Fangs"],
                evs=WALL, nature="Impish", tera="Steel")  # fmt: skip
GAMBIT_DEFIANT = mon("Kingambit", "Leftovers", "Defiant", ["Iron Head", "Kowtow Cleave", "Swords Dance", "Low Kick"],
                     tera="Dark")  # fmt: skip
VALIANT_COMP = mon("Iron Valiant", "Leftovers", "Competitive", ["Moonblast", "Close Combat", "Calm Mind", "Psychic"],
                   evs=SPEC, nature="Timid", tera="Fairy")  # fmt: skip
RIBOMBEE = mon("Ribombee", "Focus Sash", "Shield Dust", ["Sticky Web", "Moonblast", "Stun Spore", "U-turn"],
               evs=SPEC, nature="Timid", tera="Ghost")  # fmt: skip
SALAMENCE = mon("Salamence", "Leftovers", "Moxie", ["Dragon Claw", "Earthquake", "Dragon Dance", "Fire Fang"],
                evs=PHYS, nature="Jolly", tera="Dragon")  # fmt: skip
GRENINJA = mon("Greninja-Bond", "Life Orb", "Battle Bond", ["Surf", "Dark Pulse", "Ice Beam", "U-turn"], evs=SPEC,
               nature="Timid", tera="Water")  # fmt: skip
CHILLING = mon("Glastrier", "Leftovers", "Chilling Neigh", ["Icicle Crash", "High Horsepower", "Swords Dance",
               "Close Combat"], evs=PHYS, nature="Adamant", tera="Ice")  # fmt: skip
WEAK1 = mon("Smoliv", "", "Early Bird", ["Tackle", "Growl", "Absorb", "Sweet Scent"], evs="4 HP", nature="Hardy",
            tera="Grass")  # fmt: skip
WEAK2 = mon("Pichu", "", "Static", ["Thunder Shock", "Growl", "Charm", "Sweet Scent"], evs="4 HP", nature="Hardy",
            tera="Electric")  # fmt: skip
BLAZIKEN = mon("Blaziken", "Leftovers", "Speed Boost", ["Close Combat", "Flare Blitz", "Protect", "Swords Dance"],
               evs=PHYS, nature="Jolly", tera="Fire")  # fmt: skip
HAWLUCHA = mon("Hawlucha", "Sitrus Berry", "Unburden", ["Close Combat", "Brave Bird", "Swords Dance", "Acrobatics"],
               evs="252 HP / 252 Atk / 4 Spe", nature="Adamant", tera="Flying")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Knock Off", "Iron Head", "Swords Dance", "Low Kick"],
             evs="252 HP / 252 Atk / 4 Spe", tera="Dark")  # fmt: skip
TALONFLAME = mon("Talonflame", "Heavy-Duty Boots", "Gale Wings", ["Brave Bird", "Flare Blitz", "Roost", "Swords Dance"],
                 evs="252 HP / 252 Atk / 4 Spe", nature="Adamant", tera="Flying")  # fmt: skip
DRAGAPULT = mon("Dragapult", "Leftovers", "Clear Body", ["Dragon Darts", "U-turn", "Shadow Ball", "Draco Meteor"],
                evs=PHYS, nature="Jolly", tera="Ghost")  # fmt: skip
COMFEY = mon("Comfey", "Leftovers", "Triage", ["Draining Kiss", "Calm Mind", "Synthesis", "Giga Drain"], evs=SWALL,
             nature="Calm", tera="Fairy")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestStatAbilities(AbilityScenarios, unittest.TestCase):
    def test_contrary(self):
        # Leaf Storm raises SpA; Intimidate raises Atk; Haze resets
        script = [(0, 0), (0, 0), (0, 8), (0, 0), (0, 3)]
        self.play(team(SERPERIOR, PEX), team(PEX, LANDO), script)

    def test_drop_blockers_and_mirror_armor(self):
        # Screech into Big Pecks (blocked), Clear Body (blocked), Mirror Armor (bounced); Parting Shot into
        # Clear Body fails without switching, into Mirror Armor it is bounced but still switches the user out
        script = [(0, 0), (1, 0), (0, 0), (8, 0), (1, 8), (0, 0), (0, 9), (1, 0), (0, 0)]
        self.play(team(GRIMM, PEX), team(PELIPPER, METAGROSS, CORV_MIRROR), script)

    def test_defiant_competitive_sticky_web(self):
        # Sticky Web's Speed drop on entry triggers Defiant / Competitive; Parting Shot's two drops give +4
        script = [(0, 0), (0, 0), (8, 8), (0, 0), (8, 9), (1, 0)]
        self.play(team(RIBOMBEE, GRIMM), team(PEX, GAMBIT_DEFIANT, VALIANT_COMP), script)

    def test_moxie_and_chilling_neigh(self):
        script = [(0, 0), (0, 1), (0, 0), (8, 0), (0, 0), (0, 0)]
        self.play(team(SALAMENCE, CHILLING), team(WEAK1, WEAK2, PEX), script)

    def test_battle_bond(self):
        # Battle Bond: +1 Atk / SpA / Spe once per battle after a KO
        script = [(0, 0), (0, 0), (0, 0), (0, 0)]
        self.play(team(GRENINJA, PEX), team(WEAK1, WEAK2, PEX), script)

    def test_speed_boost(self):
        # No boost at the end of the turn it switched in; +1 Spe every turn after
        script = [(0, 0), (8, 0), (8, 0), (2, 0), (2, 0), (3, 0)]
        self.play(team(PEX, BLAZIKEN), team(PEX, LANDO), script)

    def test_unburden(self):
        # Unburden doubles Speed once its Sitrus Berry is eaten or knocked off
        script = [(0, 0), (2, 0), (0, 0), (0, 1), (0, 0)]
        self.play(team(HAWLUCHA, PEX), team(GAMBIT, PEX), script)

    def test_gale_wings_and_triage(self):
        # Gale Wings: Flying moves +1 priority at full HP only; Triage: healing moves +3
        script = [(0, 0), (0, 0), (0, 0), (8, 8), (0, 0), (0, 0)]
        self.play(team(TALONFLAME, COMFEY), team(DRAGAPULT, GAMBIT), script)


if __name__ == "__main__":
    unittest.main()
