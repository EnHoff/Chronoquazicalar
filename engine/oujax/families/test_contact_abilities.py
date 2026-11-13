"""
L2 scenarios for the contact_abilities family (DamagingHit abilities, Poison Touch / Toxic Chain, Pickpocket,
Magician, Berserk, Synchronize, Serene Grace, Shield Dust, Shell Armor, Gulp Missile), played in OU-JAX and in
Showdown with forced outcomes (30% ability rolls succeed; Effect Spore rolls sleep) and compared after every step.
"""

import unittest

from engine.oujax.engine import Config
from engine.oujax.families.ability_checks import AbilityScenarios
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 SpA / 4 SpD / 252 Spe"
SWALL = "252 HP / 4 Def / 252 SpD"

TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Close Combat"],
           tera="Ground")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Iron Head", "Kowtow Cleave", "Swords Dance", "Low Kick"],
             tera="Dark")  # fmt: skip
ZAPDOS = mon("Zapdos", "Heavy-Duty Boots", "Static", ["Roost", "Heat Wave", "Volt Switch", "Hydro Pump"],
             evs=WALL, nature="Bold", tera="Water")  # fmt: skip
VOLCARONA = mon("Volcarona", "Heavy-Duty Boots", "Flame Body", ["Quiver Dance", "Fiery Dance", "Bug Buzz", "Giga Drain"],
                evs=WALL, nature="Bold", tera="Water")  # fmt: skip
GARCHOMP = mon("Garchomp", "Leftovers", "Rough Skin", ["Earthquake", "Stealth Rock", "Spikes", "Dragon Claw"],
               evs=WALL, nature="Impish", tera="Steel")  # fmt: skip
GOODRA = mon("Goodra", "Leftovers", "Gooey", ["Draco Meteor", "Fire Blast", "Sludge Bomb", "Thunderbolt"],
             evs=WALL, nature="Bold", tera="Water")  # fmt: skip
DRAGAPULT = mon("Dragapult", "Leftovers", "Cursed Body", ["Dragon Darts", "U-turn", "Shadow Ball", "Draco Meteor"],
                evs=WALL, nature="Impish", tera="Fairy")  # fmt: skip
SKARMORY = mon("Skarmory", "Leftovers", "Weak Armor", ["Spikes", "Roost", "Brave Bird", "Body Press"], evs=WALL,
               nature="Impish", tera="Water")  # fmt: skip
LUCARIO = mon("Lucario", "Leftovers", "Justified", ["Close Combat", "Meteor Mash", "Swords Dance", "Extreme Speed"],
              evs=WALL, nature="Impish", tera="Normal")  # fmt: skip
GLIMMORA = mon("Glimmora", "Leftovers", "Toxic Debris", ["Power Gem", "Sludge Wave", "Stealth Rock", "Earth Power"],
               evs=WALL, nature="Bold", tera="Grass")  # fmt: skip
OKIDOGI = mon("Okidogi", "Leftovers", "Toxic Chain", ["Drain Punch", "Knock Off", "Bulk Up", "Gunk Shot"],
              tera="Dark")  # fmt: skip
SNEASLER = mon("Sneasler", "Leftovers", "Poison Touch", ["Close Combat", "Poison Jab", "Swords Dance", "Gunk Shot"],
               evs=PHYS, nature="Jolly", tera="Fighting")  # fmt: skip
COAT_DUST = mon("Volcarona", "Covert Cloak", "Swarm", ["Quiver Dance", "Fiery Dance", "Bug Buzz", "Giga Drain"],
                evs=WALL, nature="Bold", tera="Water")  # fmt: skip
SHIELD_DUST = mon("Ribombee", "Leftovers", "Shield Dust", ["Moonblast", "Bug Buzz", "Sticky Web", "Psychic"],
                  evs=WALL, nature="Bold", tera="Ghost")  # fmt: skip
WEAVILE = mon("Weavile", "", "Pickpocket", ["Triple Axel", "Knock Off", "Swords Dance", "Ice Shard"], evs=WALL,
              nature="Impish", tera="Ice")  # fmt: skip
HOOPA = mon("Hoopa-Unbound", "", "Magician", ["Close Combat", "Psychic", "Gunk Shot", "Drain Punch"], evs=PHYS,
            nature="Jolly", tera="Dark")  # fmt: skip
CORV = mon("Corviknight", "Rocky Helmet", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
DRAMPA = mon("Hydreigon", "Sitrus Berry", "Berserk", ["Draco Meteor", "Hyper Voice", "Fire Blast", "Roost"], evs=SPEC,
             nature="Modest", tera="Normal")  # fmt: skip
MEW = mon("Mew", "Leftovers", "Synchronize", ["Psychic", "Will-O-Wisp", "Roost", "Thunder Wave"], evs=WALL,
          nature="Bold", tera="Psychic")  # fmt: skip
STATUSER = mon("Clefable", "Leftovers", "Magic Guard", ["Thunder Wave", "Will-O-Wisp", "Moonblast", "Toxic"],
               evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
TOGEKISS = mon("Clefable", "Leftovers", "Serene Grace", ["Crush Claw", "Diamond Storm", "Roost", "Nasty Plot"],
               evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
SHELL = mon("Cloyster", "Leftovers", "Shell Armor", ["Icicle Spear", "Rock Blast", "Protect", "Substitute"],
            evs=WALL, nature="Impish", tera="Ice")  # fmt: skip
CRAMORANT = mon("Cramorant", "Leftovers", "Gulp Missile", ["Surf", "Roost", "Brave Bird", "Defog"], evs=WALL,
                nature="Bold", tera="Water")  # fmt: skip
BRELOOM = mon("Breloom", "Leftovers", "Effect Spore", ["Spore", "Bullet Seed", "Protect", "Mach Punch"], evs=WALL,
              nature="Impish", tera="Fire")  # fmt: skip

DONDOZO = mon("Dondozo", "Leftovers", "Unaware", ["Waterfall", "Body Press", "Earthquake", "Protect"], evs=WALL,
              nature="Impish", tera="Water")  # fmt: skip
TINGLU = mon("Ting-Lu", "Leftovers", "Vessel of Ruin", ["Earthquake", "Stealth Rock", "Spikes", "Protect"],
             evs=SWALL, nature="Careful", tera="Water")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestContactAbilities(AbilityScenarios, unittest.TestCase):
    def test_static_flame_body(self):
        script = [(0, 0), (2, 0), (8, 8), (0, 0)]
        self.play(team(TUSK, GAMBIT), team(ZAPDOS, VOLCARONA), script)

    def test_rough_skin_gooey_cursed_body(self):
        # Rough Skin 1/8 on contact; Gooey -1 Spe; Cursed Body disables the move that hit it (not a non-contact
        # check), and the Disable runs out after four end-of-turns
        script = [(0, 0), (0, 0), (8, 8), (0, 1), (8, 9), (0, 2), (2, 2), (2, 2), (2, 2), (2, 2), (0, 2)]
        self.play(team(GAMBIT, TUSK), team(GARCHOMP, GOODRA, DRAGAPULT), script)

    def test_weak_armor_justified_toxic_debris(self):
        script = [(0, 0), (0, 1), (8, 8), (1, 2), (8, 9), (0, 2), (0, 0)]
        self.play(team(GAMBIT, TUSK), team(SKARMORY, LUCARIO, GLIMMORA), script)

    def test_toxic_chain_poison_touch_covert_cloak(self):
        # Toxic Chain badly poisons on any hit; Poison Touch poisons on contact; Covert Cloak and Shield Dust
        # block both
        script = [(0, 0), (0, 0), (0, 8), (8, 9), (0, 0), (0, 10), (0, 0)]
        self.play(team(SNEASLER, OKIDOGI), team(DONDOZO, COAT_DUST, SHIELD_DUST, TINGLU), script)

    def test_pickpocket_and_magician(self):
        # Pickpocket takes a contact attacker's item; Magician takes the target's item after its hit
        script = [(0, 0), (1, 0), (8, 8), (0, 0), (2, 0)]
        self.play(team(GAMBIT, HOOPA), team(WEAVILE, CORV), script)

    def test_berserk(self):
        # Berserk: +1 SpA when a hit takes it from above half HP to half or below; Sitrus Berry after that
        script = [(0, 0), (0, 3), (0, 3), (0, 3), (0, 3)]
        self.play(team(TUSK, GAMBIT), team(DRAMPA, CORV), script)

    def test_synchronize(self):
        # Synchronize passes burn / paralysis / poison back; a Synchronized status doesn't bounce again
        script = [(0, 0), (0, 1), (8, 0), (0, 3), (0, 0)]
        self.play(team(STATUSER, MEW), team(MEW, STATUSER), script)

    def test_serene_grace(self):
        # Serene Grace doubles 50% chances to 100% (the forced roll misses everything below 100)
        script = [(0, 0), (0, 0), (1, 0), (0, 0)]
        self.play(team(TOGEKISS, GAMBIT), team(GAMBIT, TUSK), script)

    def test_shield_dust_with_secondaries(self):
        # With secondaries forced on: Shield Dust blocks a foe's secondary effects, Covert Cloak too
        script = [(0, 0), (0, 1), (8, 0), (0, 1)]
        self.play(team(SHIELD_DUST, COAT_DUST), team(TOGEKISS, GAMBIT), script, cfg=Config(forced=True, secondaries=True),
                  forced_opts={"secondaries": True})  # fmt: skip

    def test_shell_armor_with_crits(self):
        # With critical hits forced on, Shell Armor still takes regular hits
        script = [(0, 0), (0, 0), (8, 0), (0, 0)]
        self.play(team(SHELL, GAMBIT), team(GAMBIT, TUSK), script, cfg=Config(forced=True, crit=True),
                  forced_opts={"crit": True})  # fmt: skip

    def test_gulp_missile(self):
        # Surf picks up Arrokuda (above half HP) or Pikachu (at half or below); being hit spits it out:
        # 1/4 damage and -1 Defense or paralysis
        script = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        self.play(team(CRAMORANT, CORV), team(GAMBIT, TUSK), script)

    def test_effect_spore(self):
        # Effect Spore puts a contact attacker to sleep (the forced random(100) answers 0); Grass types and
        # Overcoat are immune
        script = [(0, 0), (0, 3), (0, 3)]
        self.play(team(GAMBIT, TUSK), team(BRELOOM, CORV), script)


if __name__ == "__main__":
    unittest.main()
