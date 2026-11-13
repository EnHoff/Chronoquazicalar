"""
L2 scenarios for the volatiles and move_misc families, played in OU-JAX and in Showdown with forced outcomes and
compared after every step (see engine/oujax/test_l2.py for the action encoding). The bridge also compares the
volatiles these families model (hooks.SHOWDOWN_VOLATILES) and the legal moves Showdown offers.
"""

import unittest

from engine.oujax.engine import Config
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import CORV, PEX, MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"

ZAPDOS = mon("Zapdos", "Heavy-Duty Boots", "Pressure", ["Hurricane", "Roost", "Volt Switch", "Heat Wave"],
             evs="248 HP / 8 Def / 252 Spe", nature="Timid", tera="Electric")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Supreme Overlord", ["Iron Head", "Swords Dance", "Sucker Punch", "Upper Hand"],
             tera="Dark")  # fmt: skip
PELIPPER = mon("Pelipper", "Damp Rock", "Drizzle", ["Hurricane", "Surf", "U-turn", "Roost"], evs=WALL, nature="Bold",
               tera="Water")  # fmt: skip
PECHARUNT = mon("Pecharunt", "Heavy-Duty Boots", "Poison Puppeteer", ["Malignant Chain", "Toxic", "Shadow Ball",
                "Recover"], evs=WALL, nature="Bold", tera="Dark")  # fmt: skip
RILLA = mon("Rillaboom", "Leftovers", "Grassy Surge", ["Fake Out", "Wood Hammer", "U-turn", "Grassy Glide"],
            tera="Grass")  # fmt: skip
CLEFABLE = mon("Clefable", "Leftovers", "Misty Surge", ["Moonblast", "Swagger", "Protect", "Moonlight"], evs=WALL,
               nature="Bold", tera="Water")  # fmt: skip
GRIMM = mon("Grimmsnarl", "Light Clay", "Frisk", ["Taunt", "Encore", "Spirit Break", "Thunder Wave"], tera="Dark")
GHOLD = mon("Gholdengo", "Leftovers", "Frisk", ["Nasty Plot", "Shadow Ball", "Recover", "Flash Cannon"], evs=SPEC,
            nature="Modest", tera="Steel")  # fmt: skip
PRIMARINA = mon("Primarina", "Leftovers", "Torrent", ["Psychic Noise", "Hyper Voice", "Moonblast", "Surf"], evs=SPEC,
                nature="Modest", tera="Water")  # fmt: skip
GAMBIT_CHOP = mon("Kingambit", "Leftovers", "Supreme Overlord", ["Throat Chop", "Iron Head", "Swords Dance",
                  "Sucker Punch"], tera="Dark")  # fmt: skip
DRAGAPULT_SITRUS = mon("Dragapult", "Sitrus Berry", "Infiltrator", ["Shadow Ball", "Draco Meteor", "Hex", "Protect"],
                       evs="252 HP / 4 Def / 252 Spe", nature="Timid", tera="Ghost")  # fmt: skip
GARGANACL = mon("Garganacl", "Leftovers", "Purifying Salt", ["Salt Cure", "Curse", "Protect", "Recover"], evs=WALL,
                nature="Impish", tera="Water")  # fmt: skip
GHOLD_CURSE = mon("Gholdengo", "Leftovers", "Frisk", ["Curse", "Shadow Ball", "Recover", "Nasty Plot"], evs=SPEC,
                  nature="Modest", tera="Steel")  # fmt: skip
VENUSAUR = mon("Venusaur", "Leftovers", "Chlorophyll", ["Leech Seed", "Giga Drain", "Sludge Bomb", "Synthesis"],
               evs=SPEC, nature="Modest", tera="Grass")  # fmt: skip
DRAGAPULT_DB = mon("Dragapult", "Heavy-Duty Boots", "Infiltrator", ["Destiny Bond", "Shadow Ball", "Will-O-Wisp",
                   "Draco Meteor"], evs="4 HP / 252 SpA / 252 Spe", nature="Timid", tera="Ghost")  # fmt: skip
GENGAR = mon("Gengar", "Life Orb", "Frisk", ["Destiny Bond", "Shadow Ball", "Sludge Bomb", "Will-O-Wisp"],
             evs="4 HP / 252 SpA / 252 Spe", nature="Timid", tera="Ghost")  # fmt: skip
GAMBIT_SD = mon("Kingambit", "Leftovers", "Supreme Overlord", ["Kowtow Cleave", "Iron Head", "Sucker Punch",
                "Swords Dance"], tera="Dark")  # fmt: skip
BOLT = mon("Raging Bolt", "Leftovers", "Protosynthesis", ["Thunderclap", "Calm Mind", "Dragon Pulse", "Protect"],
           evs=SPEC, nature="Modest", tera="Fairy")  # fmt: skip
WEAVILE = mon("Weavile", "Heavy-Duty Boots", "Pressure", ["Ice Spinner", "Knock Off", "Fake Out", "First Impression"],
              evs="252 Atk / 4 SpD / 252 Spe", nature="Jolly", tera="Ice")  # fmt: skip
WEAVILE_ENDURE = mon("Weavile", "Heavy-Duty Boots", "Pressure", ["Ice Spinner", "Knock Off", "Swords Dance", "Endure"],
                     evs="252 Atk / 4 SpD / 252 Spe", nature="Jolly", tera="Ice")  # fmt: skip
PEX_BUNKER = mon("Toxapex", "Black Sludge", "Regenerator", ["Baneful Bunker", "Recover", "Surf", "Haze"], evs=WALL,
                 nature="Bold", tera="Fairy")  # fmt: skip
CHESNAUGHT = mon("Chesnaught", "Leftovers", "Overgrow", ["Spiky Shield", "Body Press", "Drain Punch", "Leech Seed"],
                 evs=WALL, nature="Impish", tera="Steel")  # fmt: skip

CONFUSE = {"secondaries": True, "confusionSelfHit": True}
LANTURN_SLOW = mon("Lanturn", "Leftovers", "Volt Absorb", ["Volt Switch", "Scald", "Protect", "Rest"],
                   evs="252 HP / 4 SpA / 252 SpD", nature="Quiet", tera="Water")  # fmt: skip
GENGAR_BOND = mon("Gengar", "Choice Scarf", "Cursed Body", ["Destiny Bond", "Shadow Ball", "Sludge Bomb", "Focus Blast"],
                  evs="252 SpA / 4 SpD / 252 Spe", nature="Timid", tera="Ghost").replace("Ability:", "Level: 40\nAbility:")  # fmt: skip
DOZO = mon("Dondozo", "Leftovers", "Unaware", ["Curse", "Crunch", "Rest", "Sleep Talk"], evs=WALL, nature="Impish",
            tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestVolatiles(MirrorScenarios, unittest.TestCase):
    def test_confusion_from_hurricane_with_self_hits(self):
        # Hurricane confuses (2 turns when forced); the confused Kingambit hits itself, then snaps out of it
        script = [(0, 0), (0, 1), (1, 1), (0, 0), (1, 0), (0, 1)]
        self.play(team(ZAPDOS, PEX), team(GAMBIT, CORV), script,
                  cfg=Config(forced=True, secondaries=True, confusion_hit=True), forced_opts=CONFUSE)  # fmt: skip

    def test_strange_steam_and_water_pulse_confuse(self):
        # Their 20% confusion (forced on); confusion can't be stacked; the confused target hits itself
        steam = mon("Primarina", "Leftovers", "Torrent", ["Strange Steam", "Water Pulse", "Protect", "Moonblast"],
                    evs="252 HP / 252 SpA / 4 SpD", nature="Modest", tera="Water")  # fmt: skip
        script = [(0, 0), (0, 1), (1, 1), (2, 0), (1, 1), (0, 0)]
        self.play(team(steam, PEX), team(GAMBIT, CORV),
                  script, cfg=Config(forced=True, secondaries=True, confusion_hit=True), forced_opts=CONFUSE)  # fmt: skip

    def test_hurricane_never_misses_in_rain(self):
        # Every accuracy check fails, except Hurricane's in rain
        script = [(0, 0), (0, 0), (1, 0), (0, 1)]
        self.play(team(PELIPPER, ZAPDOS), team(GAMBIT, CORV), script, cfg=Config(forced=True, hit=False),
                  forced_opts={"hit": False})  # fmt: skip

    def test_poison_puppeteer_and_confusion_countdown(self):
        # Pecharunt's Toxic also confuses; the confusion counts down without self-hits and ends; a Steel type
        # can't be poisoned, so no confusion
        script = [(0, 0), (1, 0), (2, 0), (2, 1), (8, 8), (1, 1), (1, 0)]
        self.play(team(PECHARUNT, ZAPDOS), team(RILLA, CORV), script)

    def test_swagger_and_misty_terrain(self):
        # Swagger's Attack boost lands in Misty Terrain but its confusion doesn't on a grounded target; a
        # Flying type gets confused (and hits itself)
        script = [(0, 0), (1, 0), (1, 8), (0, 1), (1, 1)]
        self.play(team(CLEFABLE, PEX), team(GAMBIT, CORV), script,
                  cfg=Config(forced=True, confusion_hit=True), forced_opts={"confusionSelfHit": True})  # fmt: skip

    def test_taunt_and_encore(self):
        # Taunt after the target moved lasts an extra turn; Encore locks into the last move; both disable moves;
        # a faster Encore replaces the move the target chose this turn
        script = [(0, 0), (0, 0), (1, 1), (2, 1), (3, 1), (2, 1), (2, 0), (2, 8), (2, 1), (1, 0), (2, 1), (2, 1)]
        self.play(team(GRIMM, ZAPDOS), team(GHOLD, PEX), script)

    def test_psychic_noise_heal_block_and_throat_chop(self):
        # Psychic Noise blocks Recover and Black Sludge healing for 2 turns; Throat Chop blocks sound moves
        script = [(0, 0), (0, 1), (3, 0), (3, 1), (2, 8), (1, 0), (2, 1), (2, 1), (1, 1)]
        self.play(team(PRIMARINA, ZAPDOS), team(PEX, GAMBIT_CHOP), script)

    def test_heal_block_delays_sitrus_berry(self):
        # The berry isn't eaten under Heal Block; it is eaten right when the block ends at the end of the turn
        script = [(0, 0), (0, 0), (0, 0), (0, 0), (2, 0)]
        self.play(team(PRIMARINA, ZAPDOS), team(DRAGAPULT_SITRUS, CORV), script)

    def test_salt_cure_curse_and_leech_seed(self):
        # Salt Cure on a Steel type (1/4), non-Ghost Curse; a Ghost type's Curse halves its HP; Leech Seed drains
        # into whoever is active on the seeder's side; repeats fail
        script = [(0, 0), (0, 1), (1, 0), (8, 0), (0, 1), (0, 0), (1, 2), (0, 1)]
        self.play(team(GARGANACL, VENUSAUR), team(CORV, GHOLD_CURSE), script)

    def test_parting_shot_lowers_stats_and_switches(self):
        # Parting Shot's drops are in onHit, not the boosts field; the user only switches if they land
        pech = mon(
            "Pecharunt",
            "Heavy-Duty Boots",
            "Poison Puppeteer",
            ["Parting Shot", "Shadow Ball", "Recover", "Foul Play"],
            evs=WALL,
            nature="Bold",
            tera="Poison",
        )
        self.play(team(GAMBIT, PEX), team(pech, CORV), [(0, 0), (0, 0)])

    def test_destiny_bond(self):
        # A second Destiny Bond in a row fails; using another move drops it; a KO on a bonded target takes the
        # attacker down too
        script = [(0, 3), (0, 3), (1, 3), (1, 0), (0, 0), (0, 0)]
        self.play(team(DRAGAPULT_DB, GENGAR), team(GAMBIT_SD, CORV), [(0, 0)] + script)


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestMoveMisc(MirrorScenarios, unittest.TestCase):
    def test_sucker_punch_thunderclap_upper_hand(self):
        script = [(0, 0), (0, 1), (1, 0), (0, 0), (3, 0), (3, 2), (0, 3), (2, 3)]
        self.play(team(GAMBIT, ZAPDOS), team(BOLT, CORV), script)

    def test_fake_out_first_impression_ice_spinner(self):
        # Fake Out flinches only on the first turn out; First Impression works on Weavile's first move after
        # switching in; Ice Spinner clears Grassy Terrain
        script = [(0, 0), (0, 1), (0, 1), (8, 1), (3, 1), (0, 1), (3, 1), (2, 1)]
        self.play(team(RILLA, WEAVILE), team(CORV, PEX), script)

    def test_baneful_bunker_spiky_shield_endure(self):
        # Contact into Spiky Shield costs 1/8 of max HP; Endure holds at 1 HP twice in a row; contact into
        # Baneful Bunker poisons
        script = [(0, 0), (1, 0), (3, 1), (3, 1), (1, 8), (1, 0), (1, 0)]
        self.play(team(WEAVILE_ENDURE, GHOLD), team(CHESNAUGHT, PEX_BUNKER), script)

    def test_curse_from_a_non_ghost_ignores_pressure(self):
        # A non-Ghost's Curse targets the user (onModifyMove), so Pressure takes no extra PP, used directly or
        # called by Sleep Talk (regression from L3a battle 6)
        self.play(team(DOZO, PEX), team(CORV, PEX), [(0, 0), (0, 3), (2, 3), (3, 3), (3, 3), (0, 1)])

    def test_pivot_user_fainting_to_destiny_bond_does_not_switch(self):
        # Volt Switch KOs a Destiny Bond user and its user faints too: both sides replace fainted Pokemon, the
        # pivot is gone (regression from L3a battle 2)
        self.play(team(LANTURN_SLOW, PEX), team(GENGAR_BOND, CORV), [(0, 0), (0, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
