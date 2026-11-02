"""
L2 scenarios for the damage_moves family, played in OU-JAX and in Showdown with forced outcomes and compared
after every step (see engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"

WEAVILE = mon("Weavile", "Heavy-Duty Boots", "Pressure", ["Triple Axel", "Knock Off", "Swords Dance", "Protect"],
              evs=PHYS, nature="Jolly", tera="Ice")  # fmt: skip
MAUSHOLD = mon("Maushold", "Loaded Dice", "Technician", ["Population Bomb", "Tidy Up", "Protect", "U-turn"],
               evs=PHYS, nature="Jolly")  # fmt: skip
CLOYSTER = mon("Cloyster", "Heavy-Duty Boots", "Skill Link", ["Icicle Spear", "Rock Blast", "Protect", "Substitute"],
               tera="Ice")  # fmt: skip
HUSTLER = mon("Kleavor", "Leftovers", "Hustle", ["Iron Head", "Knock Off", "Swords Dance", "Protect"], evs=PHYS,
             nature="Jolly", tera="Steel")  # fmt: skip
GAMBIT = mon("Kingambit", "Leftovers", "Pressure", ["Comeuppance", "Metal Burst", "Swords Dance", "Iron Head"],
             tera="Dark")  # fmt: skip
GAMBIT_WALL = mon("Kingambit", "Leftovers", "Pressure", ["Payback", "Avalanche", "Stomping Tantrum", "Protect"],
                  evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
PEX_COUNTER = mon("Toxapex", "Black Sludge", "Regenerator", ["Counter", "Mirror Coat", "Recover", "Haze"], evs=WALL,
                  nature="Bold", tera="Fairy")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip
BOLT = mon("Raging Bolt", "Booster Energy", "Protosynthesis", ["Dragon Pulse", "Draco Meteor", "Thunderbolt", "Calm Mind"],
           evs="252 HP / 252 SpA / 4 SpD", nature="Modest", tera="Fairy")  # fmt: skip
CORV = mon("Corviknight", "Rocky Helmet", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
TYRAN = mon("Tyranitar", "Sitrus Berry", "Sand Stream", ["Stone Edge", "Knock Off", "Stealth Rock", "Protect"],
            tera="Ghost")  # fmt: skip
BRELOOM = mon("Breloom", "Loaded Dice", "Technician", ["Spore", "Bullet Seed", "Protect", "Swords Dance"], evs=PHYS,
              nature="Jolly", tera="Fire")  # fmt: skip
VALIANT = mon("Iron Valiant", "Leftovers", "Quark Drive", ["Brick Break", "High Jump Kick", "Clear Smog", "Protect"],
              evs="252 HP / 252 Atk / 4 Spe", tera="Fighting")  # fmt: skip
ARCHALUDON = mon("Archaludon", "Leftovers", "Pressure", ["Steel Beam", "Body Press", "Protect", "Draco Meteor"],
                 evs="252 HP / 252 SpA / 4 SpD", nature="Modest", tera="Fairy")  # fmt: skip
GRIMM = mon("Grimmsnarl", "Light Clay", "Pressure", ["Reflect", "Light Screen", "Protect", "Swords Dance"], evs=WALL,
            nature="Impish", tera="Dark")  # fmt: skip
GHOLDENGO = mon("Gholdengo", "Leftovers", "Pressure", ["Nasty Plot", "Shadow Ball", "Protect", "Recover"], evs=WALL,
                nature="Bold", tera="Steel")  # fmt: skip
VENUSAUR = mon("Venusaur", "Leftovers", "Chlorophyll", ["Growth", "Strength Sap", "Heal Bell", "Sludge Bomb"],
               evs="252 HP / 252 SpA / 4 Spe", nature="Modest", tera="Grass")  # fmt: skip
AZUMARILL = mon("Azumarill", "Leftovers", "Huge Power", ["Belly Drum", "Aqua Jet", "Play Rough", "Take Heart"],
                evs="252 HP / 252 Atk / 4 SpD", tera="Water")  # fmt: skip
KOMMO = mon("Kommo-o", "Leftovers", "Pressure", ["Clangorous Soul", "Close Combat", "Substitute", "Tidy Up"],
            evs=PHYS, nature="Jolly", tera="Steel")  # fmt: skip
TORKOAL = mon("Torkoal", "Leftovers", "Drought", ["Stealth Rock", "Will-O-Wisp", "Rapid Spin", "Body Press"], evs=WALL,
              nature="Relaxed", tera="Fire")  # fmt: skip
RILLABOOM = mon("Rillaboom", "Leftovers", "Grassy Surge", ["Grassy Glide", "Wood Hammer", "U-turn", "Endeavor"],
                evs="252 HP / 252 Atk / 4 SpD", tera="Grass")  # fmt: skip
HYDRAPPLE = mon("Hydrapple", "Leftovers", "Regenerator", ["Fickle Beam", "Protect", "Draco Meteor", "Earth Power"],
                evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
DRAGAPULT = mon("Dragapult", "Heavy-Duty Boots", "Infiltrator", ["Dragon Darts", "U-turn", "Shadow Ball", "Draco Meteor"],
                evs="252 HP / 4 Def / 252 Spe", nature="Timid", tera="Ghost")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestDamageMoves(MirrorScenarios, unittest.TestCase):
    def test_multi_hit_moves(self):
        # Triple Axel's rising power, Skill Link (5 hits), Loaded Dice (Population Bomb: 10 hits, through
        # Protect and a Substitute), Hustle's Attack boost
        script = [(0, 0), (0, 0), (8, 1), (0, 2), (0, 3), (0, 8), (1, 0)]
        self.play(team(WEAVILE, MAUSHOLD), team(CLOYSTER, HUSTLER), script)

    def test_counter_mirror_coat_metal_burst_comeuppance(self):
        # Comeuppance after a Knock Off; Metal Burst with nothing to return fails; Mirror Coat returns a
        # Thunderbolt; Counter fails against a special move and a switch, then returns a Knock Off
        script = [(0, 0), (0, 2), (1, 3), (8, 8), (1, 2), (0, 0), (0, 8), (0, 2)]
        self.play(team(GAMBIT, PEX_COUNTER), team(TUSK, BOLT), script)

    def test_payback_avalanche_stomping_tantrum(self):
        # Stomping Tantrum into a Flying type fails, so next turn's doubles (into a switch-in); being blocked by
        # Protect is not a failure; Payback doubles behind a faster target; Avalanche doubles only when hit
        script = [(0, 0), (2, 2), (2, 8), (0, 3), (2, 1), (0, 0), (1, 2), (1, 1)]
        self.play(team(GAMBIT_WALL, PEX_COUNTER), team(CORV, TYRAN), script)

    def test_stomping_tantrum_after_sleep_and_loaded_dice(self):
        # Not moving because of sleep counts as a failure; Loaded Dice makes Bullet Seed hit 4-5 times
        self.play(team(GAMBIT_WALL, PEX_COUNTER), team(BRELOOM, TUSK), [(0, 0), (2, 0), (2, 1), (2, 1)])

    def test_screens_clear_smog_crash_and_steel_beam(self):
        # Clear Smog resets Swords Dance; Brick Break shatters Reflect, and Light Screen even into a Ghost type;
        # High Jump Kick crashes into Protect and into a Ghost type; Steel Beam's recoil, also into Protect
        script = [(0, 0), (2, 3), (2, 0), (0, 1), (1, 2), (0, 8), (1, 0), (8, 1), (0, 0), (0, 2)]
        self.play(team(VALIANT, ARCHALUDON), team(GRIMM, GHOLDENGO), script)

    def test_status_moves(self):
        # Growth in sun, Heal Bell, Strength Sap, Belly Drum, Take Heart, Clangorous Soul, Tidy Up (hazards and
        # Substitute)
        script = [(0, 0), (0, 1), (2, 3), (1, 2), (8, 1), (0, 2), (3, 3), (9, 0), (0, 2), (2, 2), (3, 3)]
        self.play(team(VENUSAUR, AZUMARILL, KOMMO), team(TORKOAL, TUSK), script)

    def test_grassy_glide_endeavor_fickle_beam_dragon_darts(self):
        script = [(0, 0), (0, 2), (3, 1), (8, 8), (0, 0), (1, 1)]
        self.play(team(RILLABOOM, HYDRAPPLE), team(TUSK, DRAGAPULT), script)


if __name__ == "__main__":
    unittest.main()
