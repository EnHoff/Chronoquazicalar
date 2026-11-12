"""
L2 scenarios for the held_items family, played in OU-JAX and in Showdown with forced outcomes and compared after
every step (see engine/oujax/test_l2.py for the action encoding).
"""

import unittest

from engine.oujax.engine import Config
from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

PHYS = "252 Atk / 4 SpD / 252 Spe"
WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"

WEAVILE = mon("Weavile", "Heavy-Duty Boots", "Pressure", ["Triple Axel", "Knock Off", "Swords Dance", "Protect"],
              evs=PHYS, nature="Jolly", tera="Ice")  # fmt: skip
MAUSHOLD = mon("Maushold", "Loaded Dice", "Technician", ["Population Bomb", "Tidy Up", "Protect", "U-turn"],
               evs=PHYS, nature="Jolly")  # fmt: skip
CORV = mon("Corviknight", "Rocky Helmet", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip


def tusk(item, moves=("Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock")):
    return mon("Great Tusk", item, "Protosynthesis", list(moves), tera="Ground")


def gambit(item, moves=("Kowtow Cleave", "Iron Head", "Swords Dance", "Protect"), evs="252 HP / 252 Atk / 4 Spe"):
    return mon("Kingambit", item, "Pressure", list(moves), evs=evs, tera="Dark")


GLIM_SASH = mon("Glimmora", "Focus Sash", "Corrosion", ["Spikes", "Protect", "Earth Power", "Sludge Bomb"],
                evs="4 HP / 252 SpA / 252 Spe", nature="Timid", tera="Ghost")  # fmt: skip
URSALUNA = mon("Ursaluna", "Flame Orb", "Guts", ["Facade", "Headlong Rush", "Protect", "Swords Dance"], tera="Normal")  # fmt: skip
CORV_TOX = mon("Corviknight", "Toxic Orb", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
               nature="Impish", tera="Dragon")  # fmt: skip
PEX_BARB = mon("Toxapex", "Sticky Barb", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
               nature="Bold", tera="Fairy")  # fmt: skip
CHANSEY = mon("Chansey", "Eviolite", "Natural Cure", ["Seismic Toss", "Soft-Boiled", "Stealth Rock", "Toxic"],
              evs=WALL, nature="Bold", tera="Fairy")  # fmt: skip
GAMBIT_PS = mon("Kingambit", "Leftovers", "Pressure", ["Parting Shot", "Kowtow Cleave", "Iron Head", "Sticky Web"],
                tera="Dark")  # fmt: skip
BOLT_SPRAY = mon("Raging Bolt", "Throat Spray", "Protosynthesis", ["Hyper Voice", "Thunderbolt", "Calm Mind", "Protect"],
                 evs=SPEC, nature="Modest", tera="Fairy")  # fmt: skip
ZAPDOS = mon("Zapdos", "Heavy-Duty Boots", "Pressure", ["Discharge", "Roost", "Heat Wave", "Protect"],
             evs="248 HP / 8 Def / 252 Spe", nature="Timid", tera="Electric")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestHeldItems(MirrorScenarios, unittest.TestCase):
    def test_rocky_helmet_per_hit(self):
        # Rocky Helmet hurts once per contact hit: Triple Axel takes three chunks; Population Bomb's ten hits stop
        # once the attacker faints from the Helmet
        script = [(0, 0), (0, 1), (8, 1), (0, 1)]
        self.play(team(WEAVILE, MAUSHOLD), team(CORV, PEX), script)

    def test_protective_pads_ignore_rocky_helmet(self):
        # Rapid Spin and Knock Off make contact, but not for a Protective Pads holder
        self.play(team(tusk("Protective Pads"), PEX), team(CORV, PEX), [(0, 0), (1, 1), (2, 1)])

    def test_focus_sash(self):
        # Great Tusk's Headlong Rush would OHKO Glimmora: the Sash leaves it at 1 HP (consumed); next time it faints.
        # A second Glimmora with a Sash loses it to Stealth Rock damage first, so the Sash can't save it
        script = [(0, 0), (0, 0), (1, 3), (0, 0), (0, 0)]
        self.play(team(GLIM_SASH, GLIM_SASH.replace("Glimmora", "Glim (Glimmora)", 1)), team(TUSK, PEX), script)

    def test_orbs_and_sticky_barb(self):
        # Flame Orb burns Ursaluna at the end of the turn (Guts Facade); Toxic Orb does nothing to Steel-type
        # Corviknight; Sticky Barb hurts its holder each turn and jumps to an item-less contact attacker
        script = [(0, 0), (3, 1), (0, 1), (8, 8), (0, 1), (0, 1)]
        p1 = team(URSALUNA, tusk(None, ("Headlong Rush", "Rapid Spin", "Close Combat", "Stealth Rock")))
        self.play(p1, team(CORV_TOX, PEX_BARB), script)

    def test_weakness_policy(self):
        # Seismic Toss (fixed damage) doesn't trigger it; Great Tusk's super-effective Headlong Rush does
        script = [(0, 0), (2, 0), (2, 8), (2, 0), (0, 1)]
        self.play(team(gambit("Weakness Policy"), PEX), team(CHANSEY, TUSK), script)

    def test_clear_amulet(self):
        # Parting Shot can't lower its stats (so Kingambit stays in); its own Close Combat drops still apply; a
        # Clear Amulet switch-in ignores Sticky Web
        amulet = tusk("Clear Amulet", ("Headlong Rush", "Rapid Spin", "Close Combat", "Stealth Rock"))
        script = [(0, 0), (1, 0), (1, 3), (8, 1), (2, 2)]
        self.play(
            team(amulet, amulet.replace("Great Tusk", "Tusk (Great Tusk)", 1), PEX), team(GAMBIT_PS, CORV), script
        )

    def test_throat_spray(self):
        # Hyper Voice into Protect doesn't use it; Hyper Voice that hits does (+1 SpA)
        self.play(team(BOLT_SPRAY, PEX), team(ZAPDOS, CORV), [(0, 0), (0, 3), (0, 1), (0, 1)])

    def test_drain_before_rocky_helmet_and_into_substitute(self):
        # Drain heals as the hit lands (spreadDamage), before Rocky Helmet's DamagingHit: at full HP it heals
        # nothing, and the Helmet chip stays; into a Substitute it heals ceil(damage / 2)
        drainer = tusk(None, ("Drain Punch", "Rapid Spin", "Knock Off", "Stealth Rock"))
        sub = mon("Corv (Corviknight)", "Leftovers", "Pressure", ["Substitute", "Roost", "Defog", "Brave Bird"], evs=WALL,
                  nature="Impish", tera="Dragon")  # fmt: skip
        self.play(team(drainer, PEX), team(CORV, sub), [(0, 0), (0, 3), (0, 8), (0, 0), (0, 0)])

    def test_big_root(self):
        drainer = tusk("Big Root", ("Drain Punch", "Rapid Spin", "Knock Off", "Stealth Rock"))
        # Corviknight's Brave Bird hurts Great Tusk, whose Drain Punch then heals 5324/4096 as much
        self.play(team(drainer, PEX), team(CORV, PEX), [(0, 0), (0, 3), (0, 3), (0, 1)])

    def test_custap_berry(self):
        # Belly Drum then Substitute leave Kingambit at exactly 1/4 HP: the Custap Berry moves it ahead of Great
        # Tusk (and is eaten when the turn starts)
        drummer = gambit("Custap Berry", ("Kowtow Cleave", "Belly Drum", "Substitute", "Protect"))
        self.play(team(drummer, PEX), team(TUSK, CORV), [(0, 0), (1, 3), (2, 3), (0, 0)])


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestHeldItemsSecondaries(MirrorScenarios, unittest.TestCase):
    """Every secondary effect applies (forced secondaries on)."""

    def play(self, p1, p2, script):
        super().play(p1, p2, script, cfg=Config(forced=True, secondaries=True), forced_opts={"secondaries": True})

    def test_covert_cloak(self):
        # Iron Head's flinch and Discharge's paralysis don't reach the Covert Cloak holder; they do reach the other
        slow = mon("Toxapex", "Covert Cloak", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
                   nature="Relaxed", tera="Fairy")  # fmt: skip
        plain = slow.replace("Covert Cloak", "Black Sludge").replace("Toxapex", "Pex (Toxapex)", 1)
        fast = gambit("Leftovers").replace("Adamant", "Jolly")
        script = [(0, 0), (2, 0), (2, 8), (2, 2), (8, 2), (2, 2), (2, 8), (2, 0)]
        self.play(team(slow, plain), team(ZAPDOS, fast), script)


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestQuickClaw(MirrorScenarios, unittest.TestCase):
    def test_quick_claw_activates(self):
        # forced: the harness answers Quick Claw's 1-in-5 roll with its thaw setting; it can't beat priority
        claw = gambit("Quick Claw", ("Kowtow Cleave", "Iron Head", "Swords Dance", "Protect"))
        bullet = mon("Scizor", "Leftovers", "Technician", ["Bullet Punch", "U-turn", "Swords Dance", "Roost"], tera="Steel")  # fmt: skip
        script = [(0, 0), (2, 0), (1, 8), (2, 0)]
        self.play(team(claw, PEX), team(TUSK, bullet), script, cfg=Config(forced=True, thaw=True),
                  forced_opts={"thaw": True})  # fmt: skip

    def test_quick_claw_idle(self):
        claw = gambit("Quick Claw", ("Kowtow Cleave", "Iron Head", "Swords Dance", "Protect"))
        self.play(team(claw, PEX), team(TUSK, CORV), [(0, 0), (2, 0), (0, 0)])


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestBlunderPolicy(MirrorScenarios, unittest.TestCase):
    def test_blunder_policy(self):
        # Every accuracy check misses: Headlong Rush misses and the Blunder Policy gives +2 Spe once; Swords Dance
        # has no accuracy check
        blunder = tusk("Blunder Policy", ("Headlong Rush", "Rapid Spin", "Swords Dance", "Stealth Rock"))
        self.play(team(blunder, PEX), team(gambit("Leftovers"), CORV), [(0, 0), (0, 2), (2, 0), (0, 0)],
                  cfg=Config(forced=True, hit=False), forced_opts={"hit": False})  # fmt: skip


if __name__ == "__main__":
    unittest.main()
