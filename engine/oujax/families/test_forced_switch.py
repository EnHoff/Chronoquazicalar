"""
L2 scenarios for the forced_switch family (phazing, Eject Button, Red Card, Eject Pack), played in OU-JAX and in
Showdown with forced outcomes and compared after every step (see engine/oujax/test_l2.py for the action encoding).
Forced random replacements pick the first benched Pokemon in Showdown's current team order.
"""

import unittest

from engine.oujax.ref import harness_available
from engine.oujax.test_l2 import MirrorScenarios, mon, team

WALL = "252 HP / 252 Def / 4 SpD"

CORV_PHAZE = mon("Corviknight", "Leftovers", "Pressure", ["Roar", "Stealth Rock", "Dragon Tail", "Whirlwind"],
                 evs=WALL, nature="Impish", tera="Dragon")  # fmt: skip
PEX = mon("Toxapex", "Black Sludge", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
          nature="Bold", tera="Fairy")  # fmt: skip
PEX_ROAR = mon("Toxapex", "Black Sludge", "Regenerator", ["Roar", "Recover", "Surf", "Haze"], evs=WALL,
               nature="Bold", tera="Fairy")  # fmt: skip
TUSK = mon("Great Tusk", "Leftovers", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
           tera="Ground")  # fmt: skip
TUSK_LO = mon("Great Tusk", "Life Orb", "Protosynthesis", ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
              tera="Ground")  # fmt: skip
GHOLDENGO = mon("Gholdengo", "Leftovers", "Pressure", ["Nasty Plot", "Shadow Ball", "Whirlwind", "Recover"],
                evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
GHOLDENGO_EB = mon("Gholdengo", "Eject Button", "Pressure", ["Nasty Plot", "Shadow Ball", "Protect", "Recover"],
                   evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
GAMBIT_RC = mon("Kingambit", "Red Card", "Pressure", ["Kowtow Cleave", "Iron Head", "Swords Dance", "Protect"],
                evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
PEX_EB = mon("Toxapex", "Eject Button", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
             nature="Bold", tera="Fairy")  # fmt: skip
CORV = mon("Corviknight", "Leftovers", "Pressure", ["U-turn", "Roost", "Defog", "Brave Bird"], evs=WALL,
           nature="Impish", tera="Dragon")  # fmt: skip
BOLT_PACK = mon("Raging Bolt", "Eject Pack", "Protosynthesis", ["Dragon Pulse", "Draco Meteor", "Thunderbolt", "Calm Mind"],
                evs="252 HP / 252 SpA / 4 SpD", nature="Modest", tera="Fairy")  # fmt: skip
PEX_PACK = mon("Toxapex", "Eject Pack", "Regenerator", ["Toxic Spikes", "Recover", "Surf", "Haze"], evs=WALL,
               nature="Bold", tera="Fairy")  # fmt: skip
LANDO_SF = mon("Landorus", "Life Orb", "Sheer Force", ["Rock Slide", "Sludge Wave", "Stone Edge", "Protect"],
               evs="252 HP / 252 SpA / 4 SpD", nature="Mild", tera="Ground")  # fmt: skip
GAMBIT_PS = mon("Kingambit", "Leftovers", "Pressure", ["Parting Shot", "Kowtow Cleave", "Iron Head", "Swords Dance"],
                tera="Dark")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestForcedSwitch(MirrorScenarios, unittest.TestCase):
    def test_roar_dragon_tail_whirlwind(self):
        # Corviknight's Roar beats the slower Toxapex's Roar and drags it out first (its move is lost); Dragon
        # Tail and Whirlwind drag in the first benched Pokemon in Showdown's order, which changes with every
        # switch; the dragged-in Pokemon takes Stealth Rock
        script = [(0, 0), (0, 0), (1, 2), (2, 0), (3, 1), (3, 0)]
        self.play(team(CORV_PHAZE, PEX), team(PEX_ROAR, TUSK, GHOLDENGO), script)

    def test_whirlwind_fails_without_a_bench(self):
        self.play(team(GHOLDENGO, PEX), team(TUSK), [(0, 0), (2, 1), (2, 0)])

    def test_eject_button_and_red_card(self):
        # Eject Button sends Toxapex out before it moves (Kingambit comes in); Red Card drags the Life Orb
        # attacker out (no recoil); Eject Button cancels Corviknight's U-turn
        script = [(0, 0), (1, 0), (1, 1), (0, 9), (3, 0)]
        self.play(team(TUSK_LO, CORV), team(PEX_EB, GAMBIT_RC, GHOLDENGO_EB), script)

    def test_sheer_force_suppresses_eject_button_and_red_card(self):
        # Sheer Force moves skip AfterMoveSecondary (and Life Orb); Stone Edge has no secondary, so it ejects
        # Toxapex and then gets the Sheer Force user dragged out by Red Card
        script = [(0, 0), (1, 1), (2, 1), (0, 2), (2, 2)]
        self.play(team(LANDO_SF, PEX), team(PEX_EB, GAMBIT_RC), script)

    def test_knock_off_removes_eject_button_first(self):
        # Knock Off (onAfterHit) takes the Eject Button before AfterMoveSecondary could use it
        self.play(team(TUSK, CORV), team(PEX_EB, GAMBIT_RC), [(0, 0), (2, 0), (2, 0)])

    def test_eject_pack(self):
        # Draco Meteor's self-drop ejects Raging Bolt after its move, before Kingambit moves; Parting Shot's drops
        # don't eject Toxapex; Defog's evasion drop does
        script = [(0, 0), (1, 0), (1, 2), (0, 0)]
        self.play(team(BOLT_PACK, PEX_PACK, TUSK), team(GAMBIT_PS, CORV), script)


if __name__ == "__main__":
    unittest.main()
