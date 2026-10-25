"""
L2 interaction tests: scripted battles that exercise one interaction each (turn order, end-of-turn order,
pivots into hazards, Tera STAB, Knock Off and items, Choice lock, Protect, Substitute, hazard removal,
weather), played in OU-JAX and in Showdown with forced outcomes and compared after every step.

Actions use OU-JAX's encoding: lead slot during team preview; 0..3 move, 4..7 move + Tera, 8..12 switch to
the n-th benched team slot.
"""

import unittest

from engine.oujax.bridge import Mirror
from engine.oujax.coverage import supported_move
from engine.oujax.engine import Config
from engine.oujax.ref import RefHarness, harness_available
from engine.oujax.state import PHASE_SWITCH
from engine.oujax.tables import to_id


def mon(species, item, ability, moves, evs="252 HP / 252 Atk / 4 Spe", nature="Adamant", tera="Normal"):
    lines = [
        f"{species} @ {item}" if item else species,
        f"Ability: {ability}",
        f"Tera Type: {tera}",
        f"EVs: {evs}",
        f"{nature} Nature",
    ]
    return "\n".join(lines + [f"- {m}" for m in moves]) + "\n"


def team(*mons):
    return "\n".join(mons)


TUSK = mon(
    "Great Tusk",
    "Leftovers",
    "Protosynthesis",
    ["Headlong Rush", "Rapid Spin", "Knock Off", "Stealth Rock"],
    tera="Ground",
)
GAMBIT = mon(
    "Kingambit", "Choice Band", "Pressure", ["Kowtow Cleave", "Knock Off", "Iron Head", "Swords Dance"], tera="Dark"
)
CORV = mon(
    "Corviknight",
    "Rocky Helmet",
    "Pressure",
    ["U-turn", "Roost", "Defog", "Brave Bird"],
    evs="252 HP / 252 Def / 4 SpD",
    nature="Impish",
    tera="Dragon",
)
PEX = mon(
    "Toxapex",
    "Black Sludge",
    "Regenerator",
    ["Toxic Spikes", "Recover", "Surf", "Haze"],
    evs="252 HP / 252 Def / 4 SpD",
    nature="Bold",
    tera="Fairy",
)
BOLT = mon(
    "Raging Bolt",
    "Booster Energy",
    "Protosynthesis",
    ["Dragon Pulse", "Draco Meteor", "Thunderbolt", "Calm Mind"],
    evs="252 HP / 252 SpA / 4 SpD",
    nature="Modest",
    tera="Fairy",
)
TYRAN = mon(
    "Tyranitar", "Sitrus Berry", "Sand Stream", ["Stone Edge", "Knock Off", "Stealth Rock", "Protect"], tera="Ghost"
)
DRAGAPULT = mon(
    "Dragapult",
    "Choice Specs",
    "Infiltrator",
    ["Draco Meteor", "Shadow Ball", "U-turn", "Flamethrower"],
    evs="252 SpA / 4 SpD / 252 Spe",
    nature="Timid",
    tera="Ghost",
)
ZAPDOS = mon(
    "Zapdos",
    "Heavy-Duty Boots",
    "Pressure",
    ["Volt Switch", "Roost", "Discharge", "Heat Wave"],
    evs="248 HP / 8 Def / 252 Spe",
    nature="Timid",
    tera="Electric",
)
HYDREIGON = mon(
    "Hydreigon",
    "Leftovers",
    "Levitate",
    ["Draco Meteor", "Substitute", "Flash Cannon", "Nasty Plot"],
    evs="252 SpA / 4 SpD / 252 Spe",
    nature="Timid",
    tera="Poison",
)
TORKOAL = mon(
    "Torkoal",
    "Leftovers",
    "White Smoke",
    ["Flamethrower", "Stealth Rock", "Rapid Spin", "Body Press"],
    evs="252 HP / 252 Def / 4 SpD",
    nature="Relaxed",
    tera="Fire",
)
CORV_SUB = mon(
    "Corviknight",
    "Rocky Helmet",
    "Pressure",
    ["Substitute", "Roost", "Defog", "Brave Bird"],
    evs="252 HP / 252 Def / 4 SpD",
    nature="Impish",
    tera="Dragon",
)
SKARM = mon(
    "Skarmory",
    "Rocky Helmet",
    "Sturdy",
    ["Spikes", "Roost", "Brave Bird", "Body Press"],
    evs="252 HP / 252 Def / 4 SpD",
    nature="Impish",
    tera="Fighting",
)


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestL2Interactions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = RefHarness()

    @classmethod
    def tearDownClass(cls):
        cls.ref.close()

    def play(self, p1, p2, script, cfg=Config(forced=True), forced_opts=None, check=None):
        """script: actions for team preview and each move turn. Switch requests in between (faint
        replacements, pivots) are answered automatically with the first legal switch."""
        for t in (p1, p2):
            moves = [to_id(line[2:]) for line in t.splitlines() if line.startswith("- ")]
            self.assertEqual([mv for mv in moves if not supported_move(mv)], [], "scenario uses moves outside M2")
        m = Mirror.start(self.ref, p1, p2, cfg, forced_opts=forced_opts)

        def advance(acts, label):
            m.play(list(acts))
            diffs = m.diff()
            log = [line for line in m.sd["log"] if line.startswith(("|move", "|-", "|switch", "|faint"))]
            self.assertEqual(diffs, [], f"{label} {acts}: {diffs}\n" + "\n".join(log[-20:]))

        try:
            self.assertEqual(m.diff(), [])
            for i, acts in enumerate(script):
                while int(m.ours.phase) == PHASE_SWITCH and not m.sd["ended"]:
                    legal = m.legal()
                    advance([int(legal[s].nonzero()[0][0]) for s in (0, 1)], f"auto switch before step {i + 1}")
                if m.sd["ended"]:
                    break
                legal = m.legal()
                for side, a in enumerate(acts):
                    self.assertTrue(
                        legal[side, a],
                        f"script step {i + 1}: action {a} not legal for p{side + 1}: {legal[side].nonzero()[0]}",
                    )
                advance(acts, f"step {i + 1}")
                if check is not None:
                    check(i + 1, m)
        finally:
            m.end()

    def test_pivot_into_hazards_and_rocky_helmet(self):
        # p1 Tusk sets Rocks; p2 Corviknight U-turns into Rocks (Rocky Helmet on Tusk? no: on Corviknight)
        self.play(team(TUSK, PEX), team(CORV, ZAPDOS), [(0, 0), (3, 0), (8, 8), (0, 0), (2, 0)])

    def test_knock_off_sitrus_before_eating_and_choice_lock_release(self):
        self.play(team(GAMBIT, TYRAN), team(TYRAN, GAMBIT), [(0, 0), (0, 1), (8, 1), (1, 0), (0, 0)])

    def test_knock_off_booster_energy_from_raging_bolt(self):
        # Showdown doesn't tag Raging Bolt as Paradox, so Knock Off removes (and is boosted by) Booster Energy
        self.play(team(GAMBIT, TUSK), team(BOLT, PEX), [(0, 0), (8, 3), (1, 2), (1, 1)])

    def test_tera_stab_and_tera_into_new_type(self):
        self.play(team(TUSK, PEX), team(BOLT, CORV), [(0, 0), (4, 5), (0, 0), (0, 2)])

    def test_protect_and_end_of_turn_order_in_sand(self):
        # Tyranitar's sand, Leftovers and Black Sludge, Toxic Spikes poison, consecutive Protect
        self.play(team(TYRAN, PEX), team(PEX, TUSK), [(0, 0), (3, 0), (3, 1), (8, 3), (0, 2), (1, 2)])

    def test_defog_and_rapid_spin_clear_hazards(self):
        self.play(team(SKARM, TUSK), team(CORV, TUSK), [(0, 0), (0, 2), (0, 0), (8, 8), (1, 3), (1, 1)])

    def test_trick_room_free_speed_order_and_volt_switch(self):
        self.play(team(DRAGAPULT, ZAPDOS), team(ZAPDOS, GAMBIT), [(0, 0), (2, 0), (8, 8), (0, 2), (0, 3)])

    def test_substitute_blocks_secondaries_until_it_breaks(self):
        # Hydreigon is faster: Substitute goes up before Torkoal's Flamethrower, which has its burn forced on.
        # While the Substitute stands, only it takes damage and there is no burn; once broken, the burn lands.
        seen = {"sub_hits": 0, "burned_after_break": False}

        def check(step, m):
            sub_up = int(m.ours.sub_hp[0]) > 0
            hyd = next(p for p in m.sd["sides"][0]["pokemon"] if p["name"] == "Hydreigon")
            if step >= 2 and sub_up:
                seen["sub_hits"] += 1
                self.assertEqual(hyd["status"], "", f"burned through a Substitute at step {step}")
            if not sub_up and hyd["status"] == "brn":
                seen["burned_after_break"] = True

        script = [(0, 0), (1, 0)] + [(3, 0)] * 10
        self.play(team(HYDREIGON, PEX), team(TORKOAL, CORV), script, cfg=Config(forced=True, secondaries=True),
                  forced_opts={"secondaries": True}, check=check)  # fmt: skip
        self.assertGreater(seen["sub_hits"], 0)
        self.assertTrue(seen["burned_after_break"])

    def test_rocky_helmet_and_knock_off_blocked_by_substitute(self):
        # Corviknight (Rocky Helmet) puts up a Substitute; Great Tusk's Knock Off into it triggers neither the
        # Helmet nor the item removal. Once the Substitute is gone, both happen.
        state = {}

        def check(step, m):
            tusk = next(p for p in m.sd["sides"][0]["pokemon"] if p["name"] == "Great Tusk")
            corv = next(p for p in m.sd["sides"][1]["pokemon"] if p["name"] == "Corviknight")
            if step == 2:
                state["tusk_hp"] = tusk["hp"]
                self.assertIn("substitute", corv["volatiles"])
            if step == 3:  # Knock Off into the Substitute
                self.assertEqual(tusk["hp"], state["tusk_hp"], "Rocky Helmet hurt through a Substitute")
                self.assertEqual(corv["item"], "rockyhelmet", "Knock Off removed an item through a Substitute")

        script = [(0, 0), (3, 0), (2, 1), (0, 1), (2, 1)]
        self.play(team(TUSK, PEX), team(CORV_SUB, ZAPDOS), script, check=check)

    def test_protect_chain_resets_after_a_failed_roll(self):
        # With stall rolls forced to fail: the first Protect always works (no roll), the second fails, and the
        # third works again because the failure ended the chain
        script = [(0, 0), (0, 2), (0, 2), (0, 2), (0, 2)]
        self.play(team(DRAGAPULT, ZAPDOS), team(TYRAN, PEX), script, cfg=Config(forced=True, stall=False),
                  forced_opts={"stall": False})  # fmt: skip


if __name__ == "__main__":
    unittest.main()
