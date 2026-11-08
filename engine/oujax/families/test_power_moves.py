"""
L1 checks for moves whose power, type, category or effectiveness depends on the Pokemon or the field (Electro
Ball, Gyro Ball, Rising Voltage, Expanding Force, Psyblade, Barb Barrage, Infernal Parade, Revelation Dance,
Raging Bull, Photon Geyser, Shell Side Arm, Flying Press, Reversal, Flail, Stone Axe with Sheer Force) and for Tera's 60 BP floor skipping
variable-power moves, against
@smogon/calc on random pool attackers and defenders (as engine/oujax/test_damage.py). The battle-state parts
(Charge, Water Shuriken's hits) are L2 scenarios.
"""

import json
import random
import unittest

import jax.numpy as jnp
import numpy as np

from engine.oujax.damage import M_0_5, M_1_5, Combatant, FieldInfo, chain_if, modify
from engine.oujax.effects import T
from engine.oujax.ref import RefHarness, harness_available
from engine.oujax.test_damage import (
    POOL,
    TERRAINS,
    WEATHERS,
    CaseBuilder,
    batch_damage,
    calc_case,
    combatant,
    stack,
)
from engine.oujax.test_l2 import CORV, MirrorScenarios, mon, team

L1_MOVES = (
    "electroball", "gyroball", "risingvoltage", "expandingforce", "psyblade", "barbbarrage", "infernalparade",
    "revelationdance", "ragingbull", "photongeyser", "flyingpress", "lowkick", "grassknot", "heavyslam",
    "eruption", "waterspout", "stoneaxe", "reversal", "flail", "shellsidearm",
)  # fmt: skip
# Attackers that actually use the type-changing moves, so their own rule is exercised
USERS = {
    "revelationdance": ["oricorio", "oricoriopompom", "oricoriopau", "oricoriosensu"],
    "ragingbull": ["taurospaldeacombat", "taurospaldeablaze", "taurospaldeaaqua", "tauros"],
    "photongeyser": ["necrozma", "cresselia", "dragonite", "gholdengo"],
}
USERS = {m: [sp for sp in names if sp in T.species_ids] for m, names in USERS.items()}
PER_MOVE = 120


def speed(c):
    """getStat('spe') as the test's sides allow: boosts, paralysis, Quark Drive / Protosynthesis on Speed."""
    stat = int(c["stats"][5])
    b = c["boosts"][4]
    table = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    stat = int(stat * table[b]) if b >= 0 else int(stat / table[-b])
    mod = 4096
    mod = int(chain_if(mod, c["status"] == "par", M_0_5))
    mod = int(chain_if(mod, c["paradox"] == 5, M_1_5))
    return int(modify(stat, mod))


def with_spe(c) -> Combatant:
    return combatant(c)._replace(spe=speed(c))


class Builder(CaseBuilder):
    def move_case(self, move: str):
        r = self.rng
        case = self.case()
        users = USERS.get(move)
        if users:
            att_set = dict(r.choice(self.sets))
            att_set["species"] = r.choice(users)
            case["att"] = self.side(att_set, case["weather"], case["terrain"], True)
        # Tera often, so the 60 BP floor is checked
        if move in ("lowkick", "grassknot", "heavyslam", "eruption", "waterspout") and r.random() < 0.7:
            mtype = T.dex["moves"][move]["type"].lower()
            case["att"] = dict(case["att"], tera=True, set=dict(case["att"]["set"], tera=mtype))
        # The defender's Protosynthesis / Quark Drive on the stat this move hits rounds differently in calc
        if case["dfn"]["paradox"] in (2, 4):
            case["dfn"] = dict(case["dfn"], ability="pressure", paradox=0)
        # Revelation Dance under Stellar Tera: Showdown keeps the original types (getTypes), calc makes it Stellar;
        # OU-JAX follows Showdown, and the L2 scenarios cover it
        if move == "revelationdance" and case["att"]["tera"] and case["att"]["set"]["tera"] == "stellar":
            case["att"] = dict(case["att"], tera=False)
        # Raging Bull breaks screens before it hits (calc folds that in; the engine's move_try_hit hook does it)
        if move == "ragingbull":
            case.update(reflect=False, light_screen=False, aurora_veil=False)
        # Sheer Force counts Stone Axe's empty secondary
        if move == "stoneaxe" and r.random() < 0.5:
            case["att"] = dict(case["att"], ability="sheerforce", paradox=0)
        case["move"] = T.move_ids[move]
        return case


@unittest.skipUnless(harness_available() and POOL.exists(), "needs the reference harness and M1 team pool")
class TestPowerMovesAgainstCalc(unittest.TestCase):
    def test_moves_match_calc(self):
        builder = Builder(seed=11)
        builder.rng = random.Random(11)
        cases = [builder.move_case(m) for m in L1_MOVES for _ in range(PER_MOVE)]
        with RefHarness() as ref:
            results = ref.calcs([calc_case(c) for c in cases])
        w_id = {w: i for i, w in enumerate(WEATHERS)}
        t_id = {t: i for i, t in enumerate(TERRAINS)}
        field = FieldInfo(
            weather=jnp.array([w_id[c["weather"]] for c in cases]),
            terrain=jnp.array([t_id[c["terrain"]] for c in cases]),
            reflect=jnp.array([c["reflect"] for c in cases]),
            light_screen=jnp.array([c["light_screen"] for c in cases]),
            aurora_veil=jnp.array([c["aurora_veil"] for c in cases]),
        )
        ours = np.asarray(
            batch_damage(
                jnp.array([c["move"] for c in cases]),
                stack([with_spe(c["att"]) for c in cases]),
                stack([with_spe(c["dfn"]) for c in cases]),
                field,
                jnp.array([c["crit"] for c in cases]),
                jnp.array([c["roll"] for c in cases]),
            )
        )
        mismatches = []
        for i, (c, r) in enumerate(zip(cases, results)):
            self.assertNotIn("error", r, r.get("error"))
            dmg = r["damage"]
            want = dmg[15 - c["roll"]] if isinstance(dmg, list) else int(dmg)
            if int(ours[i]) != want:
                mismatches.append((int(ours[i]), want, r["desc"] or json.dumps(calc_case(c))))
        msg = f"{len(mismatches)}/{len(cases)} mismatches, e.g. " + "\n".join(map(str, mismatches[:40]))
        self.assertEqual(mismatches, [], msg)


WALL = "252 HP / 252 Def / 4 SpD"
SPEC = "252 HP / 252 SpA / 4 SpD"
ZAPDOS = mon("Zapdos", "Leftovers", "Pressure", ["Electro Ball", "Charge", "Rising Voltage", "Thunder Wave"],
             evs="248 HP / 8 Def / 252 Spe", nature="Timid", tera="Electric")  # fmt: skip
GAMBIT = mon("Kingambit", "Choice Scarf", "Pressure", ["Gyro Ball", "Iron Head", "Sucker Punch", "Kowtow Cleave"],
             evs=WALL, nature="Impish", tera="Dark")  # fmt: skip
ROTOM = mon("Rotom-Wash", "Leftovers", "Levitate", ["Electric Terrain", "Hydro Pump", "Gyro Ball", "Protect"],
            evs=WALL, nature="Bold", tera="Steel")  # fmt: skip
ORICORIO = mon("Oricorio-Pom-Pom", "Leftovers", "Frisk", ["Revelation Dance", "Photon Geyser", "Swords Dance",
               "Flying Press"], evs="252 Atk / 4 SpD / 252 Spe", nature="Jolly", tera="Stellar")  # fmt: skip
PEX_COUNTER = mon("Toxapex", "Black Sludge", "Regenerator", ["Counter", "Mirror Coat", "Recover", "Haze"], evs=WALL,
                  nature="Bold", tera="Fairy")  # fmt: skip
GRENINJA = mon("Greninja", "Life Orb", "Torrent", ["Water Shuriken", "Expanding Force", "Psychic Terrain", "Toxic"],
               evs="252 SpA / 4 SpD / 252 Spe", nature="Timid", tera="Water")  # fmt: skip
PEX_BB = mon("Toxapex", "Black Sludge", "Regenerator", ["Barb Barrage", "Infernal Parade", "Recover", "Haze"],
             evs=WALL, nature="Impish", tera="Fairy")  # fmt: skip
GALLADE = mon("Gallade", "Leftovers", "Sharpness", ["Psyblade", "Electric Terrain", "Infernal Parade", "Barb Barrage"],
              evs=WALL, nature="Impish", tera="Fighting")  # fmt: skip


@unittest.skipUnless(harness_available(), "reference harness not installed")
class TestPowerMovesL2(MirrorScenarios, unittest.TestCase):
    def test_speed_power_charge_rising_voltage(self):
        # Electro Ball and Gyro Ball from the engine's Speed (Choice Scarf, paralysis); Charge doubles the next
        # Electric move and is used up by it; Rising Voltage doubles against a grounded target in Electric Terrain
        script = [(0, 0), (0, 0), (3, 0), (0, 0), (1, 0), (2, 0), (8, 0), (0, 0), (8, 0), (2, 0)]
        self.play(team(ZAPDOS, ROTOM), team(GAMBIT, CORV), script)

    def test_rising_voltage_misses_out_on_airborne_targets(self):
        script = [(0, 0), (8, 1), (0, 1), (8, 1), (2, 3), (2, 3)]
        self.play(team(ZAPDOS, ROTOM), team(CORV, GAMBIT), script)

    def test_stellar_revelation_dance_photon_geyser_flying_press(self):
        # Stellar Tera keeps Revelation Dance Electric; Photon Geyser turns physical at +2 Atk (Counter returns
        # it); Flying Press adds Flying effectiveness
        script = [(0, 0), (4, 2), (2, 2), (3, 2), (1, 0), (1, 1)]
        self.play(team(ORICORIO, CORV), team(PEX_COUNTER, GAMBIT), script)

    def test_terrain_power_moves_and_water_shuriken(self):
        # Water Shuriken (2-5 hits, +1 priority) fails into Psychic Terrain on a grounded target; Expanding Force
        # is boosted there; Psyblade in Electric Terrain; Barb Barrage doubles on a poisoned target and Infernal
        # Parade on any status
        script = [(0, 0), (0, 0), (2, 0), (0, 0), (1, 1), (1, 0), (3, 2), (8, 3), (0, 2), (1, 0)]
        self.play(team(GRENINJA, PEX_BB), team(GALLADE, PEX_COUNTER), script)

    def test_shell_side_arm_contact_when_physical(self):
        # Physical against Corviknight's lower Def / SpD ratio (so Rocky Helmet hurts), special into Toxapex
        slowbro = mon("Slowbro-Galar", "Leftovers", "Frisk", ["Shell Side Arm", "Slack Off", "Reversal", "Protect"],
                      evs="252 HP / 252 Atk / 4 SpD", nature="Adamant", tera="Poison")  # fmt: skip
        self.play(team(slowbro, ZAPDOS), team(CORV, PEX_COUNTER), [(0, 0), (0, 1), (0, 8), (0, 2), (2, 2)])


if __name__ == "__main__":
    unittest.main()
