"""
L1 damage tests for OU-JAX: thousands of random attacker / defender / move / field cases drawn from the M1
team pool, compared roll by roll against @smogon/calc through the reference harness.

Items and abilities outside the implemented damage modifiers are neutralised on both sides (no item;
Pressure, which has no damage effect) so the comparison isolates what OU-JAX implements.
"""

import json
import random
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from engine.oujax.damage import Combatant, FieldInfo, blocked_by_psychic_terrain, damage
from engine.oujax.effects import _ITEM_EFFECTS as ITEM_EFFECTS
from engine.oujax.effects import MOVE, T
from engine.oujax.ref import RefHarness, harness_available
from engine.oujax.tables import MC, SC, STATS
from engine.oujax.team import calc_stats

REPO = Path(__file__).resolve().parents[2]
POOL = REPO / "data" / "ou" / "generated" / "team_pool.jsonl"
N_CASES = 4000

DAMAGE_ITEMS = {
    "choiceband", "choicespecs", "lifeorb", "expertbelt", "assaultvest", "airballoon", "boosterenergy",
    "wellspringmask", "hearthflamemask", "cornerstonemask", "souldew", "wiseglasses", "muscleband", "punchingglove",
    "eviolite", *ITEM_EFFECTS["type_boost_items"], *ITEM_EFFECTS["resist_berries"],
}  # fmt: skip
DAMAGE_ABILITIES = {
    "protosynthesis", "quarkdrive", "levitate", "guts", "technician", "ironfist", "reckless", "toughclaws",
    "sharpness", "strongjaw", "megalauncher", "supremeoverlord", "sandforce", "sheerforce", "punkrock", "hugepower",
    "purepower", "torrent", "blaze", "overgrow", "swarm", "transistor", "dragonsmaw", "steelworker", "rockypayload",
    "waterbubble", "solarpower", "adaptability", "tintedlens", "sniper", "liquidvoice", "multiscale", "shadowshield",
    "filter", "solidrock", "prismarmor", "fluffy", "icescales", "thickfat", "heatproof", "purifyingsalt", "furcoat",
    "vesselofruin", "tabletsofruin", "swordofruin", "beadsofruin", "unaware", "moldbreaker", "teravolt",
    "turboblaze", "infiltrator",
}  # fmt: skip
SPECIAL_MOVES = {
    MOVE.knockoff, MOVE.facade, MOVE.hex, MOVE.acrobatics, MOVE.weatherball, MOVE.lowkick, MOVE.grassknot,
    MOVE.heavyslam, MOVE.heatcrash, MOVE.eruption, MOVE.waterspout, MOVE.dragonenergy, MOVE.storedpower,
    MOVE.powertrip, MOVE.ruination, MOVE.superfang, MOVE.freezedry, MOVE.collisioncourse, MOVE.electrodrift,
    MOVE.ivycudgel,
}  # fmt: skip
WEATHERS = ["", "Sun", "Rain", "Sand", "Snow"]
TERRAINS = ["", "Electric", "Grassy", "Misty", "Psychic"]
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe")


def supported_damage_move(mid: int) -> bool:
    row = T.moves[mid]
    if row[MC["category"]] == 2 or row[MC["bp"]] == 0:
        return False  # incl. Ruination / Super Fang, which @smogon/calc doesn't model (Showdown L2 covers them)
    if row[MC["multihit_max"]] or row[MC["fixed_damage"]] or row[MC["ohko"]]:
        return False
    return not row[MC["has_handlers"]] or mid in SPECIAL_MOVES


def load_sets(n_teams=3000):
    with open(POOL) as f:
        lines = f.readlines()
    rng = random.Random(1)
    sets = []
    for line in rng.sample(lines, min(n_teams, len(lines))):
        sets += json.loads(line)["sets"]
    return sets


def paradox_stat(ability, item, weather, terrain, stats, boosts):
    """Which stat Protosynthesis / Quark Drive boosts (0 = inactive): best stat including boosts."""
    active = (ability == "protosynthesis" and (weather == "Sun" or item == "boosterenergy")) or (
        ability == "quarkdrive" and (terrain == "Electric" or item == "boosterenergy")
    )
    if not active:
        return 0
    table = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    best, best_val = 0, -1
    for i in range(1, 6):
        b = boosts[i - 1]
        val = int(stats[i] * table[b]) if b >= 0 else int(stats[i] / table[-b])
        if val > best_val:
            best, best_val = i, val
    return best


class CaseBuilder:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.sets = load_sets()

    def side(self, s, weather, terrain, as_attacker):
        r = self.rng
        item = s["item"] if s["item"] in DAMAGE_ITEMS else ""
        ability = s["ability"] if s["ability"] in DAMAGE_ABILITIES else "pressure"
        boosts = [r.choice([0, 0, 0, -1, 1, 2, -2]) for _ in BOOST_KEYS]
        status = r.choice(["", "", "brn", "par"]) if as_attacker else r.choice(["", "", "par"])
        tera = r.random() < 0.3
        if tera and as_attacker and r.random() < 0.15:
            s = dict(s, tera="stellar")  # Stellar Tera (first use of each type is boosted)
        sid = T.species_ids[s["species"]]
        base = T.species[sid, SC["base_hp"] : SC["base_spe"] + 1]
        stats = calc_stats(
            base, np.array(s["evs"]), np.array(s["ivs"]), np.array(100), T.natures[T.nature_ids[s["nature"]]]
        )
        hp = int(stats[0]) if r.random() < 0.5 else r.randint(1, int(stats[0]))
        fallen = r.randint(0, 5) if as_attacker else 0
        return dict(
            set=s, item=item, ability=ability, boosts=boosts, status=status, tera=tera, sid=sid, stats=stats,
            paradox=paradox_stat(ability, item, weather, terrain, stats, boosts), hp=hp, fallen=fallen,
        )  # fmt: skip

    def case(self):
        r = self.rng
        while True:
            a_set, d_set = r.choice(self.sets), r.choice(self.sets)
            moves = [T.move_ids[m] for m in a_set["moves"] if supported_damage_move(T.move_ids[m])]
            if moves:
                break
        weather, terrain = r.choice(WEATHERS), r.choice(TERRAINS)
        move = r.choice(moves)
        # @smogon/calc applies defensive Protosynthesis / Quark Drive as 5324/4096, Showdown as 5325/4096, so they
        # can round differently; OU-JAX follows Showdown and the full-battle L3 tests cover it
        def_stat_for_move = 2 if T.moves[move, MC["category"]] == 0 else 4
        if T.moves[move, MC["def_stat"]]:
            def_stat_for_move = int(T.moves[move, MC["def_stat"]])
        if move == MOVE.knockoff and d_set["item"] == "boosterenergy":
            # @smogon/calc treats Booster Energy as unremovable from every Paradox Pokemon, Showdown only from
            # species tagged Paradox in its dex (not Raging Bolt, Gouging Fire, Iron Boulder, Iron Crown).
            # Showdown is checked separately in the L2 tests.
            d_set = dict(d_set, item="leftovers")
        att = self.side(a_set, weather, terrain, True)
        dfn = self.side(d_set, weather, terrain, False)
        if dfn["paradox"] == def_stat_for_move:
            dfn = dict(dfn, ability="pressure", paradox=0)
        return dict(
            move=move,
            att=att,
            dfn=dfn,
            weather=weather, terrain=terrain,
            reflect=r.random() < 0.15, light_screen=r.random() < 0.15, aurora_veil=r.random() < 0.1,
            crit=r.random() < 0.2, roll=r.randrange(16),
        )  # fmt: skip


def calc_side(c):
    s = c["set"]
    dex = T.dex
    out = {
        "species": dex["species"][s["species"]]["name"],
        "level": 100,
        "item": dex["items"][c["item"]]["name"] if c["item"] else "",
        "ability": dex["abilities"][c["ability"]]["name"],
        "nature": dex["natures"][s["nature"]]["name"],
        "evs": dict(zip(STATS, s["evs"])),
        "ivs": dict(zip(STATS, s["ivs"])),
        "boosts": dict(zip(BOOST_KEYS, c["boosts"])),
        "status": c["status"],
        "curHP": c["hp"],
        "alliesFainted": c["fallen"],
    }
    if c["tera"]:
        out["teraType"] = dex["types"][s["tera"]]["name"]
    if c["paradox"]:
        out["boostedStat"] = "auto"  # calc only applies Protosynthesis / Quark Drive when asked to
    return out


def calc_case(case):
    return {
        "attacker": calc_side(case["att"]),
        "defender": calc_side(case["dfn"]),
        "move": {
            "name": T.dex["moves"][[k for k, v in T.move_ids.items() if v == case["move"]][0]]["name"],
            "isCrit": case["crit"],
            "isStellarFirstUse": case["att"]["tera"] and case["att"]["set"]["tera"] == "stellar",
        },
        "field": {
            "weather": case["weather"] or None,
            "terrain": case["terrain"] or None,
            "defenderSide": {
                "isReflect": case["reflect"],
                "isLightScreen": case["light_screen"],
                "isAuroraVeil": case["aurora_veil"],
            },
        },
    }


def combatant(c) -> Combatant:
    s = c["set"]
    sid = c["sid"]
    from engine.oujax.tables import STATUS_ID

    return Combatant(
        species=sid,
        level=100,
        stats=c["stats"].astype(np.int32),
        boosts=np.array(c["boosts"] + [0, 0], np.int32),
        base_types=T.species[sid, [SC["type1"], SC["type2"]]].astype(np.int32),
        tera_type=T.type_ids[s["tera"]],
        terastallized=c["tera"],
        status=STATUS_ID[c["status"]],
        item=T.item_ids[c["item"]],
        ability=T.ability_ids[c["ability"]],
        paradox_stat=c["paradox"],
        hp=c["hp"],
        fallen=c["fallen"],
    )


def stack(items):
    return jax.tree.map(lambda *xs: jnp.asarray(np.stack([np.asarray(x) for x in xs])), *items)


@jax.jit
def batch_damage(move, att, dfn, field, crit, roll):
    def one(m, a, d, f, c, r):
        return jnp.where(blocked_by_psychic_terrain(m, d, f), 0, damage(m, a, d, f, c, r))

    return jax.vmap(one)(move, att, dfn, field, crit, roll)


@unittest.skipUnless(harness_available() and POOL.exists(), "needs the reference harness and M1 team pool")
class TestDamageAgainstCalc(unittest.TestCase):
    def test_random_cases_match_calc(self):
        builder = CaseBuilder(seed=0)
        cases = [builder.case() for _ in range(N_CASES)]
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
        ours = batch_damage(
            jnp.array([c["move"] for c in cases]),
            stack([combatant(c["att"]) for c in cases]),
            stack([combatant(c["dfn"]) for c in cases]),
            field,
            jnp.array([c["crit"] for c in cases]),
            jnp.array([c["roll"] for c in cases]),
        )
        ours = np.asarray(ours)
        mismatches = []
        for i, (c, r) in enumerate(zip(cases, results)):
            self.assertNotIn("error", r, r.get("error"))
            dmg = r["damage"]
            want = dmg[15 - c["roll"]] if isinstance(dmg, list) else int(dmg)
            if int(ours[i]) != want:
                mismatches.append((int(ours[i]), want, r["desc"] or json.dumps(calc_case(c))))
        rate = len(mismatches) / len(cases)
        msg = f"{len(mismatches)}/{len(cases)} mismatches, e.g. " + "\n".join(map(str, mismatches[:8]))
        self.assertEqual(mismatches, [], msg)
        self.assertEqual(rate, 0.0)


if __name__ == "__main__":
    unittest.main()
