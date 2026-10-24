"""
data/ou/test_ou_data.py
Unit tests for the OU data pipeline: set parsing/normalisation, validation, dedup, archetype tags,
set archetypes, usage marginals and meta coverage. Small fixtures only, no network. The Showdown
validator test runs only when a native node and the local Showdown clone are available.
"""

import json
import tempfile
import unittest
from pathlib import Path

from data.ou.archetypes import _Cluster, build_set_archetypes, ev_bucket, merge_clusters
from data.ou.coverage import build_meta_coverage, team_elements
from data.ou.sources import DEFAULT_SHOWDOWN_DIR
from data.ou.stats import build_usage_marginals, species_marginals
from data.ou.teams import (
    archetype_tags,
    dedupe,
    normalize_set,
    recency_weight,
    set_role,
    set_similarity,
    team_key,
    team_similarity,
)
from data.ou.validator import find_node, problem_kind, validate_teams


def mon(species, item="leftovers", ability="pressure", tera="steel", nature="bold", evs=None, moves=None):
    return {
        "species": species,
        "item": item,
        "ability": ability,
        "tera": tera,
        "nature": nature,
        "evs": evs or [252, 0, 252, 0, 4, 0],
        "ivs": [31] * 6,
        "moves": sorted(moves or ["protect", "toxic", "roost", "uturn"]),
    }


def team(names, weight=1.0, **overrides):
    return {"sets": [mon(n, **overrides) for n in names], "recency_weight": weight}


SIX = ["a", "b", "c", "d", "e", "f"]

SAMPLE_EXPORT = """Great Tusk @ Booster Energy
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Knock Off
"""


class TestNormalise(unittest.TestCase):
    def test_normalize_showdown_set(self):
        s = normalize_set(
            {
                "species": "Great Tusk",
                "item": "Booster Energy",
                "ability": "Protosynthesis",
                "teraType": "Steel",
                "nature": "Jolly",
                "evs": {"hp": 0, "atk": 252, "def": 0, "spa": 0, "spd": 4, "spe": 252},
                "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
                "moves": ["Rapid Spin", "Headlong Rush", "Ice Spinner", "Knock Off"],
            }
        )
        self.assertEqual(s["species"], "greattusk")
        self.assertEqual(s["item"], "boosterenergy")
        self.assertEqual(s["tera"], "steel")
        self.assertEqual(s["evs"], [0, 252, 0, 0, 4, 252])
        self.assertEqual(s["moves"], ["headlongrush", "icespinner", "knockoff", "rapidspin"])

    def test_missing_ivs_default_to_31(self):
        s = normalize_set({"species": "Ditto", "moves": ["Transform"], "ivs": {"atk": 0}})
        self.assertEqual(s["ivs"], [31, 0, 31, 31, 31, 31])


class TestDedup(unittest.TestCase):
    def test_team_key_ignores_order(self):
        t1 = team(SIX)["sets"]
        self.assertEqual(team_key(t1), team_key(list(reversed(t1))))

    def test_set_similarity(self):
        a = mon("x")
        self.assertEqual(set_similarity(a, a), 1.0)
        b = dict(a, item="choicescarf", moves=sorted(["protect", "toxic", "roost", "knockoff"]))
        self.assertAlmostEqual(set_similarity(a, b), 7 / 9)

    def test_exact_and_near_duplicates(self):
        base = team(SIX, weight=0.5)
        exact = team(list(reversed(SIX)), weight=1.0)
        near = team(SIX, weight=0.25)
        near["sets"][0] = dict(near["sets"][0], item="choicescarf")  # 1 of 54 fields differs
        other_species = team(["a", "b", "c", "d", "e", "g"])
        different = team(SIX, nature="adamant", item="lifeorb", tera="fire")  # 3 of 9 fields differ per set
        kept, stats = dedupe([base, exact, near, other_species, different])
        self.assertEqual(stats, {"exact": 1, "near": 1})
        self.assertEqual(len(kept), 3)
        rep = next(t for t in kept if team_similarity(t["sets"], base["sets"]) == 1.0)
        self.assertEqual(rep["copies"], 3)
        self.assertEqual(rep["recency_weight"], 1.0)  # newest copy is kept

    def test_different_species_are_never_similar(self):
        self.assertEqual(team_similarity(team(SIX)["sets"], team(["a", "b", "c", "d", "e", "g"])["sets"]), 0.0)


class TestRecency(unittest.TestCase):
    def test_half_life(self):
        self.assertEqual(recency_weight("2026-09", "2026-09"), 1.0)
        self.assertAlmostEqual(recency_weight("2026-06", "2026-09"), 0.5)
        self.assertAlmostEqual(recency_weight("2025-12", "2026-09"), 0.125)
        self.assertEqual(recency_weight(None, "2026-09"), 0.5)


class TestArchetypeTags(unittest.TestCase):
    def test_weather_and_hazards(self):
        sets = team(SIX)["sets"]
        sets[0] = mon("pelipper", ability="drizzle", moves=["hurricane", "uturn", "roost", "surf"])
        sets[1] = mon("tinglu", moves=["stealthrock", "ruination", "earthquake", "whirlwind"])
        sets[2] = mon("glimmora", moves=["spikes", "toxicspikes", "mortalspin", "powergem"])
        tags = archetype_tags(sets)
        for tag in ("rain", "hazard_stack", "hazard_removal"):
            self.assertIn(tag, tags)

    def test_styles(self):
        attacker = dict(evs=[0, 252, 0, 0, 4, 252], item="choiceband", moves=["closecombat"])
        wall = dict(evs=[252, 0, 252, 0, 4, 0], item="leftovers", moves=["recover", "toxic"])
        self.assertEqual(set_role(mon("x", **attacker)), "offensive")
        self.assertEqual(set_role(mon("x", **wall)), "defensive")
        self.assertIn("hyper_offence", archetype_tags([mon(n, **attacker) for n in SIX]))
        self.assertIn("stall", archetype_tags([mon(n, **wall) for n in SIX]))
        mixed = [mon(n, **attacker) for n in SIX[:3]] + [mon(n, **wall) for n in SIX[3:]]
        self.assertIn("balance", archetype_tags(mixed))


class TestSetArchetypes(unittest.TestCase):
    def test_ev_bucket(self):
        self.assertEqual(ev_bucket([252, 0, 4, 0, 0, 252]), "hp+spe")
        self.assertEqual(ev_bucket([4, 0, 0, 0, 0, 0]), "none")

    def test_merge_requires_three_shared_moves_and_same_item(self):
        def cluster(moves, item="leftovers", w=1.0):
            c = _Cluster((item, tuple(sorted(moves)), "hp+def"))
            c.add(mon("x", item=item, moves=moves), w, "pool")
            return c

        a = cluster(["a", "b", "c", "d"], w=3.0)
        b = cluster(["a", "b", "c", "e"])  # 3 shared -> merges into a
        c = cluster(["a", "b", "x", "y"])  # 2 shared -> own archetype
        d = cluster(["a", "b", "c", "d"], item="choicescarf")  # different item -> own archetype
        merged = merge_clusters([a, b, c, d], max_clusters=10)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0].weight, 4.0)
        self.assertEqual(merge_clusters([cluster(["a", "b", "c", "d"]), cluster(["w", "x", "y", "z"])], 1)[0].weight, 2)

    def test_seed_sets_fill_missing_ability(self):
        smogon = {"Great Tusk": {"Spinner": {"moves": ["Rapid Spin", ["Headlong Rush", "Earthquake"]],
                                             "item": "Heavy-Duty Boots", "nature": "Jolly",
                                             "evs": {"atk": 252, "spe": 252}, "teratypes": ["Steel"]}}}  # fmt: skip
        out = build_set_archetypes([], smogon_sets=smogon, usage_marginals={
            "species": {"greattusk": {"abilities": {"protosynthesis": 1.0}}}})  # fmt: skip
        s = out["greattusk"]["sets"][0]
        self.assertEqual(s["abilities"], {"protosynthesis": 1.0})
        self.assertEqual(s["moves"], ["headlongrush", "rapidspin"])
        self.assertEqual(s["ev_bucket"], "atk+spe")


class TestUsageMarginals(unittest.TestCase):
    def entry(self, raw, usage, ability="pressure"):
        return {
            "Raw count": raw,
            "usage": usage,
            "Abilities": {ability: 100.0},
            "Moves": {"roost": 90.0, "uturn": 50.0, "": 10.0},
            "Items": {"leftovers": 80.0, "heavydutyboots": 20.0},
            "Tera Types": {"steel": 100.0},
            "Spreads": {"Bold:252/0/252/0/4/0": 100.0},
            "Teammates": {"Great Tusk": 30.0},
        }

    def test_species_marginals_normalise_by_species_count(self):
        m = species_marginals(self.entry(1000, 0.1))
        self.assertAlmostEqual(m["moves"]["roost"], 0.9)
        self.assertNotIn("", m["moves"])
        self.assertEqual(m["teammates"], {"greattusk": 0.3})

    def test_cutoff_fallback_and_recency(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for month, usage in (("2026-09", 0.2), ("2026-06", 0.1)):
                paths[month] = {}
                for cutoff, raw in ((1825, 50), (1695, 5000)):  # too few at 1825 -> 1695
                    p = Path(d) / f"{month}-{cutoff}.json"
                    p.write_text(json.dumps({"data": {"Corviknight": self.entry(raw, usage + cutoff / 1e5)}}))
                    paths[month][cutoff] = p
            out = build_usage_marginals(paths)
        rec = out["species"]["corviknight"]
        self.assertEqual(set(rec["cutoff_by_month"].values()), {1695})
        # weights 1.0 (2026-09) and 0.5 (2026-06)
        self.assertAlmostEqual(rec["usage"], (1.0 * 0.21695 + 0.5 * 0.11695) / 1.5, places=5)


class TestCoverage(unittest.TestCase):
    def test_team_elements_skip_empty(self):
        elems = team_elements([mon("x", item="", moves=["a", "b"])])
        self.assertEqual(elems, {("move", "a"), ("move", "b"), ("ability", "pressure")})

    def test_greedy_cover(self):
        common = dict(item="leftovers", ability="pressure", moves=["a", "b", "c", "d"])
        pool = [
            {"weight": 9.0, "sets": [mon("x", **common)]},
            {"weight": 1.0, "sets": [mon("y", item="lifeorb", ability="levitate", moves=["a", "b", "c", "z"])]},
        ]
        cov = build_meta_coverage(pool, {"species": {}}, targets=(0.9, 1.0))
        self.assertEqual(cov["covers"]["90"]["counts"], {"move": 4, "item": 1, "ability": 1})
        self.assertAlmostEqual(cov["covers"]["90"]["coverage"], 0.9)
        self.assertEqual(cov["covers"]["100"]["counts"], {"move": 5, "item": 2, "ability": 2})
        self.assertEqual(cov["ranking"][-1]["cum_coverage"], 1.0)


class TestValidation(unittest.TestCase):
    def test_problem_kind(self):
        self.assertEqual(problem_kind("Kyurem's move Tera Blast is banned."), "banned")
        self.assertEqual(problem_kind("Pikachu can't learn Earthquake."), "illegal move")
        self.assertEqual(problem_kind("import failed: no sets parsed"), "unparseable")

    @unittest.skipUnless((DEFAULT_SHOWDOWN_DIR / "dist" / "sim").exists(), "local Showdown clone not built")
    def test_showdown_validator(self):
        try:
            find_node()
        except FileNotFoundError:
            self.skipTest("native node not installed")
        teams = {
            "valid": SAMPLE_EXPORT,
            "banned": SAMPLE_EXPORT.replace("Knock Off", "Tera Blast"),
            "garbage": "",
        }
        res = validate_teams(teams, DEFAULT_SHOWDOWN_DIR, workers=2)
        self.assertTrue(res["valid"]["valid"], res["valid"]["problems"])
        self.assertEqual(res["valid"]["sets"][0]["species"], "Great Tusk")
        self.assertIn("BoosterEnergy", res["valid"]["packed"])
        self.assertFalse(res["banned"]["valid"])
        self.assertEqual(problem_kind(res["banned"]["problems"][0]), "banned")
        self.assertFalse(res["garbage"]["valid"])


if __name__ == "__main__":
    unittest.main()
