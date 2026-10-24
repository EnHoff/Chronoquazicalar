"""
data/ou/coverage.py
Meta coverage: ranks every move, item and ability seen in the team pool and chaos stats by usage weight,
and finds the smallest prefix of that ranking that fully covers 95% / 99% of usage-weighted pool teams
(a team is covered when all of its moves, items and abilities are in the set). The same is reported per
Pokémon set (a set is covered when its 4 moves, item and ability are in), which is the gentler measure
of how much of what the engine will actually see is implemented.

Elements are ranked by the weighted share of pool teams that use them, which is the order in which an
engine would implement them. Taking a prefix of that ranking is a greedy approximation of the true
minimal cover, and in practice close to it because a team only becomes covered once its rarest
element is in.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

KINDS = ("move", "item", "ability")
PLURAL = {"move": "moves", "item": "items", "ability": "abilities"}
TARGETS = (0.95, 0.99)
IGNORED_IDS = {"", "nothing", "none", "noability"}

Element = Tuple[str, str]


def team_elements(sets: List[Dict]) -> Set[Element]:
    out: Set[Element] = set()
    for s in sets:
        out.update(("move", m) for m in s["moves"])
        out.add(("item", s["item"]))
        out.add(("ability", s["ability"]))
    return {e for e in out if e[1] not in IGNORED_IDS}


def chaos_shares(usage_marginals: Dict) -> Dict[Element, float]:
    """Expected number of team members using each element, from chaos usage x per-species probability."""
    shares: Dict[Element, float] = defaultdict(float)
    for rec in usage_marginals.get("species", {}).values():
        u = rec.get("usage", 0.0)
        for kind, key in (("move", "moves"), ("item", "items"), ("ability", "abilities")):
            for eid, p in rec.get(key, {}).items():
                if eid not in IGNORED_IDS:
                    shares[(kind, eid)] += u * p
    return shares


def build_meta_coverage(pool: List[Dict], usage_marginals: Dict, targets=TARGETS) -> Dict:
    weights = [t["weight"] for t in pool]
    total = sum(weights) or 1.0
    team_elems = [team_elements(t["sets"]) for t in pool]

    pool_share: Dict[Element, float] = defaultdict(float)
    for w, elems in zip(weights, team_elems):
        for e in elems:
            pool_share[e] += w / total
    chaos = chaos_shares(usage_marginals)

    elements = set(pool_share) | set(chaos)
    ranked = sorted(elements, key=lambda e: (-pool_share.get(e, 0.0), -chaos.get(e, 0.0), e))
    rank = {e: i for i, e in enumerate(ranked)}

    # A team (or set) is covered once its worst-ranked element is included
    team_covered_at: Dict[int, float] = defaultdict(float)
    set_covered_at: Dict[int, float] = defaultdict(float)
    n_sets = sum(len(t["sets"]) for t in pool) or 1
    set_total = sum(w * len(t["sets"]) for w, t in zip(weights, pool)) or 1.0
    for w, t, elems in zip(weights, pool, team_elems):
        if elems:
            team_covered_at[max(rank[e] for e in elems)] += w / total
        for s in t["sets"]:
            set_elems = team_elements([s])
            if set_elems:
                set_covered_at[max(rank[e] for e in set_elems)] += w / set_total

    rows, cum_team, cum_set = [], 0.0, 0.0
    for i, e in enumerate(ranked):
        cum_team += team_covered_at.get(i, 0.0)
        cum_set += set_covered_at.get(i, 0.0)
        rows.append(
            {
                "kind": e[0],
                "id": e[1],
                "pool_share": round(pool_share.get(e, 0.0), 6),
                "chaos_share": round(chaos.get(e, 0.0), 6),
                "cum_coverage": round(cum_team, 6),
                "cum_set_coverage": round(cum_set, 6),
            }
        )

    def cover(field: str, target: float) -> Dict:
        n = next((i + 1 for i, r in enumerate(rows) if r[field] >= target - 1e-9), len(rows))
        chosen = rows[:n]
        return {
            "coverage": chosen[-1][field] if chosen else 0.0,
            "counts": {k: sum(r["kind"] == k for r in chosen) for k in KINDS},
            **{PLURAL[k]: [r["id"] for r in chosen if r["kind"] == k] for k in KINDS},
        }

    return {
        "method": "greedy prefix of elements ranked by usage-weighted pool share",
        "n_teams": len(pool),
        "n_sets": n_sets,
        "n_elements": {k: sum(e[0] == k for e in elements) for k in KINDS},
        "covers": {f"{round(t * 100)}": cover("cum_coverage", t) for t in targets},
        "set_covers": {f"{round(t * 100)}": cover("cum_set_coverage", t) for t in targets},
        "ranking": rows,
    }
