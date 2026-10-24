"""
data/ou/archetypes.py
Per-species set archetypes: full sets grouped by item, EV bucket and moveset, each with a weight and
distributions over moves, ability, Tera type, nature and exact spread. Seeded with Showdown's Battle Factory
OU sets and Smogon analysis sets, and filled from the validated team pool.

Metamon fills unrevealed moves from usage stats, so exact 4-move combinations fragment badly (Great Tusk has
~1,800). Sets are therefore merged into the heaviest earlier archetype with the same item and EV bucket that
shares at least 3 of 4 moves, and at most MAX_CLUSTERS archetypes are kept per species; any further sets are
folded into their closest archetype.
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from data.ou.teams import STATS, to_id

EV_BUCKET_MIN = 100
MAX_CLUSTERS = 20
MIN_SHARED_MOVES = 3
# Share of a species' archetype mass given to each seed source when the pool also has that species
SEED_SHARE_WITH_POOL = 0.1
SEED_SHARE_WITHOUT_POOL = 0.5


def ev_bucket(evs: List[int]) -> str:
    """Stats with a real investment, in stat order, e.g. [252,0,4,0,0,252] -> 'hp+spe'."""
    return "+".join(stat for stat, v in zip(STATS, evs) if v >= EV_BUCKET_MIN) or "none"


def spread_str(nature: str, evs: List[int]) -> str:
    return f"{nature}:{'/'.join(map(str, evs))}"


def _first(x):
    while isinstance(x, list):
        if not x:
            return None
        x = x[0]
    return x


def _evs(d: Optional[Dict]) -> List[int]:
    d = _first(d) if isinstance(d, list) else d
    return [int((d or {}).get(k, 0)) for k in STATS]


def factory_seed_sets(factory_ou: Dict) -> Iterable[Tuple[Dict, float]]:
    """(normalised set, relative weight) for Showdown's gen9 Battle Factory OU sets, first options only."""
    for species_entry in factory_ou.values():
        for s in species_entry.get("sets", []):
            yield (
                {
                    "species": to_id(s["species"]),
                    "item": to_id(_first(s.get("item"))),
                    "ability": to_id(_first(s.get("ability"))),
                    "tera": to_id(_first(s.get("teraType"))),
                    "nature": to_id(_first(s.get("nature"))),
                    "evs": _evs(s.get("evs")),
                    "moves": sorted(to_id(_first(slot)) for slot in s.get("moves", []) if _first(slot)),
                },
                float(s.get("weight", 1)),
            )


def smogon_seed_sets(analysis: Dict) -> Iterable[Tuple[Dict, float]]:
    """(normalised set, weight 1) for pkmn.github.io Smogon analysis sets, first options only."""
    for species, named_sets in analysis.items():
        for s in named_sets.values():
            yield (
                {
                    "species": to_id(species),
                    "item": to_id(_first(s.get("item"))),
                    "ability": to_id(_first(s.get("ability"))),
                    "tera": to_id(_first(s.get("teratypes"))),
                    "nature": to_id(_first(s.get("nature"))),
                    "evs": _evs(s.get("evs")),
                    "moves": sorted(to_id(_first(m)) for m in s.get("moves", []) if _first(m)),
                },
                1.0,
            )


def cluster_key(s: Dict) -> Tuple:
    return (s["item"], tuple(s["moves"]), ev_bucket(s["evs"]))


class _Cluster:
    def __init__(self, key: Tuple):
        self.key = key
        self.weight = 0.0
        self.moves: Counter = Counter()
        self.abilities: Counter = Counter()
        self.tera: Counter = Counter()
        self.natures: Counter = Counter()
        self.spreads: Counter = Counter()
        self.sources: Counter = Counter()

    def add(self, s: Dict, w: float, source: str) -> None:
        self.weight += w
        for m in s["moves"]:
            self.moves[m] += w
        self.abilities[s["ability"]] += w
        self.tera[s["tera"]] += w
        self.natures[s["nature"]] += w
        self.spreads[spread_str(s["nature"], s["evs"])] += w
        self.sources[source] += 1

    def merge(self, other: "_Cluster") -> None:
        self.weight += other.weight
        for mine, theirs in (
            (self.moves, other.moves),
            (self.abilities, other.abilities),
            (self.tera, other.tera),
            (self.natures, other.natures),
            (self.spreads, other.spreads),
            (self.sources, other.sources),
        ):
            mine.update(theirs)

    def affinity(self, other: "_Cluster") -> Tuple[int, int]:
        """(shared moves, matching item + EV bucket) between two clusters' defining keys."""
        shared = len(set(self.key[1]) & set(other.key[1]))
        return shared, (self.key[0] == other.key[0]) + (self.key[2] == other.key[2])

    def to_json(self, total: float) -> Dict:
        def dist(c: Counter, k: int = 5) -> Dict[str, float]:
            s = sum(c.values()) or 1.0
            return {key: round(v / s, 4) for key, v in c.most_common(k) if key}

        item, _, bucket = self.key
        move_probs = {m: round(v / self.weight, 4) for m, v in self.moves.most_common(10)} if self.weight else {}
        return {
            "weight": round(self.weight / total, 5),
            "item": item,
            "moves": sorted(m for m, _ in self.moves.most_common(4)),
            "move_probs": move_probs,
            "ev_bucket": bucket,
            "abilities": dist(self.abilities),
            "tera": dist(self.tera),
            "natures": dist(self.natures),
            "spreads": dist(self.spreads, 3),
            "sources": dict(self.sources),
        }


def merge_clusters(raw: List[_Cluster], max_clusters: int = MAX_CLUSTERS) -> List[_Cluster]:
    """Greedy merge of exact-key clusters, heaviest first (see module docstring)."""
    centers: List[_Cluster] = []
    for c in sorted(raw, key=lambda c: -c.weight):
        match = next(
            (k for k in centers if k.affinity(c)[1] == 2 and k.affinity(c)[0] >= MIN_SHARED_MOVES),
            None,
        )
        if match is None and len(centers) >= max_clusters:
            match = max(centers, key=lambda k: k.affinity(c))
        if match is None:
            centers.append(c)
        else:
            match.merge(c)
    return sorted(centers, key=lambda c: -c.weight)


def build_set_archetypes(
    pool: List[Dict],
    factory_ou: Optional[Dict] = None,
    smogon_sets: Optional[Dict] = None,
    max_clusters: int = MAX_CLUSTERS,
    usage_marginals: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """pool: team_pool records with normalised "sets" and "weight". Returns {species: {...}}.

    usage_marginals (optional) supplies the most likely ability for seed sets that omit it
    (Smogon analysis sets leave out single-ability species' abilities)."""
    pool_mass: Dict[str, float] = defaultdict(float)
    pool_abilities: Dict[str, Counter] = defaultdict(Counter)
    clusters: Dict[str, Dict[Tuple, _Cluster]] = defaultdict(dict)

    def add(s: Dict, w: float, source: str) -> None:
        key = cluster_key(s)
        c = clusters[s["species"]].get(key)
        if c is None:
            c = clusters[s["species"]][key] = _Cluster(key)
        c.add(s, w, source)

    for team in pool:
        for s in team["sets"]:
            pool_mass[s["species"]] += team["weight"]
            pool_abilities[s["species"]][s["ability"]] += team["weight"]
            add(s, team["weight"], "pool")

    def likely_ability(species: str) -> str:
        if pool_abilities.get(species):
            return pool_abilities[species].most_common(1)[0][0]
        usage = ((usage_marginals or {}).get("species", {}).get(species) or {}).get("abilities") or {}
        return max(usage, key=usage.get) if usage else ""

    for source, seeds in (
        ("factory", list(factory_seed_sets(factory_ou or {}))),
        ("smogon", list(smogon_seed_sets(smogon_sets or {}))),
    ):
        seed_totals: Dict[str, float] = defaultdict(float)
        for s, w in seeds:
            seed_totals[s["species"]] += w
        for s, w in seeds:
            if not s["ability"]:
                s = dict(s, ability=likely_ability(s["species"]))
            share = SEED_SHARE_WITH_POOL if pool_mass.get(s["species"]) else SEED_SHARE_WITHOUT_POOL
            # Seed mass is expressed relative to the pool mass so the share holds after normalising
            base = pool_mass.get(s["species"]) or 1.0
            add(s, base * share * w / seed_totals[s["species"]], source)

    out = {}
    for species, by_key in clusters.items():
        merged = merge_clusters(list(by_key.values()), max_clusters)
        total = sum(c.weight for c in merged) or 1.0
        out[species] = {
            "pool_weight": round(pool_mass.get(species, 0.0), 4),
            "n_exact_sets": len(by_key),
            "sets": [c.to_json(total) for c in merged],
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["pool_weight"]))
