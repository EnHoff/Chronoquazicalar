"""
data/ou/build.py
Builds the Gen 9 OU data artifacts in data/ou/generated/:
  team_pool.jsonl, set_archetypes.json, usage_marginals.json, meta_coverage.json, eval_teams/, report.md
and copies manifest.json + report.md to data/ou/ (committed) so each build is documented in git.

Usage:
    python -m data.ou.build                # fetch sources (cached) and build everything
    python -m data.ou.build --offline      # rebuild from data/ou/cache without network
    python -m data.ou.build --refresh      # re-download all sources
"""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from data.ou import sources
from data.ou.archetypes import build_set_archetypes
from data.ou.coverage import build_meta_coverage
from data.ou.stats import build_usage_marginals, chaos_paths
from data.ou.teams import STYLE_TAGS, archetype_tags, dedupe, normalize_set, recency_weight
from data.ou.validator import problem_kind, showdown_commit, validate_teams

N_EVAL_TEAMS = 20
MAX_COMPETITIVE_EVAL = 10
MAX_EVAL_SPECIES_OVERLAP = 3
EVAL_TAG_ORDER = (
    "rain", "sun", "sand", "snow", "stall", "hyper_offence", "webs", "trick_room",
    "hazard_stack", "volt_turn", "offence", "balance",
)  # fmt: skip


def collect_candidates() -> Dict[str, Dict]:
    """{team id: {"text", "source", "month"}} from every team source."""
    out = {}
    for set_name, month in sources.METAMON_SETS.items():
        for fname, text in sources.metamon_team_texts(set_name).items():
            out[f"metamon/{set_name}/{fname}"] = {"text": text, "source": f"metamon/{set_name}", "month": month}
    return out


def build_pool(candidates: Dict[str, Dict], results: Dict[str, Dict], ref_month: str):
    drops: Counter = Counter()
    drop_examples: Dict[str, str] = {}
    per_source: Dict[str, Counter] = {}
    valid = []
    for tid, cand in candidates.items():
        src = per_source.setdefault(cand["source"], Counter())
        src["candidates"] += 1
        r = results.get(tid)
        if r is None:
            reason, detail = "not validated", ""
        elif not r["valid"]:
            reason = problem_kind(r["problems"][0]) if r["problems"] else "other"
            detail = "; ".join(r["problems"][:2])
        elif len(r["sets"]) != 6:
            reason, detail = "fewer than 6 Pokemon", str(len(r["sets"]))
        else:
            reason = None
        if reason:
            drops[reason] += 1
            drop_examples.setdefault(reason, f"{tid}: {detail}"[:200])
            src["dropped"] += 1
            continue
        src["valid"] += 1
        valid.append(
            {
                "id": tid,
                "source": cand["source"],
                "month": cand["month"],
                "rating": None,  # Metamon team files carry no per-game rating
                "revealed_fraction": None,  # ...nor which set fields were revealed vs predicted
                "recency_weight": round(recency_weight(cand["month"], ref_month), 4),
                "packed": r["packed"],
                "export": r["export"].strip() + "\n",
                "sets": [normalize_set(s) for s in r["sets"]],
            }
        )
    pool, dedup_stats = dedupe(valid)
    for t in pool:
        t["weight"] = round(t["recency_weight"] * t["copies"], 4)
        t["species"] = [s["species"] for s in t["sets"]]
        t["tags"] = archetype_tags(t["sets"])
        per_source[t["source"]]["pool"] += 1
    pool.sort(key=lambda t: -t["weight"])
    return pool, {
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "drops": dict(drops),
        "drop_examples": drop_examples,
        "dedup": dedup_stats,
    }


def pick_eval_teams(pool: List[Dict], n: int = N_EVAL_TEAMS) -> List[Dict]:
    """~n diverse teams: expert sample teams first, then the most-copied high-ladder team for each
    archetype tag, then the heaviest remaining teams, limiting species overlap between picks."""
    chosen: List[Dict] = []

    def ok(t: Dict) -> bool:
        return all(len(set(t["species"]) & set(c["species"])) <= MAX_EVAL_SPECIES_OVERLAP for c in chosen)

    for t in [t for t in pool if t["source"] == "metamon/competitive"][:MAX_COMPETITIVE_EVAL]:
        if ok(t):
            chosen.append(t)
    ladder = [t for t in pool if t["source"] != "metamon/competitive"]
    for tag in EVAL_TAG_ORDER:
        if len(chosen) >= n:
            break
        if any(tag in c["tags"] for c in chosen):
            continue
        t = next((t for t in ladder if tag in t["tags"] and t not in chosen and ok(t)), None)
        if t:
            chosen.append(t)
    for t in ladder:
        if len(chosen) >= n:
            break
        if t not in chosen and ok(t):
            chosen.append(t)
    return chosen


def write_eval_teams(teams: List[Dict], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    index = []
    for i, t in enumerate(teams, 1):
        style = next((s for s in STYLE_TAGS if s in t["tags"]), "balance")
        name = f"{i:02d}_{style}_{'-'.join(t['species'][:2])}.txt"
        (out_dir / name).write_text(t["export"], encoding="utf-8")
        index.append({"file": name, "id": t["id"], "source": t["source"], "tags": t["tags"], "copies": t["copies"]})
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def write_report(path: Path, manifest: Dict, pool: List[Dict], pool_stats: Dict, coverage: Dict, eval_teams) -> str:
    total_w = sum(t["weight"] for t in pool) or 1.0
    lines = [
        "# Gen 9 OU data build report",
        "",
        f"Built {manifest['built']} against Showdown `{manifest['showdown_commit']}` (format `gen9ou`). "
        f"Stats months: {', '.join(manifest['stats_months'])}; recency reference month {manifest['ref_month']}.",
        "",
        "## Sources",
        "",
        "| Source | Candidates | Valid | Dropped | In pool after dedup |",
        "|---|---|---|---|---|",
    ]
    for src, c in pool_stats["per_source"].items():
        lines.append(
            f"| {src} | {c.get('candidates', 0)} | {c.get('valid', 0)} | {c.get('dropped', 0)} | {c.get('pool', 0)} |"
        )
    lines += [
        "",
        "Metamon team files are plain Showdown exports: they do not mark which set fields were revealed in the "
        "replay and which were predicted from usage stats, and carry no per-team rating or date. "
        "`revealed_fraction` and `rating` are therefore null; `hl_05_26` is dated 2026-05 from its name, "
        "the expert `competitive` set is undated.",
        "",
        "Other sources (no teams, used for set archetypes and marginals): "
        + ", ".join(f"`{f['name']}`" for f in manifest["sources"]["files"] if not f["name"].startswith("metamon/")),
        "",
        "## Validation",
        "",
        "Every candidate team was validated with Showdown's `TeamValidator` for `gen9ou` (in-process, batched). "
        "Invalid teams are dropped:",
        "",
    ]
    if pool_stats["drops"]:
        lines += ["| Reason | Teams | Example |", "|---|---|---|"]
        for reason, n in sorted(pool_stats["drops"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {n} | {pool_stats['drop_examples'].get(reason, '')} |")
    else:
        lines.append("No teams were dropped.")
    d = pool_stats["dedup"]
    lines += [
        "",
        f"Deduplication removed {d['exact']} exact and {d['near']} near duplicates (same six species, "
        f">= 90% identical set fields). **Pool size: {len(pool)} teams**, total weight {total_w:.1f} "
        "(weight = recency weight x copies).",
        "",
        "## Archetypes",
        "",
        "| Tag | Teams | Share of pool weight |",
        "|---|---|---|",
    ]
    tag_n, tag_w = Counter(), Counter()
    for t in pool:
        for tag in t["tags"]:
            tag_n[tag] += 1
            tag_w[tag] += t["weight"]
    for tag, n in tag_n.most_common():
        lines.append(f"| {tag} | {n} | {tag_w[tag] / total_w:.1%} |")
    sp_w: Counter = Counter()
    for t in pool:
        for s in t["species"]:
            sp_w[s] += t["weight"]
    lines += ["", "## Top 30 species by pool weight", "", "| # | Species | Share of teams |", "|---|---|---|"]
    for i, (s, w) in enumerate(sp_w.most_common(30), 1):
        lines.append(f"| {i} | {s} | {w / total_w:.1%} |")
    lines += [
        "",
        "## Meta coverage",
        "",
        f"{coverage['n_elements']['move']} moves, {coverage['n_elements']['item']} items and "
        f"{coverage['n_elements']['ability']} abilities appear in the pool or chaos stats. "
        "Smallest greedy sets (elements ranked by usage-weighted pool share) that fully cover a share of "
        "usage-weighted pool teams:",
        "",
        "| Target | Covered | Moves | Items | Abilities |",
        "|---|---|---|---|---|",
    ]
    for target, c in coverage["covers"].items():
        k = c["counts"]
        lines.append(f"| {target}% | {c['coverage']:.2%} | {k['move']} | {k['item']} | {k['ability']} |")
    lines += [
        "",
        "Per Pokémon set instead (a set is covered when its moves, item and ability are all in), "
        "taking prefixes of the same ranking:",
        "",
        "| Target | Covered | Moves | Items | Abilities |",
        "|---|---|---|---|---|",
    ]
    for target, c in coverage["set_covers"].items():
        k = c["counts"]
        lines.append(f"| {target}% of sets | {c['coverage']:.2%} | {k['move']} | {k['item']} | {k['ability']} |")
    for target, c in coverage["covers"].items():
        lines += [
            "",
            f"### {target}% coverage set",
            "",
            f"**Moves ({len(c['moves'])}):** " + ", ".join(c["moves"]),
            "",
            f"**Items ({len(c['items'])}):** " + ", ".join(c["items"]),
            "",
            f"**Abilities ({len(c['abilities'])}):** " + ", ".join(c["abilities"]),
        ]
    lines += ["", "## Evaluation teams", "", "| File | Source | Tags |", "|---|---|---|"]
    for e in eval_teams:
        lines.append(f"| {e['file']} | {e['source']} | {', '.join(e['tags'])} |")
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def build(showdown_dir: Path, offline: bool = False, refresh: bool = False, workers: int = 6) -> Dict:
    t0 = time.time()
    if offline:
        src_manifest = sources.load_sources_manifest()
    else:
        src_manifest = sources.fetch_all(showdown_dir, refresh=refresh)
    months = src_manifest["stats_months"]
    ref_month = months[0]
    commit = showdown_commit(showdown_dir)
    out = sources.GENERATED_DIR
    out.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates()
    results = validate_teams({k: v["text"] for k, v in candidates.items()}, showdown_dir, workers=workers)
    pool, pool_stats = build_pool(candidates, results, ref_month)
    with open(out / "team_pool.jsonl", "w", encoding="utf-8") as f:
        for t in pool:
            f.write(json.dumps(t) + "\n")

    usage = build_usage_marginals(chaos_paths(sources.CACHE_DIR, months), ref_month)
    (out / "usage_marginals.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")

    factory = json.loads((sources.CACHE_DIR / "factory" / "gen9-factory-ou.json").read_text(encoding="utf-8"))
    smogon = json.loads((sources.CACHE_DIR / "smogon" / "gen9ou-sets.json").read_text(encoding="utf-8"))
    archetypes = build_set_archetypes(pool, factory, smogon, usage_marginals=usage)
    (out / "set_archetypes.json").write_text(json.dumps(archetypes, indent=1), encoding="utf-8")

    coverage = build_meta_coverage(pool, usage)
    (out / "meta_coverage.json").write_text(json.dumps(coverage, indent=1), encoding="utf-8")

    eval_teams = pick_eval_teams(pool)
    write_eval_teams(eval_teams, out / "eval_teams")
    eval_index = json.loads((out / "eval_teams" / "index.json").read_text(encoding="utf-8"))

    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
        "showdown_commit": commit,
        "format": "gen9ou",
        "stats_months": months,
        "ref_month": ref_month,
        "sources": src_manifest,
        "counts": {
            "candidates": len(candidates),
            "valid": sum(1 for r in results.values() if r["valid"]),
            "pool": len(pool),
            "dropped": pool_stats["drops"],
            "dedup": pool_stats["dedup"],
            "per_source": pool_stats["per_source"],
            "archetype_species": len(archetypes),
            "usage_species": len(usage["species"]),
            "eval_teams": len(eval_teams),
        },
        "coverage": {k: {"coverage": v["coverage"], **v["counts"]} for k, v in coverage["covers"].items()},
        "set_coverage": {k: {"coverage": v["coverage"], **v["counts"]} for k, v in coverage["set_covers"].items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out / "report.md", manifest, pool, pool_stats, coverage, eval_index)
    for name in ("manifest.json", "report.md"):
        shutil.copy(out / name, sources.OU_DIR / name)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Gen 9 OU team and set data")
    parser.add_argument("--showdown-dir", type=Path, default=sources.DEFAULT_SHOWDOWN_DIR)
    parser.add_argument("--offline", action="store_true", help="Use cached sources only")
    parser.add_argument("--refresh", action="store_true", help="Re-download all sources")
    parser.add_argument("--workers", type=int, default=6, help="Parallel Node validator processes")
    args = parser.parse_args(argv)
    manifest = build(args.showdown_dir, offline=args.offline, refresh=args.refresh, workers=args.workers)
    print(json.dumps({"counts": manifest["counts"], "coverage": manifest["coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
