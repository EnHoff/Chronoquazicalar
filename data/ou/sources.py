"""
data/ou/sources.py
Downloads and caches the raw OU data sources, recording URL, retrieval time, size and SHA-256
of every file so a build can be reproduced from its manifest.

Sources:
- Metamon team sets (Hugging Face dataset jakegrigsby/metamon-teams), gen9ou files only
- Smogon chaos usage stats for gen9ou at the 1695 and 1825 cut-offs, most recent months
- Showdown's gen9 Battle Factory sets (OU section), from the local pinned Showdown clone
- Smogon analysis sets via pkmn.github.io
"""

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

OU_DIR = Path(__file__).resolve().parent
CACHE_DIR = OU_DIR / "cache"
GENERATED_DIR = OU_DIR / "generated"
DEFAULT_SHOWDOWN_DIR = Path(os.environ.get("SHOWDOWN_DIR", OU_DIR.parents[2] / "pokemon-showdown"))

METAMON_REPO = "jakegrigsby/metamon-teams"
METAMON_SETS = {
    # set name -> month the teams represent (None when undated)
    "hl_05_26": "2026-05",
    "competitive": None,
}
STATS_INDEX_URL = "https://www.smogon.com/stats/"
CHAOS_URL = "https://www.smogon.com/stats/{month}/chaos/gen9ou-{cutoff}.json"
CHAOS_CUTOFFS = (1695, 1825)
SMOGON_SETS_URL = "https://pkmn.github.io/smogon/data/sets/gen9ou.json"
FACTORY_SETS_PATH = Path("data") / "random-battles" / "gen9" / "factory-sets.json"

USER_AGENT = "Chronoquazicalar-OU-data/0.1 (+https://github.com/EnHoff/Chronoquazicalar)"


@dataclass
class SourceFile:
    name: str
    url: str
    path: str
    sha256: str
    bytes: int
    retrieved: str
    month: Optional[str] = None
    note: Optional[str] = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(name: str, url: str, path: Path, month: Optional[str] = None, note: Optional[str] = None) -> SourceFile:
    return SourceFile(
        name=name,
        url=url,
        path=str(path.relative_to(OU_DIR)),
        sha256=sha256_of(path),
        bytes=path.stat().st_size,
        retrieved=_now(),
        month=month,
        note=note,
    )


def http_download(url: str, dest: Path, refresh: bool = False) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    return dest


def latest_stats_months(n: int = 3) -> List[str]:
    """The n most recent YYYY-MM folders listed on smogon.com/stats, newest first."""
    req = urllib.request.Request(STATS_INDEX_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")
    months = sorted(set(re.findall(r'href="(\d{4}-\d{2})/"', html)), reverse=True)
    return months[:n]


def fetch_metamon(refresh: bool = False) -> List[SourceFile]:
    from huggingface_hub import hf_hub_download

    out = []
    dest_root = CACHE_DIR / "metamon"
    readme = Path(hf_hub_download(METAMON_REPO, "README.md", repo_type="dataset", local_dir=dest_root))
    out.append(_record("metamon/README", f"hf://datasets/{METAMON_REPO}/README.md", readme))
    for set_name, month in METAMON_SETS.items():
        rel = f"{set_name}/gen9ou.tar.gz"
        archive = dest_root / rel
        if refresh or not archive.exists():
            archive = Path(hf_hub_download(METAMON_REPO, rel, repo_type="dataset", local_dir=dest_root))
        out.append(_record(f"metamon/{set_name}", f"hf://datasets/{METAMON_REPO}/{rel}", archive, month=month))
    return out


def fetch_chaos(months: List[str], refresh: bool = False) -> List[SourceFile]:
    out = []
    for month in months:
        for cutoff in CHAOS_CUTOFFS:
            url = CHAOS_URL.format(month=month, cutoff=cutoff)
            dest = CACHE_DIR / "chaos" / f"{month}-gen9ou-{cutoff}.json"
            http_download(url, dest, refresh=refresh)
            out.append(_record(f"chaos/{month}/{cutoff}", url, dest, month=month))
    return out


def fetch_smogon_sets(refresh: bool = False) -> List[SourceFile]:
    dest = http_download(SMOGON_SETS_URL, CACHE_DIR / "smogon" / "gen9ou-sets.json", refresh=refresh)
    return [_record("smogon/analysis_sets", SMOGON_SETS_URL, dest)]


def fetch_factory_sets(showdown_dir: Path) -> List[SourceFile]:
    src = showdown_dir / FACTORY_SETS_PATH
    commit = subprocess.run(
        ["git", "-C", str(showdown_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    with open(src, "r", encoding="utf-8") as f:
        ou = json.load(f)["OU"]
    dest = CACHE_DIR / "factory" / "gen9-factory-ou.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(ou, indent=1), encoding="utf-8")
    return [_record("showdown/factory_sets_ou", f"pokemon-showdown@{commit}:{FACTORY_SETS_PATH}", dest)]


def metamon_team_texts(set_name: str) -> Dict[str, str]:
    """Returns {team file name: export text} for one Metamon gen9ou archive."""
    archive = CACHE_DIR / "metamon" / set_name / "gen9ou.tar.gz"
    teams = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith("_team"):
                continue
            f = tar.extractfile(member)
            if f is not None:
                teams[Path(member.name).name] = f.read().decode("utf-8", "replace")
    return teams


def fetch_all(showdown_dir: Path = DEFAULT_SHOWDOWN_DIR, months: Optional[List[str]] = None, refresh=False) -> Dict:
    months = months or latest_stats_months(3)
    files = fetch_metamon(refresh) + fetch_chaos(months, refresh) + fetch_smogon_sets(refresh)
    files += fetch_factory_sets(showdown_dir)
    manifest = {"fetched": _now(), "stats_months": months, "files": [asdict(f) for f in files]}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "sources_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_sources_manifest() -> Dict:
    return json.loads((CACHE_DIR / "sources_manifest.json").read_text(encoding="utf-8"))
