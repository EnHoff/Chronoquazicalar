# Local setup (WSL2 / Linux, NVIDIA GPU)

Tested on Ubuntu 24.04 under WSL2 with an RTX 4060 Laptop GPU (driver 592.82, CUDA 13.1).

## Python environment

```bash
conda create -n chrono python=3.12
conda activate chrono
pip install "jax[cuda12]" flax optax numpy scipy poke-env requests pytest ruff huggingface_hub
# Don't let JAX grab 75% of GPU memory up front (fails on 8 GB laptop GPUs under WSL)
conda env config vars set XLA_PYTHON_CLIENT_PREALLOCATE=false
conda deactivate && conda activate chrono
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

Run the unit tests with `python -m pytest engine models search`. The throughput benchmark threshold can be
raised for faster hardware with `CHRONO_MIN_TPS=50000`.

## Node.js

Install Node 22 natively in WSL; the Windows `npm` under `/mnt/c` will not work.

```bash
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. ~/.nvm/nvm.sh
nvm install 22 && nvm alias default 22
which node   # should be ~/.nvm/versions/node/v22.x/bin/node
```

## Local Pokémon Showdown server

Pinned commit: `c0845f33d381402dfcb18ce324c20a94f18af075` (smogon/pokemon-showdown master, 2026-10-23).

```bash
cd ..   # sibling of this repo
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
git checkout c0845f33d381402dfcb18ce324c20a94f18af075
npm install
cp config/config-example.js config/config.js
```

Start it (background, port 8000, log in `../showdown.log`):

```bash
scripts/start_showdown.sh
```

Validate an OU team: `./pokemon-showdown pack-team < team.txt | ./pokemon-showdown validate-team gen9ou`
(run from the Showdown directory; exit code 0 means valid).

## Playing gen9ou locally

```bash
python bot_client.py --test-local --format gen9ou --team teams/sample_ou.txt --battles 5
python -m evaluation.run_local_ou --games 100 --opponent random
```

## OU team and set data

`python -m data.ou.build` downloads the sources into `data/ou/cache/` and writes `team_pool.jsonl`,
`set_archetypes.json`, `usage_marginals.json`, `meta_coverage.json`, `eval_teams/` and `report.md` to
`data/ou/generated/` (both directories are gitignored, about 60 MB and 160 MB). It needs `huggingface_hub`
(`pip install huggingface_hub`) and the local Showdown clone, whose `TeamValidator` checks every team.
`--offline` rebuilds from the cache; `--refresh` re-downloads everything. Each build's `manifest.json` and
`report.md` are copied to `data/ou/` so the numbers are tracked in git.

Play against rotating eval teams with:

```bash
python -m evaluation.run_local_ou --games 20 --opponent heuristic --rotate-teams
```

## OU-JAX reference harness

`tools/ref/ref.js` runs Showdown's own simulator and `@smogon/calc` (pinned in `tools/ref/package.json`)
behind a JSON-lines protocol; `engine/oujax/ref.py` is the Python bridge used by the OU-JAX tests. Install
its one dependency once:

```bash
cd tools/ref && npm install && cd ../..
```

Tests that need the harness skip themselves when node, the built Showdown clone or `node_modules` are missing.
`node tools/ref/export_dex.js ../pokemon-showdown > engine/oujax/data/dex_gen9.json` regenerates the game data
export from the pinned Showdown commit.
