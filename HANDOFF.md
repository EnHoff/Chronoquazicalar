# Gen 9 OU bot: project handoff

This is the single source of truth for continuing the project in a new tool. It was written on 2026-11-01. M3 progress after the parallel subagents launched on 2026-10-25 is **not** captured here, so check the repo first (see "First steps").

## Goal

Build a Pokémon Showdown **Gen 9 OU** bot on top of Chronoquazicalar (fork: https://github.com/EnHoff/Chronoquazicalar, upstream TurboRx/Chronoquazicalar).
- Replace its randbats knowledge with real, recently popular OU teams.
- Follow the Jaxcalibur recipe (https://jaxcalibur.github.io/): a GPU-native JAX simulator, a small transformer, and PPO self-play.
- Use the PokeJAX paper's verification method (arXiv 2603.12145): L1–L4 checks against Showdown.

The original plan doc is https://claude.ai/artifact/QmiTc67H4GwBjzCG7BPDjc. The essentials are copied below.

## Decisions (Ian's, non-negotiable unless he changes them)

- **No test-time search.** The bot plays straight from the policy network. Set inference only feeds the policy's input tokens.
- **Independent randomness.** Don't match Showdown's PRNG call order. Every random check is an independent draw.
- **"An efficient engine that makes a good model, not an exact replica."** Mechanics that change decisions must be right: damage, speed order, items, abilities, Tera, hazards, pivoting. Rare or low-impact details may be simplified or skipped. L4 (policy transfer) is the final judge.
- **Speed wins ties.** If a shortcut is noticeably faster and the tests still pass, keep it.
- **No public ladder.** The real-world test is a challenge match against Ian's friend on the Showdown website, from a clearly named bot account, with private challenges only.
- **Compute.** Prototype and train on Ian's laptop RTX 4060 (8 GB) under WSL2. Rent a bigger GPU (for example Colab A100) only if the run plateaus.
- **Ian merges every PR.** The agent never merges to main.

## Milestones and gates

| | Milestone | Gate | Status |
|---|---|---|---|
| M0 | Harness and baseline: pinned local Showdown, OU client port (team preview, team file) | 100 local OU games without errors | **Done**, PR #1 merged. 98% vs Random, 27% vs SimpleHeuristics (untrained) |
| M1 | Data pipeline: team_pool, set_archetypes, usage_marginals, meta coverage | All pool teams pass the validator; coverage list for 95/99% | **Done**, PR #2 merged |
| M2 | OU-JAX core | L1 and L2 green, L3 forced and statistical runs pass on covered mechanics | **Done**, PR #3 merged |
| M3 | OU-JAX meta coverage | 95% strict usage-weighted team coverage; L3a 100+ forced battles on **real pool teams** match turn by turn; L3b passes; JIT and throughput reported; full suite and ruff clean | **In progress** on `m3-meta-coverage` |
| M4 | Model and BC: new tokenizer and heads, in-engine observation masking, optional BC warm start from Metamon replays | Beats SimpleHeuristicsPlayer on real Showdown | Not started |
| M5 | Self-play PPO: 1/0 reward with no shaping, GAE, entropy plus zero-avoiding KL, opponent-action aux loss, league of past checkpoints, team specialization | L4 TOST passes (Δ=0.02, no sim-to-sim gap) and beats Metamon Kakuna locally above 55% | Not started |
| M6 | Set inference: OUKnowledgeBase with team prior, archetype prior, hard and soft evidence, fused with the aux heads, feeding the opponent and matchup tokens | Calibration acceptable on held-out replays; beats the same policy without it | Not started |
| M7 | Friend match on the Showdown website | Games complete with no timeouts or illegal moves | Not started |

### M1 results
- **Pool:** 43,707 valid gen9ou teams, from Metamon `hl_05_26` (high-ladder replays from May 2026) plus 11 expert sample teams.
- **Dropped:** 5 expert teams for Tera Blast, plus 68 near-duplicates.
- **Generated data:** set_archetypes for 388 species; usage_marginals for 350 species (Smogon stats, July–September 2026).
- **Coverage:** 95% of sets need 247 moves, 49 items and 88 abilities. Strict 95% team coverage needs 336, 64 and 138.
- **Eval teams:** 20.
- **Rebuild:** `python -m data.ou.build` (about 45 s). The generated outputs in `data/ou/generated/` are gitignored. `data/ou/report.md` and `data/ou/manifest.json` are committed.

### M2 results
- **L1:** stats exact for 3,120 Pokémon; 16,000 damage cases exact against @smogon/calc.
- **L2:** 10 of 10 interaction scenarios.
- **L3a:** 80 of 80 forced battles.
- **L3b:** 72 statistical tests with 0 failures (lowest p 0.061, Bonferroni).
- **Throughput:** about 1.0M steps/s at batch 1K, 3.4M at 4K, 9.5M at 16K and 10.9M at 64K. JIT takes about 50 s, and the run uses under 300 MB of GPU memory.

### M3 state as of 2026-11-02 (checked against git)
- **Branches:** all work is on `m3-meta-coverage` (pushed). The sub-branches `m3-volatiles`, `m3-items-switch` and `m3-abilities` have **no commits of their own** (the first two point at 7af3dd2, the third at d732d63), so their 2026-10-25 subagent work was never committed and is lost; there is nothing to merge. Their worktrees under `.claude/worktrees/` only held untracked links. The `worktree-agent-*` branches point at main.
- **Commits on m3-meta-coverage since main:** a921597 (batch 1), 7af3dd2 (hooks refactor), d732d63 (Unaware, Mold Breaker family, Infiltrator), cbe5db7 (damage_moves family: multi-hit and per-hit accuracy, Loaded Dice / Skill Link, accuracy and crit items/abilities, Counter / Mirror Coat / Metal Burst / Comeuppance, Payback / Avalanche / Stomping Tantrum, Brick Break, crash moves, Steel Beam, Belly Drum and other status moves, Gen 5+ 2-5 hit odds).
- **Tests:** full suite 82 passed (about 11 min), `ruff check .` and `ruff format --check .` clean.
- **Coverage** (`python -m engine.oujax.coverage`): 70,822 of 262,242 pool sets (27.0%); strict team coverage 8 of 43,707 teams (0.02% usage-weighted). 345 elements are still missing. `python -m engine.oujax.coverage --worklist` lists them in usage order with the cumulative strict coverage: about 170 more reach 95%.
- **JIT and throughput** (`python -m engine.oujax.bench`, RTX 4060): JIT about 46 s; 0.58M steps/s at 1K, 2.1M at 4K, 4.7M at 16K. Population Bomb's 10-hit loop (was 5) costs about 20-25% of that (2.6M / 5.8M with a 5-hit cap); revisit once the per-hit contact hook is in.
- **Known gap:** Axe Kick is counted as covered but its 30% confusion secondary isn't modelled until confusion lands. Rocky Helmet triggers once per move rather than once per hit of a multi-hit move.

### M3 state as of 2026-10-25 (historical)
- 483 pool elements were unsupported. Adding them in usage order gives this strict team coverage:

  | Additions | Strict team coverage |
  |---|---|
  | 50 | 7% |
  | 100 | 41% |
  | 150 | 71% |
  | 200 | 87% |
  | 300 | 97% |

- **Batch 1 is merged into m3-meta-coverage** (commit a921597):
  - About 45 damage abilities, type-boost items, Ogerpon masks, resist berries and BP-callback moves.
  - L1: 20,000 calc cases with 0 mismatches.
  - Fixed bugs: Ruination dealing no damage, and Air Balloon not popping when its Substitute is hit.
  - Damage was split into a per-move part plus a per-hit fori_loop. JIT is now 36–40 s, and throughput is about 3.0M steps/s at 4K and 7.7M at 16K.
  - Set coverage is about 20%; strict team coverage is about 0%.
- **Refactor 7af3dd2:**
  - `engine/oujax/hooks.py` is an event registry: before_move, move_fails, try_hit, damage_taken, after_hit, status_move, after_move, switch_in/out, residual by order, legal_moves, speed, priority, status_immune, can_heal and boost_change.
  - Generic VOL/MON/SIDE/FIELD state slots.
  - `engine/oujax/families/` holds one module and one test file per family.
  - `SHOWDOWN_DIR` override for worktrees.
- **Parallel sub-branches** (each in its own worktree), launched on 2026-10-25. Their state is unknown:
  - **`m3-volatiles`:** Encore, Taunt, confusion/Hurricane/lockedmove, Sucker Punch/Thunderclap, Fake Out/First Impression, Heal Block, Salt Cure, Leech Seed, Yawn, Curse, Destiny Bond, crit stages, Ice Spinner, recharge/charge moves, Protect variants.
  - **`m3-items-switch`:** Trick, Focus Sash, Orbs, Eject Button/Red Card/Eject Pack, phazing, Wish/Healing Wish, Future Sight, Pain Split, berries and herbs, Weakness Policy, Covert Cloak, Shed Tail/Baton Pass, self-destruct moves, Custap/Quick Claw, seeds. This branch owns the request-flow changes for forced switches.
  - **`m3-abilities`:** Good as Gold, absorbs, Dauntless Shield, Magic Bounce, Intimidate and its reactions, contact abilities, Magic Guard/Poison Heal/Sturdy/Disguise, Prankster/Gale Wings/Triage, residual abilities, the Moxie family, Libero/Protean, and the non-damage part of Mold Breaker.
  - **Integrator (main checkout):** damage.py extras (Unaware, the damage parts of Mold Breaker and Infiltrator, accuracy and crit items, multi-hit specials), plus merging and the serial L3 runs.
- **Remaining work list:** `python -m engine.oujax.coverage`.

## Architecture map

- **`data/ou/`:** M1 pipeline. Its build CLI is `python -m data.ou.build`. Outputs go to `generated/` (gitignored), including team_pool.jsonl (159 MB, sorted by weight) and eval_teams/.
- **`engine/oujax/`:** the new engine. The old toy engine `engine/jax_battle_engine.py` is still there and isn't used for OU.
  - **Data:** `data/dex_gen9.json` (exported from Showdown, including handler names), `tables.py` (compact int arrays and id maps) and `team.py` (packed-team parser and stat precompute).
  - **Mechanics:** `damage.py` (Showdown getDamage/modifyDamage in int32 with exact 4096 rounding) and the state and step modules: fixed-shape int8/int16 PyTree, request flow, 13 actions plus team preview, and legality masks.
  - **Extension points:** `hooks.py` and `families/`.
  - **Coverage:** `coverage.py` (the M2/M3 implemented set, the covered-team builder and the work list).
  - **Bridge:** `ref.py`, the Python bridge to the Node harness.
- **`tools/ref/ref.js`:** a JSON-lines server over the pinned Showdown plus @smogon/calc 0.12.0.
  - Commands: stats, calc (16 rolls), and start/choose/end battle with full per-turn state.
  - **Forced mode** overrides battle.random/randomChance/sample/prng.shuffle. It always hits, never crits, rolls max damage, fires no secondaries (configurable), has no full paralysis or thaw, always passes Protect stall rolls, uses minimum durations, and lets p1 win speed ties.
- **Tests:** L1/L2 per family, `test_l3.py` (forced battles plus statistical scenarios), the L3a/L3b runners and the throughput benchmark.
- **Docs:** `docs/OUJAX.md` and `docs/SETUP.md`.
- **Client and eval:** `bot_client.py` (with `--format` and `--team`) and `evaluation/run_local_ou.py` (rotating eval teams). `engine/belief_sampler.py` uses previewed species.

## Environment (Ian's laptop)

- WSL2 Ubuntu 24.04, repo at `/home/ian/Pokemon/Chronoquazicalar`.
- conda env `chrono` (Python 3.12, JAX 0.12 cuda12), with `XLA_PYTHON_CLIENT_PREALLOCATE=false` set in the env. The default 75% preallocation runs out of memory on 8 GB under WSL.
- Node 22 via nvm.
- Showdown at `~/Pokemon/pokemon-showdown`, pinned to `c0845f33d381402dfcb18ce324c20a94f18af075`. `scripts/start_showdown.sh` starts it on port 8000. Set `SHOWDOWN_DIR` when working in a worktree.
- poke-env 0.16.1.
- `gh` is authenticated over HTTPS with a classic PAT (repo, read:org).
- RTX 4060 8 GB and 14 GB RAM. **Run at most 3–4 parallel agents.** Run L3 jobs serially. Use `JAX_PLATFORMS=cpu` for small tests to keep the GPU free.
- Commands: see `docs/SETUP.md` and `docs/OUJAX.md` for the exact test, L3a/L3b, benchmark and eval commands. In outline:
  - `pytest`, then `ruff check .`
  - `python -m data.ou.build`
  - `python -m engine.oujax.coverage`
  - `python evaluation/run_local_ou.py` (with Showdown running)
  - Run times: L3b takes about 45 min, and 50 L3a battles take about 25 min.

## Process rules

- Every L3 mismatch becomes an L1 or L2 regression test **before** the fix.
- Never skip, weaken or quarantine a test to get green.
- Keep move effects table-driven, with special cases grouped into effect families via hooks. Never write a giant per-move `lax.switch`. Write a single-instance step and vmap it, with no Python in the hot path.
- Watch JIT time (target under about 1 minute) and throughput after every merge.
- Commit and push after every batch, because the laptop may be shut down at any time.
- Use one branch per milestone or chunk, with PRs into main. Ian merges.
- Before a high-usage mechanic is simplified, or before a design change alters the plan's goals, ask Ian.

## Git conventions

- Commit author (repo-local config): `Ian Hoff <111913350+EnHoff@users.noreply.github.com>`. **Never put Ian's personal email in history.**
- The trailer used so far is `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Cursor may use its own attribution if Ian prefers.

## Gotchas already hit

- **JAX preallocation:** it runs out of memory under WSL. Keep preallocation off.
- **gh device flow:** it failed with `slow_down` (gh 2.45). Use a PAT instead.
- **Metamon teams:** unrevealed details are filled from usage stats with no seen/guessed marking, and there's no per-game rating or date.
- **Calc vs Showdown, Knock Off:** calc never boosts Knock Off against Booster Energy on any Paradox Pokémon. Showdown's dex doesn't tag Raging Bolt, Gouging Fire, Iron Boulder or Iron Crown as Paradox, so Showdown does boost it. **Follow Showdown.**
- **Calc vs Showdown, Protosynthesis:** calc uses 5324 for defensive Protosynthesis and Showdown uses 5325. Follow Showdown.
- **L3b seeding:** seeding Showdown's Gen 5 LCG with near-identical seeds correlates the draws. In our runs a 10% burn showed 0 of 200. Give each sample 4 independent random seed words.
- **Protect chain:** the first Protect has no stall roll, and a failed roll deletes the chain, so the next Protect always works.
- **Showdown rules our engine got wrong at first, now fixed and tested:**
  - Turn flow: residuals stop once a side is wiped, and on a double wipe the side whose Pokémon fainted last wins.
  - Status and PP: fainted status is shown as `fnt`, and Pressure targets cover field moves, plus foe-side moves only with `mustpressure`.
  - Moves: Struggle is typeless with no STAB, Defog drops evasion, Knock Off releases Choice lock, and Sitrus activates after after-hit effects.
  - Status and ability: Rest works after waking, and Sand Rush and other sand-immune abilities ignore sand chip.
  - Stellar Tera: the first use gets 2x or 1.2x and a 60 BP floor.
- **No whole pool team** is covered by M2 alone. Test teams were assembled from covered sets and Showdown-validated. **M3's gate requires real pool teams.**

## Yardsticks

- Metamon Kakuna has a 71% GXE on gen9ou. Beating it locally above 55% is the bar.
- Foul Play (which uses search) is a stretch goal.
- Jaxcalibur trained about 100M games in about a week on one H100. On the 4060, aim for a 3–5M parameter model and a resumable multi-day run that checkpoints often. Measure games per hour at a few model sizes in M4, and go up to about 8M if that's affordable.
