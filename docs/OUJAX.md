# OU-JAX

A Gen 9 OU battle engine in JAX for self-play training, verified against Pokémon Showdown. Training uses
only OU-JAX (thousands of battles per GPU step); Showdown is the test oracle and the evaluation ground truth.

## Layout

| Path | What |
|---|---|
| `engine/oujax/data/dex_gen9.json` | Gen 9 dex exported from the pinned Showdown commit (`tools/ref/export_dex.js`) |
| `engine/oujax/tables.py` | Compact int tables + id maps: species, moves (60 columns), natures, type chart |
| `engine/oujax/team.py` | Packed-team parsing and exact stat precompute |
| `engine/oujax/state.py` | `Battle`: one fixed-shape PyTree per battle (int8/int16, side-major) |
| `engine/oujax/damage.py` | Showdown's damage formula with 4096-based rounding |
| `engine/oujax/engine.py` | `step(state, actions, key, Config)`, `legal_actions(state)` |
| `engine/oujax/effects.py`, `coverage.py` | Implemented moves/items/abilities (the M2 set) and the covered-team builder |
| `tools/ref/ref.js`, `engine/oujax/ref.py` | Reference harness: Showdown's sim and `@smogon/calc` over JSON lines |
| `engine/oujax/bridge.py` | Runs a battle in both engines with identical choices and diffs them every step |

## Interface

```python
from engine.oujax.team import load_packed_team
from engine.oujax.state import init_battle
from engine.oujax.engine import Config, legal_actions, step

st = init_battle(load_packed_team(p1_packed), load_packed_team(p2_packed))
mask = legal_actions(st)  # (2, 13) bool
st = step(st, actions, key, Config())  # actions: int (2,)
```

Requests follow Showdown: `st.phase` is team preview (actions = lead slot 0..5), a move turn (0..3 move,
4..7 move + Terastallize, 8..12 switch to the n-th benched team slot), or a switch request (`st.need_switch`
says which sides act; used for faint replacements and the mid-turn switch after U-turn / Volt Switch / Flip
Turn / Parting Shot / Teleport / Chilly Reception, after which the rest of the turn runs). `st.winner` is -1
while the battle is on, 0/1 for a win, 2 for a tie. Everything is branch-free, so `jax.vmap(step)` runs many
battles at once (see `engine/oujax/bench.py`).

Every random check is an independent draw from the step key. `Config(forced=True)` fixes them instead
(always hit, no crit, max roll, no secondaries below 100%, no full paralysis or thaw, Protect always
succeeds, minimum durations, speed ties to p1), matching the harness's forced-outcome mode.

## What M2 implements

- Stats, boosts, accuracy/evasion stages, the full damage formula (crits, rolls, STAB and Tera/Stellar STAB,
  type chart, burn, weather, terrain, screens, stat-override moves), speed order with priority, Trick Room,
  paralysis, Choice Scarf, Tailwind, weather speed abilities and Protosynthesis/Quark Drive.
- Switching, team preview, faint replacement, pivots with Showdown's request flow, Struggle.
- Hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web), Heavy-Duty Boots, removal (Rapid Spin, Mortal
  Spin, Defog, Court Change).
- Status (burn, paralysis, poison, toxic, sleep with Sleep Clause, freeze), terrain/type status immunities.
- Weather and terrain with setters (moves and abilities), durations and rock/extender items; Trick Room.
- Tera, Protect, Substitute, recovery (fractions, weather-dependent, Roost, Rest), drain, recoil, Haze,
  screens, Knock Off, secondary effects (status, stat changes, flinch), self stat changes.
- Items: Leftovers, Black Sludge, Life Orb, Choice Band/Specs/Scarf, Booster Energy, Heavy-Duty Boots, Rocky
  Helmet, Assault Vest, Sitrus Berry, Air Balloon, Expert Belt, Light Clay, weather rocks, Terrain Extender.
- Abilities: Protosynthesis, Quark Drive, Drizzle, Drought, Sand Stream, Snow Warning, the four Surges,
  Levitate, Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Guts, Regenerator, Natural Cure, Pressure, plus
  a few with no battle effect.

`python -m engine.oujax.coverage` prints how much of the M1 pool this covers and the most common missing
moves, items and abilities (the M3 work list). Moves are covered when they are table-driven (no Showdown
handler code beyond what the engine models) or hand-written.

## Verification

| Level | What | How to run |
|---|---|---|
| L1 | Stats, damage | `pytest engine/oujax/test_tables.py engine/oujax/test_damage.py` (3,120 Pokemon; 4,000 random damage cases vs `@smogon/calc`) |
| L2 | Interactions | `pytest engine/oujax/test_l2.py` (scripted scenarios vs Showdown, forced outcomes) |
| L3a | Full battles, forced | `python -m engine.oujax.l3_check --battles 50` (random legal policy, turn-by-turn diff) |
| L3b | Full battles, statistics | `python -m engine.oujax.stats_check --scenarios 20 --samples 200` (KS / chi-square) |
| Speed | Throughput | `python -m engine.oujax.bench --batch 1024 4096 16384` |

The harness needs `cd tools/ref && npm install` once and the Showdown clone next to the repo
(`docs/SETUP.md`). Known Showdown-vs-calc difference: calc never lets Knock Off remove Booster Energy from a
Paradox Pokemon, but Showdown's dex doesn't tag Raging Bolt / Gouging Fire / Iron Boulder / Iron Crown as
Paradox; OU-JAX follows Showdown (pinned by an L2 test).
