"""
engine/oujax/bridge.py
Runs the same battle in OU-JAX and in Showdown (through the reference harness) and compares them after every
step: translates OU-JAX actions into Showdown choices, and diffs HP, status, boosts, items, PP, Tera, field,
side conditions, requests and legal actions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.oujax.engine import A_SWITCH, A_TERA, Config, choice_locked, legal_actions, step, switch_target
from engine.oujax.ref import RefHarness
from engine.oujax.state import (
    HAZ_ROCKS,
    HAZ_SPIKES,
    HAZ_TSPIKES,
    HAZ_WEBS,
    PHASE_MOVE,
    PHASE_PREVIEW,
    PHASE_SWITCH,
    SCR_LIGHT_SCREEN,
    SCR_REFLECT,
    SCR_VEIL,
    Battle,
    init_battle,
)
from engine.oujax.tables import STATUS, TERRAINS, WEATHERS, load_tables
from engine.oujax.team import load_packed_team

T = load_tables()
ITEM_NAMES = {v: k for k, v in T.item_ids.items()}
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

_jit_step = jax.jit(step, static_argnums=3)
_jit_legal = jax.jit(legal_actions)


@dataclass
class Mirror:
    """One battle played in both engines with identical choices."""

    ref: RefHarness
    packed: Tuple[str, str]
    cfg: Config
    names: List[List[str]] = field(default_factory=list)  # names[s][slot]: Showdown nickname of our team slot
    battle: int = 0
    sd: Dict = field(default_factory=dict)
    ours: Optional[Battle] = None
    key: jax.Array = None

    @classmethod
    def start(cls, ref: RefHarness, p1: str, p2: str, cfg: Config, forced_opts=None, seed: int = 0) -> "Mirror":
        packed = tuple(ref.pack([p1, p2]))
        m = cls(ref=ref, packed=packed, cfg=cfg)
        forced = forced_opts if forced_opts is not None else (True if cfg.forced else None)
        m.battle, m.sd = ref.start(packed[0], packed[1], forced=forced)
        m.names = [[p["name"] for p in side["pokemon"]] for side in m.sd["sides"]]
        m.ours = init_battle(load_packed_team(packed[0]), load_packed_team(packed[1]))
        m.key = jax.random.PRNGKey(seed)
        return m

    # --- actions ---------------------------------------------------------------------------------

    def sd_position(self, s: int, slot: int) -> int:
        """1-based position of our team slot in Showdown's current (reordered) team list."""
        name = self.names[s][slot]
        return next(i for i, p in enumerate(self.sd["sides"][s]["pokemon"]) if p["name"] == name) + 1

    def choice(self, s: int, action: int) -> Optional[str]:
        st = self.ours
        phase = int(st.phase)
        if phase == PHASE_PREVIEW:
            return f"team {action + 1}"
        if phase == PHASE_SWITCH and not bool(st.need_switch[s]):
            return None
        if action >= A_SWITCH:
            slot = int(switch_target(st, s, action))
            return f"switch {self.sd_position(s, slot)}"
        move = action % 4 + 1
        return f"move {move}" + (" terastallize" if A_TERA <= action < A_SWITCH else "")

    def legal(self) -> np.ndarray:
        return np.asarray(_jit_legal(self.ours))

    def play(self, actions) -> None:
        choices = [self.choice(s, int(actions[s])) for s in (0, 1)]
        self.sd = self.ref.choose(self.battle, p1=choices[0], p2=choices[1])
        self.key, k = jax.random.split(self.key)
        self.ours = _jit_step(self.ours, jnp.asarray(actions, jnp.int32), k, self.cfg)

    def end(self) -> None:
        self.ref.end(self.battle)

    # --- comparison ------------------------------------------------------------------------------

    def diff(self) -> List[str]:
        st, sd = self.ours, self.sd
        out = []

        def check(label, ours, theirs):
            if ours != theirs:
                out.append(f"{label}: ours={ours!r} showdown={theirs!r}")

        check("turn", int(st.turn), sd["turn"])
        winner = {"": -1, "p1": 0, "p2": 1}.get(sd["winner"], 2) if sd["ended"] else -1
        if sd["ended"] and not sd["winner"]:
            winner = 2
        check("winner", int(st.winner), winner)
        for s, side in enumerate(sd["sides"]):
            by_name = {p["name"]: p for p in side["pokemon"]}
            for slot, name in enumerate(self.names[s]):
                p = by_name[name]
                tag = f"p{s + 1}.{name}"
                check(f"{tag}.hp", int(st.hp[s, slot]), p["hp"])
                if p["hp"] == 0 or int(st.hp[s, slot]) == 0:
                    continue  # Showdown clears a fainted Pokemon's status and Tera; nothing else matters then
                check(f"{tag}.status", STATUS[int(st.status[s, slot])], p["status"])
                if p["status"] == "slp" and p["statusTurns"] is not None:
                    check(f"{tag}.sleep_turns", int(st.status_turns[s, slot]), p["statusTurns"])
                if p["status"] == "tox":
                    check(f"{tag}.toxic_stage", int(st.status_turns[s, slot]), p["toxicStage"])
                check(f"{tag}.item", ITEM_NAMES[int(st.item[s, slot])], p["item"])
                check(f"{tag}.tera", bool(st.terastallized[s, slot]), bool(p["terastallized"]))
                check(f"{tag}.pp", [int(x) for x in st.pp[s, slot][: len(p["moves"])]], [m["pp"] for m in p["moves"]])
            active_idx = side["active"][0]
            if active_idx is not None and int(st.active[s]) >= 0:
                p = side["pokemon"][active_idx]
                tag = f"p{s + 1}.active"
                check(tag, self.names[s][int(st.active[s])], p["name"])
                if self.names[s][int(st.active[s])] == p["name"] and not p["fainted"]:
                    check(f"{tag}.boosts", [int(x) for x in st.boosts[s]], [p["boosts"][k] for k in BOOST_KEYS])
                    vol = set(p["volatiles"])
                    check(f"{tag}.substitute", bool(st.sub_hp[s] > 0), "substitute" in vol)
                    check(f"{tag}.choicelock", bool(choice_locked(st, s)), "choicelock" in vol)
                    check(f"{tag}.paradox", bool(st.paradox_stat[s] > 0), bool({"protosynthesis", "quarkdrive"} & vol))
            if sd["ended"]:
                continue  # Showdown stops mid-turn when the battle is decided; leftover counters don't matter
            conds = side["sideConditions"]
            hz = [int(x) for x in st.hazards[s]]
            check(f"p{s + 1}.stealthrock", hz[HAZ_ROCKS], conds.get("stealthrock", {}).get("layers", 0))
            check(f"p{s + 1}.spikes", hz[HAZ_SPIKES], conds.get("spikes", {}).get("layers", 0))
            check(f"p{s + 1}.toxicspikes", hz[HAZ_TSPIKES], conds.get("toxicspikes", {}).get("layers", 0))
            check(f"p{s + 1}.stickyweb", hz[HAZ_WEBS], conds.get("stickyweb", {}).get("layers", 0))
            for name, idx in (("reflect", SCR_REFLECT), ("lightscreen", SCR_LIGHT_SCREEN), ("auroraveil", SCR_VEIL)):
                check(f"p{s + 1}.{name}", int(st.screens[s, idx]), (conds.get(name) or {}).get("duration") or 0)
            check(f"p{s + 1}.tailwind", int(st.tailwind[s]), (conds.get("tailwind") or {}).get("duration") or 0)
        f = sd["field"]
        if sd["ended"]:
            return out
        check("weather", WEATHERS[int(st.weather)], f["weather"])
        if f["weather"]:
            check("weather_turns", int(st.weather_turns), f["weatherTurns"])
        check("terrain", TERRAINS[int(st.terrain)], f["terrain"])
        if f["terrain"]:
            check("terrain_turns", int(st.terrain_turns), f["terrainTurns"])
        check("trickroom", int(st.trick_room), f["pseudoWeather"].get("trickroom") or 0)
        out += self.request_diff()
        return out

    def request_diff(self) -> List[str]:
        """Compare who has to act and with which options."""
        st, sd = self.ours, self.sd
        if sd["ended"]:
            return []
        out = []
        legal = self.legal()
        phase = int(st.phase)
        for s in (0, 1):
            req = sd["requests"][f"p{s + 1}"] or {}
            kind = (
                "preview" if req.get("teamPreview") else
                "switch" if req.get("forceSwitch") else
                "wait" if req.get("wait") else
                "move" if req.get("active") else "none"
            )  # fmt: skip
            ours_kind = (
                "preview" if phase == PHASE_PREVIEW else
                "move" if phase == PHASE_MOVE else
                "switch" if bool(st.need_switch[s]) else "wait"
            )  # fmt: skip
            if kind != ours_kind:
                out.append(f"p{s + 1}.request: ours={ours_kind} showdown={kind}")
                continue
            if kind == "move":
                active = req["active"][0]
                sd_moves = [not m.get("disabled") and m.get("pp", 1) > 0 for m in active["moves"]]
                if len(active["moves"]) == 1 and active["moves"][0]["id"] == "struggle":
                    sd_moves = [True]
                ours_moves = [bool(x) for x in legal[s, : len(sd_moves)]]
                if ours_moves != sd_moves:
                    out.append(f"p{s + 1}.legal_moves: ours={ours_moves} showdown={sd_moves}")
                can_tera = bool(active.get("canTerastallize"))
                if bool(legal[s, A_TERA : A_TERA + 4].any()) != can_tera:
                    out.append(f"p{s + 1}.can_tera: ours={not can_tera} showdown={can_tera}")
            if kind in ("move", "switch"):
                side_mons = req["side"]["pokemon"]
                sd_switch = sorted(
                    p["ident"].split(": ")[1]
                    for p in side_mons
                    if not p["active"] and not p["condition"].endswith(" fnt")
                )
                ours_switch = sorted(
                    self.names[s][int(switch_target(st, s, A_SWITCH + n))] for n in range(5) if legal[s, A_SWITCH + n]
                )
                if req.get("active") and req["active"][0].get("trapped"):
                    sd_switch = []
                if ours_switch != sd_switch:
                    out.append(f"p{s + 1}.legal_switches: ours={ours_switch} showdown={sd_switch}")
        return out


def random_policy(rng: np.random.Generator, legal: np.ndarray, tera_rate: float = 0.15) -> np.ndarray:
    """Random legal action per side, with Tera picked only occasionally so battles see both states."""
    acts = []
    for s in range(2):
        ok = np.flatnonzero(legal[s])
        non_tera = [a for a in ok if not (A_TERA <= a < A_SWITCH)]
        tera = [a for a in ok if A_TERA <= a < A_SWITCH]
        pool = tera if tera and rng.random() < tera_rate else (non_tera or ok)
        acts.append(int(rng.choice(pool)))
    return np.array(acts)


def play_mirrored(ref: RefHarness, p1: str, p2: str, cfg: Config, seed: int, max_steps: int = 400, forced_opts=None):
    """Plays one battle in both engines; returns (steps played, first diff list or [] if they never diverged,
    log of the Showdown lines leading to the divergence)."""
    rng = np.random.default_rng(seed)
    m = Mirror.start(ref, p1, p2, cfg, forced_opts=forced_opts, seed=seed)
    try:
        diffs = m.diff()
        if diffs:
            return 0, diffs, m.sd["log"]
        for i in range(max_steps):
            acts = random_policy(rng, m.legal())
            m.play(acts)
            diffs = m.diff()
            if diffs:
                return i + 1, [f"after actions {acts.tolist()} at step {i + 1}"] + diffs, m.sd["log"]
            if m.sd["ended"]:
                return i + 1, [], []
        return max_steps, [], []
    finally:
        m.end()
