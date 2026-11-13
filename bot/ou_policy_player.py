"""
bot/ou_policy_player.py
A poke-env Player that plays Gen 9 OU on Showdown straight from a training/ou_ppo.py checkpoint (no search). The
live battle is mapped into the same Obs fields training/ou_obs.py builds from OU-JAX, and the 13 policy actions
(moves 0-3, moves + Tera 4-7, switches to the n-th benched team member 8-12, leads 0-5) map back to orders.

Differences from the simulator's view (the policy trains on OU-JAX): Showdown reveals foe items and moves as they
show up (OU-JAX's observation only counts a foe item once it changed, so a foe item stays UNKNOWN here until the
foe has none), screen / Trick Room turns are counted from their start turn, and foe move slots are unordered.
"""

import pickle
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from poke_env.battle import Effect, Field, SideCondition, Weather
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from training.ou_obs import N_ABILITIES, N_ITEMS, N_MOVES, N_TYPES, Obs, T
from training.ou_ppo import Policy

STATUS = ("", "brn", "par", "psn", "tox", "slp", "frz")
WEATHER = {Weather.SUNNYDAY: 1, Weather.DESOLATELAND: 1, Weather.RAINDANCE: 2, Weather.PRIMORDIALSEA: 2,
           Weather.SANDSTORM: 3, Weather.SNOWSCAPE: 4, Weather.SNOW: 4, Weather.HAIL: 4}  # fmt: skip
TERRAIN = {Field.ELECTRIC_TERRAIN: 1, Field.GRASSY_TERRAIN: 2, Field.MISTY_TERRAIN: 3, Field.PSYCHIC_TERRAIN: 4}
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def _id(table, name: Optional[str], unknown: int) -> int:
    if name is None:
        return unknown
    key = "".join(c for c in str(name).lower() if c.isalnum())
    return table.get(key, unknown)


def _type_id(t) -> int:
    return T.type_ids.get(t.name.lower(), N_TYPES) if t is not None else N_TYPES


def load_policy(path: str):
    ck = pickle.load(open(path, "rb"))
    args = ck.get("args", {})
    d, layers = args.get("d", 192), args.get("layers", 4)
    model = Policy(d=d, layers=layers, heads=max(1, d // 32))
    return model, ck["params"]


class OUPolicyPlayer(Player):
    def __init__(self, checkpoint: str, *args, greedy: bool = True, seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.model, self.params = load_policy(checkpoint)
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)
        self._apply = jax.jit(lambda p, o, m: self.model.apply(p, o, m))

    # --- observation ------------------------------------------------------------------------------------

    def _team_order(self, battle):
        return list(battle.team.values())  # request order = the order the team was submitted in

    def _mon_rows(self, mons, own: bool, active, battle):
        rows = []
        for mon in mons[:6]:
            sp = T.dex["species"].get(_species_key(mon.species), {})
            types = [T.type_ids.get(t.lower(), N_TYPES) for t in sp.get("types", [])] + [N_TYPES]
            if len(sp.get("types", [])) == 1:
                types[1] = types[0]  # OU-JAX stores a single-typed Pokemon's type twice
            moves = list(mon.moves.values())[:4]
            if own:
                mv_ids = [_id(T.move_ids, m.id, N_MOVES) for m in moves]
                pp = [min(max(m.current_pp / 16.0, 0.0), 4.0) / 4.0 for m in moves]
                item = _id(T.item_ids, mon.item, N_ITEMS) if mon.item else 0
                ability = _id(T.ability_ids, mon.ability, N_ABILITIES)
                tera = _type_id(mon.tera_type)
                stats = [mon.max_hp or 0] + [(mon.stats or {}).get(k) or 0 for k in ("atk", "def", "spa", "spd", "spe")]
            else:
                mv_ids = [_id(T.move_ids, m.id, N_MOVES) for m in moves]
                pp = [0.0] * len(moves)
                item = 0 if mon.item == "" else N_ITEMS
                ability = N_ABILITIES
                tera = _type_id(mon.tera_type) if mon.is_terastallized else N_TYPES
                stats = [0] * 6
            known = [1.0] * len(mv_ids)
            while len(mv_ids) < 4:
                mv_ids.append(N_MOVES if not own else 0)
                pp.append(0.0)
                known.append(0.0)
            status = np.zeros(7)
            st_name = mon.status.name.lower() if mon.status else ""
            status[STATUS.index(st_name) if st_name in STATUS else 0] = 1.0  # fainted (fnt) shows no status
            is_active = mon is active
            boosts = [mon.boosts.get(k, 0) / 6.0 if is_active else 0.0 for k in BOOSTS]
            mon_float = np.concatenate(
                [
                    [mon.current_hp_fraction],
                    status,
                    [float(is_active)],
                    [float(mon.fainted)],
                    [float(mon.is_terastallized)],
                    boosts,
                    [mon.level / 100.0],
                    (np.array(stats, np.float32) / 500.0) if own else np.zeros(6),
                ]  # fmt: skip
            )
            rows.append(dict(species=_id(T.species_ids, _species_key(mon.species), 0), item=item, ability=ability,
                             moves=mv_ids, move_float=np.stack([pp, known], -1), types=types[:2], tera=tera,
                             mon_float=mon_float))  # fmt: skip
        while len(rows) < 6:
            rows.append(dict(species=0, item=0, ability=0, moves=[0] * 4, move_float=np.zeros((4, 2)),
                             types=[0, 0], tera=0, mon_float=np.zeros(25)))  # fmt: skip
        return rows

    def observe(self, battle, preview: bool = False) -> Obs:
        team = self._team_order(battle)
        own_active = None if preview else battle.active_pokemon
        foe = list(battle.opponent_team.values())
        known = {m.species for m in foe}
        foe += [m for m in battle.teampreview_opponent_team if m.species not in known]
        rows = self._mon_rows(team, True, own_active, battle) + self._mon_rows(
            foe, False, None if preview else battle.opponent_active_pokemon, battle
        )

        def side_field(conds):
            hz = [float(SideCondition.STEALTH_ROCK in conds), conds.get(SideCondition.SPIKES, 0),
                  conds.get(SideCondition.TOXIC_SPIKES, 0), float(SideCondition.STICKY_WEB in conds)]  # fmt: skip

            def left(c, n):
                return max(n - (battle.turn - conds[c]), 0) if c in conds else 0

            scr = [
                left(SideCondition.REFLECT, 5),
                left(SideCondition.LIGHT_SCREEN, 5),
                left(SideCondition.AURORA_VEIL, 5),
            ]
            return np.array(hz) / 3.0, np.array(scr) / 8.0, left(SideCondition.TAILWIND, 4) / 4.0

        own_hz, own_scr, own_tw = side_field(battle.side_conditions)
        foe_hz, foe_scr, foe_tw = side_field(battle.opponent_side_conditions)
        weather = np.zeros(5)
        weather[max([WEATHER.get(w, 0) for w in battle.weather] + [0])] = 1.0
        terrain = np.zeros(5)
        terrain[max([TERRAIN.get(f, 0) for f in battle.fields] + [0])] = 1.0
        tr = battle.fields.get(Field.TRICK_ROOM)
        trick_room = max(5 - (battle.turn - tr), 0) / 5.0 if tr is not None else 0.0
        phase = np.zeros(3)
        phase[0 if preview else (2 if battle.force_switch else 1)] = 1.0
        subs = [float(m is not None and Effect.SUBSTITUTE in m.effects) for m in (own_active, None if preview else battle.opponent_active_pokemon)]  # fmt: skip
        field = np.concatenate([weather, terrain, [trick_room], own_hz, foe_hz, own_scr, foe_scr, [own_tw, foe_tw],
                                [battle.turn / 100.0], phase, [float(bool(battle.force_switch)), 0.0], subs])  # fmt: skip

        slot = team.index(own_active) if own_active in team else -1
        own_onehot = np.array([float(i == slot) for i in range(6)])
        benched = [i for i in range(6) if i != slot]
        bench_slot = np.zeros((5, 6))
        for n, i in enumerate(benched[:5]):
            bench_slot[n, i] = 1.0

        def col(k, dtype):
            return jnp.asarray(np.array([r[k] for r in rows]), dtype)

        return Obs(
            species=col("species", jnp.int32), item=col("item", jnp.int32), ability=col("ability", jnp.int32),
            moves=col("moves", jnp.int32), move_float=col("move_float", jnp.float32), types=col("types", jnp.int32),
            tera=col("tera", jnp.int32), mon_float=col("mon_float", jnp.float32),
            side=jnp.asarray([0] * 6 + [1] * 6, jnp.int32), field=jnp.asarray(field, jnp.float32),
            own_active=jnp.asarray(own_onehot, jnp.float32), bench_slot=jnp.asarray(bench_slot, jnp.float32),
            preview=jnp.asarray(preview),
        )  # fmt: skip

    # --- actions ----------------------------------------------------------------------------------------

    def _pick(self, obs, mask):
        logits, _ = self._apply(self.params, obs, jnp.asarray(mask))
        logits = np.asarray(logits, np.float64)
        if self.greedy:
            return int(np.argmax(logits))
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return int(self.rng.choice(len(p), p=p))

    def teampreview(self, battle):
        obs = self.observe(battle, preview=True)
        mask = np.zeros(13, bool)
        mask[: min(6, len(battle.team))] = True
        lead = self._pick(obs, mask)
        order = [lead] + [i for i in range(len(battle.team)) if i != lead]
        return "/team " + "".join(str(i + 1) for i in order)

    def choose_move(self, battle) -> BattleOrder:
        team = self._team_order(battle)
        active = battle.active_pokemon
        mask = np.zeros(13, bool)
        targets = {}
        avail = {m.id: m for m in battle.available_moves}
        if active is not None and not battle.force_switch:
            own_moves = list(active.moves.values())[:4]
            for i, m in enumerate(own_moves):
                if m.id in avail:
                    mask[i] = True
                    targets[i] = (avail[m.id], False)
                    if battle.can_tera:
                        mask[4 + i] = True
                        targets[4 + i] = (avail[m.id], True)
            if not mask[:4].any() and battle.available_moves:  # Struggle / recharge / locked move
                mask[0] = True
                targets[0] = (battle.available_moves[0], False)
        slot = team.index(active) if active in team else -1
        benched = [team[i] for i in range(len(team)) if i != slot]
        switchable = {id(m) for m in battle.available_switches}
        for n, mon in enumerate(benched[:5]):
            if id(mon) in switchable:
                mask[8 + n] = True
                targets[8 + n] = (mon, False)
        if not mask.any():
            return self.choose_random_move(battle)
        a = self._pick(self.observe(battle), mask)
        what, tera = targets[a]
        return self.create_order(what, terastallize=tera)


def _species_key(species: str) -> str:
    return "".join(c for c in species.lower() if c.isalnum())
