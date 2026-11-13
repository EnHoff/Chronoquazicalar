"""
training/ou_ppo.py
PPO self-play for Gen 9 OU on OU-JAX: everything (battles, observations, policy, GAE, updates) stays on the GPU.

- Teams: real pool teams whose every set OU-JAX covers (data/ou/generated/covered_pool.jsonl, built on first use).
- Observation (obs.py): one player's view. The foe's six species are known from team preview; its moves, item,
  ability and Tera type stay hidden until revealed.
- Model: a small transformer over 12 Pokemon tokens and a field token; 13 action logits and a value.
- Opponents: the current policy (both sides learn), past checkpoints (a league), and a uniform random policy.
- Reward: +1 win, -1 loss, at the end only. GAE, clipped PPO, entropy bonus, zero-avoiding penalty.
- Checkpoints every --ckpt-minutes to checkpoints/ou_ppo/ (resumed automatically); metrics as JSON lines.

    python -m training.ou_ppo --batch 1024 --steps 64 --hours 60
"""

import argparse
import json
import pickle
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from engine.oujax.engine import N_ACTIONS, Config, legal_actions, step
from engine.oujax.state import init_battle
from engine.oujax.team import load_packed_team
from training.ou_obs import VOCAB, Obs, Seen, init_seen, observe, update_seen

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "data" / "ou" / "generated" / "team_pool.jsonl"
COVERED_POOL = REPO / "data" / "ou" / "generated" / "covered_pool.jsonl"
CKPT_DIR = REPO / "checkpoints" / "ou_ppo"

OPP_SELF, OPP_LEAGUE, OPP_RANDOM = 0, 1, 2


# --- teams ----------------------------------------------------------------------------------------------------


def covered_pool(path: Path = COVERED_POOL):
    """[(packed, weight)] for every pool team OU-JAX fully covers (cached to covered_pool.jsonl)."""
    if not path.exists():
        from engine.oujax.coverage import covered_set

        with open(POOL) as f, open(path, "w") as out:
            for line in f:
                t = json.loads(line)
                if all(covered_set(s) for s in t["sets"]):
                    out.write(json.dumps({"packed": t["packed"], "weight": t.get("recency_weight", 1.0)}) + "\n")
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [(r["packed"], r["weight"]) for r in rows]


def battle_bank(n: int, seed: int = 0):
    """n starting battles between covered pool teams drawn by recency weight, stacked on a leading axis."""
    pool = covered_pool()
    w = np.array([wt for _, wt in pool], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=2 * n, p=w / w.sum())
    teams = [load_packed_team(pool[i][0]) for i in idx]
    battles = [init_battle(teams[2 * i], teams[2 * i + 1]) for i in range(n)]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *battles), len(pool)


# --- model ----------------------------------------------------------------------------------------------------


class Block(nn.Module):
    d: int
    heads: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads)(h, h)
        h = nn.LayerNorm()(x)
        return x + nn.Dense(self.d)(nn.gelu(nn.Dense(4 * self.d)(h)))


class Policy(nn.Module):
    """Pokemon tokens (6 own, 6 foe) plus a field token -> 13 action logits and a value in [-1, 1]."""

    d: int = 192
    layers: int = 4
    heads: int = 6
    vocab: tuple = VOCAB  # species, items, abilities, moves, types (each + 1 for UNKNOWN)

    @nn.compact
    def __call__(self, obs: Obs, mask):
        n_sp, n_it, n_ab, n_mv, n_ty = (v + 1 for v in self.vocab)
        e = self.d
        sp = nn.Embed(n_sp, e)(obs.species)
        it = nn.Embed(n_it, e // 2)(obs.item)
        ab = nn.Embed(n_ab, e // 2)(obs.ability)
        mv_emb = nn.Embed(n_mv, e // 2)
        mv = mv_emb(obs.moves)  # (12, 4, e/2)
        mv_feat = jnp.concatenate([mv, obs.move_float], axis=-1)
        mv_tok = nn.Dense(e)(mv_feat)  # per move slot
        ty = nn.Embed(n_ty, e // 4)
        types = ty(obs.types).reshape(obs.types.shape[:-1] + (-1,))  # (12, 2*e/4)
        tera = ty(obs.tera)
        mon = jnp.concatenate([it, ab, types, tera, obs.mon_float, mv_tok.mean(axis=-2)], axis=-1)
        mon = sp + nn.Dense(e)(mon) + nn.Embed(2, e)(obs.side)  # side: 0 own, 1 foe
        field = nn.Dense(e)(obs.field)[None]
        x = jnp.concatenate([field, mon], axis=0)  # (13, e)
        for _ in range(self.layers):
            x = Block(e, self.heads)(x)
        x = nn.LayerNorm()(x)
        g = x[0]
        own = x[1:7]
        # Action heads: moves score the own active's move tokens, switches / leads score own Pokemon tokens
        act_mon = jnp.sum(own * obs.own_active[:, None], axis=0)
        mv_q = nn.Dense(e)(jnp.concatenate([g, act_mon]))
        own_moves = jnp.sum(mv_tok[:6] * obs.own_active[:, None, None], axis=0)  # (4, e)
        tera_bias = nn.Dense(1)(jnp.concatenate([g, act_mon]))
        move_logits = own_moves @ mv_q / jnp.sqrt(e)
        sw_q = nn.Dense(e)(g)
        mon_logits = own @ sw_q / jnp.sqrt(e)  # (6,) per team slot
        bench_logits = jnp.sum(obs.bench_slot * mon_logits[None], axis=-1)  # (5,) n-th benched slot
        lead_logits = mon_logits
        moves13 = jnp.concatenate([move_logits, move_logits + tera_bias, bench_logits])
        preview13 = jnp.concatenate([lead_logits, jnp.full((N_ACTIONS - 6,), -1e9)])
        logits = jnp.where(obs.preview, preview13, moves13)
        logits = jnp.where(mask, logits, -1e9)
        value = jnp.tanh(nn.Dense(1)(nn.gelu(nn.Dense(e)(g)))[0])
        return logits, value


# --- rollout --------------------------------------------------------------------------------------------------


class Env(NamedTuple):
    st: object  # Battle
    seen: Seen
    opp_kind: jnp.ndarray  # () int32: OPP_* for side 1
    opp_idx: jnp.ndarray  # () int32: league slot for OPP_LEAGUE


class Traj(NamedTuple):
    obs: Obs  # (T, B, 2, ...)
    mask: jnp.ndarray  # (T, B, 2, 13)
    act: jnp.ndarray  # (T, B, 2)
    logp: jnp.ndarray  # (T, B, 2)
    value: jnp.ndarray  # (T, B, 2)
    reward: jnp.ndarray  # (T, B, 2)
    done: jnp.ndarray  # (T, B)
    learn: jnp.ndarray  # (T, B, 2) bool: this side's data trains the policy


def make_fns(model: Policy, cfg: Config, bank, bank_n: int, league_size: int, mix):
    """Jitted rollout and update functions. mix = (p_self, p_league, p_random) for each new episode's opponent."""

    def policy_both(params, obs, mask):
        return jax.vmap(lambda o, m: model.apply(params, o, m))(obs, mask)

    def new_episode(key, env_template):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        idx = jax.random.randint(k1, (), 0, bank_n)
        st = jax.tree.map(lambda b: b[idx], bank)
        kind = jax.random.choice(k2, 3, p=jnp.asarray(mix))
        return Env(st, init_seen(st), kind.astype(jnp.int32), jax.random.randint(k3, (), 0, league_size))

    def env_step(params, league, env: Env, key):
        k_a, k_o, k_s, k_r = jax.random.split(key, 4)
        st = env.st
        mask = legal_actions(st)  # (2, 13)
        obs = jax.vmap(lambda p: observe(st, env.seen, p))(jnp.arange(2))
        logits, value = policy_both(params, obs, mask)
        # Side 1's opponent policy: league checkpoint or uniform random
        league_p = jax.tree.map(lambda x: x[env.opp_idx], league)
        opp_logits, _ = model.apply(league_p, jax.tree.map(lambda x: x[1], obs), mask[1])
        rand_logits = jnp.where(mask[1], 0.0, -1e9)
        side1 = jnp.select(
            [env.opp_kind == OPP_LEAGUE, env.opp_kind == OPP_RANDOM], [opp_logits, rand_logits], logits[1]
        )
        use_logits = logits.at[1].set(side1)
        acts = jax.random.categorical(k_a, use_logits, axis=-1)
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), acts[:, None], -1)[:, 0]
        nxt = step(st, acts.astype(jnp.int32), k_s, cfg)
        seen = update_seen(env.seen, st, nxt)
        done = nxt.winner >= 0
        win = jnp.stack([nxt.winner == 0, nxt.winner == 1]).astype(jnp.float32)
        reward = jnp.where(done, win - win[::-1], 0.0)
        learn = jnp.array([True, False]) | (env.opp_kind == OPP_SELF)
        fresh = new_episode(k_r, env)
        nenv = jax.tree.map(lambda a, b: jnp.where(done, b, a), Env(nxt, seen, env.opp_kind, env.opp_idx), fresh)
        out = (obs, mask, acts, logp, value, reward, done, learn, env.opp_kind, nxt.winner)
        return nenv, out

    @partial(jax.jit, static_argnums=(4,))
    def rollout(params, league, envs, key, n_steps):
        def body(envs, k):
            keys = jax.random.split(k, envs.opp_kind.shape[0])
            return jax.vmap(env_step, in_axes=(None, None, 0, 0))(params, league, envs, keys)

        envs, out = jax.lax.scan(body, envs, jax.random.split(key, n_steps))
        obs, mask, acts, logp, value, reward, done, learn, kind, winner = out
        # Bootstrap values for the last state
        last_mask = jax.vmap(legal_actions)(envs.st)
        last_obs = jax.vmap(lambda st, seen: jax.vmap(lambda p: observe(st, seen, p))(jnp.arange(2)))(
            envs.st, envs.seen
        )
        _, last_v = jax.vmap(lambda o, m: policy_both(params, o, m))(last_obs, last_mask)
        stats = {
            "episodes": jnp.sum(done),
            "rand_games": jnp.sum(done & (kind == OPP_RANDOM)),
            "rand_wins": jnp.sum(done & (kind == OPP_RANDOM) & (winner == 0)),
            "league_games": jnp.sum(done & (kind == OPP_LEAGUE)),
            "league_wins": jnp.sum(done & (kind == OPP_LEAGUE) & (winner == 0)),
        }
        return envs, Traj(obs, mask, acts, logp, value, reward, done, learn), last_v, stats

    return rollout, new_episode, policy_both


def gae(traj: Traj, last_v, gamma=0.997, lam=0.95):
    def back(carry, x):
        adv_next, v_next = carry
        r, v, d = x
        nd = 1.0 - d[:, None].astype(jnp.float32)
        delta = r + gamma * v_next * nd - v
        adv = delta + gamma * lam * nd * adv_next
        return (adv, v), adv

    _, adv = jax.lax.scan(back, (jnp.zeros_like(last_v), last_v), (traj.reward, traj.value, traj.done), reverse=True)
    return adv, adv + traj.value


def make_update(model: Policy, opt, clip=0.2, ent_coef=0.01, vf_coef=0.5, za_coef=0.002):
    def loss_fn(params, b):
        logits, value = jax.vmap(lambda o, m: model.apply(params, o, m))(b["obs"], b["mask"])
        logp_all = jax.nn.log_softmax(logits, -1)
        logp = jnp.take_along_axis(logp_all, b["act"][:, None], -1)[:, 0]
        ratio = jnp.exp(logp - b["logp"])
        adv = b["adv"]
        w = b["learn"].astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        pg = -jnp.sum(w * jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv)) / n
        vf = 0.5 * jnp.sum(w * (value - b["ret"]) ** 2) / n
        probs = jnp.exp(logp_all)
        valid = b["mask"].astype(jnp.float32)
        ent = -jnp.sum(w * jnp.sum(probs * jnp.where(b["mask"], logp_all, 0.0), -1)) / n
        za = -jnp.sum(w * jnp.sum(valid * jnp.maximum(logp_all, -20.0), -1) / jnp.maximum(valid.sum(-1), 1.0)) / n
        total = pg + vf_coef * vf - ent_coef * ent + za_coef * za
        return total, {"pg": pg, "vf": vf, "entropy": ent, "za": za, "clipfrac": jnp.mean(jnp.abs(ratio - 1) > clip)}

    @jax.jit
    def update(params, opt_state, batch):
        (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, aux

    return update


def flatten(traj: Traj, adv, ret):
    """(T, B, 2, ...) -> (T*B*2, ...) with advantages normalized over the learning samples."""

    def f(x):
        return x.reshape((-1,) + x.shape[3:])

    w = traj.learn.reshape(-1)
    a = adv.reshape(-1)
    mean = jnp.sum(a * w) / jnp.maximum(w.sum(), 1)
    std = jnp.sqrt(jnp.sum(w * (a - mean) ** 2) / jnp.maximum(w.sum(), 1)) + 1e-8
    return {
        "obs": jax.tree.map(f, traj.obs),
        "mask": f(traj.mask),
        "act": f(traj.act),
        "logp": f(traj.logp),
        "adv": (a - mean) / std,
        "ret": ret.reshape(-1),
        "learn": w,
    }


# --- training loop --------------------------------------------------------------------------------------------


def save(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(obj), f)
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024, help="parallel battles")
    ap.add_argument("--steps", type=int, default=64, help="rollout length per update")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bank", type=int, default=4096, help="starting battles kept on device")
    ap.add_argument("--league", type=int, default=8, help="past checkpoints kept as opponents")
    ap.add_argument("--league-every", type=int, default=200, help="updates between league snapshots")
    ap.add_argument("--mix", type=float, nargs=3, default=(0.7, 0.2, 0.1), help="self / league / random")
    ap.add_argument("--hours", type=float, default=60.0)
    ap.add_argument("--minutes", type=float, default=None, help="stop after this many minutes (smoke tests)")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0)
    ap.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"
    cfg = Config()
    model = Policy(d=args.d, layers=args.layers, heads=max(1, args.d // 32))
    t_setup = time.time()
    bank, n_pool = battle_bank(args.bank, seed=args.seed)
    print(f"covered pool: {n_pool} teams; bank {args.bank} battles", flush=True)

    key = jax.random.PRNGKey(args.seed)
    st0 = jax.tree.map(lambda b: b[0], bank)
    obs0 = observe(st0, init_seen(st0), 0)
    params = model.init(key, obs0, legal_actions(st0)[0])
    n_params = sum(x.size for x in jax.tree.util.tree_leaves(params))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = opt.init(params)
    league = jax.tree.map(lambda x: jnp.stack([x] * args.league), params)
    update_i, env_steps, league_next = 0, 0, 0
    latest = ckpt_dir / "latest.pkl"
    if latest.exists():
        ck = pickle.load(open(latest, "rb"))
        params, opt_state, league = ck["params"], ck["opt_state"], ck["league"]
        update_i, env_steps, league_next = ck["update"], ck["env_steps"], ck["league_next"]
        print(f"resumed from update {update_i} ({env_steps} env steps)", flush=True)
    print(f"params: {n_params / 1e6:.2f}M", flush=True)

    rollout, new_episode, _ = make_fns(model, cfg, bank, args.bank, args.league, tuple(args.mix))
    update = make_update(model, opt)
    key, k = jax.random.split(key)
    envs = jax.vmap(new_episode, in_axes=(0, None))(jax.random.split(k, args.batch), None)

    t_start = time.time()
    last_ckpt = time.time()
    deadline = t_start + (args.minutes * 60 if args.minutes else args.hours * 3600)
    win_hist = []
    while time.time() < deadline:
        t0 = time.time()
        key, k_roll, k_perm = jax.random.split(key, 3)
        envs, traj, last_v, stats = rollout(params, league, envs, k_roll, args.steps)
        adv, ret = gae(traj, last_v)
        flat = flatten(traj, adv, ret)
        n = flat["act"].shape[0]
        mb = n // args.minibatches
        for ep in range(args.epochs):
            perm = jax.random.permutation(jax.random.fold_in(k_perm, ep), n)
            for i in range(args.minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                batch = jax.tree.map(lambda x: x[idx], flat)
                params, opt_state, aux = update(params, opt_state, batch)
        jax.block_until_ready(params)
        dt = time.time() - t0
        update_i += 1
        env_steps += args.batch * args.steps
        if update_i % args.league_every == 0:
            league = jax.tree.map(lambda lg, p: lg.at[league_next % args.league].set(p), league, params)
            league_next += 1
        s = {k: int(v) for k, v in jax.device_get(stats).items()}
        if s["rand_games"]:
            win_hist.append((s["rand_wins"], s["rand_games"]))
        win_hist = win_hist[-20:]
        rw, rg = sum(w for w, _ in win_hist), sum(g for _, g in win_hist)
        row = {
            "update": update_i,
            "env_steps": env_steps,
            "hours": round((time.time() - t_start) / 3600, 3),
            "steps_per_s": round(args.batch * args.steps / dt),
            "episodes": s["episodes"],
            "winrate_vs_random": round(rw / rg, 3) if rg else None,
            "random_games_window": rg,
            "league_wr": round(s["league_wins"] / s["league_games"], 3) if s["league_games"] else None,
            **{k: round(float(v), 4) for k, v in jax.device_get(aux).items()},
        }
        if update_i == 1:
            row["setup_s"] = round(t0 - t_setup)
            row["first_update_s"] = round(dt, 1)
            try:
                row["gpu_peak_mb"] = jax.devices()[0].memory_stats()["peak_bytes_in_use"] // 2**20
            except Exception:
                pass
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if update_i % 10 == 1:
            print(json.dumps(row), flush=True)
        if time.time() - last_ckpt > args.ckpt_minutes * 60 or time.time() >= deadline:
            ck = {"params": params, "opt_state": opt_state, "league": league, "update": update_i,
                  "env_steps": env_steps, "league_next": league_next, "args": vars(args)}  # fmt: skip
            save(latest, ck)
            save(ckpt_dir / f"policy_{update_i:06d}.pkl", {"params": params, "args": vars(args), "update": update_i})
            last_ckpt = time.time()
            print(f"checkpoint at update {update_i}", flush=True)


if __name__ == "__main__":
    main()
