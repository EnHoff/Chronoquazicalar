"""
engine/oujax/families/ability_checks.py
Test helpers for the ability families' L2 scenarios: on top of the bridge's diff (HP, status, boosts, items,
...), compare each active Pokemon's working ability, whether it is suppressed (Neutralizing Gas), its current
types (Libero / Protean, Tera) and its disabled moves with Showdown after every step.
"""

from engine.oujax.engine import types_now, vol
from engine.oujax.tables import TYPE_NONE, load_tables
from engine.oujax.test_l2 import MirrorScenarios

T = load_tables()
ABILITY_NAMES = {v: k for k, v in T.ability_ids.items()}
TYPE_NAMES = {v: k for k, v in T.type_ids.items()}


def ability_diff(m):
    """Differences in the active Pokemon's ability, suppression and types between OU-JAX and Showdown."""
    st, sd = m.ours, m.sd
    out = []
    if sd["ended"]:
        return out
    for s, side in enumerate(sd["sides"]):
        idx = side["active"][0]
        slot = int(st.active[s])
        if idx is None or slot < 0:
            continue
        p = side["pokemon"][idx]
        if p["fainted"] or p["name"] != m.names[s][slot]:
            continue
        tag = f"p{s + 1}.{p['name']}"
        cur = vol(st, s, "ab_cur") - 1
        ours_ab = ABILITY_NAMES[int(cur)] if int(cur) >= 0 else ABILITY_NAMES[int(st.ability[s, slot])]
        if ours_ab != p["ability"]:
            out.append(f"{tag}.ability: ours={ours_ab!r} showdown={p['ability']!r}")
        off = bool(vol(st, s, "ab_off") > 0)
        if "abilitySuppressed" in p and off != p["abilitySuppressed"]:
            out.append(f"{tag}.ability_suppressed: ours={off} showdown={p['abilitySuppressed']}")
        types = [TYPE_NAMES[int(t)] for t in types_now(st, s, slot) if int(t) != TYPE_NONE]
        theirs = [t.lower() for t in p.get("types", [])]
        if "types" in p and types != theirs:
            out.append(f"{tag}.types: ours={types} showdown={theirs}")
    return out


class AbilityScenarios(MirrorScenarios):
    """MirrorScenarios that also compare abilities and types after every step."""

    def play(self, p1, p2, script, cfg=None, forced_opts=None, check=None):
        def both(step, m):
            self.assertEqual(ability_diff(m), [], f"step {step}")
            if check is not None:
                check(step, m)

        kwargs = {} if cfg is None else {"cfg": cfg}
        return super().play(p1, p2, script, forced_opts=forced_opts, check=both, **kwargs)
