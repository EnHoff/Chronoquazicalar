"""
engine/oujax/families/held_items.py
Held items with in-battle triggers: Rocky Helmet (per contact hit).
"""

from engine.oujax import hooks
from engine.oujax.effects import ITEM
from engine.oujax.engine import MOVES, _i, lose_hp
from engine.oujax.tables import FLAG_BIT, MC


def makes_contact(st, s, slot, mid):
    """checkMoveMakesContact: the contact flag, except punches thrown with a Punching Glove."""
    flags = MOVES[mid, MC["flags"]]
    punch = (flags & FLAG_BIT["punch"]) > 0
    return ((flags & FLAG_BIT["contact"]) > 0) & ~(punch & (st.item[s, slot] == ITEM.punchingglove))


@hooks.hook("damaging_hit")
def _rocky_helmet(st, ctx, hit, dmg):
    s, slot = ctx.s, ctx.slot
    helmet = ctx.landed & (st.item[ctx.o, ctx.tslot] == ITEM.rockyhelmet) & makes_contact(st, s, slot, ctx.mid)
    return lose_hp(st, s, slot, _i(st.stats[s, slot, 0]) // 6, helmet)
