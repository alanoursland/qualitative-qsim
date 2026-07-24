"""Regenerate every figure used in the tutorial.

Run from anywhere:  ``python docs/tutorial/make_figures.py``

Each figure is produced from a real ``qrlib`` run (behavior graphs and
timelines via ``qrlib.viz``) or, for the two concept diagrams, from a small
inline SVG builder — so nothing here is hand-drawn or out of sync with the
library. Output lands in ``docs/tutorial/figures/``.
"""

from __future__ import annotations

import os
from html import escape

import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag
from qrlib.analysis import causal
from qrlib.viz import timeline_svg, tree_svg

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


def write(name: str, svg: str) -> None:
    with open(os.path.join(FIG, name), "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", name, f"({len(svg)} bytes)")


# --- the running examples --------------------------------------------------


def bathtub():
    """Water flows in at a constant rate and drains faster as it deepens."""
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=(Landmark("0", value=0.0), Landmark("FULL", value=10.0)))
    m.variable("level", landmarks=(Landmark("0", value=0.0), Landmark("TOP", value=5.0)))
    m.variable("outflow", landmarks=(Landmark("0", value=0.0), Landmark("OMAX", value=3.0)))
    m.variable("inflow", landmarks=(Landmark("0", value=0.0), Landmark("IF", value=2.0)))
    m.variable("netflow", landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
    m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    m.constrain(qr.Deriv("amount", "netflow"))
    m.constrain(qr.Constant("inflow"))
    initial = m.state(
        time=TimeTag.POINT,
        amount=("0", Qdir.INC),
        level=("0", Qdir.INC),
        outflow=("0", Qdir.INC),
        inflow=("IF", Qdir.STD),
        netflow=(("0", "+inf"), Qdir.DEC),
    )
    return m, initial


def spring():
    """A frictionless spring: x'' = -x."""
    m = qr.Model("spring")
    for name in ("x", "v", "a"):
        m.variable(name, landmarks=(Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Deriv("x", "v"))
    m.constrain(qr.Deriv("v", "a"))
    m.constrain(qr.Minus("x", "a"))
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.STD),
        a=("0", Qdir.DEC),
    )
    return m, initial


def overflow_bathtub():
    """A bathtub that, on reaching the brim, crosses into a steady-overflow
    region (excess spills; the level holds at the top)."""
    m = qr.Model("overflow bathtub")
    m.variable("amount", landmarks=("0", "FULL"))
    m.variable("level", landmarks=("0", "TOP"))
    m.variable("outflow", landmarks=("0", "OMAX"))
    m.variable("inflow", landmarks=("0", "IF"))
    m.variable("netflow", landmarks=("0",), unbounded=True)
    c_al = m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"), ("FULL", "TOP"))))
    c_lo = m.constrain(qr.MPlus("level", "outflow", cvals=(("0", "0"), ("TOP", "OMAX"))))
    c_add = m.constrain(qr.Add("netflow", "outflow", "inflow"))
    c_der = m.constrain(qr.Deriv("amount", "netflow"))
    c_inf = m.constrain(qr.Constant("inflow"))
    c_amt = m.constrain(qr.Constant("amount"))  # held constant while overflowing
    m.region("filling", constraints=(c_al, c_lo, c_add, c_der, c_inf))
    m.region("overflowing", constraints=(c_al, c_lo, c_add, c_inf, c_amt))
    m.transition(
        "filling", "overflowing",
        when=(("amount", "==", "FULL"), ("netflow", ">", "0")),
    )
    initial = m.state(
        time=TimeTag.POINT,
        amount=("0", Qdir.INC),
        level=("0", Qdir.INC),
        outflow=("0", Qdir.INC),
        inflow=("IF", Qdir.STD),
        netflow=(("0", "+inf"), Qdir.DEC),
    )
    return m, initial


# --- concept diagrams (small inline SVG) -----------------------------------


def quantity_space_svg() -> str:
    """A number line: landmarks, the intervals between them, and rank ids."""
    marks = [("−∞", None), ("0", 0), ("(0, FULL)", 1), ("FULL", 2), ("+∞", None)]
    W, H, x0, gap = 560, 140, 60, 110
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        'font-family="monospace" font-size="13">',
        '<text x="16" y="24">quantity space of "amount"</text>',
        f'<line x1="40" y1="70" x2="{W-20}" y2="70" stroke="#333" stroke-width="2"/>',
    ]
    xs = [x0 + i * gap for i in range(len(marks))]
    for i, (label, rank) in enumerate(marks):
        x = xs[i]
        landmark = rank is not None and rank % 2 == 0
        if landmark or label in ("−∞", "+∞"):
            p.append(f'<line x1="{x}" y1="60" x2="{x}" y2="80" stroke="#333" stroke-width="2"/>')
            p.append(f'<text x="{x}" y="52" text-anchor="middle">{escape(label)}</text>')
        else:  # an interval sits between two landmarks
            p.append(f'<text x="{x}" y="98" text-anchor="middle" fill="#b26a00">{escape(label)}</text>')
        if rank is not None:
            p.append(f'<text x="{x}" y="120" text-anchor="middle" fill="#777">rank {rank}</text>')
    p.append('<text x="16" y="136" fill="#777">even rank = at a landmark · odd rank = strictly between two</text>')
    p.append("</svg>")
    return "".join(p)


def point_interval_svg() -> str:
    """QSIM time alternates: instant (point) then a stretch (interval)."""
    seq = [("•", "point", "an instant"), ("~", "interval", "a stretch of time"),
           ("•", "point", ""), ("~", "interval", ""), ("•", "point", "")]
    W, H, x0, gap = 560, 120, 70, 100
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        'font-family="monospace" font-size="13">',
        '<text x="16" y="24">qualitative time alternates point / interval</text>',
        f'<line x1="40" y1="66" x2="{W-20}" y2="66" stroke="#ccc"/>',
    ]
    for i, (mark, kind, note) in enumerate(seq):
        x = x0 + i * gap
        color = "#2a6" if kind == "point" else "#26a"
        p.append(f'<text x="{x}" y="72" text-anchor="middle" font-size="20" fill="{color}">{mark}</text>')
        p.append(f'<text x="{x}" y="92" text-anchor="middle" fill="#777">{kind}</text>')
        if note:
            p.append(f'<text x="{x}" y="110" text-anchor="middle" fill="#999" font-size="11">{escape(note)}</text>')
    p.append("</svg>")
    return "".join(p)


def causal_svg(model) -> str:
    """The bathtub's instantaneous causal chain, laid out by depth level."""
    order = causal.causal_order(model)
    level = {}
    for v in order.exogenous:
        level[v] = 0
    for v, lvl in order.levels.items():
        level[v] = lvl
    by_level: dict[int, list[str]] = {}
    for v, lvl in level.items():
        by_level.setdefault(lvl, []).append(v)
    for v in order.state_variables:
        by_level.setdefault(max(by_level) + 1 if by_level else 0, []).append(v)
    col_w, row_h, r = 150, 70, 26
    maxrow = max(len(vs) for vs in by_level.values())
    W = 40 + (max(by_level) + 1) * col_w
    H = 60 + maxrow * row_h
    pos = {}
    for lvl, vs in by_level.items():
        for i, v in enumerate(sorted(set(vs))):
            pos[v] = (40 + lvl * col_w + 40, 50 + i * row_h)
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        'font-family="monospace" font-size="12">',
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="3" '
        'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#666"/></marker></defs>',
        '<text x="12" y="20">what determines what (instantaneous causal order)</text>',
    ]
    for link in order.links:
        if link.variable not in pos:
            continue
        tx, ty = pos[link.variable]
        for src in link.inputs:
            if src not in pos:
                continue
            sx, sy = pos[src]
            style = ' stroke-dasharray="4"' if link.integral else ""
            p.append(
                f'<line x1="{sx+38}" y1="{sy}" x2="{tx-38}" y2="{ty}" '
                f'stroke="#666"{style} marker-end="url(#a)"/>'
            )
    for v, (x, y) in pos.items():
        fill = "#eef" if v in order.exogenous else ("#efe" if v in order.state_variables else "#fff")
        p.append(f'<ellipse cx="{x}" cy="{y}" rx="38" ry="18" fill="{fill}" stroke="#333"/>')
        p.append(f'<text x="{x}" y="{y+4}" text-anchor="middle">{escape(v)}</text>')
    p.append('<text x="12" y="'+str(H-8)+'" fill="#777">dashed = integration (feedback over time); blue = input, green = state</text>')
    p.append("</svg>")
    return "".join(p)


# --- main ------------------------------------------------------------------


def main() -> None:
    os.makedirs(FIG, exist_ok=True)

    # concept diagrams
    write("quantity-space.svg", quantity_space_svg())
    write("point-interval.svg", point_interval_svg())

    # bathtub
    m, init = bathtub()
    result = qr.qsim(m, init)
    write("bathtub-tree.svg", tree_svg(result.graph))
    write("bathtub-causal.svg", causal_svg(m))
    overflow = next(b for b in result.behaviors()
                    if b.terminal is qr.TerminalClass.DOMAIN_EXIT)
    write("bathtub-timeline.svg", timeline_svg(result.graph, overflow))

    # semi-quantitative refinement of the below-FULL equilibrium (feasible:
    # the drain balances the inflow before the brim)
    def final_amount(b):
        node = result.graph.nodes[b.node_ids[-1]]
        return node.model.spaces[node.model.index("amount")].describe(
            b.states[-1]["amount"].mag
        )

    below = next(b for b in result.behaviors()
                 if b.terminal is qr.TerminalClass.QUIESCENT
                 and final_amount(b) != "FULL")
    ref = qr.semiquant.refine(result.graph, below)
    write("semiquant-timeline.svg", timeline_svg(result.graph, below, refinement=ref))

    # spring: the true cycle (discovery off)
    ms, si = spring()
    cyc = qr.qsim(ms, si, config=qr.SimConfig(discover_landmarks=False))
    write("spring-cycle.svg", tree_svg(cyc.graph))
    (loop,) = cyc.behaviors()
    write("spring-timeline.svg", timeline_svg(cyc.graph, loop))

    # spring: spurious branching (discovery on, capped) vs tamed (energy)
    wild = qr.qsim(ms, si, max_states=22)
    write("spring-intractable.svg", tree_svg(wild.graph))
    tamed = qr.qsim(ms, si, config=qr.SimConfig(successor_filters=(qr.EnergyFilter(),)))
    write("spring-tamed.svg", tree_svg(tamed.graph))

    # operating regions: filling -> steady overflow
    mv, iv = overflow_bathtub()
    vr = qr.qsim(mv, iv)
    write("regions-tree.svg", tree_svg(vr.graph))

    print("\nall figures written to", FIG)


if __name__ == "__main__":
    main()
