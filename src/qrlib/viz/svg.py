"""Dependency-free SVG rendering of the viz data.

Small, self-contained renderers over :mod:`qrlib.viz.data` — enough to see
a behavior at a glance in a notebook or report without pulling in a
plotting stack. Hosts wanting styled output should consume the data
exports directly.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from ..behavior import Behavior, BehaviorGraph
from .data import timeline_bands, tree_layout

__all__ = ["timeline_svg", "tree_svg"]

_DIR_COLOR = {
    "inc": "#2f7d4f",
    "dec": "#b23a3a",
    "std": "#555555",
    "ign": "#999999",
}


def timeline_svg(
    graph: BehaviorGraph,
    behavior: Behavior,
    *,
    refinement=None,
    col_width: int = 90,
    row_height: int = 56,
) -> str:
    """One behavior as a per-variable qualitative timeline (SVG string)."""
    data = timeline_bands(graph, behavior, refinement=refinement)
    n_states = len(data["regions"])
    n_vars = len(data["variables"])
    left, top = 110, 30
    width = left + n_states * col_width + 20
    height = top + n_vars * row_height + 20

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="monospace" font-size="11">',
        f'<text x="8" y="16">{escape(data["model"])} — '
        f'{escape(data["terminal"])}</text>',
    ]
    for si in range(n_states):
        tag = data["variables"][0]["segments"][si]["time_tag"]
        mark = "•" if tag == "point" else "~"
        x = left + si * col_width + col_width // 2
        parts.append(f'<text x="{x}" y="{top - 6}" text-anchor="middle">{mark}</text>')

    for vrow, var in enumerate(data["variables"]):
        y0 = top + vrow * row_height
        parts.append(
            f'<text x="8" y="{y0 + row_height // 2}">{escape(var["name"])}</text>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y0 + row_height - 8}" '
            f'x2="{width - 20}" y2="{y0 + row_height - 8}" stroke="#ddd"/>'
        )
        for seg in var["segments"]:
            si = seg["state"]
            frac = (
                seg["rank"] / (seg["num_ranks"] - 1)
                if seg["num_ranks"] > 1
                else 0.5
            )
            y = y0 + (row_height - 12) * (1 - frac) + 4
            x0 = left + si * col_width
            color = _DIR_COLOR[seg["dir"]]
            title = f'<title>{escape(var["name"])}={escape(seg["label"])} ({seg["dir"]})</title>'
            if seg["time_tag"] == "point":
                cx = x0 + col_width // 2
                parts.append(
                    f'<circle cx="{cx}" cy="{y:.1f}" r="4" fill="{color}">{title}</circle>'
                )
            else:
                parts.append(
                    f'<line x1="{x0 + 8}" y1="{y:.1f}" x2="{x0 + col_width - 8}" '
                    f'y2="{y:.1f}" stroke="{color}" stroke-width="4">{title}</line>'
                )
    parts.append("</svg>")
    return "".join(parts)


def tree_svg(graph: BehaviorGraph, *, node_gap: int = 150, layer_gap: int = 64) -> str:
    """A behavior graph as a layered SVG drawing (cycle edges dashed)."""
    layout = tree_layout(graph)
    pos = {
        n["id"]: (40 + n["x"] * node_gap, 30 + n["y"] * layer_gap)
        for n in layout["nodes"]
    }
    width = int(max(x for x, _ in pos.values()) + node_gap)
    height = int(max(y for _, y in pos.values()) + layer_gap)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="monospace" font-size="11">'
    ]
    for a, b in layout["edges"]:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1 + 10:.1f}" x2="{x2:.1f}" '
            f'y2="{y2 - 10:.1f}" stroke="#888"/>'
        )
    for a, b in layout["cycle_edges"]:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#b23a3a" stroke-dasharray="5,4"/>'
        )
    for n in layout["nodes"]:
        x, y = pos[n["id"]]
        mark = "•" if n["time_tag"] == "point" else "~"
        text = f'n{n["id"]} {mark}'
        if n["terminal"]:
            text += f' [{n["terminal"]}]'
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#eee" stroke="#444">'
            f'<title>{escape(n["label"])}</title></circle>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle">'
            f"{escape(text)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)
