"""Visualization: data-first exports + a dependency-free SVG renderer.

``timeline_bands``/``tree_layout`` produce plain data for host plotting
stacks (Surface 4 of docs/host-integration.md); ``timeline_svg``/
``tree_svg`` render them without any plotting dependency. The behavior
graph's dot export lives on :class:`qrlib.BehaviorGraph` itself.
"""

from .data import timeline_bands, tree_layout
from .svg import timeline_svg, tree_svg

__all__ = ["timeline_bands", "tree_layout", "timeline_svg", "tree_svg"]
