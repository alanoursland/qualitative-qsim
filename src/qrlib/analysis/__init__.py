"""Analysis over models and behavior graphs.

- ``queries`` answers reachability/quiescence/cycle questions over a
  behavior graph with plain data;
- ``explain`` narrates behaviors as structured step records + prose;
- ``causal`` derives a model's causal ordering (which variable determines
  which) from its constraints alone;
- ``compare`` derives comparative statics (how an equilibrium shifts when a
  parameter is perturbed) by sign propagation over the constraints.

(docs/host-integration.md, Surface 4 and cross-cutting conventions.)
"""

from . import causal, compare, explain, queries

__all__ = ["causal", "compare", "explain", "queries"]
