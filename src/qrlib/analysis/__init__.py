"""Analysis over models and behavior graphs.

- ``queries`` answers reachability/quiescence/cycle questions over a
  behavior graph with plain data;
- ``explain`` narrates behaviors as structured step records + prose;
- ``causal`` derives a model's causal ordering (which variable determines
  which) from its constraints alone;
- ``compare`` derives comparative statics (how an equilibrium shifts when a
  parameter is perturbed) by sign propagation over the constraints.
- ``monotonicity`` checks whether M+/M-/Minus relationships admit one
  consistent orthant ordering and returns a contradictory signed cycle when
  they do not.

"""

from . import causal, compare, explain, monotonicity, queries

__all__ = ["causal", "compare", "explain", "monotonicity", "queries"]
