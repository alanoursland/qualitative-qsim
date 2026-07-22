"""Analysis over behavior graphs.

``queries`` answers reachability/quiescence/cycle questions with plain
data; ``explain`` narrates behaviors as structured step records + prose
(docs/host-integration.md, Surface 4 and cross-cutting conventions).
"""

from . import explain, queries

__all__ = ["explain", "queries"]
