"""Analysis over behavior graphs.

``queries`` answers reachability/quiescence/cycle questions with plain data
(docs/host-integration.md, Surface 4). ``explain`` (structured + prose
behavior narration) is planned for phase 7.
"""

from . import queries

__all__ = ["queries"]
