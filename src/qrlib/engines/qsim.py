"""Reference QSIM engine (phase 1 — not yet implemented).

Planned entry point::

    qsim(model, initial, *, max_states=..., max_depth=..., filters=...)
        -> BehaviorGraph

Algorithm plan and acceptance tests: docs/qsim.md.
"""

from __future__ import annotations

__all__ = ["qsim"]


def qsim(model, initial, **kwargs):
    raise NotImplementedError(
        "The reference QSIM engine is phase 1 of docs/roadmap.md; "
        "see docs/qsim.md for the implementation plan."
    )
