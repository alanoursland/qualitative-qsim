"""Tensor codecs and compiled constraint tables.

The bridge between symbolic states and integer tensors:

- **qcode**: a qualitative value packs into one integer,
  ``mag_rank * 4 + dir`` (direction base 4 so states carrying ``IGN`` also
  encode; candidate values are always concrete).
- **Frontier encoding**: the canonical ``(B, 2V)`` interleaved
  ``[mag_0, dir_0, mag_1, dir_1, ...]`` layout (matching
  ``QState.encode``), plus ``(B, V)`` qcode form for table lookups.
- **TableSet**: per-frame dense boolean lookup tables, one per compiled
  constraint, indexed by the qcodes of its variables. Tables are built by
  exhaustively evaluating the *reference predicate*
  (:func:`qrlib.engines.filters.check`) over the frame's concrete value
  combinations — agreement with the reference engine holds by
  construction, and the equivalence tests re-verify it. Value spaces are
  tiny (tens of ranks), so tables are at most a few hundred thousand
  booleans and build in milliseconds.

Frames are content-hashable, so table sets cache per frame
(:func:`tables_for`).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import torch

from ..engines import filters
from ..model import CompiledModel
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag

__all__ = [
    "DIRS",
    "qcode",
    "qval_of_code",
    "encode_frontier",
    "decode_frontier",
    "TableSet",
    "tables_for",
]

DIRS = 4  # DEC, STD, INC, IGN
_CONCRETE = (Qdir.DEC, Qdir.STD, Qdir.INC)


def qcode(qv: QVal) -> int:
    return qv.mag * DIRS + int(qv.dir)


def qval_of_code(code: int) -> QVal:
    return QVal(code // DIRS, Qdir(code % DIRS))


def encode_frontier(states, var_order) -> torch.Tensor:
    """Stack states into the canonical ``(B, 2V)`` interleaved int tensor."""
    return torch.tensor(
        [state.encode(tuple(var_order)) for state in states], dtype=torch.long
    )


def decode_frontier(frontier: torch.Tensor, var_order, time: TimeTag) -> list[QState]:
    out = []
    for row in frontier.tolist():
        values = {
            name: QVal(row[2 * i], Qdir(row[2 * i + 1]))
            for i, name in enumerate(var_order)
        }
        out.append(QState.from_dict(values, time))
    return out


class TableSet:
    """Dense boolean consistency tables for one frame's constraints."""

    def __init__(self, frame: CompiledModel):
        self.frame = frame
        self.tables: list[torch.Tensor] = []
        for cc in frame.constraints:
            dims = [frame.spaces[vi].num_ranks * DIRS for vi in cc.vars]
            table = torch.zeros(dims, dtype=torch.bool)
            axes = [
                [
                    QVal(m, d)
                    for m in range(frame.spaces[vi].num_ranks)
                    for d in _CONCRETE
                ]
                for vi in cc.vars
            ]
            for combo in product(*axes):
                if filters.check(cc, combo):
                    table[tuple(qcode(qv) for qv in combo)] = True
            self.tables.append(table)

    def mask(self, ci: int, cand: torch.Tensor) -> torch.Tensor:
        """Consistency of candidate tuples: ``cand`` is ``(N, arity)`` long
        qcodes aligned with constraint ``ci``'s variables; returns ``(N,)``
        bool."""
        table = self.tables[ci]
        return table[tuple(cand[:, k] for k in range(cand.shape[1]))]


@lru_cache(maxsize=64)
def tables_for(frame: CompiledModel) -> TableSet:
    return TableSet(frame)
