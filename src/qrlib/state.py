"""Qualitative states.

A :class:`QState` assigns a :class:`~qrlib.quantity.QVal` to every variable
of a model at one qualitative time — either a distinguished time *point* or
the open *interval* between two distinguished points (QSIM time alternates
between the two). States are immutable values: hashable, comparable, and
(given a compiled model's frozen landmark→rank mapping) encodable as a flat
integer vector — see docs/gpu-tensorization.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .quantity import QVal

__all__ = ["TimeTag", "QState"]


class TimeTag(Enum):
    POINT = "point"
    INTERVAL = "interval"


@dataclass(frozen=True)
class QState:
    """An immutable mapping ``variable name -> QVal`` plus a time tag."""

    values: tuple[tuple[str, QVal], ...]
    time: TimeTag

    @classmethod
    def from_dict(cls, values: dict[str, QVal], time: TimeTag) -> "QState":
        return cls(tuple(sorted(values.items())), time)

    def __getitem__(self, var: str) -> QVal:
        for name, val in self.values:
            if name == var:
                return val
        raise KeyError(var)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.values)

    def as_dict(self) -> dict[str, QVal]:
        return dict(self.values)

    def encode(self, order: tuple[str, ...]) -> tuple[int, ...]:
        """Flat interleaved ``(mag_0, dir_0, mag_1, dir_1, ...)`` encoding.

        ``order`` is the compiled model's frozen variable order. This tuple
        is the row layout of the ``(B, 2*V)`` frontier tensors in
        ``qrlib.tensor``.
        """
        vals = self.as_dict()
        out: list[int] = []
        for name in order:
            qv = vals[name]
            out.extend((qv.mag, int(qv.dir)))
        return tuple(out)
