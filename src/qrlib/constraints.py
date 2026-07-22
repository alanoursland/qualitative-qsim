"""Constraint types for qualitative differential equations (QDEs).

Constraints are inert data: they name the variables they relate and carry
optional *corresponding values* (tuples of landmark names, one per variable,
known to co-occur — e.g. ``f(0) = 0`` for an ``MPlus``). Consistency
predicates and their compiled table forms live with the engines (phase 1;
see docs/qsim.md §3), keeping the model description engine-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Constraint",
    "MPlus",
    "MMinus",
    "Add",
    "Mult",
    "Minus",
    "Deriv",
    "Constant",
    "At",
]

CorrespondingValues = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Constraint:
    """Base class. ``variables`` lists referenced variable names in order."""

    @property
    def variables(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def corresponding_values(self) -> CorrespondingValues:
        return getattr(self, "cvals", ())

    def _check_cvals(self) -> None:
        for cv in self.corresponding_values:
            if len(cv) != len(self.variables):
                raise ValueError(
                    f"corresponding value {cv!r} arity != {len(self.variables)}"
                )


@dataclass(frozen=True)
class MPlus(Constraint):
    """``y = f(x)`` for some monotonically increasing f."""

    x: str
    y: str
    cvals: CorrespondingValues = field(default=())

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x, self.y)

    def __post_init__(self) -> None:
        self._check_cvals()


@dataclass(frozen=True)
class MMinus(Constraint):
    """``y = f(x)`` for some monotonically decreasing f."""

    x: str
    y: str
    cvals: CorrespondingValues = field(default=())

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x, self.y)

    def __post_init__(self) -> None:
        self._check_cvals()


@dataclass(frozen=True)
class Add(Constraint):
    """``x + y = z``."""

    x: str
    y: str
    z: str
    cvals: CorrespondingValues = field(default=())

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x, self.y, self.z)

    def __post_init__(self) -> None:
        self._check_cvals()


@dataclass(frozen=True)
class Mult(Constraint):
    """``x * y = z``."""

    x: str
    y: str
    z: str
    cvals: CorrespondingValues = field(default=())

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x, self.y, self.z)

    def __post_init__(self) -> None:
        self._check_cvals()


@dataclass(frozen=True)
class Minus(Constraint):
    """``y = -x``."""

    x: str
    y: str
    cvals: CorrespondingValues = field(default=())

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x, self.y)

    def __post_init__(self) -> None:
        self._check_cvals()


@dataclass(frozen=True)
class Deriv(Constraint):
    """``d(x)/dt = y``."""

    x: str
    y: str

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Constant(Constraint):
    """``x`` is constant over time (qdir is STD everywhere)."""

    x: str

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x,)


@dataclass(frozen=True)
class At(Constraint):
    """``x`` is pinned at ``landmark`` — an operating-point assertion.

    Constrains magnitude only; pair with :class:`Constant` to also pin the
    direction (otherwise successors that move off the landmark are filtered
    into dead ends). The landmark rides the corresponding-values machinery,
    so per-branch rank shifts from landmark discovery are handled
    automatically. Introduced for fault-mode modeling (a nominal operating
    point is what makes 'stuck at zero' distinguishable from 'normally
    zero'), and generally useful for assumption pinning."""

    x: str
    landmark: str

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.x,)

    @property
    def corresponding_values(self) -> tuple[tuple[str, ...], ...]:
        return ((self.landmark,),)
