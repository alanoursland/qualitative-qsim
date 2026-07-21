"""Declarative QDE model descriptions.

A :class:`Model` is pure data — variables (each owning a
:class:`~qrlib.quantity.QuantitySpace`) plus constraints — consumed by every
engine. ``Model.compile()`` (phase 1) will freeze name→index and
landmark→rank mappings and precompute constraint tables; the editable
``Model`` itself stays serialization-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import Constraint
from .quantity import Landmark, Qdir, QuantitySpace, QVal
from .state import QState, TimeTag

__all__ = ["Variable", "Model"]


@dataclass(frozen=True)
class Variable:
    name: str
    space: QuantitySpace


@dataclass
class Model:
    """A qualitative differential equation: variables + constraints."""

    name: str = "model"
    variables: dict[str, Variable] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)

    def variable(
        self,
        name: str,
        *,
        landmarks: tuple[str | Landmark, ...] = ("0",),
        lower_unbounded: bool = False,
        upper_unbounded: bool = False,
        unbounded: bool = False,
    ) -> Variable:
        """Declare a variable and return its handle."""
        if name in self.variables:
            raise ValueError(f"variable {name!r} already declared")
        space = QuantitySpace(
            landmarks,
            lower_unbounded=lower_unbounded or unbounded,
            upper_unbounded=upper_unbounded or unbounded,
        )
        var = Variable(name, space)
        self.variables[name] = var
        return var

    def constrain(self, constraint: Constraint) -> Constraint:
        for ref in constraint.variables:
            if ref not in self.variables:
                raise ValueError(
                    f"constraint {constraint!r} references undeclared variable {ref!r}"
                )
        for cv in constraint.corresponding_values:
            for ref, landmark in zip(constraint.variables, cv):
                if landmark not in self.variables[ref].space.effective_landmarks:
                    raise ValueError(
                        f"corresponding value {landmark!r} is not a landmark of {ref!r}"
                    )
        self.constraints.append(constraint)
        return constraint

    def state(self, time: TimeTag = TimeTag.POINT, **values: tuple[str, Qdir] | QVal) -> QState:
        """Build a state from keyword args, e.g. ``amount=("0", Qdir.INC)``.

        Magnitude may be a landmark name or an ``"(a, b)"``-free shorthand:
        pass a landmark name for an at-landmark magnitude; interval
        magnitudes should be built via ``QVal`` directly for now.
        """
        out: dict[str, QVal] = {}
        for name, v in values.items():
            if name not in self.variables:
                raise ValueError(f"unknown variable {name!r}")
            if isinstance(v, QVal):
                out[name] = v
            else:
                landmark, qdir = v
                rank = self.variables[name].space.rank_of(landmark)
                out[name] = QVal(rank, qdir)
        missing = set(self.variables) - set(out)
        if missing:
            raise ValueError(f"state is missing variables: {sorted(missing)}")
        return QState.from_dict(out, time)

    def compile(self):
        """Freeze mappings and precompute constraint tables. Phase 1."""
        raise NotImplementedError(
            "Model.compile() lands with the reference QSIM engine "
            "(docs/roadmap.md, phase 1)"
        )
