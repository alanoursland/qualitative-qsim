"""Declarative QDE model descriptions.

A :class:`Model` is pure data — variables (each owning a
:class:`~qrlib.quantity.QuantitySpace`) plus constraints — consumed by every
engine. :meth:`Model.compile` freezes the name -> index and landmark -> rank
mappings and resolves constraints into :class:`CompiledConstraint` records
(variable indices, corresponding-value ranks, zero/infinity ranks). The
compiled artifacts are inert data; the consistency predicates that interpret
them live with the engines (``qrlib.engines.filters``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import Add, Constant, Constraint, Deriv, Minus, MMinus, MPlus, Mult
from .quantity import Landmark, Qdir, QuantitySpace, QVal
from .state import QState, TimeTag

__all__ = ["Variable", "Model", "CompiledConstraint", "CompiledModel"]

_KIND: dict[type, str] = {
    MPlus: "mplus",
    MMinus: "mminus",
    Add: "add",
    Mult: "mult",
    Minus: "minus",
    Deriv: "deriv",
    Constant: "constant",
}

ZERO = "0"


@dataclass(frozen=True)
class Variable:
    name: str
    space: QuantitySpace


@dataclass(frozen=True)
class CompiledConstraint:
    """A constraint resolved against frozen variable/landmark mappings.

    ``vars`` are indices into the compiled variable order; ``cvals`` are
    corresponding-value tuples as magnitude ranks (including any implicit
    zero tuples added at compile time); ``zeros`` is the rank of the ``0``
    landmark per constrained variable (or None); ``infs`` is the pair
    (rank of -inf, rank of +inf) per constrained variable (None = bounded).
    """

    kind: str
    vars: tuple[int, ...]
    cvals: tuple[tuple[int, ...], ...]
    zeros: tuple[int | None, ...]
    infs: tuple[tuple[int | None, int | None], ...]
    source: Constraint


@dataclass(frozen=True)
class CompiledModel:
    """Frozen, engine-consumable form of a :class:`Model`."""

    name: str
    var_order: tuple[str, ...]
    spaces: tuple[QuantitySpace, ...]
    constraints: tuple[CompiledConstraint, ...]

    def index(self, var: str) -> int:
        return self.var_order.index(var)

    def inf_ranks(self, vi: int) -> tuple[int | None, int | None]:
        space = self.spaces[vi]
        lo = 0 if space.lower_unbounded else None
        hi = space.num_ranks - 1 if space.upper_unbounded else None
        return (lo, hi)


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

    def state(
        self,
        time: TimeTag = TimeTag.POINT,
        **values: tuple[str | tuple[str, str], Qdir] | QVal,
    ) -> QState:
        """Build a state from keyword args.

        Each value is either a ``QVal``, or a ``(magnitude, qdir)`` pair
        where magnitude is a landmark name (``"0"``) for an at-landmark
        value or a pair of adjacent landmark names (``("0", "FULL")``) for
        an open-interval value.
        """
        out: dict[str, QVal] = {}
        for name, v in values.items():
            if name not in self.variables:
                raise ValueError(f"unknown variable {name!r}")
            if isinstance(v, QVal):
                out[name] = v
                continue
            mag_spec, qdir = v
            space = self.variables[name].space
            if isinstance(mag_spec, tuple):
                rank = space.rank_between(*mag_spec)
            else:
                rank = space.rank_of(mag_spec)
            out[name] = QVal(rank, qdir)
        missing = set(self.variables) - set(out)
        if missing:
            raise ValueError(f"state is missing variables: {sorted(missing)}")
        return QState.from_dict(out, time)

    def compile(self) -> CompiledModel:
        """Freeze mappings and resolve constraints. See module docstring."""
        var_order = tuple(self.variables)
        spaces = tuple(self.variables[v].space for v in var_order)

        def zero_rank(space: QuantitySpace) -> int | None:
            eff = space.effective_landmarks
            return 2 * eff.index(ZERO) if ZERO in eff else None

        def inf_pair(space: QuantitySpace) -> tuple[int | None, int | None]:
            lo = 0 if space.lower_unbounded else None
            hi = space.num_ranks - 1 if space.upper_unbounded else None
            return (lo, hi)

        compiled: list[CompiledConstraint] = []
        for c in self.constraints:
            kind = _KIND[type(c)]
            idx = tuple(var_order.index(v) for v in c.variables)
            csp = [spaces[i] for i in idx]
            zeros = tuple(zero_rank(s) for s in csp)
            cvals = [
                tuple(csp[k].rank_of(nm) for k, nm in enumerate(cv))
                for cv in c.corresponding_values
            ]
            if kind == "deriv" and zeros[1] is None:
                raise ValueError(
                    f"{c!r}: the derivative variable {c.variables[1]!r} needs a "
                    f"'0' landmark (its sign drives the other's direction)"
                )
            if kind == "mult" and any(z is None for z in zeros):
                raise ValueError(
                    f"{c!r}: MULT requires a '0' landmark in every operand space"
                )
            # Implicit corresponding values at zero: x+y=z pins (0,0,0);
            # y=-x pins (0,0). These carry the sign algebra of the constraint.
            if kind == "add" and all(z is not None for z in zeros):
                implicit3 = (zeros[0], zeros[1], zeros[2])
                if implicit3 not in cvals:
                    cvals.append(implicit3)
            if kind == "minus" and all(z is not None for z in zeros):
                implicit2 = (zeros[0], zeros[1])
                if implicit2 not in cvals:
                    cvals.append(implicit2)
            compiled.append(
                CompiledConstraint(
                    kind, idx, tuple(cvals), zeros, tuple(inf_pair(s) for s in csp), c
                )
            )
        return CompiledModel(self.name, var_order, spaces, tuple(compiled))
