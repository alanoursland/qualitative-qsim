"""Declarative energy and Lyapunov successor filters.

Research lineage: Fouché and Kuipers (1992).

The canonical spurious-behavior killer is an *energy argument*: a
conserved or dissipated quantity constrains how far a system can swing.
QSIM's pluggable ``successor_filters`` hook can express it, but the
point/interval landmark semantics are fiddly and shouldn't be reinvented
per host — so this ships it first-class.

:class:`EnergyFilter` is a ready-made ``SuccessorFilter``: construct it,
drop it in ``SimConfig(successor_filters=(EnergyFilter(...),))``, and it
prunes successors whose amplitude would violate the declared energy trend.
It works *through landmark discovery*: as a variable's turning points are
minted into named landmarks, those landmarks become the amplitude bounds
the argument enforces. The classic result — the frictionless spring, whose
authentic QSIM run branches into spurious growing/shrinking oscillations —
collapses to its single true cycle.

Two trends:

- ``CONSERVED`` (default): every turning point on a given side of the
  reference must coincide with *the* discovered extremum on that side. A
  variable coming to rest (steady) at a fresh unnamed amplitude, or moving
  outward past a discovered extremum, is refused — energy conservation pins
  the swing to one amplitude per side.
- ``NONINCREASING`` (dissipative): amplitude may shrink but never grow.
  Resting *inside* a discovered extremum is allowed; only crossing beyond
  it — a fresh larger turning point, or accelerating outward at the peak —
  is refused.

Point/interval care: the direction (``qdir``) and rank parity carry the
semantics — a variable *at* a landmark (even rank) with an outward
direction is mid-swing past a peak; a variable *steady* (``STD``) in an
interval (odd rank) is a turning point at an unnamed amplitude. Both are
handled explicitly.

Soundness: like a declared constraint, an energy filter is only as valid
as the physics it asserts. Applied to a genuinely conservative system it
removes only spurious behaviors (regression-tested against the numeric
soundness harness); asserting conservation of a non-conserved system would
remove real ones. The filter never adds behaviors, so a wrong declaration
can only over-prune, never fabricate.

For asymptotic-stability arguments, :class:`LyapunovCertificate` handles a
different issue.  A scalar Lyapunov variable can be strictly decreasing
numerically while remaining in one open qualitative magnitude interval.
Ordinary qualitative cycle detection would then mistake repeated interval
descriptions for a recurrent physical state.  The certificate:

* enforces a minimum at a declared equilibrium;
* enforces nonincrease everywhere and strict decrease under a declarative
  landmark condition; and
* lets QSIM reject a repeated qualitative cycle when strict decrease occurs
  somewhere around it.

For a damped mechanical system, for example, ``strict_when`` can state
``velocity != 0``.  Isolated turning points where the derivative is zero are
allowed, but a complete non-equilibrium oscillation cannot recur because its
Lyapunov scalar fell during the moving part of the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable, Mapping

from .model import CompiledModel
from .quantity import Qdir, QVal
from .state import QState

__all__ = ["Trend", "EnergyFilter", "LyapunovCertificate"]

_OPS = ("==", "!=", "<", ">", "<=", ">=")


class Trend(Enum):
    CONSERVED = "conserved"
    NONINCREASING = "nonincreasing"


@dataclass(frozen=True)
class EnergyFilter:
    """A declarative energy-argument successor filter.

    - ``variables``: the amplitude-contributing variables; empty means
      *all* of the model's variables (the usual case — every oscillating
      state variable is bounded).
    - ``reference``: the landmark amplitude is measured from (``"0"``).
    - ``trend``: :class:`Trend` (or its string value)."""

    variables: tuple[str, ...] = ()
    reference: str = "0"
    trend: Trend = Trend.CONSERVED

    def __post_init__(self) -> None:
        if isinstance(self.trend, str):
            object.__setattr__(self, "trend", Trend(self.trend))
        object.__setattr__(self, "variables", tuple(self.variables))

    def __call__(self, parent: QState, cand: QState, frame: CompiledModel) -> bool:
        """``SuccessorFilter`` protocol: keep ``cand`` (True) or veto it."""
        names = self.variables or frame.var_order
        for name in names:
            vi = frame.index(name)
            space = frame.spaces[vi]
            try:
                ref = space.rank_of(self.reference)
            except ValueError:
                continue  # no reference landmark here: nothing to bound
            qv = cand[name]
            # discovered extrema on each side of the reference (named
            # landmarks other than the reference itself)
            pos = [space.rank_of(n) for n in space.names if space.rank_of(n) > ref]
            neg = [space.rank_of(n) for n in space.names if space.rank_of(n) < ref]
            if not self._admits(qv, ref, pos, neg):
                return False
        return True

    def _admits(self, qv: QVal, ref: int, pos: list[int], neg: list[int]) -> bool:
        mag, at_landmark = qv.mag, qv.mag % 2 == 0
        # --- turning point: steady at an unnamed amplitude ----------------
        if qv.dir is Qdir.STD and not at_landmark:
            if mag > ref and pos:
                if self.trend is Trend.CONSERVED or mag > max(pos):
                    return False
            if mag < ref and neg:
                if self.trend is Trend.CONSERVED or mag < min(neg):
                    return False
        # --- mid-swing: at a discovered extremum, still moving outward ----
        if at_landmark and mag != ref:
            if mag in pos and qv.dir is Qdir.INC:
                return False
            if mag in neg and qv.dir is Qdir.DEC:
                return False
        return True

    def describe(self) -> str:
        scope = ", ".join(self.variables) if self.variables else "all variables"
        return f"energy({self.trend.value}, {scope}, ref={self.reference!r})"

    def to_dict(self) -> dict:
        return {
            "kind": "energy",
            "trend": self.trend.value,
            "variables": list(self.variables),
            "reference": self.reference,
        }


@dataclass(frozen=True)
class LyapunovCertificate:
    """Conditional strict-decrease certificate for a scalar model variable.

    ``variable`` names the modeled Lyapunov scalar. ``minimum`` is its
    equilibrium landmark. ``equilibrium`` maps state-variable names to the
    landmarks defining the equilibrium. ``strict_when`` is an optional
    conjunction of ``(variable, operator, landmark)`` predicates. When it is
    omitted, strict decrease is required everywhere outside ``equilibrium``.

    Put the certificate in :attr:`qrlib.SimConfig.successor_filters`. It
    behaves as a local filter and also gives QSIM the path information needed
    to refute recurrence inside one qualitative magnitude interval.
    """

    variable: str
    equilibrium: tuple[tuple[str, str], ...] | Mapping[str, str]
    minimum: str = "0"
    strict_when: tuple[tuple[str, str, str], ...] | Iterable[
        tuple[str, str, str]
    ] = ()

    def __post_init__(self) -> None:
        if not self.variable:
            raise ValueError("Lyapunov variable must be non-empty")
        if not self.minimum:
            raise ValueError("Lyapunov minimum must be non-empty")
        equilibrium = (
            tuple(self.equilibrium.items())
            if isinstance(self.equilibrium, Mapping)
            else tuple(self.equilibrium)
        )
        if not equilibrium:
            raise ValueError("Lyapunov equilibrium must name at least one variable")
        if any(not name or not landmark for name, landmark in equilibrium):
            raise ValueError("Lyapunov equilibrium names and landmarks must be non-empty")
        if len({name for name, _ in equilibrium}) != len(equilibrium):
            raise ValueError("Lyapunov equilibrium contains duplicate variables")
        strict_when = tuple(self.strict_when)
        for predicate in strict_when:
            if len(predicate) != 3:
                raise ValueError(
                    "Lyapunov strict_when predicates must be "
                    "(variable, operator, landmark) triples"
                )
            name, op, landmark = predicate
            if not name or not landmark:
                raise ValueError(
                    "Lyapunov strict_when names and landmarks must be non-empty"
                )
            if op not in _OPS:
                raise ValueError(
                    f"Lyapunov predicate operator must be one of {_OPS}, got {op!r}"
                )
        object.__setattr__(self, "equilibrium", equilibrium)
        object.__setattr__(self, "strict_when", strict_when)

    def validate(self, frame: CompiledModel) -> None:
        """Validate all variable and landmark references against ``frame``."""
        references = [(self.variable, self.minimum), *self.equilibrium]
        references.extend((name, landmark) for name, _, landmark in self.strict_when)
        for name, landmark in references:
            try:
                vi = frame.index(name)
            except ValueError as exc:
                raise ValueError(
                    f"Lyapunov certificate references unknown variable {name!r}"
                ) from exc
            if landmark not in frame.spaces[vi].effective_landmarks:
                raise ValueError(
                    f"Lyapunov certificate references unknown landmark "
                    f"{landmark!r} of {name!r}"
                )

    def is_equilibrium(self, state: QState, frame: CompiledModel) -> bool:
        """Whether ``state`` matches every declared equilibrium landmark."""
        return all(
            state[name].mag == frame.spaces[frame.index(name)].rank_of(landmark)
            for name, landmark in self.equilibrium
        )

    def is_strict(self, state: QState, frame: CompiledModel) -> bool:
        """Whether strict decrease is required at ``state``."""
        if not self.strict_when:
            return not self.is_equilibrium(state, frame)
        return all(
            self._predicate_holds(state, frame, name, op, landmark)
            for name, op, landmark in self.strict_when
        )

    def admits_cycle(
        self, states: Iterable[tuple[QState, CompiledModel]]
    ) -> bool:
        """Return False when declared strict progress refutes recurrence."""
        progress = False
        for state, frame in states:
            qv = state[self.variable]
            if qv.dir is Qdir.INC:
                return True
            if self.is_strict(state, frame) and qv.dir is Qdir.DEC:
                progress = True
        return not progress

    def __call__(self, parent: QState, cand: QState, frame: CompiledModel) -> bool:
        """Keep candidates consistent with the declared Lyapunov argument."""
        self.validate(frame)
        qv = cand[self.variable]
        vi = frame.index(self.variable)
        minimum = frame.spaces[vi].rank_of(self.minimum)
        if qv.mag < minimum or qv.dir is Qdir.INC:
            return False
        if self.is_equilibrium(cand, frame):
            return qv.mag == minimum and qv.dir is Qdir.STD
        if qv.mag == minimum:
            return False  # positive definiteness: only equilibrium has V=min
        if self.is_strict(cand, frame):
            return qv.dir is Qdir.DEC
        return True

    def _predicate_holds(
        self,
        state: QState,
        frame: CompiledModel,
        name: str,
        op: str,
        landmark: str,
    ) -> bool:
        vi = frame.index(name)
        rank = state[name].mag
        reference = frame.spaces[vi].rank_of(landmark)
        return {
            "==": rank == reference,
            "!=": rank != reference,
            "<": rank < reference,
            ">": rank > reference,
            "<=": rank <= reference,
            ">=": rank >= reference,
        }[op]

    def describe(self) -> str:
        equilibrium = ", ".join(
            f"{name}={landmark}" for name, landmark in self.equilibrium
        )
        condition = (
            " and ".join(
                f"{name}{op}{landmark}" for name, op, landmark in self.strict_when
            )
            if self.strict_when
            else "outside equilibrium"
        )
        return (
            f"lyapunov({self.variable}, min={self.minimum!r}, "
            f"equilibrium=({equilibrium}), strict when {condition})"
        )

    def to_dict(self) -> dict:
        return {
            "kind": "lyapunov",
            "variable": self.variable,
            "minimum": self.minimum,
            "equilibrium": {
                name: landmark for name, landmark in self.equilibrium
            },
            "strict_when": [list(predicate) for predicate in self.strict_when],
        }
