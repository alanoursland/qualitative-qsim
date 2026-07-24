"""Declarative energy filter (open-questions.md #7).

Research lineage: [FoucheKuipers1992] in ``docs/references.md``.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model import CompiledModel
from .quantity import Qdir, QVal
from .state import QState

__all__ = ["Trend", "EnergyFilter"]


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
