"""Comparative analysis: how an equilibrium depends on its parameters.

Research lineage: Chiu and Kuipers (1992). This module implements a narrower
equilibrium comparative-statics procedure.

Weld's *differential qualitative* analysis asks the perturbation question —
*change a parameter in a known direction; which way does the behavior move?*
— e.g. "raise the
inflow: does the equilibrium level rise?". This module answers the sound,
well-defined core of that question, **comparative statics**: compare two
equilibria of the same model whose exogenous parameters differ
infinitesimally in known sign, and derive the sign of change induced on
every other quantity.

Method — sign propagation over the constraint network under the
equilibrium condition:

- At an equilibrium every rate is zero, so for each ``Deriv(x, y)`` the
  derivative variable ``y`` is zero in *both* systems: its change is
  ``0``. That is the equilibrium seed that closes the algebraic network.
- Exogenous parameters not named in the perturbation are held fixed
  (``Constant`` and ``At`` variables get change ``0``); the perturbed ones
  get their declared sign.
- The algebraic constraints relate the *signs of change* of their
  variables and propagate to a fixpoint: ``M+`` copies a sign, ``M-`` and
  ``MINUS`` flip it, ``ADD`` is a qualitative sum (invertible), ``MULT``
  weights each operand's change by the *other* operand's sign at the base
  operating point (so it needs a ``base`` equilibrium state).

Every relation is exact, so a uniquely forced sign is sound; where the
network genuinely cannot decide — an ``ADD`` of opposing changes, a
loop with no determined entry, a ``MULT`` with no base — the result is
``INDETERMINATE`` (reported, never guessed). This is comparative statics,
not full behavioral comparison: it compares equilibria, not transient
timing (period, time-to-threshold), and it perturbs *parameters*
(exogenous variables), not landmark values or quantity-space structure.
Like every constraint-level result it inherits the model's correctness —
a wrong constraint gives a wrong comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..model import CompiledModel, Model, _KIND
from ..state import QState

__all__ = ["Change", "ComparativeResult", "compare", "narrate_comparison"]


class Change(Enum):
    """Sign of change of a quantity between the two compared equilibria."""

    INCREASE = 1
    DECREASE = -1
    UNCHANGED = 0
    INDETERMINATE = None

    @property
    def symbol(self) -> str:
        return {1: "↑", -1: "↓", 0: "·", None: "?"}[self.value]


@dataclass(frozen=True)
class ComparativeResult:
    """Signs of change of every quantity given the parameter perturbation.

    ``changes`` maps each variable to its :class:`Change`; ``indeterminate``
    lists the variables the network could not sign; ``region`` is the
    operating region the comparison was run in."""

    perturbation: tuple[tuple[str, int], ...]
    changes: dict[str, Change]
    region: str
    indeterminate: tuple[str, ...]

    def __getitem__(self, var: str) -> Change:
        return self.changes[var]

    def to_dict(self) -> dict:
        return {
            "perturbation": {k: v for k, v in self.perturbation},
            "region": self.region,
            "changes": {k: v.value for k, v in self.changes.items()},
            "indeterminate": list(self.indeterminate),
        }


# --- sign algebra ----------------------------------------------------------

# A change value is +1 / -1 / 0 (definite), or None (indeterminate: known to
# be un-signable). "Unknown" is absence from the assignment dict.

_INDET = None


def _neg(s):
    return _INDET if s is _INDET else -s


def _qsum(a, b):
    """Qualitative sum of two change signs (None = indeterminate)."""
    if a is _INDET or b is _INDET:
        return _INDET
    if a == 0:
        return b
    if b == 0:
        return a
    if a == b:
        return a
    return _INDET  # opposing changes: cannot decide


def _mul(a, b):
    if a is _INDET or b is _INDET:
        return _INDET
    return a * b


def compare(
    model: Model | CompiledModel,
    perturbation: dict[str, int],
    *,
    base: QState | None = None,
    region: str | None = None,
) -> ComparativeResult:
    """Comparative statics: sign of change of every quantity when the
    ``perturbation`` parameters shift by the given signs (``+1``/``-1``).

    ``base`` is an equilibrium state whose operating-point signs resolve
    ``MULT`` constraints (required only when the model has products whose
    operands can be either sign). ``region`` selects the operating region
    for a multi-region model."""
    frame = model.compile() if isinstance(model, Model) else model
    if region is None:
        region = frame.initial_region
    active = [
        cc for cc in frame.constraints_of(region) if cc.kind != "negligible"
    ]
    for name, s in perturbation.items():
        if name not in frame.var_order:
            raise ValueError(f"perturbation names unknown variable {name!r}")
        if s not in (-1, 0, 1):
            raise ValueError(f"perturbation sign of {name!r} must be -1, 0 or +1")

    names = frame.var_order
    base_sign = _base_signs(frame, base, names) if base is not None else None

    # seeds ----------------------------------------------------------------
    change: dict[str, object] = {}

    def assign(var: str, s) -> None:
        if var in change and change[var] is not _INDET and s is not _INDET:
            if change[var] != s:
                raise ValueError(
                    f"contradictory change signs for {var!r}: "
                    f"{change[var]} vs {s} (is the perturbation over-specified?)"
                )
            return
        if var in change and change[var] is not _INDET and s is _INDET:
            return  # keep the definite value
        change[var] = s

    for name, s in perturbation.items():
        assign(name, s)
    for cc in active:
        if cc.kind == "deriv":  # equilibrium: the rate is zero in both systems
            assign(names[cc.vars[1]], 0)
        elif cc.kind in ("constant", "at"):
            v = names[cc.vars[0]]
            if v not in perturbation:
                assign(v, 0)

    # propagate to a fixpoint ---------------------------------------------
    changed = True
    guard = 0
    max_iter = (len(names) + 1) * (len(active) + 1) + 8
    while changed and guard < max_iter:
        changed = False
        guard += 1
        for cc in active:
            vs = [names[i] for i in cc.vars]
            before = {v: change.get(v, "?") for v in vs}
            _propagate(cc, vs, change, base_sign, assign)
            if any(change.get(v, "?") != before[v] for v in vs):
                changed = True

    changes = {
        n: Change(change[n]) if change.get(n, _INDET) is not _INDET
        else Change.INDETERMINATE
        for n in names
    }
    indeterminate = tuple(n for n in names if changes[n] is Change.INDETERMINATE)
    return ComparativeResult(
        tuple(sorted(perturbation.items())), changes, region, indeterminate
    )


def _propagate(cc, vs, change, base_sign, assign) -> None:
    kind = cc.kind

    def known(v):
        return v in change

    if kind == "mplus":
        x, y = vs
        if known(x):
            assign(y, change[x])
        if known(y):
            assign(x, change[y])
    elif kind in ("mminus", "minus"):
        x, y = vs
        if known(x):
            assign(y, _neg(change[x]))
        if known(y):
            assign(x, _neg(change[y]))
    elif kind == "add":  # x + y = z
        x, y, z = vs
        if known(x) and known(y):
            assign(z, _qsum(change[x], change[y]))
        if known(z) and known(x):
            assign(y, _qsum(change[z], _neg(change[x])))
        if known(z) and known(y):
            assign(x, _qsum(change[z], _neg(change[y])))
    elif kind == "mult":  # z = x * y ; dz = y0*dx + x0*dy
        x, y, z = vs
        if base_sign is None:
            if known(x) and known(y):
                # a product with unknown operating point: undecidable in general
                if change[x] != 0 or change[y] != 0:
                    assign(z, _INDET)
                else:
                    assign(z, 0)
            return
        x0, y0 = base_sign[x], base_sign[y]
        if known(x) and known(y):
            assign(z, _qsum(_mul(y0, change[x]), _mul(x0, change[y])))
    # deriv / constant / at contribute only seeds (handled before the loop)


def _base_signs(frame: CompiledModel, base: QState, names) -> dict[str, int]:
    """Operating-point sign (value vs the ``0`` landmark) of each variable in
    the base equilibrium state, for resolving MULT."""
    out: dict[str, int] = {}
    for vi, name in enumerate(names):
        space = frame.spaces[vi]
        try:
            zero = space.rank_of("0")
        except ValueError:
            out[name] = 0
            continue
        mag = base[name].mag
        out[name] = (mag > zero) - (mag < zero)
    return out


def narrate_comparison(result: ComparativeResult) -> str:
    """One-line-per-quantity prose summary."""
    params = ", ".join(
        f"{k} {'up' if v > 0 else 'down' if v < 0 else 'fixed'}"
        for k, v in result.perturbation
    )
    lines = [f"Comparing equilibria with {params} (region {result.region!r}):"]
    verb = {
        Change.INCREASE: "increases",
        Change.DECREASE: "decreases",
        Change.UNCHANGED: "is unchanged",
        Change.INDETERMINATE: "is indeterminate",
    }
    for name, ch in result.changes.items():
        lines.append(f"  {name} {verb[ch]} {ch.symbol}")
    return "\n".join(lines)
