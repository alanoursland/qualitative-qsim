"""Process-centric model authoring (after Forbus's Qualitative Process
Theory).

Research lineage: [Forbus1984] in ``docs/references.md``.

A :class:`System` is written in QPT vocabulary: **quantities**,
**qualitative proportionalities** (``target`` moves monotonically with
``source``), and **processes** that exert **direct influences**
(``I+``/``I-``: the process contributes its rate to a quantity's time
derivative) while their activation conditions hold. :meth:`System.build`
compiles this down to an ordinary :class:`~qrlib.model.Model`:

- each influenced quantity ``q`` gets a derivative variable ``q'`` and a
  ``Deriv(q, q')`` constraint;
- **influence resolution** turns the set of influences active in a given
  process combination into constraints on ``q'``: no influences pins
  ``q' = 0`` (QPT's sole-mechanism assumption: nothing else moves a
  quantity), a single influence ties ``q'`` to its rate through a
  sign-preserving monotonic link, and a pair combines through ``Add``;
- **activation conditions** become operating regions — one per on/off
  combination of the conditioned processes — with boundary transitions
  where a condition's landmark is reached (both directions; the engine's
  Zeno guard keeps mutual boundary transitions from ping-ponging).

Compiling to QDE constraints necessarily *strengthens* some QPT
statements (a proportionality becomes a definite monotonic function, a
pair of influences a definite sum); that is the standard price of
feeding a QDE engine and is stated here rather than hidden.

Current limits (raise, never silently degrade): one proportionality per
target (introduce an explicit combining quantity for more), at most two
influences per quantity per region and not both negative (flip a rate's
sign in its own definition instead), single-comparison activation
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from ..constraints import Add, At, Constant, Constraint, Deriv, MMinus, MPlus
from ..model import Model
from ..state import QState

__all__ = ["Process", "System"]

_OPS = (">", "<", ">=", "<=")


@dataclass
class Process:
    """A named mechanism: direct influences, active while ``when`` holds
    (``when=()`` means always active)."""

    name: str
    when: tuple[tuple[str, str, str], ...] = ()
    influences: list[tuple[str, int, str]] = field(default_factory=list)

    def influence(self, quantity: str, sign: int, rate: str) -> "Process":
        """``I+``/``I-``: while active, contribute ``sign * rate`` to
        ``d(quantity)/dt``."""
        if sign not in (1, -1):
            raise ValueError(f"influence sign must be +1 or -1, got {sign!r}")
        self.influences.append((quantity, sign, rate))
        return self


class System:
    """A QPT-style system description; :meth:`build` emits the Model."""

    def __init__(self, name: str = "system"):
        self.name = name
        self._model = Model(name)
        self._props: list[tuple[str, str, int, tuple]] = []
        self._processes: list[Process] = []
        self._built: Model | None = None

    # --- authoring ---------------------------------------------------------

    def quantity(self, name: str, **kwargs) -> str:
        """Declare a quantity (same keywords as ``Model.variable``)."""
        self._model.variable(name, **kwargs)
        return name

    def proportional(
        self, target: str, source: str, sign: int = 1, *, cvals: tuple = ()
    ) -> None:
        """``target`` ∝Q± ``source``: target is a monotonic function of
        source (compiled to ``MPlus``/``MMinus``)."""
        if sign not in (1, -1):
            raise ValueError(f"proportionality sign must be +1 or -1, got {sign!r}")
        if any(t == target for t, _s, _g, _c in self._props):
            raise ValueError(
                f"{target!r} already has a proportionality; combine multiple "
                f"antecedents through an explicit quantity (e.g. a sum)"
            )
        self._props.append((target, source, sign, tuple(cvals)))

    def constrain(self, constraint: Constraint | str) -> Constraint:
        """Escape hatch: structural equations stated directly or as compact
        string expressions."""
        return self._model.constrain(constraint)

    def process(
        self, name: str, *, when: tuple[tuple[str, str, str], ...] = ()
    ) -> Process:
        if any(p.name == name for p in self._processes):
            raise ValueError(f"process {name!r} already declared")
        if len(when) > 1:
            raise ValueError(
                "multi-condition activations are not supported yet; "
                "introduce a single deciding quantity"
            )
        for var, op, landmark in when:
            if op not in _OPS:
                raise ValueError(f"condition op must be one of {_OPS}, got {op!r}")
            if var not in self._model.variables:
                raise ValueError(f"condition references unknown quantity {var!r}")
            if landmark not in self._model.variables[var].space.effective_landmarks:
                raise ValueError(
                    f"condition landmark {landmark!r} is not a landmark of {var!r}"
                )
        p = Process(name, tuple(when))
        self._processes.append(p)
        return p

    # --- compilation -------------------------------------------------------

    def build(self) -> Model:
        """Resolve influences and emit the Model (with activation regions
        when any process is conditioned)."""
        m = self._model
        for target, source, sign, cvals in self._props:
            m.constrain((MPlus if sign > 0 else MMinus)(source, target, cvals=cvals))

        influenced = []
        for p in self._processes:
            for q, _sign, rate in p.influences:
                for ref in (q, rate):
                    if ref not in m.variables:
                        raise ValueError(f"influence references unknown quantity {ref!r}")
                if q not in influenced:
                    influenced.append(q)
        rate_of = {}
        base: list[Constraint] = list(m.constraints)
        for q in influenced:
            rate_of[q] = f"{q}'"
            m.variable(rate_of[q], landmarks=("0",), unbounded=True)
            base.append(m.constrain(Deriv(q, rate_of[q])))

        conditioned = [p for p in self._processes if p.when]
        combos = list(product((True, False), repeat=len(conditioned))) or [()]
        resolution_cache: dict[tuple, tuple[Constraint, ...]] = {}
        region_constraints: dict[str, list[Constraint]] = {}
        for combo in combos:
            active = [p for p in self._processes if not p.when] + [
                p for p, on in zip(conditioned, combo) if on
            ]
            cs: list[Constraint] = []
            for q in influenced:
                infl = tuple(
                    (sign, rate)
                    for p in active
                    for iq, sign, rate in p.influences
                    if iq == q
                )
                key = (q, infl)
                if key not in resolution_cache:
                    resolution_cache[key] = tuple(
                        m.constrain(c) for c in self._resolve(rate_of[q], infl)
                    )
                cs.extend(resolution_cache[key])
            region_constraints[self._region_name(conditioned, combo)] = cs

        if conditioned:
            for combo in combos:
                name = self._region_name(conditioned, combo)
                m.region(name, constraints=tuple(base + region_constraints[name]))
            for combo in combos:
                source = self._region_name(conditioned, combo)
                for k, p in enumerate(conditioned):
                    target = self._region_name(
                        conditioned, combo[:k] + (not combo[k],) + combo[k + 1 :]
                    )
                    (var, _op, landmark) = p.when[0]
                    m.transition(source, target, when=((var, "==", landmark),))
        self._built = m
        return m

    @staticmethod
    def _region_name(conditioned: list[Process], combo: tuple[bool, ...]) -> str:
        if not conditioned:
            return "default"
        return ",".join(
            p.name if on else f"!{p.name}" for p, on in zip(conditioned, combo)
        )

    @staticmethod
    def _resolve(rate_var: str, infl: tuple[tuple[int, str], ...]) -> list[Constraint]:
        """Influence resolution: constraints tying ``rate_var`` to the
        active influences' rates."""
        zero = (("0", "0"),)
        if not infl:  # sole-mechanism: nothing moves the quantity
            return [At(rate_var, "0"), Constant(rate_var)]
        if len(infl) == 1:
            (sign, r) = infl[0]
            return [(MPlus if sign > 0 else MMinus)(r, rate_var, cvals=zero)]
        if len(infl) == 2:
            (s1, r1), (s2, r2) = infl
            if s1 > 0 and s2 > 0:
                return [Add(r1, r2, rate_var)]
            if s1 > 0 and s2 < 0:
                return [Add(rate_var, r2, r1)]
            if s1 < 0 and s2 > 0:
                return [Add(rate_var, r1, r2)]
        raise ValueError(
            "influence resolution supports at most two influences per "
            "quantity per region, not both negative — fold signs into the "
            "rate definitions or combine rates explicitly"
        )

    # --- runtime helper ----------------------------------------------------

    def active_region(self, state: QState) -> str:
        """The region whose process activations match ``state`` — assign
        it to ``model.initial_region`` before simulating."""
        if self._built is None:
            raise ValueError("build() the system first")
        conditioned = [p for p in self._processes if p.when]
        if not conditioned:
            return "default"

        def holds(cond: tuple[str, str, str]) -> bool:
            var, op, landmark = cond
            space = self._built.variables[var].space
            rank = state[var].mag
            ref = 2 * space.effective_landmarks.index(landmark)
            return {
                ">": rank > ref,
                "<": rank < ref,
                ">=": rank >= ref,
                "<=": rank <= ref,
            }[op]

        combo = tuple(holds(p.when[0]) for p in conditioned)
        return self._region_name(conditioned, combo)
