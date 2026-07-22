"""Consistency-based model-based diagnosis over qualitative dynamic models.

The GDE/Sherlock lineage extended to continuous dynamics in the manner of
QDOCS (docs/literature-survey.md §2): components carry **behavioral modes**
(a normal mode plus fault modes, each a set of constraints); a **candidate**
assigns one mode to every component; a candidate is *consistent* when the
model under that assignment — simulated from the initial state — **covers**
every observation (the coverage oracle is the consistency check, and its
refutations are the conflicts). Diagnoses are the consistent candidates
with minimal faulted-component sets, searched in order of fault cardinality.

Soundness semantics (the guaranteed-coverage asymmetry, see
docs/literature-survey.md, governing caveat):

- **Refutations are sound**: QSIM predicts every real behavior of a
  candidate's model, so an observation not covered genuinely rules that
  candidate out (modulo the abstraction parameters embedded in the
  observation).
- **Diagnoses are "not refuted"**: a covering path may be spurious, so a
  consistent candidate is *possible*, not proven. In particular, refuting
  the all-normal candidate soundly establishes that *something* is wrong;
  the returned diagnoses are the minimal explanations the observations
  cannot exclude. A diagnosis whose evidence rests on a truncated frontier
  is flagged ``vacuous``.

Fault modes are assumed present throughout the observation window (no
mid-behavior fault occurrence). Operating points matter: use
:class:`~qrlib.constraints.At` (paired with ``Constant``) in mode
definitions so that e.g. "supply stuck at zero" is distinguishable from
"normally operating at zero inflow".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

from .behavior import SimConfig
from .bridge.abstraction import AbstractedBehavior
from .bridge.coverage import CoverageResult, check
from .constraints import Constraint
from .engines.qsim import qsim
from .model import Model
from .state import QState, TimeTag

__all__ = ["Component", "CandidateCheck", "DiagnosisResult", "diagnose"]


@dataclass(frozen=True)
class Component:
    """A diagnosable component: named behavioral modes, each a constraint set.

    ``modes`` maps mode name -> constraints active in that mode (the normal
    mode's constraints are the component's nominal physics; fault modes
    replace them). All mode constraints must reference variables of the
    base model."""

    name: str
    modes: Mapping[str, tuple[Constraint, ...]]
    normal: str = "ok"

    def __post_init__(self) -> None:
        if self.normal not in self.modes:
            raise ValueError(
                f"component {self.name!r}: normal mode {self.normal!r} "
                f"is not among its modes {sorted(self.modes)}"
            )

    @property
    def fault_modes(self) -> tuple[str, ...]:
        return tuple(m for m in self.modes if m != self.normal)


@dataclass(frozen=True)
class CandidateCheck:
    """One candidate (mode assignment) and the outcome of its check.

    ``evidence`` holds one CoverageResult per observation (up to the first
    refuting one when inconsistent); ``vacuous`` marks consistency that
    rests on a truncated simulation frontier (unrefuted, not endorsed)."""

    modes: tuple[tuple[str, str], ...]  # (component, mode), declaration order
    faulted: tuple[str, ...]
    consistent: bool
    reason: str | None
    evidence: tuple[CoverageResult, ...]
    vacuous: bool = False

    def to_dict(self) -> dict:
        return {
            "modes": dict(self.modes),
            "faulted": list(self.faulted),
            "consistent": self.consistent,
            "reason": self.reason,
            "vacuous": self.vacuous,
            "evidence": [
                {
                    "covered": e.covered,
                    "matched": e.matched,
                    "total": e.total,
                    "witness": list(e.witness) if e.witness else None,
                    "note": e.note,
                }
                for e in self.evidence
            ],
        }


@dataclass(frozen=True)
class DiagnosisResult:
    """Outcome of a diagnosis run.

    ``normal`` is the all-normal candidate's check — when it is consistent
    the observations give no reason to suspect a fault and no candidates
    are searched. Otherwise ``diagnoses`` are the consistent candidates
    with minimal faulted sets found within ``max_faults`` (different fault
    modes of the same minimal set are reported separately)."""

    normal: CandidateCheck
    diagnoses: tuple[CandidateCheck, ...]
    candidates_checked: int
    max_faults: int
    note: str | None = None

    @property
    def fault_detected(self) -> bool:
        return not self.normal.consistent

    def to_dict(self) -> dict:
        return {
            "fault_detected": self.fault_detected,
            "normal": self.normal.to_dict(),
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "candidates_checked": self.candidates_checked,
            "max_faults": self.max_faults,
            "note": self.note,
        }


def diagnose(
    model: Model,
    components: Sequence[Component],
    observations,
    *,
    initial: QState | None = None,
    max_faults: int = 2,
    config: SimConfig | None = None,
) -> DiagnosisResult:
    """Diagnose which component modes are consistent with the observations.

    ``model`` holds the variables and the always-active constraints;
    ``components`` contribute per-mode constraints. ``observations`` is one
    observation or a sequence of them (each an
    :class:`~qrlib.bridge.abstraction.AbstractedBehavior` or a QState
    sequence); a candidate must cover all of them. ``initial`` is the
    (nominal) initial point state; when omitted it is taken from the first
    observation's first state, which must be a point state.
    """
    obs_list = _as_observations(observations)
    if not obs_list:
        raise ValueError("at least one observation is required")
    if initial is None:
        first = _states_of(obs_list[0])[0]
        if first.time is not TimeTag.POINT:
            raise ValueError(
                "the first observation starts mid-interval; pass the "
                "initial point state explicitly (initial=...)"
            )
        initial = first
    _validate(model, components, initial)
    cfg = config or SimConfig()

    def assignment_of(faulted: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (c.name, faulted.get(c.name, c.normal)) for c in components
        )

    checked = 0

    def run(modes: tuple[tuple[str, str], ...]) -> CandidateCheck:
        nonlocal checked
        checked += 1
        return _check_candidate(model, components, dict(modes), initial, obs_list, cfg)

    normal = run(assignment_of({}))
    if normal.consistent:
        return DiagnosisResult(normal, (), checked, max_faults)

    diagnoses: list[CandidateCheck] = []
    minimal_sets: list[frozenset[str]] = []
    for k in range(1, max_faults + 1):
        for combo in combinations(components, k):
            names = frozenset(c.name for c in combo)
            if any(found < names for found in minimal_sets):
                continue  # strict superset of a known minimal diagnosis
            for choice in product(*(c.fault_modes for c in combo)):
                cand = run(
                    assignment_of(dict(zip((c.name for c in combo), choice)))
                )
                if cand.consistent:
                    diagnoses.append(cand)
                    if names not in minimal_sets:
                        minimal_sets.append(names)
    note = None
    if not diagnoses:
        note = (
            f"the observations refute normal behavior, but no candidate "
            f"with at most {max_faults} faulted component(s) explains them"
        )
    return DiagnosisResult(normal, tuple(diagnoses), checked, max_faults, note)


# --- internals -------------------------------------------------------------


def _as_observations(observations) -> list:
    if isinstance(observations, AbstractedBehavior):
        return [observations]
    seq = list(observations)
    if seq and isinstance(seq[0], QState):
        return [tuple(seq)]
    return seq


def _states_of(obs) -> tuple[QState, ...]:
    return obs.states if isinstance(obs, AbstractedBehavior) else tuple(obs)


def _validate(
    model: Model, components: Sequence[Component], initial: QState
) -> None:
    seen: set[str] = set()
    for comp in components:
        if comp.name in seen:
            raise ValueError(f"duplicate component name {comp.name!r}")
        seen.add(comp.name)
        for mode, constraints in comp.modes.items():
            for c in constraints:
                for ref in c.variables:
                    if ref not in model.variables:
                        raise ValueError(
                            f"component {comp.name!r} mode {mode!r}: "
                            f"constraint references undeclared variable {ref!r}"
                        )
                for cv in c.corresponding_values:
                    for ref, lm in zip(c.variables, cv):
                        if lm not in model.variables[ref].space.effective_landmarks:
                            raise ValueError(
                                f"component {comp.name!r} mode {mode!r}: "
                                f"{lm!r} is not a landmark of {ref!r}"
                            )
    missing = set(model.variables) - set(initial.variables)
    extra = set(initial.variables) - set(model.variables)
    if missing or extra:
        raise ValueError(
            f"initial state variables do not match the model "
            f"(missing: {sorted(missing)}, unknown: {sorted(extra)})"
        )


def _candidate_model(
    base: Model, components: Sequence[Component], assignment: Mapping[str, str]
) -> Model:
    label = ",".join(f"{n}:{m}" for n, m in assignment.items())
    mode_constraints = [
        c for comp in components for c in comp.modes[assignment[comp.name]]
    ]
    m = Model(f"{base.name}[{label}]")
    m.variables = dict(base.variables)
    m.constraints = list(base.constraints) + mode_constraints
    if base.regions:
        # mode constraints are active in every region
        for rname, subset in base.regions.items():
            m.regions[rname] = (
                None if subset is None else tuple(subset) + tuple(mode_constraints)
            )
        m.region_transitions = list(base.region_transitions)
        m.initial_region = base.initial_region
    return m


def _check_candidate(
    base: Model,
    components: Sequence[Component],
    assignment: Mapping[str, str],
    initial: QState,
    obs_list: list,
    cfg: SimConfig,
) -> CandidateCheck:
    modes = tuple((c.name, assignment[c.name]) for c in components)
    faulted = tuple(
        c.name for c in components if assignment[c.name] != c.normal
    )
    cand_model = _candidate_model(base, components, assignment)
    try:
        result = qsim(cand_model, initial, config=cfg)
    except ValueError as e:
        if "violates constraint" not in str(e):
            raise  # structural problem, not a candidate refutation
        return CandidateCheck(
            modes, faulted, False, f"initial state inconsistent: {e}", ()
        )
    evidence: list[CoverageResult] = []
    for obs in obs_list:
        cov = check(obs, result.graph)
        evidence.append(cov)
        if not cov.covered:
            return CandidateCheck(
                modes, faulted, False, cov.diagnosis, tuple(evidence)
            )
    vacuous = any(e.note and "truncated" in e.note for e in evidence)
    return CandidateCheck(modes, faulted, True, None, tuple(evidence), vacuous)
