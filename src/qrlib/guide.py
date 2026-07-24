"""Guided qualitative simulation: temporal-logic trajectory constraints.

Research lineage: [BrajnikClancy1998] and [ShultsKuipers1997] in
``docs/references.md``.

TeQSIM-style guidance (docs/literature-survey.md §3): a propositional
linear-temporal-logic specification over qualitative states is treated as
part of the model. Simulation is interleaved with **formula progression**
(each behavior-graph node carries the spec's residual formula; a successor
whose residual collapses to ``FALSE`` is a *bad prefix* — no extension can
satisfy the spec — and is pruned soundly), and completed behaviors receive
an exact **verdict** with the appropriate suffix semantics:

- ``CYCLE`` terminals: lasso (loop) evaluation;
- ``QUIESCENT`` / ``DIVERGENT``: the final state repeats forever
  (a self-loop lasso);
- ``DOMAIN_EXIT`` / ``REGION_EXIT`` / ``DEADEND``: finite-trace semantics;
- ``SPEC_PRUNED``: violated by construction (every physically consistent
  continuation was excluded by the spec);
- ``TRUNCATED``: three-valued — ``UNDETERMINED`` unless the prefix already
  decides the spec.

Uses this enables (per TeQSIM): focusing a large simulation on behaviors
of interest; **exogenous time-varying inputs** expressed as temporal
constraints over otherwise-unconstrained input variables; boundary-condition
problems; and folding observations into simulation.

Soundness (the guaranteed-coverage asymmetry — docs/literature-survey.md,
governing caveat): pruning only removes behaviors whose *every* extension
violates the spec, so the guided behavior set still covers every real
behavior that satisfies the spec. An empty ``violated`` (and empty
``undetermined``) set proves the **universal** claim "all real behaviors
satisfy the spec"; a non-empty ``satisfied`` set does **not** prove
existence — a satisfying behavior may be spurious.

Formulas compose with operators: ``~f`` (not), ``f & g``, ``f | g``,
``f >> g`` (implies), plus ``X`` (next), ``G`` (always), ``F``
(eventually), ``U`` (until) constructors. Atoms: :func:`mag` compares a
variable's magnitude against a *named landmark*; :func:`dir_is` tests its
direction. Landmark names are stable across discovery frames, so specs
survive landmark minting; specs may not reference directions of
``ignore_qdir`` variables (their direction is untracked).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .behavior import Behavior, BehaviorGraph, SimConfig, SimResult, TerminalClass
from .model import CompiledModel, Model
from .quantity import Qdir
from .state import QState

__all__ = [
    "Formula",
    "TRUE",
    "FALSE",
    "mag",
    "dir_is",
    "X",
    "G",
    "F",
    "U",
    "progress",
    "format_formula",
    "Verdict",
    "GuidedResult",
    "guided",
    "classify",
]

_MAG_OPS = ("==", "<", ">", "<=", ">=")
_DIRS = {"inc": Qdir.INC, "std": Qdir.STD, "dec": Qdir.DEC}


class Formula:
    """Base class; composes with ``~ & | >>`` and the X/G/F/U constructors."""

    def __invert__(self) -> "Formula":
        return _not(self)

    def __and__(self, other: "Formula") -> "Formula":
        return _and(self, other)

    def __or__(self, other: "Formula") -> "Formula":
        return _or(self, other)

    def __rshift__(self, other: "Formula") -> "Formula":  # implication
        return _or(_not(self), other)

    def __repr__(self) -> str:
        return format_formula(self)


@dataclass(frozen=True)
class _True(Formula):
    pass


@dataclass(frozen=True)
class _False(Formula):
    pass


TRUE = _True()
FALSE = _False()


@dataclass(frozen=True)
class Atom(Formula):
    var: str
    op: str  # one of _MAG_OPS, or "dir"
    value: str  # landmark name, or "inc"/"std"/"dec"


@dataclass(frozen=True)
class Not(Formula):
    sub: Formula


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Next(Formula):
    sub: Formula


@dataclass(frozen=True)
class Always(Formula):
    sub: Formula


@dataclass(frozen=True)
class Eventually(Formula):
    sub: Formula


@dataclass(frozen=True)
class Until(Formula):
    left: Formula
    right: Formula


def mag(var: str, op: str, landmark: str) -> Formula:
    """Atom: the variable's magnitude compared against a named landmark."""
    if op not in _MAG_OPS:
        raise ValueError(f"magnitude op must be one of {_MAG_OPS}, got {op!r}")
    return Atom(var, op, landmark)


def dir_is(var: str, direction: str | Qdir) -> Formula:
    """Atom: the variable's direction is inc/std/dec."""
    name = direction.name.lower() if isinstance(direction, Qdir) else direction
    if name not in _DIRS:
        raise ValueError(f"direction must be one of {sorted(_DIRS)}, got {direction!r}")
    return Atom(var, "dir", name)


def X(sub: Formula) -> Formula:
    return Next(sub)


def G(sub: Formula) -> Formula:
    return Always(sub)


def F(sub: Formula) -> Formula:
    return Eventually(sub)


def U(left: Formula, right: Formula) -> Formula:
    return Until(left, right)


# --- simplifying constructors ----------------------------------------------


def _not(f: Formula) -> Formula:
    if f is TRUE or isinstance(f, _True):
        return FALSE
    if f is FALSE or isinstance(f, _False):
        return TRUE
    if isinstance(f, Not):
        return f.sub
    return Not(f)


def _and(a: Formula, b: Formula) -> Formula:
    if isinstance(a, _False) or isinstance(b, _False):
        return FALSE
    if isinstance(a, _True):
        return b
    if isinstance(b, _True):
        return a
    if a == b:
        return a
    return And(a, b)


def _or(a: Formula, b: Formula) -> Formula:
    if isinstance(a, _True) or isinstance(b, _True):
        return TRUE
    if isinstance(a, _False):
        return b
    if isinstance(b, _False):
        return a
    if a == b:
        return a
    return Or(a, b)


# --- progression ------------------------------------------------------------


def _eval_atom(atom: Atom, state: QState, frame: CompiledModel) -> bool:
    qv = state[atom.var]
    if atom.op == "dir":
        if qv.dir is Qdir.IGN:
            raise ValueError(
                f"spec references direction of {atom.var!r}, whose direction "
                f"is untracked (ignore_qdir)"
            )
        return qv.dir is _DIRS[atom.value]
    vi = frame.index(atom.var)
    ref = 2 * frame.spaces[vi].effective_landmarks.index(atom.value)
    return {
        "==": qv.mag == ref,
        "<": qv.mag < ref,
        ">": qv.mag > ref,
        "<=": qv.mag <= ref,
        ">=": qv.mag >= ref,
    }[atom.op]


def progress(f: Formula, state: QState, frame: CompiledModel) -> Formula:
    """One step of formula progression through ``state`` (Bacchus-Kabanza).

    ``FALSE`` means the prefix already violates the spec (no extension can
    satisfy it); ``TRUE`` means every extension satisfies it."""
    if isinstance(f, (_True, _False)):
        return f
    if isinstance(f, Atom):
        return TRUE if _eval_atom(f, state, frame) else FALSE
    if isinstance(f, Not):
        return _not(progress(f.sub, state, frame))
    if isinstance(f, And):
        return _and(progress(f.left, state, frame), progress(f.right, state, frame))
    if isinstance(f, Or):
        return _or(progress(f.left, state, frame), progress(f.right, state, frame))
    if isinstance(f, Next):
        return f.sub
    if isinstance(f, Always):
        return _and(progress(f.sub, state, frame), f)
    if isinstance(f, Eventually):
        return _or(progress(f.sub, state, frame), f)
    if isinstance(f, Until):
        return _or(
            progress(f.right, state, frame),
            _and(progress(f.left, state, frame), f),
        )
    raise TypeError(f"not a formula: {f!r}")


def format_formula(f: Formula) -> str:
    if isinstance(f, _True):
        return "true"
    if isinstance(f, _False):
        return "false"
    if isinstance(f, Atom):
        if f.op == "dir":
            return f"dir({f.var})={f.value}"
        return f"{f.var}{f.op}{f.value}"
    if isinstance(f, Not):
        return f"¬({format_formula(f.sub)})"
    if isinstance(f, And):
        return f"({format_formula(f.left)} ∧ {format_formula(f.right)})"
    if isinstance(f, Or):
        return f"({format_formula(f.left)} ∨ {format_formula(f.right)})"
    if isinstance(f, Next):
        return f"X({format_formula(f.sub)})"
    if isinstance(f, Always):
        return f"G({format_formula(f.sub)})"
    if isinstance(f, Eventually):
        return f"F({format_formula(f.sub)})"
    if isinstance(f, Until):
        return f"({format_formula(f.left)} U {format_formula(f.right)})"
    raise TypeError(f"not a formula: {f!r}")


# --- exact verdicts ---------------------------------------------------------


class Verdict(Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNDETERMINED = "undetermined"


def verdict(f: Formula, graph: BehaviorGraph, behavior: Behavior) -> Verdict:
    """Exact spec verdict for one behavior, with per-terminal suffix
    semantics (module docstring)."""
    word = [
        (graph.nodes[nid].state, graph.nodes[nid].model)
        for nid in behavior.node_ids
    ]
    term = behavior.terminal
    if term is TerminalClass.SPEC_PRUNED:
        return Verdict.VIOLATED
    if term is TerminalClass.TRUNCATED:
        residual = f
        for state, frame in word:
            residual = progress(residual, state, frame)
            if isinstance(residual, _False):
                return Verdict.VIOLATED
            if isinstance(residual, _True):
                return Verdict.SATISFIED
        return Verdict.UNDETERMINED
    if term is TerminalClass.CYCLE and behavior.cycle_target is not None:
        loop = behavior.node_ids.index(behavior.cycle_target)
        word = word[:-1]  # the closing state duplicates the loop entry
        ok = _eval_word(f, word, loop)
    elif term in (TerminalClass.QUIESCENT, TerminalClass.DIVERGENT):
        ok = _eval_word(f, word, len(word) - 1)  # final state repeats forever
    else:  # DOMAIN_EXIT, REGION_EXIT, DEADEND: finite trace
        ok = _eval_word(f, word, None)
    return Verdict.SATISFIED if ok else Verdict.VIOLATED


def _subformulas(f: Formula, acc: list[Formula]) -> None:
    for child in _children(f):
        _subformulas(child, acc)
    if f not in acc:
        acc.append(f)


def _children(f: Formula) -> tuple[Formula, ...]:
    if isinstance(f, (Not, Next, Always, Eventually)):
        return (f.sub,)
    if isinstance(f, (And, Or, Until)):
        return (f.left, f.right)
    return ()


def _eval_word(
    f: Formula, word: list[tuple[QState, CompiledModel]], loop: int | None
) -> bool:
    """Tabulated LTL evaluation over a finite trace (``loop=None``) or a
    lasso word looping back to position ``loop``. Temporal fixpoints are
    iterated to stability (F/U from below, G from above)."""
    n = len(word)
    order: list[Formula] = []
    _subformulas(f, order)
    table: dict[Formula, list[bool]] = {}

    def nxt(i: int) -> int | None:
        if i + 1 < n:
            return i + 1
        return loop

    for sub in order:
        if isinstance(sub, _True):
            table[sub] = [True] * n
        elif isinstance(sub, _False):
            table[sub] = [False] * n
        elif isinstance(sub, Atom):
            table[sub] = [
                _eval_atom(sub, state, frame) for state, frame in word
            ]
        elif isinstance(sub, Not):
            table[sub] = [not v for v in table[sub.sub]]
        elif isinstance(sub, And):
            table[sub] = [
                a and b for a, b in zip(table[sub.left], table[sub.right])
            ]
        elif isinstance(sub, Or):
            table[sub] = [
                a or b for a, b in zip(table[sub.left], table[sub.right])
            ]
        elif isinstance(sub, Next):
            child = table[sub.sub]
            table[sub] = [
                child[j] if (j := nxt(i)) is not None else False
                for i in range(n)
            ]
        else:  # temporal fixpoints, iterated to stability
            if isinstance(sub, Always):
                row = [True] * n
            else:
                row = [False] * n
            for _ in range(2 * n + 2):
                changed = False
                for i in range(n - 1, -1, -1):
                    j = nxt(i)
                    succ_g = row[j] if j is not None else True
                    succ_fu = row[j] if j is not None else False
                    if isinstance(sub, Always):
                        v = table[sub.sub][i] and succ_g
                    elif isinstance(sub, Eventually):
                        v = table[sub.sub][i] or succ_fu
                    else:  # Until
                        v = table[sub.right][i] or (
                            table[sub.left][i] and succ_fu
                        )
                    if v != row[i]:
                        row[i] = v
                        changed = True
                if not changed:
                    break
            table[sub] = row
    return table[f][0]


# --- guided simulation ------------------------------------------------------


@dataclass(frozen=True)
class GuidedResult:
    """A simulation classified against a spec.

    ``universal`` is True when *no* generated behavior violates the spec or
    is undetermined — a sound proof that every real behavior from the
    initial state satisfies it. ``satisfied`` behaviors are consistent with
    the spec but not proven to exist (they may be spurious)."""

    result: SimResult
    spec: Formula
    satisfied: tuple[Behavior, ...]
    violated: tuple[Behavior, ...]
    undetermined: tuple[Behavior, ...]

    @property
    def universal(self) -> bool:
        return not self.violated and not self.undetermined

    def to_dict(self) -> dict:
        return {
            "spec": format_formula(self.spec),
            "universal": self.universal,
            "satisfied": len(self.satisfied),
            "violated": len(self.violated),
            "undetermined": len(self.undetermined),
            "result": self.result.to_dict(),
        }


def guided(
    model: Model | CompiledModel,
    initial: QState,
    spec: Formula,
    *,
    config: SimConfig | None = None,
    **overrides,
) -> GuidedResult:
    """Simulate with the spec pruning bad prefixes, then classify exactly."""
    from dataclasses import replace

    from .engines.qsim import qsim

    compiled = model.compile() if isinstance(model, Model) else model
    cfg = config or SimConfig()
    _validate_spec(spec, compiled, cfg.ignore_qdir)
    cfg = replace(cfg, guide=spec)
    if cfg.dynamic_chatter:
        # directions the spec mentions must stay tracked
        atoms: list[Formula] = []
        _subformulas(spec, atoms)
        dir_vars = tuple(
            a.var
            for a in atoms
            if isinstance(a, Atom) and a.op == "dir" and a.var not in cfg.track_qdir
        )
        cfg = replace(cfg, track_qdir=cfg.track_qdir + dir_vars)
    result = qsim(compiled, initial, config=cfg, **overrides)
    return classify(result, spec, _validated=True)


def classify(
    result: SimResult, spec: Formula, *, _validated: bool = False
) -> GuidedResult:
    """Classify an existing (guided or unguided) result's behaviors against
    a spec — temporal-logic model checking over the behavior graph."""
    graph = result.graph
    if not _validated:
        root = graph.nodes[graph.root].model
        _validate_spec(spec, root, result.config.ignore_qdir)
    buckets: dict[Verdict, list[Behavior]] = {v: [] for v in Verdict}
    for behavior in result.behaviors():
        buckets[verdict(spec, graph, behavior)].append(behavior)
    return GuidedResult(
        result,
        spec,
        tuple(buckets[Verdict.SATISFIED]),
        tuple(buckets[Verdict.VIOLATED]),
        tuple(buckets[Verdict.UNDETERMINED]),
    )


def _validate_spec(
    spec: Formula, compiled: CompiledModel, ignore_qdir: tuple[str, ...]
) -> None:
    atoms: list[Formula] = []
    _subformulas(spec, atoms)
    for a in atoms:
        if not isinstance(a, Atom):
            continue
        if a.var not in compiled.var_order:
            raise ValueError(f"spec references unknown variable {a.var!r}")
        if a.op == "dir":
            if a.var in ignore_qdir:
                raise ValueError(
                    f"spec references direction of {a.var!r}, which is in "
                    f"ignore_qdir (direction untracked)"
                )
        else:
            vi = compiled.index(a.var)
            if a.value not in compiled.spaces[vi].effective_landmarks:
                raise ValueError(
                    f"spec references {a.value!r}, not a landmark of {a.var!r}"
                )
