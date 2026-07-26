"""QDE induction: learn qualitative models from trajectory data.

Research lineage: Richards, Kraan, and Kuipers (1992), and Ramachandran,
Mooney, and Kuipers (1994). The implementation is a qrlib-specific
parsimony-ranked procedure, not a reproduction of MISQ.

The inverse of simulation. Instead of authoring a :class:`~qrlib.model.Model`
and predicting its behaviors, :func:`induce` searches for the model whose
structure the observed trajectories are consistent with — the GENMODEL/MISQ
lineage, realized here as **structure selection over a parsimony ladder,
validated by the data-consistency checker**. It is
diagnosis with the whole model as the unknown, reusing the same parts: the
least-squares sign fit and the per-constraint consistency test
(``bridge.signs``).

The pipeline:

1. **Fit** each variable's rate against the others by least squares
   (``dx_i ~ sum_j a_ij x_j``), giving a signed influence coefficient and a
   confidence per pair — exact signs for linear systems, average
   monotonicity for monotone nonlinear ones.
2. **Ladder**: sweep a descending confidence threshold. Each level keeps
   the influences it is at-least-that-confident in and drops the rest,
   giving candidate structures from dense (every influence) through sparse
   to empty (state-independent constant rates). Duplicate structures collapse.
3. **Compile** each structure to a QDE: an influenced variable gets a
   derivative variable driven by the qualitative sum of its signed
   monotone influences (``Deriv`` + ``M+``/``M-`` + ``Add``); an
   uninfluenced rate becomes ``Constant(d_x)`` plus ``Deriv(x, d_x)``.
   Latent influence-term and derivative variables carry fitted or observed
   rate values, so every constraint the structure asserts is checkable
   against the data.
4. **Validate** with :func:`~qrlib.bridge.signs.check_consistency`: each
   constraint's violation mass against the (augmented) data. A structure
   that drops a real influence leaves its derivative rows disagreeing with
   the observed motion; a structure asserting a false influence disagrees
   in the corresponding monotone row. A candidate is **consistent** when
   every constraint's violation stays under ``tol``.
5. **Rank** consistent candidates ahead of inconsistent ones, then by
   **parsimony** (fewer influences), then by mean violation.

Soundness follows the guaranteed-coverage asymmetry. Inconsistency is a
sound refutation: if the data violates a constraint the
structure asserts, that structure is wrong (modulo the tolerance and the
sampling). Consistency is **not** proof of uniqueness — weak influences are
nearly invisible qualitatively, so several structures are typically
consistent, and ``tol`` sets the sparsity/fidelity trade-off. :func:`induce`
therefore returns a **ranked set**; ``best`` is "the most parsimonious
structure the data does not refute at this tolerance", never "the true
model".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bridge.signs import (
    ConsistencyRecord,
    _coefficient_signs,
    _fit_coefficients,
    _validate_coefficient_tolerances,
    check_consistency,
)
from .constraints import Add, Constant, Deriv, MMinus, MPlus
from .model import Model
from .quantity import Landmark

__all__ = ["Candidate", "InductionResult", "induce"]


@dataclass(frozen=True)
class Candidate:
    """One induced structure and the outcome of validating it.

    ``model`` is the QDE; ``influences`` are the surviving
    ``(target, source, sign)`` rate dependencies; ``threshold`` is the
    confidence level that produced it. ``consistent`` is "every constraint's
    violation under ``tol``"; ``max_violation`` / ``mean_violation`` are the
    per-constraint violation masses (``bridge.signs.check_consistency``);
    ``worst`` names the most-violated constraint kind."""

    model: Model
    influences: tuple[tuple[str, str, int], ...]
    threshold: float
    consistent: bool
    max_violation: float
    mean_violation: float
    worst: str | None

    @property
    def n_influences(self) -> int:
        return len(self.influences)

    def to_dict(self) -> dict:
        return {
            "influences": [list(i) for i in self.influences],
            "threshold": self.threshold,
            "consistent": self.consistent,
            "n_influences": self.n_influences,
            "max_violation": self.max_violation,
            "mean_violation": self.mean_violation,
            "worst": self.worst,
        }


@dataclass(frozen=True)
class InductionResult:
    """Ranked induced candidates. ``best`` is the top-ranked consistent one
    (the most parsimonious structure no constraint refutes at ``tol``), or
    ``None`` if every candidate was refuted."""

    candidates: tuple[Candidate, ...]
    variables: tuple[str, ...]
    tol: float
    note: str | None = None

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates and self.candidates[0].consistent else None

    def to_dict(self) -> dict:
        return {
            "variables": list(self.variables),
            "tol": self.tol,
            "note": self.note,
            "candidates": [c.to_dict() for c in self.candidates],
        }


def induce(
    observations,
    variables: Sequence[str],
    *,
    derivatives=None,
    thresholds: Sequence[float] | None = None,
    tol: float = 0.02,
    max_candidates: int = 12,
    coefficient_atol: float = 1e-12,
    coefficient_rtol: float = 1e-9,
) -> InductionResult:
    """Induce candidate QDE models over ``variables`` from trajectory data.

    ``observations``: one trajectory or a list of them, each array-like
    ``(T, V)`` with columns in ``variables`` order. ``derivatives``:
    optional matching ``dx/dt`` samples (one array per trajectory, or one
    for a single trajectory); forward finite differences are used when
    omitted. ``thresholds``: the confidence ladder (descending); derived
    from the fitted confidences when omitted. ``tol``: the per-constraint
    violation a consistent structure may not exceed — the sparsity/fidelity
    knob (higher tolerates dropping weak influences). Fitted slopes below
    ``coefficient_atol`` or whose observed contribution is below
    ``coefficient_rtol`` of their rate scale are treated as numerical zero.
    """
    _validate_coefficient_tolerances(coefficient_atol, coefficient_rtol)
    trajs = _as_trajectories(observations, len(variables))
    derivs = _as_derivatives(derivatives, trajs)
    names = tuple(variables)

    A, conf = _fit(
        trajs,
        derivs,
        len(names),
        coefficient_atol,
        coefficient_rtol,
    )
    levels = _ladder(conf) if thresholds is None else sorted(thresholds, reverse=True)

    seen: set[tuple] = set()
    candidates: list[Candidate] = []
    for thr in levels:
        signed = tuple(
            tuple(_sign(A[i][j]) if conf[i][j] >= thr else 0 for j in range(len(names)))
            for i in range(len(names))
        )
        if signed in seen:
            continue
        seen.add(signed)
        candidates.append(
            _evaluate(names, A, signed, thr, trajs, derivs, tol)
        )
        if len(candidates) >= max_candidates:
            break

    candidates.sort(key=_rank_key)
    note = None
    if not any(c.consistent for c in candidates):
        note = f"no candidate structure is consistent within tol={tol}"
    return InductionResult(tuple(candidates), names, tol, note)


def _rank_key(c: Candidate) -> tuple:
    return (0 if c.consistent else 1, c.n_influences, c.mean_violation)


# --- candidate construction + validation -----------------------------------


def _build(names: tuple[str, ...], A, signed) -> tuple[Model, list, dict, tuple]:
    """Compile a signed structure to a QDE.

    Return the model, fitted latent recipes ``(name, coefficients, offset)``,
    observed constant-rate variables, and the influence tuple. Each
    influenced variable's rate is the qualitative sum of its signed monotone
    influences; an empty row means a state-independent rate, not a stationary
    state.
    """
    m = Model("induced")
    for n in names:
        m.variable(n, landmarks=(Landmark("0", value=0.0),), unbounded=True)
    recipes: list[tuple[str, dict[int, float], float]] = []
    constant_rates: dict[str, int] = {}
    influences: list[tuple[str, str, int]] = []

    def aux(n: str) -> str:
        m.variable(n, landmarks=(Landmark("0", value=0.0),), unbounded=True)
        return n

    for i, xi in enumerate(names):
        terms = [(j, signed[i][j]) for j in range(len(names)) if signed[i][j] != 0]
        if not terms:
            dname = aux(f"d_{xi}")
            m.constrain(Constant(dname))
            m.constrain(Deriv(xi, dname))
            constant_rates[dname] = i
            continue
        term_vars: list[str] = []
        for term_index, (j, sign) in enumerate(terms):
            influences.append((xi, names[j], sign))
            tname = aux(f"d_{xi}") if len(terms) == 1 else aux(f"t_{xi}_{names[j]}")
            m.constrain((MPlus if sign > 0 else MMinus)(names[j], tname))
            offset = A[i][len(names)] if term_index == 0 else 0.0
            recipes.append((tname, {j: A[i][j]}, offset))
            term_vars.append(tname)
        if len(term_vars) == 1:
            m.constrain(Deriv(xi, term_vars[0]))
            continue
        acc = term_vars[0]
        acc_coeffs = {terms[0][0]: A[i][terms[0][0]]}
        acc_offset = A[i][len(names)]
        for k in range(1, len(term_vars)):
            is_last = k == len(term_vars) - 1
            nxt = aux(f"d_{xi}" if is_last else f"s_{xi}_{k}")
            m.constrain(Add(acc, term_vars[k], nxt))
            acc_coeffs = {**acc_coeffs, terms[k][0]: A[i][terms[k][0]]}
            recipes.append((nxt, dict(acc_coeffs), acc_offset))
            acc = nxt
        m.constrain(Deriv(xi, acc))
    return m, recipes, constant_rates, tuple(influences)


def _augment(rows, derivative_rows, names, recipes, constant_rates):
    """Full-model trajectory in model-variable order: state columns from
    data, fitted latent terms, and observed state-independent rates."""
    idx = {n: i for i, n in enumerate(names)}
    order = (
        list(names)
        + [name for name, _, _ in recipes]
        + list(constant_rates)
    )
    out = []
    for sample, row in enumerate(rows):
        vals = {n: row[idx[n]] for n in names}
        for name, rate_index in constant_rates.items():
            vals[name] = derivative_rows[sample][rate_index]
        for name, coeffs, offset in recipes:
            vals[name] = offset + sum(c * row[j] for j, c in coeffs.items())
        out.append([vals[v] for v in order])
    return out, tuple(order)


def _evaluate(names, A, signed, threshold, trajs, derivs, tol) -> Candidate:
    model, recipes, constant_rates, influences = _build(names, A, signed)
    order = tuple(model.variables)

    per_kind: dict[str, list[float]] = {}
    for rows, derivative_rows in zip(trajs, derivs):
        augmented, aug_order = _augment(
            rows,
            derivative_rows,
            names,
            recipes,
            constant_rates,
        )
        # reorder augmented columns into the model's variable order
        pos = {n: i for i, n in enumerate(aug_order)}
        matrix = [[arow[pos[v]] for v in order] for arow in augmented]
        for rec in check_consistency(matrix, model):
            if rec.samples:
                per_kind.setdefault(rec.kind, []).append(rec.violation)

    all_viol = [v for vs in per_kind.values() for v in vs]
    max_v = max(all_viol) if all_viol else 0.0
    mean_v = (sum(all_viol) / len(all_viol)) if all_viol else 0.0
    worst = None
    if per_kind:
        worst = max(per_kind, key=lambda k: max(per_kind[k]))
    return Candidate(
        model, influences, threshold, max_v <= tol, max_v, mean_v, worst
    )


# --- fitting ---------------------------------------------------------------


def _fit(trajs, derivs, V, coefficient_atol, coefficient_rtol):
    """Pooled least-squares fit dx_i ~ sum_j a_ij x_j + b_i; returns the
    coefficient matrix A and a confidence matrix (t-like statistic)."""
    xs: list[list[float]] = []
    ds: list[list[float]] = []
    for rows, drow in zip(trajs, derivs):
        xs.extend(rows)
        ds.extend(drow)
    N = len(xs)
    if N < V + 2:
        raise ValueError(
            f"need at least {V + 2} pooled samples for {V} variables, got {N}"
        )
    cols = V + 1
    coefficients = _fit_coefficients(xs, ds, 1e-9)
    fitted_signs = _coefficient_signs(
        coefficients,
        xs,
        ds,
        coefficient_atol,
        coefficient_rtol,
    )

    def std(col):
        mean = sum(col) / len(col)
        return (sum((v - mean) ** 2 for v in col) / len(col)) ** 0.5

    A = [row[:] for row in coefficients]
    conf = [[0.0] * V for _ in range(V)]
    for i in range(V):
        w = coefficients[i]
        residuals = [
            ds[n][i] - sum(w[j] * xs[n][j] for j in range(V)) - w[V] for n in range(N)
        ]
        res_scale = (sum(r * r for r in residuals) / max(N - cols, 1)) ** 0.5
        for j in range(V):
            spread = std([xs[n][j] for n in range(N)])
            if fitted_signs[i][j] == 0:
                A[i][j] = 0.0
                continue
            conf[i][j] = abs(w[j]) * spread * (N**0.5) / (res_scale + 1e-30)
    return A, conf


def _ladder(conf) -> list[float]:
    vals = sorted({conf[i][j] for i in range(len(conf)) for j in range(len(conf))})
    levels = [1e-30] + [v + 1e-9 for v in vals]  # +eps: threshold just above each
    return sorted(set(levels), reverse=True)


# --- inputs ----------------------------------------------------------------


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def _to_rows(x, V) -> list[list[float]]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    rows = [[float(v) for v in row] for row in x]
    if rows and len(rows[0]) != V:
        raise ValueError(f"expected {V} columns, got {len(rows[0])}")
    return rows


def _as_trajectories(observations, V) -> list[list[list[float]]]:
    if hasattr(observations, "tolist"):
        observations = observations.tolist()
    seq = list(observations)
    if not seq:
        raise ValueError("at least one trajectory is required")
    first = seq[0]
    if first and not hasattr(first[0], "__len__"):  # a single (T, V) trajectory
        return [_to_rows(seq, V)]
    return [_to_rows(t, V) for t in seq]


def _as_derivatives(derivatives, trajs) -> list[list[list[float]]]:
    if derivatives is not None:
        if hasattr(derivatives, "tolist"):
            derivatives = derivatives.tolist()
        seq = list(derivatives)
        if seq and seq[0] and not hasattr(seq[0][0], "__len__"):
            seq = [seq]
        derivs = [[[float(v) for v in row] for row in d] for d in seq]
        if len(derivs) != len(trajs):
            raise ValueError("derivatives must match the number of trajectories")
        for d, rows in zip(derivs, trajs):
            if len(d) != len(rows):
                raise ValueError("each derivative array must match its trajectory length")
        return derivs
    # forward finite differences; align state samples with their differences
    out = []
    for k, rows in enumerate(trajs):
        if len(rows) < 2:
            raise ValueError("each trajectory needs at least 2 samples")
        out.append(
            [
                [rows[t + 1][j] - rows[t][j] for j in range(len(rows[t]))]
                for t in range(len(rows) - 1)
            ]
        )
        trajs[k] = rows[:-1]
    return out
