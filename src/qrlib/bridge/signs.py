"""Sign-structure intake and estimation (docs/host-integration.md, Surface 2)
and the downward consistency checker (Surface 6).

- :func:`model_from_signs`: compile an interaction sign matrix
  ``S[i][j] = sign of d(dx_i/dt)/dx_j`` — however the host obtained it —
  into a qualitative model (Deriv + M+/M- constraints through named
  auxiliary variables). ``UNKNOWN`` entries stay unconstrained (sound).
- :func:`estimate_signs`: estimate the matrix from ``(x, dx/dt)`` data with
  per-entry confidences (linear least-squares first cut: exact for linear
  systems, average monotonicity for monotone nonlinear ones; it asserts
  signs or UNKNOWN, never a confident zero — see docs/open-questions.md #10).
- :func:`check_consistency`: per-constraint violation masses of a model
  against sampled data — does ``outflow`` actually increase with ``level``?

Soundness note on :func:`model_from_signs`: a multi-input row compiles to a
*sum of single-variable monotone terms*. Sign-wise this imposes exactly the
qualitative constraint the sign matrix justifies (the direction of dx_i/dt
is a qualitative sum of S[i][j]-signed influences); no zero-crossing
corresponding values are asserted for the terms, because a sign matrix
does not pin where the influences vanish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..constraints import Add, Constant, Deriv, MMinus, MPlus
from ..model import Model
from ..quantity import Landmark

__all__ = [
    "UNKNOWN",
    "model_from_signs",
    "estimate_signs",
    "signs_with_threshold",
    "ConsistencyRecord",
    "check_consistency",
]

UNKNOWN = None  # sign-matrix entry: dependence direction not known


def model_from_signs(
    names: Sequence[str],
    S: Sequence[Sequence[int | None]],
    *,
    name: str = "signed",
) -> Model:
    """Compile an interaction sign matrix into a qualitative model.

    ``S[i][j]`` is the sign of ``d(dx_i/dt)/dx_j``: +1, -1, 0 (no
    dependence), or ``UNKNOWN`` (None — an unconstrained influence).
    State variables get unbounded spaces around ``0``; auxiliary variables
    are ``d_<x>`` (the derivative of x), ``t_<x>_<y>`` (y's influence term
    on x), and ``s_<x>_<k>`` (partial sums). An all-zero row compiles to
    ``Constant(x)``.
    """
    V = len(names)
    if len(S) != V or any(len(row) != V for row in S):
        raise ValueError(f"S must be {V}x{V} to match {V} variable names")
    m = Model(name)
    for n in names:
        m.variable(n, landmarks=(Landmark("0", value=0.0),), unbounded=True)

    def aux(n: str) -> str:
        m.variable(n, landmarks=(Landmark("0", value=0.0),), unbounded=True)
        return n

    for i, xi in enumerate(names):
        terms: list[tuple[str, int | None]] = [
            (names[j], S[i][j]) for j in range(V) if S[i][j] != 0
        ]
        if not terms:
            m.constrain(Constant(xi))
            continue
        term_vars: list[str] = []
        for xj, sign in terms:
            if len(terms) == 1:
                tname = aux(f"d_{xi}")
            else:
                tname = aux(f"t_{xi}_{xj}")
            if sign == 1:
                m.constrain(MPlus(xj, tname))
            elif sign == -1:
                m.constrain(MMinus(xj, tname))
            # UNKNOWN: the term exists but its relation to xj is unconstrained
            term_vars.append(tname)
        if len(term_vars) == 1:
            d_var = term_vars[0]
        else:
            acc = term_vars[0]
            for k, t in enumerate(term_vars[1:], start=1):
                is_last = k == len(term_vars) - 1
                nxt = aux(f"d_{xi}" if is_last else f"s_{xi}_{k}")
                m.constrain(Add(acc, t, nxt))
                acc = nxt
            d_var = acc
        m.constrain(Deriv(xi, d_var))
    return m


def estimate_signs(
    x,
    dx,
    *,
    ridge: float = 1e-9,
) -> tuple[list[list[int]], list[list[float]]]:
    """Estimate the interaction sign matrix from samples.

    ``x``: (N, V) state samples; ``dx``: (N, V) matching derivative samples
    (a host's vector-field evaluations, or finite differences). Fits
    ``dx_i ~ sum_j a_ij x_j + b_i`` by least squares and returns
    ``(signs, confidence)`` where ``signs[i][j] = sign(a_ij)`` and
    ``confidence[i][j]`` is a t-like statistic (|a_ij| * spread of x_j *
    sqrt(N) / residual scale). Threshold with :func:`signs_with_threshold`.
    Exact for linear systems; average monotonicity for nonlinear ones.
    """
    xs = _to_matrix(x)
    ds = _to_matrix(dx)
    if len(xs) != len(ds):
        raise ValueError(f"{len(xs)} state samples but {len(ds)} derivative samples")
    N, V = len(xs), len(xs[0])
    if N < V + 2:
        raise ValueError(f"need at least {V + 2} samples for {V} variables, got {N}")

    # design matrix with bias column; normal equations with a tiny ridge
    cols = V + 1
    A = [[0.0] * cols for _ in range(cols)]
    for row in xs:
        ext = list(row) + [1.0]
        for a in range(cols):
            for b in range(cols):
                A[a][b] += ext[a] * ext[b]
    for a in range(cols):
        A[a][a] += ridge

    def std(col: list[float]) -> float:
        mean = sum(col) / len(col)
        return (sum((v - mean) ** 2 for v in col) / len(col)) ** 0.5

    signs = [[0] * V for _ in range(V)]
    confidence = [[0.0] * V for _ in range(V)]
    for i in range(V):
        b = [0.0] * cols
        for row, drow in zip(xs, ds):
            ext = list(row) + [1.0]
            for a in range(cols):
                b[a] += ext[a] * drow[i]
        w = _solve([r[:] for r in A], b[:])
        residuals = [
            ds[n][i] - sum(w[j] * xs[n][j] for j in range(V)) - w[V]
            for n in range(N)
        ]
        res_scale = (sum(r * r for r in residuals) / max(N - cols, 1)) ** 0.5
        for j in range(V):
            spread = std([xs[n][j] for n in range(N)])
            signs[i][j] = (w[j] > 0) - (w[j] < 0)
            confidence[i][j] = (
                abs(w[j]) * spread * (N**0.5) / (res_scale + 1e-30)
            )
    return signs, confidence


def signs_with_threshold(
    signs: list[list[int]], confidence: list[list[float]], threshold: float = 5.0
) -> list[list[int | None]]:
    """Sign matrix for :func:`model_from_signs`: entries below the
    confidence threshold become ``UNKNOWN`` (never a confident zero)."""
    return [
        [
            signs[i][j] if confidence[i][j] >= threshold else UNKNOWN
            for j in range(len(signs[i]))
        ]
        for i in range(len(signs))
    ]


@dataclass(frozen=True)
class ConsistencyRecord:
    """Violation mass of one constraint against sampled data."""

    constraint: object
    kind: str
    violation: float  # fraction of applicable samples/steps violating
    samples: int  # how many were applicable


def check_consistency(
    x,
    model: Model,
    *,
    eps_rel: float = 1e-3,
) -> tuple[ConsistencyRecord, ...]:
    """Check a model's constraints directly against trajectory data.

    ``x``: (T, V) samples in model variable order. Monotonic constraints
    check step-sign agreement; arithmetic constraints check residuals
    relative to the variables' scales; ``Deriv`` checks sign agreement of
    finite-difference derivatives with the derivative variable;
    ``Constant`` checks spread. Violation masses near 0 mean the data
    respects the structure.
    """
    from ..model import _KIND  # kind names shared with compilation

    rows = _to_matrix(x)
    T = len(rows)
    order = tuple(model.variables)
    col = {name: i for i, name in enumerate(order)}
    scale = {
        name: max(abs(rows[t][col[name]]) for t in range(T)) or 1.0
        for name in order
    }

    def steps(name: str) -> list[float]:
        i = col[name]
        return [rows[t + 1][i] - rows[t][i] for t in range(T - 1)]

    records = []
    for c in model.constraints:
        kind = _KIND[type(c)]
        bad = 0
        n = 0
        if kind in ("mplus", "mminus", "minus"):
            want = 1 if kind == "mplus" else -1
            eps_x = eps_rel * scale[c.x]
            eps_y = eps_rel * scale[c.y]
            for dx_step, dy_step in zip(steps(c.x), steps(c.y)):
                if abs(dx_step) < eps_x or abs(dy_step) < eps_y:
                    continue
                n += 1
                sx = (dx_step > 0) - (dx_step < 0)
                sy = (dy_step > 0) - (dy_step < 0)
                if sx * sy != want:
                    bad += 1
            if kind == "minus":  # y = -x is a value identity too
                for t in range(T):
                    n += 1
                    if abs(rows[t][col[c.x]] + rows[t][col[c.y]]) > eps_rel * scale[c.x]:
                        bad += 1
        elif kind == "add":
            tol = eps_rel * max(scale[c.x], scale[c.y], scale[c.z])
            for t in range(T):
                n += 1
                if abs(rows[t][col[c.x]] + rows[t][col[c.y]] - rows[t][col[c.z]]) > tol:
                    bad += 1
        elif kind == "mult":
            tol = eps_rel * max(scale[c.x] * scale[c.y], scale[c.z])
            for t in range(T):
                n += 1
                if abs(rows[t][col[c.x]] * rows[t][col[c.y]] - rows[t][col[c.z]]) > tol:
                    bad += 1
        elif kind == "deriv":
            eps_d = eps_rel * scale[c.x]
            eps_y = eps_rel * scale[c.y]
            for t, dx_step in enumerate(steps(c.x)):
                y_val = rows[t][col[c.y]]
                if abs(dx_step) < eps_d and abs(y_val) < eps_y:
                    continue
                n += 1
                sd = (dx_step > 0) - (dx_step < 0)
                sy = (y_val > eps_y) - (y_val < -eps_y)
                if sd != sy:
                    bad += 1
        elif kind == "constant":
            i = col[c.x]
            mean = sum(rows[t][i] for t in range(T)) / T
            for t in range(T):
                n += 1
                if abs(rows[t][i] - mean) > eps_rel * scale[c.x]:
                    bad += 1
        records.append(
            ConsistencyRecord(c, kind, (bad / n) if n else 0.0, n)
        )
    return tuple(records)


# --- small numerics --------------------------------------------------------


def _to_matrix(x) -> list[list[float]]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    rows = [[float(v) for v in row] for row in x]
    if not rows:
        raise ValueError("empty sample matrix")
    return rows


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (tiny systems)."""
    n = len(A)
    for k in range(n):
        pivot = max(range(k, n), key=lambda r: abs(A[r][k]))
        if abs(A[pivot][k]) < 1e-300:
            raise ValueError("singular normal equations")
        A[k], A[pivot] = A[pivot], A[k]
        b[k], b[pivot] = b[pivot], b[k]
        for r in range(k + 1, n):
            f = A[r][k] / A[k][k]
            for cc in range(k, n):
                A[r][cc] -= f * A[k][cc]
            b[r] -= f * b[k]
    out = [0.0] * n
    for k in range(n - 1, -1, -1):
        out[k] = (b[k] - sum(A[k][j] * out[j] for j in range(k + 1, n))) / A[k][k]
    return out
