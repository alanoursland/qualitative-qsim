"""Landmark intake and data-driven landmark proposal.

Hosts contribute candidate landmark values from equilibrium finders, guard
thresholds, or domain knowledge as :class:`LandmarkRecord`s;
:func:`harvest_into_model`
inserts them into a model's quantity spaces by numeric value, reporting
conflicts instead of guessing. :func:`propose_landmarks` runs the other way:
sustained steady stretches in trajectory data at unnamed values suggest
landmarks the host can accept or reject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..model import CompiledModel, Model, Variable
from ..quantity import Landmark, Qdir
from .abstraction import AbstractionConfig, _directions, _quantize, _to_rows

__all__ = ["LandmarkRecord", "harvest_into_model", "propose_landmarks"]


@dataclass(frozen=True)
class LandmarkRecord:
    """A candidate landmark for one variable, from any source."""

    variable: str
    name: str
    value: float | None = None
    lo: float | None = None
    hi: float | None = None


def harvest_into_model(model: Model, records: Sequence[LandmarkRecord]) -> Model:
    """A new model with the records' landmarks inserted by numeric value.

    Requirements (violations raise ``ValueError`` naming the conflict):
    records need numeric values; the target variable's existing landmarks
    must all carry values (otherwise ordering is ambiguous); a record equal
    in value or name to an existing landmark is a conflict; a value below a
    bounded-below space (or above a bounded-above one) is a conflict.
    """
    grouped: dict[str, list[LandmarkRecord]] = {}
    for rec in records:
        if rec.variable not in model.variables:
            raise ValueError(f"landmark record for unknown variable {rec.variable!r}")
        if rec.value is None:
            raise ValueError(
                f"record {rec.name!r} for {rec.variable!r} needs a numeric value "
                f"to be ordered into the quantity space"
            )
        grouped.setdefault(rec.variable, []).append(rec)

    new_vars: dict[str, Variable] = {}
    for var, variable in model.variables.items():
        space = variable.space
        for rec in sorted(grouped.get(var, []), key=lambda r: r.value):
            if any(lm.value is None for lm in space.landmarks):
                unvalued = [lm.name for lm in space.landmarks if lm.value is None]
                raise ValueError(
                    f"cannot order {rec.name!r} into {var!r}: existing "
                    f"landmarks without numeric values: {unvalued}"
                )
            if rec.name in space.effective_landmarks:
                raise ValueError(f"{var!r} already has a landmark named {rec.name!r}")
            clash = [lm.name for lm in space.landmarks if lm.value == rec.value]
            if clash:
                raise ValueError(
                    f"conflict: {rec.name!r}={rec.value} duplicates the value of "
                    f"{clash[0]!r} on {var!r}"
                )
            below = [lm.name for lm in space.landmarks if lm.value < rec.value]
            if not below:
                if not space.lower_unbounded:
                    raise ValueError(
                        f"conflict: {rec.name!r}={rec.value} lies below the "
                        f"bounded space of {var!r}"
                    )
                after = "-inf"
            else:
                after = below[-1]
                if len(below) == len(space.landmarks) and not space.upper_unbounded:
                    raise ValueError(
                        f"conflict: {rec.name!r}={rec.value} lies above the "
                        f"bounded space of {var!r}"
                    )
            space = space.insert_landmark(
                Landmark(rec.name, value=rec.value, lo=rec.lo, hi=rec.hi),
                after=after,
            )
        new_vars[var] = Variable(var, space)
    return Model(model.name, new_vars, list(model.constraints))


def propose_landmarks(
    xs,
    model: Model | CompiledModel,
    *,
    times=None,
    config: AbstractionConfig | None = None,
    min_dwell: int = 10,
    merge_tol: float | None = None,
) -> tuple[LandmarkRecord, ...]:
    """Suggest landmarks from steady stretches in trajectory data.

    ``xs`` is one trajectory (T, V) or a batch of them. A variable holding
    steady (direction STD) at an *unnamed* magnitude for at least
    ``min_dwell`` samples proposes its mean value; proposals across
    stretches and trajectories merge when within ``merge_tol`` (default:
    2% of the variable's observed range). Suggested names are
    ``var^0, var^1, ...`` — the host decides whether to accept them
    (e.g. via :func:`harvest_into_model`).
    """
    compiled = model.compile() if isinstance(model, Model) else model
    cfg = config or AbstractionConfig()
    V = len(compiled.var_order)
    trajectories = _as_batch(xs, V)
    ts_batch = times if times is not None else [None] * len(trajectories)

    means: dict[int, list[float]] = {vi: [] for vi in range(V)}
    ranges: dict[int, tuple[float, float]] = {}
    for x, t in zip(trajectories, ts_batch):
        rows = _to_rows(x, V)
        ts = (
            [float(v) for v in t]
            if t is not None
            else [float(i) for i in range(len(rows))]
        )
        ranks = _quantize(rows, compiled, cfg)
        dirs = _directions(rows, ts, cfg)
        for vi in range(V):
            lo = min(r[vi] for r in rows)
            hi = max(r[vi] for r in rows)
            plo, phi = ranges.get(vi, (lo, hi))
            ranges[vi] = (min(lo, plo), max(hi, phi))
            start = None
            for t_i in range(len(rows) + 1):
                steady = (
                    t_i < len(rows)
                    and dirs[t_i][vi] is Qdir.STD
                    and ranks[t_i][vi] % 2 == 1
                )
                if steady and start is None:
                    start = t_i
                elif not steady and start is not None:
                    if t_i - start >= min_dwell:
                        stretch = [rows[k][vi] for k in range(start, t_i)]
                        means[vi].append(sum(stretch) / len(stretch))
                    start = None

    records: list[LandmarkRecord] = []
    for vi in range(V):
        if not means[vi]:
            continue
        lo, hi = ranges[vi]
        tol = merge_tol if merge_tol is not None else 0.02 * (hi - lo)
        clusters: list[list[float]] = []
        for m in sorted(means[vi]):
            if clusters and m - clusters[-1][-1] <= tol:
                clusters[-1].append(m)
            else:
                clusters.append([m])
        var = compiled.var_order[vi]
        existing = set(compiled.spaces[vi].names)
        k = 0
        for cluster in clusters:
            while f"{var}^{k}" in existing:
                k += 1
            records.append(
                LandmarkRecord(var, f"{var}^{k}", sum(cluster) / len(cluster))
            )
            k += 1
    return tuple(records)


def _as_batch(xs, width: int):
    """Accept one (T, V) trajectory or a sequence of them."""
    if hasattr(xs, "tolist"):
        xs = xs.tolist()
    first = xs[0]
    if hasattr(first, "tolist"):
        return list(xs)
    try:
        float(first[0])  # (T, V): first row of a single trajectory
        return [xs]
    except (TypeError, ValueError):
        return list(xs)
