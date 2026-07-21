"""Trajectory abstraction: numeric trajectories -> qualitative behaviors.

The upward bridge (docs/numeric-bridge.md): quantize sampled trajectories
against landmark values, estimate directions, segment into runs, debounce,
and emit a QSIM-style alternating point/interval state sequence with the
mapping back to sample indices.

This is the **reference implementation**: pure Python, one trajectory at a
time, exact and readable. The batched/tensorized version (phase 5) must
reproduce it. Inputs accept any array-like — nested sequences, numpy
arrays, torch tensors (anything with ``tolist()``).

Semantics notes (docs/numeric-bridge.md, gotchas):

- Landmark crossings between samples are *synthesized* as point states
  (magnitude jumps of one landmark); a jump across more than one landmark
  raises — that is undersampling, and silently inventing the intermediate
  history would be dishonest.
- The abstraction parameters are semantically meaningful (they define
  "steady" and "at the landmark") and travel with every result.
- A finite sample window ends mid-behavior: asymptotic approaches to
  equilibrium never produce the quiescent point itself (that is the t->inf
  limit). Coverage checking treats observed behaviors as prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import CompiledModel, Model
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag

__all__ = ["AbstractionConfig", "AbstractedBehavior", "abstract_trajectory", "abstract_batch"]


@dataclass(frozen=True)
class AbstractionConfig:
    """Parameters defining the numeric -> qualitative mapping.

    - ``landmark_atol``: absolute tolerance band around a landmark value
      within which a sample counts as *at* the landmark. Keep it tiny
      relative to sampling steps: transversal crossings between samples are
      synthesized as point states and need no in-band samples, while a
      large band creates artificial at-landmark dwells whose entry/exit
      times differ across constrained variables.
    - ``direction_eps``: threshold on the estimated derivative below which
      the direction is STD. With ``eps_relative`` (default) it is a
      fraction of the per-variable maximum |derivative| over the
      trajectory, making it scale-free.
    - ``debounce``: interior runs shorter than this many samples are
      dropped (numeric chatter around crossings and extrema).
    """

    landmark_atol: float = 1e-9
    direction_eps: float = 1e-4
    eps_relative: bool = True
    debounce: int = 3


@dataclass(frozen=True)
class AbstractedBehavior:
    """A trajectory's qualitative behavior plus its provenance.

    ``states`` alternate point/interval; ``spans[i]`` is the half-open
    sample-index range ``[start, end)`` the state was observed over
    (synthesized crossing points have zero width at the boundary index).
    Magnitude ranks are relative to the *model's* (root) quantity spaces.
    ``regions[i]`` carries the mode channel's region label for each state
    when one was provided (None per state otherwise); a synthesized
    boundary point belongs to the region being left.
    """

    states: tuple[QState, ...]
    spans: tuple[tuple[int, int], ...]
    var_order: tuple[str, ...]
    config: AbstractionConfig
    regions: tuple[str | None, ...] = ()


def abstract_trajectory(
    x,
    model: Model | CompiledModel,
    *,
    times=None,
    modes=None,
    config: AbstractionConfig | None = None,
) -> AbstractedBehavior:
    """Abstract one trajectory of shape (T, V), columns in model order.

    ``modes`` is the optional mode channel: one region label per sample
    (for hybrid executions). Mode changes force segment boundaries and the
    emitted states carry their region labels."""
    compiled = model.compile() if isinstance(model, Model) else model
    cfg = config or AbstractionConfig()
    rows = _to_rows(x, len(compiled.var_order))
    ts = (
        [float(t) for t in (times.tolist() if hasattr(times, "tolist") else times)]
        if times is not None
        else [float(i) for i in range(len(rows))]
    )
    if len(ts) != len(rows):
        raise ValueError(f"{len(rows)} samples but {len(ts)} timestamps")
    if modes is not None:
        modes = list(modes.tolist() if hasattr(modes, "tolist") else modes)
        if len(modes) != len(rows):
            raise ValueError(f"{len(rows)} samples but {len(modes)} mode labels")

    ranks = _quantize(rows, compiled, cfg)
    dirs = _directions(rows, ts, cfg)
    codes = [
        tuple((ranks[t][v], dirs[t][v]) for v in range(len(compiled.var_order)))
        for t in range(len(rows))
    ]
    run_modes = modes if modes is not None else [None] * len(rows)
    runs = _runs(list(zip(codes, run_modes)))
    runs = _debounce(runs, cfg.debounce)
    states, spans, regions = _emit(runs, compiled.var_order)
    return AbstractedBehavior(states, spans, compiled.var_order, cfg, regions)


def abstract_batch(
    xs, model: Model | CompiledModel, *, times=None, modes=None, config=None
) -> tuple[AbstractedBehavior, ...]:
    """Abstract a batch of trajectories (loop reference; tensorized later)."""
    ts = times if times is not None else [None] * len(xs)
    ms = modes if modes is not None else [None] * len(xs)
    return tuple(
        abstract_trajectory(x, model, times=t, modes=mo, config=config)
        for x, t, mo in zip(xs, ts, ms)
    )


# --- pipeline stages -------------------------------------------------------


def _to_rows(x, width: int) -> list[list[float]]:
    if hasattr(x, "tolist"):
        x = x.tolist()
    rows = [[float(v) for v in row] for row in x]
    if not rows:
        raise ValueError("empty trajectory")
    for t, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"sample {t} has {len(row)} values, expected {width}")
    return rows


def _quantize(rows, compiled: CompiledModel, cfg: AbstractionConfig):
    out = []
    for t, row in enumerate(rows):
        encoded = []
        for vi, val in enumerate(row):
            space = compiled.spaces[vi]
            try:
                encoded.append(space.rank_of_value(val, atol=cfg.landmark_atol))
            except ValueError as e:
                raise ValueError(
                    f"sample {t}, variable {compiled.var_order[vi]!r}: {e}"
                ) from e
        out.append(encoded)
    return out


def _directions(rows, ts, cfg: AbstractionConfig):
    T, V = len(rows), len(rows[0])
    derivs = [[0.0] * V for _ in range(T)]
    def forward3(x0, x1, x2, h1, h2):
        # second-order one-sided difference (nonuniform spacing): first-order
        # endpoint estimates are O(h)-biased exactly at critical points,
        # misreading directions at initial/final instants
        return (
            -(2 * h1 + h2) / (h1 * (h1 + h2)) * x0
            + (h1 + h2) / (h1 * h2) * x1
            - h1 / (h2 * (h1 + h2)) * x2
        )

    for v in range(V):
        for t in range(T):
            if T == 1:
                d = 0.0
            elif t == 0:
                if T >= 3:
                    d = forward3(
                        rows[0][v], rows[1][v], rows[2][v],
                        ts[1] - ts[0], ts[2] - ts[1],
                    )
                else:
                    d = (rows[1][v] - rows[0][v]) / (ts[1] - ts[0])
            elif t == T - 1:
                if T >= 3:
                    d = -forward3(
                        rows[-1][v], rows[-2][v], rows[-3][v],
                        ts[-1] - ts[-2], ts[-2] - ts[-3],
                    )
                else:
                    d = (rows[-1][v] - rows[-2][v]) / (ts[-1] - ts[-2])
            else:
                d = (rows[t + 1][v] - rows[t - 1][v]) / (ts[t + 1] - ts[t - 1])
            derivs[t][v] = d
    dirs = [[Qdir.STD] * V for _ in range(T)]
    duration = (ts[-1] - ts[0]) or 1.0
    for v in range(V):
        eps = cfg.direction_eps
        if cfg.eps_relative:
            max_deriv = max(abs(derivs[t][v]) for t in range(T))
            # floor at value-scale/duration: an (essentially) constant
            # variable has max_deriv ~ rounding noise, and a threshold
            # relative to noise would hallucinate directions
            scale = max(abs(rows[t][v]) for t in range(T))
            eps *= max(max_deriv, scale / duration)
        for t in range(T):
            d = derivs[t][v]
            dirs[t][v] = Qdir.INC if d > eps else Qdir.DEC if d < -eps else Qdir.STD
    return dirs


def _runs(codes):
    runs: list[list] = []  # [code, start, end)
    for t, code in enumerate(codes):
        if runs and runs[-1][0] == code:
            runs[-1][2] = t + 1
        else:
            runs.append([code, t, t + 1])
    return runs


def _debounce(runs, k: int):
    """Drop interior runs shorter than k samples (never the first/last),
    re-merging equal neighbors, to a fixpoint."""
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(runs) - 1:
            if runs[i][2] - runs[i][1] < k:
                del runs[i]
                if 0 < i < len(runs) and runs[i - 1][0] == runs[i][0]:
                    runs[i - 1][2] = runs[i][2]
                    del runs[i]
                changed = True
            else:
                i += 1
    return runs


def _is_point_run(entry) -> bool:
    # a variable cannot sit AT a landmark while moving over a time interval:
    # such a run is a crossing dwell — an instant, qualitatively
    code, _mode = entry
    return any(m % 2 == 0 and d is not Qdir.STD for m, d in code)


def _synth_point(c1, c2):
    """Per-variable value at the instant between two interval runs."""
    out = []
    for (m1, d1), (m2, d2) in zip(c1, c2):
        if m1 == m2:
            out.append((m1, d1 if d1 == d2 else Qdir.STD))
        elif abs(m1 - m2) == 1:
            m = m1 if m1 % 2 == 0 else m2
            out.append((m, d1 if d1 == d2 else Qdir.STD))
        elif abs(m1 - m2) == 2 and m1 % 2 == 1:
            d = d1 if d1 == d2 else (Qdir.INC if m2 > m1 else Qdir.DEC)
            out.append(((m1 + m2) // 2, d))
        else:
            raise ValueError(
                f"undersampled trajectory: magnitude jumped {m1} -> {m2} "
                f"between samples (crossed more than one landmark); "
                f"sample more finely"
            )
    return tuple(out)


def _synth_interval(c1, c2):
    """Best-effort interval between two adjacent point runs (rare)."""
    out = []
    for (m1, d1), (m2, d2) in zip(c1, c2):
        if m1 == m2:
            d = d1 if d1 == d2 else (d2 if d1 is Qdir.STD else d1 if d2 is Qdir.STD else Qdir.STD)
            out.append((m1, d))
        elif abs(m1 - m2) == 1:
            m = m1 if m1 % 2 == 1 else m2
            d = d2 if d2 is not Qdir.STD else d1
            out.append((m, d))
        elif abs(m1 - m2) == 2 and m1 % 2 == 0:
            out.append(((m1 + m2) // 2, Qdir.INC if m2 > m1 else Qdir.DEC))
        else:
            raise ValueError(
                f"undersampled trajectory: magnitude jumped {m1} -> {m2} "
                f"between instants; sample more finely"
            )
    return tuple(out)


def _emit(runs, var_order):
    states: list[QState] = []
    spans: list[tuple[int, int]] = []
    regions: list[str | None] = []

    def emit(code, tag, span, mode):
        values = {
            name: QVal(m, d) for name, (m, d) in zip(var_order, code)
        }
        states.append(QState.from_dict(values, tag))
        spans.append(span)
        regions.append(mode)

    prev_kind = None
    prev_code = None
    prev_mode = None
    for (code, mode), start, end in runs:
        kind = TimeTag.POINT if _is_point_run((code, mode)) else TimeTag.INTERVAL
        if prev_kind is not None and prev_kind == kind:
            # a synthesized boundary instant belongs to the region being left
            if kind is TimeTag.INTERVAL:
                emit(_synth_point(prev_code, code), TimeTag.POINT, (start, start), prev_mode)
            else:
                emit(_synth_interval(prev_code, code), TimeTag.INTERVAL, (start, start), prev_mode)
        emit(code, kind, (start, end), mode)
        prev_kind, prev_code, prev_mode = kind, code, mode

    assert all(
        states[i].time is not states[i + 1].time for i in range(len(states) - 1)
    ), "internal error: emitted states do not alternate point/interval"
    return tuple(states), tuple(spans), tuple(regions)
