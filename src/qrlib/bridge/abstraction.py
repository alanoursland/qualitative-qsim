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

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from ..engines import filters
from ..model import CompiledModel, Model
from ..quantity import Qdir, QVal
from ..state import QState, TimeTag

__all__ = [
    "AbstractionConfig",
    "CrossingEvent",
    "AbstractedBehavior",
    "abstract_trajectory",
    "abstract_batch",
]


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
      fraction of each variable's value scale divided by the shared
      trajectory duration. This preserves units without allowing one
      variable's early derivative spike to desensitize only that variable.
    - ``debounce``: interior runs shorter than this many samples are
      dropped (numeric chatter around crossings and extrema).
    - ``endpoint_extremum_ratio``: at a landmark endpoint, classify the
      instantaneous direction as steady when the second-order endpoint
      derivative is this fraction or less of the outgoing/incoming secant.
      This corrects finite-difference truncation at smooth extrema; set to
      zero to retain unmodified one-sided estimates.
    """

    landmark_atol: float = 1e-9
    direction_eps: float = 1e-4
    eps_relative: bool = True
    debounce: int = 3
    endpoint_extremum_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.landmark_atol < 0 or self.direction_eps < 0:
            raise ValueError("abstraction tolerances must be nonnegative")
        if self.debounce < 1:
            raise ValueError("debounce must be at least 1")
        if not 0 <= self.endpoint_extremum_ratio <= 1:
            raise ValueError("endpoint_extremum_ratio must be between 0 and 1")


@dataclass(frozen=True)
class CrossingEvent:
    """A solver-refined landmark crossing between trajectory samples.

    ``values`` is the complete state vector at ``time``, either in model
    variable order or as a mapping keyed by every model variable. Requiring
    the full vector keeps the bridge from inventing the other variables by
    interpolation at a nonlinear event. Multiple events may share a time and
    state when several landmarks are crossed simultaneously.
    """

    time: float
    variable: str
    landmark: str
    values: Sequence[float] | Mapping[str, float]
    mode: str | None = None


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
    ``time_bounds[i]`` is the physical-time support for an observed run.
    For a synthesized point its pair brackets the unknown crossing time;
    a refined crossing has equal lower and upper bounds.
    """

    states: tuple[QState, ...]
    spans: tuple[tuple[int, int], ...]
    var_order: tuple[str, ...]
    config: AbstractionConfig
    regions: tuple[str | None, ...] = ()
    time_bounds: tuple[tuple[float, float], ...] = ()


def abstract_trajectory(
    x,
    model: Model | CompiledModel,
    *,
    times=None,
    modes=None,
    crossings: Sequence[CrossingEvent] | None = None,
    config: AbstractionConfig | None = None,
) -> AbstractedBehavior:
    """Abstract one trajectory of shape (T, V), columns in model order.

    ``modes`` is the optional mode channel: one region label per sample
    (for hybrid executions). Mode changes force segment boundaries and the
    emitted states carry their region labels.

    ``crossings`` optionally supplies solver-refined states between samples.
    Each event must identify a numeric landmark and carry the complete state
    vector. The events are inserted before quantization and protected from
    debounce, so ordered crossings inside one coarse sample interval remain
    visible."""
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
    _validate_times(ts)
    if modes is not None:
        modes = list(modes.tolist() if hasattr(modes, "tolist") else modes)
        if len(modes) != len(rows):
            raise ValueError(f"{len(rows)} samples but {len(modes)} mode labels")

    source_indices: list[int | None] = list(range(len(rows)))
    protected: set[int] = set()
    if crossings:
        rows, ts, modes, source_indices, protected = _insert_crossings(
            rows, ts, modes, crossings, compiled, cfg
        )

    ranks = _quantize(rows, compiled, cfg)
    dirs = _directions(rows, ts, cfg, ranks=ranks)
    codes = [
        tuple((ranks[t][v], dirs[t][v]) for v in range(len(compiled.var_order)))
        for t in range(len(rows))
    ]
    run_modes = modes if modes is not None else [None] * len(rows)
    runs = _runs(list(zip(codes, run_modes)))
    runs = _debounce(runs, cfg.debounce, protected=protected)
    runs = _project_runs_consistent(runs, compiled)
    states, augmented_spans, regions = _emit(runs, compiled.var_order)
    states = _project_consistent(states, regions, compiled)
    spans = (
        _project_spans(augmented_spans, source_indices)
        if crossings
        else augmented_spans
    )
    time_bounds = _time_bounds(augmented_spans, ts)
    return AbstractedBehavior(
        states, spans, compiled.var_order, cfg, regions, time_bounds
    )


def abstract_batch(
    xs,
    model: Model | CompiledModel,
    *,
    times=None,
    modes=None,
    crossings=None,
    config=None,
) -> tuple[AbstractedBehavior, ...]:
    """Abstract a batch of trajectories (loop reference; tensorized later)."""
    ts = times if times is not None else [None] * len(xs)
    ms = modes if modes is not None else [None] * len(xs)
    cs = crossings if crossings is not None else [None] * len(xs)
    if len(ts) != len(xs) or len(ms) != len(xs) or len(cs) != len(xs):
        raise ValueError("batch times, modes, and crossings must match batch size")
    return tuple(
        abstract_trajectory(
            x, model, times=t, modes=mo, crossings=cr, config=config
        )
        for x, t, mo, cr in zip(xs, ts, ms, cs)
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


def _validate_times(ts: list[float]) -> None:
    if any(not isfinite(t) for t in ts):
        raise ValueError("timestamps must be finite")
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            raise ValueError("timestamps must be strictly increasing")


def _event_values(event: CrossingEvent, var_order: tuple[str, ...]) -> list[float]:
    values = event.values
    if isinstance(values, Mapping):
        missing = set(var_order) - set(values)
        extra = set(values) - set(var_order)
        if missing or extra:
            raise ValueError(
                f"crossing at {event.time}: state mapping keys must match model "
                f"variables (missing={sorted(missing)}, extra={sorted(extra)})"
            )
        row = [float(values[name]) for name in var_order]
    else:
        row = [float(value) for value in values]
        if len(row) != len(var_order):
            raise ValueError(
                f"crossing at {event.time}: state has {len(row)} values, "
                f"expected {len(var_order)}"
            )
    if any(not isfinite(value) for value in row):
        raise ValueError(f"crossing at {event.time}: state values must be finite")
    return row


def _insert_crossings(
    rows,
    ts,
    modes,
    crossings,
    compiled: CompiledModel,
    cfg: AbstractionConfig,
):
    events = list(crossings)
    if not events:
        return rows, ts, modes, list(range(len(rows))), set()
    if any(not isinstance(event, CrossingEvent) for event in events):
        raise TypeError("crossings entries must be CrossingEvent instances")
    events.sort(key=lambda event: float(event.time))

    grouped: list[tuple[float, list[float], str | None]] = []
    for event in events:
        time = float(event.time)
        if not isfinite(time) or not ts[0] < time < ts[-1]:
            raise ValueError(
                f"crossing time {time} must be finite and strictly inside "
                f"the sampled time range ({ts[0]}, {ts[-1]})"
            )
        insertion = bisect_left(ts, time)
        if ts[insertion] == time:
            raise ValueError(
                f"crossing time {time} coincides with an existing sample; "
                "put the refined state in the trajectory instead"
            )
        if event.variable not in compiled.var_order:
            raise ValueError(
                f"crossing at {time}: unknown variable {event.variable!r}"
            )
        vi = compiled.index(event.variable)
        space = compiled.spaces[vi]
        try:
            landmark = space.landmark(event.landmark)
        except KeyError as exc:
            raise ValueError(
                f"crossing at {time}: unknown landmark {event.landmark!r} "
                f"for variable {event.variable!r}"
            ) from exc
        if landmark.value is None:
            raise ValueError(
                f"crossing at {time}: landmark {event.landmark!r} has no "
                "numeric value"
            )
        row = _event_values(event, compiled.var_order)
        if abs(row[vi] - landmark.value) > cfg.landmark_atol:
            raise ValueError(
                f"crossing at {time}: {event.variable!r}={row[vi]} does not "
                f"match landmark {event.landmark!r}={landmark.value} within "
                f"landmark_atol={cfg.landmark_atol}"
            )
        if event.mode is not None and modes is None:
            raise ValueError("crossing mode requires a trajectory mode channel")
        mode = event.mode if event.mode is not None else (
            modes[insertion - 1] if modes is not None else None
        )
        if grouped and grouped[-1][0] == time:
            prior_row, prior_mode = grouped[-1][1], grouped[-1][2]
            if prior_mode != mode or any(
                abs(a - b) > cfg.landmark_atol
                for a, b in zip(prior_row, row)
            ):
                raise ValueError(
                    f"simultaneous crossings at {time} must share one state "
                    "and mode"
                )
        else:
            grouped.append((time, row, mode))

    augmented_rows: list[list[float]] = []
    augmented_ts: list[float] = []
    augmented_modes: list[str | None] | None = [] if modes is not None else None
    sources: list[int | None] = []
    protected: set[int] = set()
    event_index = 0
    for sample_index, sample_time in enumerate(ts):
        while event_index < len(grouped) and grouped[event_index][0] < sample_time:
            event_time, event_row, event_mode = grouped[event_index]
            protected.add(len(augmented_rows))
            augmented_rows.append(event_row)
            augmented_ts.append(event_time)
            sources.append(None)
            if augmented_modes is not None:
                augmented_modes.append(event_mode)
            event_index += 1
        augmented_rows.append(rows[sample_index])
        augmented_ts.append(sample_time)
        sources.append(sample_index)
        if augmented_modes is not None:
            augmented_modes.append(modes[sample_index])
    assert event_index == len(grouped)
    return augmented_rows, augmented_ts, augmented_modes, sources, protected


def _project_spans(spans, source_indices):
    projected = []
    for start, end in spans:
        originals = [
            source
            for source in source_indices[start:end]
            if source is not None
        ]
        if originals:
            projected.append((min(originals), max(originals) + 1))
        else:
            boundary = sum(
                source is not None for source in source_indices[:start]
            )
            projected.append((boundary, boundary))
    return tuple(projected)


def _time_bounds(spans, ts):
    bounds = []
    for start, end in spans:
        if start < end:
            bounds.append((ts[start], ts[end - 1]))
        elif start == 0:
            bounds.append((ts[0], ts[0]))
        elif start == len(ts):
            bounds.append((ts[-1], ts[-1]))
        else:
            bounds.append((ts[start - 1], ts[start]))
    return tuple(bounds)


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


def _directions(rows, ts, cfg: AbstractionConfig, *, ranks=None):
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

    if ranks is not None and T >= 3 and cfg.endpoint_extremum_ratio:
        ratio = cfg.endpoint_extremum_ratio
        for v in range(V):
            first_secant = (rows[1][v] - rows[0][v]) / (ts[1] - ts[0])
            if (
                ranks[0][v] % 2 == 0
                and abs(derivs[0][v]) <= ratio * abs(first_secant)
            ):
                derivs[0][v] = 0.0
            last_secant = (rows[-1][v] - rows[-2][v]) / (ts[-1] - ts[-2])
            if (
                ranks[-1][v] % 2 == 0
                and abs(derivs[-1][v]) <= ratio * abs(last_secant)
            ):
                derivs[-1][v] = 0.0
    dirs = [[Qdir.STD] * V for _ in range(T)]
    duration = (ts[-1] - ts[0]) or 1.0
    for v in range(V):
        eps = cfg.direction_eps
        if cfg.eps_relative:
            # One trajectory-wide time scale keeps related variables
            # commensurable. Scaling by each variable's own peak derivative
            # lets an early spike desensitize only that variable and can
            # assemble constraint-inconsistent joint directions.
            scale = max(abs(rows[t][v]) for t in range(T))
            eps *= scale / duration
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


def _debounce(runs, k: int, *, protected: set[int] | None = None):
    """Drop interior runs shorter than k samples (never the first/last),
    re-merging equal neighbors, to a fixpoint."""
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(runs) - 1:
            contains_protected = protected and any(
                sample in protected
                for sample in range(runs[i][1], runs[i][2])
            )
            if runs[i][2] - runs[i][1] < k and not contains_protected:
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


def _project_runs_consistent(runs, compiled: CompiledModel):
    """Project observed run directions, then coalesce newly equal neighbors."""
    projected = []
    cache = {}
    sequence = [item[0] for item in runs]
    sequence_alternatives = _sequence_alternatives(sequence)
    for index, ((code, region), start, end) in enumerate(runs):
        alternatives = sequence_alternatives[index]
        key = (code, region, alternatives)
        if key not in cache:
            state = QState.from_dict(
                {
                    name: QVal(magnitude, direction)
                    for name, (magnitude, direction) in zip(
                        compiled.var_order, code
                    )
                },
                TimeTag.INTERVAL,
            )
            repaired = _project_state_consistent(
                state, region, compiled, alternatives
            )
            cache[key] = tuple(
                (repaired[name].mag, repaired[name].dir)
                for name in compiled.var_order
            )
        repaired_code = cache[key]
        entry = [(repaired_code, region), start, end]
        if projected and projected[-1][0] == entry[0]:
            projected[-1][2] = end
        else:
            projected.append(entry)
    return projected


def _project_state_consistent(
    state: QState,
    region: str | None,
    compiled: CompiledModel,
    alternatives: tuple[tuple[Qdir, ...], ...],
) -> QState:
    try:
        active = compiled.constraints_of(
            region if region is not None else compiled.initial_region
        )
    except KeyError:
        # Mode labels are allowed to be host metadata rather than qrlib
        # region names; retain the model's initial-region semantics.
        active = compiled.constraints_of(compiled.initial_region)
    violation = filters.check_state(compiled, state, active)
    if violation is None:
        return state
    domains = []
    for index, name in enumerate(compiled.var_order):
        qv = state[name]
        domain = [qv]
        if qv.dir is Qdir.STD:
            domain.extend(
                QVal(qv.mag, direction)
                for direction in alternatives[index]
            )
        domains.append(domain)
    pruned = filters.prune_domains(compiled, domains, active)
    combo = (
        next(filters.assemble(compiled, pruned, active), None)
        if pruned is not None
        else None
    )
    if combo is None:
        return state
    return QState.from_dict(
        dict(zip(compiled.var_order, combo)), state.time
    )


def _project_consistent(states, regions, compiled: CompiledModel):
    """Repair threshold-borderline steady directions against model semantics.

    Numeric direction thresholds are per variable, so a monotone pair can
    otherwise be assembled as one moving variable plus one steady variable.
    Moving classifications remain trusted. A steady classification may be
    promoted only to a moving direction evidenced by an adjacent run/state,
    and only when needed to satisfy the active constraint set. If no such
    repair exists, the state is retained so the coverage oracle can diagnose
    genuinely model-inconsistent input.
    """
    out = []
    cache: dict[tuple, QState] = {}
    sequence = [
        (
            tuple(
                (state[name].mag, state[name].dir)
                for name in compiled.var_order
            ),
            region,
        )
        for state, region in zip(states, regions)
    ]
    sequence_alternatives = _sequence_alternatives(sequence)
    for index, (state, region) in enumerate(zip(states, regions)):
        alternatives = sequence_alternatives[index]
        key = (state, region, alternatives)
        if key in cache:
            out.append(cache[key])
            continue
        projected = _project_state_consistent(
            state, region, compiled, alternatives
        )
        cache[key] = projected
        out.append(projected)
    return tuple(out)


def _sequence_alternatives(sequence):
    """Nearest moving directions around each steady plateau, in linear time."""
    if not sequence:
        return ()
    width = len(sequence[0][0])
    alternatives: list[list[list[Qdir]]] = [
        [[] for _ in range(width)] for _ in sequence
    ]
    for indices in (range(len(sequence)), range(len(sequence) - 1, -1, -1)):
        last: list[Qdir | None] = [None] * width
        prior_region = object()
        for index in indices:
            code, region = sequence[index]
            if region != prior_region:
                last = [None] * width
                prior_region = region
            for variable, (_magnitude, direction) in enumerate(code):
                if direction is Qdir.STD:
                    if (
                        last[variable] is not None
                        and last[variable] not in alternatives[index][variable]
                    ):
                        alternatives[index][variable].append(last[variable])
                else:
                    last[variable] = direction
    return tuple(
        tuple(tuple(options) for options in row)
        for row in alternatives
    )
