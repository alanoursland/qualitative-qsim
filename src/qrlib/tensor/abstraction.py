"""Batched trajectory abstraction (docs/gpu-tensorization.md §1: the
trajectory-batch axis — the embarrassingly parallel case).

The per-sample stages of the pipeline — quantization against landmark
values, direction estimation, and run-boundary detection — vectorize over
``(B, T, V)`` tensors. Run starts and their codes are gathered into a compact
``O(runs)`` representation on-device and copied to the host in bulk.
Debounce/emission remain the small Python view layer shared with the reference
implementation; they never inspect the full sample tensors.

Parity contract: identical results to
:func:`qrlib.bridge.abstraction.abstract_trajectory` — the tensor stages
mirror the reference arithmetic expression-for-expression in float64, so
ranks and directions match exactly, and the shared tail produces equal
:class:`AbstractedBehavior` values. The equivalence tests assert this on
the soundness-harness trajectories.
"""

from __future__ import annotations

import torch

from ..bridge.abstraction import (
    AbstractedBehavior,
    AbstractionConfig,
    _debounce,
    _emit,
)
from ..model import CompiledModel, Model
from ..quantity import Qdir

__all__ = ["quantize_batch", "directions_batch", "abstract_batch_tensor"]


def _normalize_modes(modes, B: int, T: int):
    """Return mode rows plus a CPU change mask.

    Region labels may be arbitrary Python objects, so their equality cannot
    generally run as a tensor operation. Normalize them once and construct one
    compact boolean mask; the no-mode hot path does no host work here.
    """
    if modes is None:
        return None, None
    raw = modes.tolist() if hasattr(modes, "tolist") else modes
    rows = [list(row) for row in raw]
    if len(rows) != B or any(len(row) != T for row in rows):
        raise ValueError(f"modes must have shape ({B}, {T})")
    changes = [
        [rows[b][t] != rows[b][t - 1] for t in range(1, T)]
        for b in range(B)
    ]
    return rows, torch.tensor(changes, dtype=torch.bool)


def _pack_runs_for_host(
    ranks: torch.Tensor,
    dirs: torch.Tensor,
    ts: torch.Tensor,
    change: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the ragged run stream on-device, then copy it to CPU in bulk.

    The integer result has one row per actual run:
    ``[batch, start, end, ranks[V], directions[V]]``. The time result carries
    only ``[time[start], time[end - 1]]`` for each run. This avoids both
    per-run device synchronization and a ``B*T`` timestamp copy.
    """
    B, T, V = ranks.shape
    starts_mask = torch.empty((B, T), dtype=torch.bool, device=ranks.device)
    starts_mask[:, 0] = True
    if T > 1:
        starts_mask[:, 1:] = change
    coords = starts_mask.nonzero(as_tuple=False)
    batches = coords[:, 0]
    starts = coords[:, 1]
    ends = torch.full_like(starts, T)
    if len(starts) > 1:
        same_batch = batches[:-1] == batches[1:]
        ends[:-1] = torch.where(same_batch, starts[1:], ends[:-1])

    run_ranks = ranks[batches, starts]
    run_dirs = dirs[batches, starts]
    packed = torch.cat(
        (coords, ends.unsqueeze(1), run_ranks, run_dirs), dim=1
    ).to(device="cpu")
    run_times = torch.stack(
        (ts[batches, starts], ts[batches, ends - 1]), dim=1
    ).to(device="cpu")
    assert packed.shape == (len(starts), 3 + 2 * V)
    return packed, run_times


def _time_bounds_from_run_endpoints(spans, endpoint_times):
    """Physical bounds using only times gathered at original run endpoints."""
    bounds = []
    for start, end in spans:
        if start < end:
            bounds.append((endpoint_times[start], endpoint_times[end - 1]))
        elif start == 0:
            bounds.append((endpoint_times[0], endpoint_times[0]))
        else:
            bounds.append((endpoint_times[start - 1], endpoint_times[start]))
    return tuple(bounds)


def quantize_batch(
    x: torch.Tensor, compiled: CompiledModel, atol: float
) -> torch.Tensor:
    """Magnitude ranks for ``(B, T, V)`` samples (long tensor, same shape).

    Same semantics as ``QuantitySpace.rank_of_value``: within ``atol`` of a
    landmark value -> at the (first such) landmark; otherwise the open
    interval located by counting strictly-smaller landmark values;
    out-of-space values raise."""
    B, T, V = x.shape
    out = torch.empty((B, T, V), dtype=torch.long, device=x.device)
    for vi in range(V):
        space = compiled.spaces[vi]
        missing = [lm.name for lm in space.landmarks if lm.value is None]
        if missing:
            raise ValueError(
                f"variable {compiled.var_order[vi]!r}: landmarks without "
                f"numeric values: {missing}"
            )
        vals = torch.tensor(
            [lm.value for lm in space.landmarks],
            dtype=x.dtype,
            device=x.device,
        )
        xv = x[..., vi]
        near = (xv.unsqueeze(-1) - vals).abs() <= atol  # (B, T, L)
        has = near.any(-1)
        first = near.to(torch.uint8).argmax(-1)  # first landmark within atol
        below = (vals < xv.unsqueeze(-1)).sum(-1)
        off = 1 if space.lower_unbounded else 0
        if not space.lower_unbounded:
            bad = (~has) & (below == 0)
            if bad.any():
                b, t = [int(i) for i in bad.nonzero()[0]]
                raise ValueError(
                    f"sample {t}, variable {compiled.var_order[vi]!r}: "
                    f"{x[b, t, vi].item()} lies below this space"
                )
        if not space.upper_unbounded:
            bad = (~has) & (below == len(space.landmarks))
            if bad.any():
                b, t = [int(i) for i in bad.nonzero()[0]]
                raise ValueError(
                    f"sample {t}, variable {compiled.var_order[vi]!r}: "
                    f"{x[b, t, vi].item()} lies above this space"
                )
        rank = torch.where(has, 2 * (first + off), 2 * (below - 1 + off) + 1)
        out[..., vi] = rank
    return out


def directions_batch(
    x: torch.Tensor, ts: torch.Tensor, cfg: AbstractionConfig
) -> torch.Tensor:
    """Direction codes (0=DEC, 1=STD, 2=INC) for ``(B, T, V)`` samples.

    Mirrors the reference estimator expression-for-expression: central
    differences, second-order one-sided endpoints, and the relative
    threshold with its value-scale floor."""
    B, T, V = x.shape
    d = torch.zeros_like(x)
    if T >= 2:
        tspan = (ts[:, 2:] - ts[:, :-2]).unsqueeze(-1)
        if T >= 3:
            d[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / tspan

            def forward3(x0, x1, x2, h1, h2):
                return (
                    -(2 * h1 + h2) / (h1 * (h1 + h2)) * x0
                    + (h1 + h2) / (h1 * h2) * x1
                    - h1 / (h2 * (h1 + h2)) * x2
                )

            h1 = (ts[:, 1] - ts[:, 0]).unsqueeze(-1)
            h2 = (ts[:, 2] - ts[:, 1]).unsqueeze(-1)
            d[:, 0] = forward3(x[:, 0], x[:, 1], x[:, 2], h1, h2)
            g1 = (ts[:, -1] - ts[:, -2]).unsqueeze(-1)
            g2 = (ts[:, -2] - ts[:, -3]).unsqueeze(-1)
            d[:, -1] = -forward3(x[:, -1], x[:, -2], x[:, -3], g1, g2)
        else:
            step = ((x[:, 1] - x[:, 0]) / (ts[:, 1] - ts[:, 0]).unsqueeze(-1))
            d[:, 0] = step
            d[:, 1] = step

    eps = torch.full(
        (B, 1, V), cfg.direction_eps, dtype=x.dtype, device=x.device
    )
    if cfg.eps_relative:
        max_deriv = d.abs().amax(dim=1, keepdim=True)
        scale = x.abs().amax(dim=1, keepdim=True)
        duration = (ts[:, -1] - ts[:, 0]).reshape(B, 1, 1)
        duration = torch.where(
            duration == 0, torch.ones_like(duration), duration
        )
        eps = cfg.direction_eps * torch.maximum(max_deriv, scale / duration)
    codes = torch.full(
        (B, T, V), int(Qdir.STD), dtype=torch.long, device=x.device
    )
    codes[d > eps] = int(Qdir.INC)
    codes[d < -eps] = int(Qdir.DEC)
    return codes


def abstract_batch_tensor(
    x,
    model: Model | CompiledModel,
    *,
    times=None,
    modes=None,
    config: AbstractionConfig | None = None,
) -> tuple[AbstractedBehavior, ...]:
    """Batched abstraction: ``x`` is ``(B, T, V)`` (or ``(T, V)`` for a
    single trajectory); ``times`` broadcasts ``(T,)`` or ``(B, T)``;
    ``modes`` is an optional per-trajectory list of per-sample region
    labels. Results equal the reference per-trajectory pipeline."""
    compiled = model.compile() if isinstance(model, Model) else model
    cfg = config or AbstractionConfig()
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, T, V = x.shape
    if V != len(compiled.var_order):
        raise ValueError(f"trajectories have {V} columns, model has {len(compiled.var_order)}")
    if T == 0:
        raise ValueError("empty trajectories")
    if times is None:
        ts = (
            torch.arange(T, dtype=torch.float64, device=x.device)
            .unsqueeze(0)
            .expand(B, T)
        )
    else:
        ts = torch.as_tensor(times, dtype=torch.float64, device=x.device)
        ts = ts.unsqueeze(0).expand(B, T) if ts.dim() == 1 else ts
    if ts.shape != (B, T):
        raise ValueError(f"times must have shape ({T},) or ({B}, {T})")

    ranks = quantize_batch(x, compiled, cfg.landmark_atol)
    dirs = directions_batch(x, ts, cfg)
    mode_rows, mode_change = _normalize_modes(modes, B, T)
    # Run boundaries and row gathering stay in tensor land. Python receives
    # one compact O(runs) stream, not B*T samples or per-run device views.
    if T > 1:
        change = ((ranks[:, 1:] != ranks[:, :-1]) | (dirs[:, 1:] != dirs[:, :-1])).any(-1)
        if mode_change is not None:
            change |= mode_change.to(device=x.device)
    else:
        change = torch.zeros((B, 0), dtype=torch.bool, device=x.device)
    packed, packed_times = _pack_runs_for_host(ranks, dirs, ts, change)

    runs_by_batch = [[] for _ in range(B)]
    endpoint_times = [dict() for _ in range(B)]
    for record, time_pair in zip(packed.tolist(), packed_times.tolist()):
        b, start, end = record[:3]
        row_r = record[3 : 3 + V]
        row_d = record[3 + V :]
        code = tuple((row_r[v], Qdir(row_d[v])) for v in range(V))
        mode = mode_rows[b][start] if mode_rows is not None else None
        runs_by_batch[b].append([(code, mode), start, end])
        endpoint_times[b][start] = time_pair[0]
        endpoint_times[b][end - 1] = time_pair[1]

    out = []
    for b, runs in enumerate(runs_by_batch):
        runs = _debounce(runs, cfg.debounce)
        states, spans, regions = _emit(runs, compiled.var_order)
        time_bounds = _time_bounds_from_run_endpoints(
            spans, endpoint_times[b]
        )
        out.append(
            AbstractedBehavior(
                states, spans, compiled.var_order, cfg, regions, time_bounds
            )
        )
    return tuple(out)
