"""Batched trajectory abstraction (docs/gpu-tensorization.md §1: the
trajectory-batch axis — the embarrassingly parallel case).

The per-sample stages of the pipeline — quantization against landmark
values and direction estimation — vectorize over ``(B, T, V)`` tensors.
Segmentation/debounce/emission are inherently ragged and reuse the
reference implementation per trajectory (their cost is O(runs), not
O(samples)).

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


def quantize_batch(
    x: torch.Tensor, compiled: CompiledModel, atol: float
) -> torch.Tensor:
    """Magnitude ranks for ``(B, T, V)`` samples (long tensor, same shape).

    Same semantics as ``QuantitySpace.rank_of_value``: within ``atol`` of a
    landmark value -> at the (first such) landmark; otherwise the open
    interval located by counting strictly-smaller landmark values;
    out-of-space values raise."""
    B, T, V = x.shape
    out = torch.empty((B, T, V), dtype=torch.long)
    for vi in range(V):
        space = compiled.spaces[vi]
        missing = [lm.name for lm in space.landmarks if lm.value is None]
        if missing:
            raise ValueError(
                f"variable {compiled.var_order[vi]!r}: landmarks without "
                f"numeric values: {missing}"
            )
        vals = torch.tensor(
            [lm.value for lm in space.landmarks], dtype=torch.float64
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

    eps = torch.full((B, 1, V), cfg.direction_eps, dtype=torch.float64)
    if cfg.eps_relative:
        max_deriv = d.abs().amax(dim=1, keepdim=True)
        scale = x.abs().amax(dim=1, keepdim=True)
        duration = (ts[:, -1] - ts[:, 0]).reshape(B, 1, 1)
        duration = torch.where(
            duration == 0, torch.ones_like(duration), duration
        )
        eps = cfg.direction_eps * torch.maximum(max_deriv, scale / duration)
    codes = torch.full((B, T, V), int(Qdir.STD), dtype=torch.long)
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
    if times is None:
        ts = torch.arange(T, dtype=torch.float64).unsqueeze(0).expand(B, T)
    else:
        ts = torch.as_tensor(times, dtype=torch.float64)
        ts = ts.unsqueeze(0).expand(B, T) if ts.dim() == 1 else ts

    ranks = quantize_batch(x, compiled, cfg.landmark_atol)
    dirs = directions_batch(x, ts, cfg)
    # run boundaries in tensor land, so Python only touches O(runs), not O(T)
    if T > 1:
        change = ((ranks[:, 1:] != ranks[:, :-1]) | (dirs[:, 1:] != dirs[:, :-1])).any(-1)
    else:
        change = torch.zeros((B, 0), dtype=torch.bool)

    out = []
    for b in range(B):
        starts = [0] + (change[b].nonzero().flatten() + 1).tolist()
        mo = list(modes[b]) if modes is not None else None
        if mo is not None:
            starts = sorted(
                set(starts) | {t for t in range(1, T) if mo[t] != mo[t - 1]}
            )
        runs = []
        for i, s0 in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else T
            row_r = ranks[b, s0].tolist()
            row_d = dirs[b, s0].tolist()
            code = tuple((row_r[v], Qdir(row_d[v])) for v in range(V))
            runs.append([(code, mo[s0] if mo is not None else None), s0, end])
        runs = _debounce(runs, cfg.debounce)
        states, spans, regions = _emit(runs, compiled.var_order)
        out.append(
            AbstractedBehavior(states, spans, compiled.var_order, cfg, regions)
        )
    return tuple(out)
