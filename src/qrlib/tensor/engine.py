"""Tensorized successor filtering (docs/gpu-tensorization.md §3).

The combinatorial core of QSIM — prune per-variable candidate domains,
then filter the cross-product of interpretations — as batched tensor ops
over the compiled constraint tables. Semantics contract: for identical
inputs, :func:`filtered_combos` returns exactly the combinations the
reference pipeline (``filters.prune_domains`` + ``filters.assemble``)
yields, in the same (``itertools.product``) order — the engine's results
are bit-identical either way, which the equivalence tests assert on whole
behavior graphs.

``torch.cartesian_prod`` enumerates rows in the same lexicographic order
as ``itertools.product``; filtering a padded grid to per-state valid
entries preserves that order, so batching never reorders results.

Oversized interpretation products fall back to the reference generator
(the "fallback backtracker"): correctness never depends on the fast path.
"""

from __future__ import annotations

import torch

from ..engines import filters
from ..model import CompiledModel
from ..quantity import QVal
from .encoding import qcode, tables_for

__all__ = ["ASSEMBLE_CAP", "filtered_combos", "filtered_combos_batch"]

ASSEMBLE_CAP = 1 << 18  # interpretation-product size above which we fall back


def filtered_combos(
    frame: CompiledModel,
    domains: list[list[QVal]],
    active_idx: tuple[int, ...],
    *,
    telemetry: dict | None = None,
) -> list[tuple[QVal, ...]]:
    """All consistent complete assignments from per-variable candidate
    domains, under the region's active constraints."""
    ts = tables_for(frame)
    doms = [list(d) for d in domains]
    if any(not d for d in doms):
        return []
    if telemetry is not None:
        telemetry["input_product"] = _product_size(doms)

    # tuple + Waltz pruning via table lookups, to a fixpoint
    changed = True
    while changed:
        changed = False
        for ci in active_idx:
            cc = frame.constraints[ci]
            local = [doms[vi] for vi in cc.vars]
            grid = torch.cartesian_prod(
                *[torch.arange(len(d)) for d in local]
            ).reshape(-1, len(local))
            codes = torch.stack(
                [
                    torch.tensor([qcode(q) for q in d], dtype=torch.long)[
                        grid[:, k]
                    ]
                    for k, d in enumerate(local)
                ],
                dim=1,
            )
            ok = ts.mask(ci, codes)
            if not ok.any():
                return []
            for k, vi in enumerate(cc.vars):
                keep = sorted(set(grid[ok][:, k].tolist()))
                if len(keep) != len(doms[vi]):
                    doms[vi] = [doms[vi][i] for i in keep]
                    changed = True

    total = _product_size(doms)
    if telemetry is not None:
        telemetry["pruned_product"] = total
    if total > ASSEMBLE_CAP:  # fallback backtracker: correctness first
        if telemetry is not None:
            telemetry["fallback"] = "oversized_product"
        active = tuple(frame.constraints[i] for i in active_idx)
        return list(filters.assemble(frame, doms, active))

    grid = torch.cartesian_prod(*[torch.arange(len(d)) for d in doms]).reshape(
        -1, len(doms)
    )
    col_codes = [
        torch.tensor([qcode(q) for q in d], dtype=torch.long)[grid[:, v]]
        for v, d in enumerate(doms)
    ]
    keep = torch.ones(grid.shape[0], dtype=torch.bool)
    for ci in active_idx:
        cc = frame.constraints[ci]
        cand = torch.stack([col_codes[vi] for vi in cc.vars], dim=1)
        keep &= ts.mask(ci, cand)
        if not keep.any():
            return []
    return [
        tuple(doms[v][i] for v, i in enumerate(row))
        for row in grid[keep].tolist()
    ]


def _product_size(domains: list[list[QVal]]) -> int:
    total = 1
    for domain in domains:
        total *= len(domain)
    return total


def filtered_combos_batch(
    frame: CompiledModel,
    domains_list: list[list[list[QVal]]],
    active_idx: tuple[int, ...],
) -> list[list[tuple[QVal, ...]]]:
    """Batched frontier expansion: filter many states' candidate domains in
    one shot over a shared padded grid. Per-state results are identical to
    :func:`filtered_combos` (order included)."""
    B = len(domains_list)
    if B == 0:
        return []
    V = len(frame.var_order)
    if any(any(not d for d in doms) for doms in domains_list):
        return [
            [] if any(not d for d in doms) else filtered_combos(frame, doms, active_idx)
            for doms in domains_list
        ]
    K = max(len(d) for doms in domains_list for d in doms)
    if K**V > ASSEMBLE_CAP:
        return [filtered_combos(frame, doms, active_idx) for doms in domains_list]

    codes = torch.zeros((B, V, K), dtype=torch.long)
    lens = torch.zeros((B, V), dtype=torch.long)
    for b, doms in enumerate(domains_list):
        for v, d in enumerate(doms):
            lens[b, v] = len(d)
            for k, qv in enumerate(d):
                codes[b, v, k] = qcode(qv)

    grid = torch.cartesian_prod(*([torch.arange(K)] * V)).reshape(-1, V)  # (G, V)
    valid = (grid.unsqueeze(0) < lens.unsqueeze(1)).all(-1)  # (B, G)
    col_codes = [codes[:, v, :][:, grid[:, v]] for v in range(V)]  # each (B, G)

    ts = tables_for(frame)
    keep = valid
    for ci in active_idx:
        cc = frame.constraints[ci]
        cand = torch.stack([col_codes[vi] for vi in cc.vars], dim=-1)  # (B, G, a)
        mask = ts.mask(ci, cand.reshape(-1, cand.shape[-1])).reshape(B, -1)
        keep = keep & mask

    out: list[list[tuple[QVal, ...]]] = []
    for b, doms in enumerate(domains_list):
        rows = grid[keep[b]].tolist()
        out.append(
            [tuple(doms[v][i] for v, i in enumerate(row)) for row in rows]
        )
    return out
