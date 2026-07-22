"""Differentiable constraint losses.

The boolean constraint predicates (``engines.filters``) answer *consistent
or not*; their violation-mass sibling (``bridge.signs.check_consistency``)
answers *how badly, per constraint, on data*. This module is the
**differentiable** rendering: a smooth, torch-autograd penalty over a
*numeric* trajectory that is zero exactly when the trajectory qualitatively
satisfies a constraint and grows with the degree of violation. Gradients
flow back to whatever produced the trajectory, so a qualitative model can
act as a training signal:

- **fit** numeric parameters (or a whole trajectory) so they obey a
  qualitative model, by gradient descent on the loss;
- **regularize** a differentiable simulator / neural model toward the
  model's monotonicity and sign structure, as an added loss term;
- **score** a batch's soft consistency (a differentiable analogue of the
  coverage/consistency fraction).

This layer sits strictly *above* the exact core — it never feeds the sound
boolean engine, and the engine never depends on it. The penalties are
chosen so that ``loss == 0`` (at ``margin == 0``) iff every constraint's
qualitative assertion holds on the numeric data, matching
``check_consistency``'s zero:

- **value identities** — ``ADD`` (``x+y=z``), ``MINUS`` (``y=-x``),
  ``MULT`` (``z=x*y``), ``At`` (``x=landmark``) — squared residuals,
  scale-normalized.
- **monotone relations** — ``M+``/``M-`` — a step-agreement hinge: adjacent
  samples of the two variables must move the same way (``M+``) or opposite
  ways (``M-``); the penalty is the rectified wrong-signed product of their
  increments.
- **integration** — ``Deriv(x, y)`` — the finite-difference increment of
  ``x`` must share the sign of the rate ``y`` (rectified disagreement).
- **constancy** — ``Constant`` — squared increments (zero iff flat).
- **order of magnitude** — ``Negligible`` — a rectified-margin hinge on
  ``|small| - |large|``.

Increments and reductions are ordinary torch ops, so the whole thing is
differentiable and batched. Scale normalizers are detached (they balance
the terms, they are not optimized through). A model whose constraints are
wrong will train the data to the wrong thing — the loss inherits the
model's correctness, like every constraint-level tool here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..model import CompiledModel, Model, _KIND

__all__ = ["LossConfig", "ConstraintLoss", "constraint_loss", "per_constraint_loss"]

_EPS = 1e-12


@dataclass(frozen=True)
class LossConfig:
    """- ``margin``: slack added to the hinge (monotone/deriv/negligible)
      penalties, pushing *strict* agreement rather than mere non-violation.
    - ``reduction``: ``"mean"`` | ``"sum"`` over the batch/time elements of
      each constraint's penalty (``"none"`` returns the raw per-element
      tensors, un-aggregated across constraints is not meaningful, so
      ``per_constraint_loss`` is the way to get that).
    - ``scale_floor``: lower bound on the per-variable normalizers, so a
      variable that is ~0 throughout does not blow up the residuals."""

    margin: float = 0.0
    reduction: str = "mean"
    scale_floor: float = 1e-6


@dataclass(frozen=True)
class ConstraintLoss:
    """One constraint's differentiable penalty."""

    constraint: object
    kind: str
    loss: torch.Tensor  # scalar (reduction != "none") or per-element tensor


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """(B, T, V) from (T, V) or (B, T, V), float."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"trajectory must be (T, V) or (B, T, V), got {tuple(x.shape)}")
    return x.to(torch.get_default_dtype()) if not x.is_floating_point() else x


def _reduce(t: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return t
    if reduction == "sum":
        return t.sum()
    if reduction == "mean":
        return t.mean() if t.numel() else t.new_zeros(())
    raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")


def _scale(x: torch.Tensor, floor: float) -> torch.Tensor:
    """Per-variable magnitude normalizer (detached: balances, not trained)."""
    return x.abs().amax(dim=(0, 1), keepdim=True).clamp_min(floor).detach()


def per_constraint_loss(
    trajectory: torch.Tensor,
    model: Model | CompiledModel,
    *,
    config: LossConfig | None = None,
) -> list[ConstraintLoss]:
    """Per-constraint differentiable penalties over ``trajectory`` (columns
    in the model's variable order). Each is zero iff the numeric data
    respects that constraint's qualitative assertion (at ``margin == 0``)."""
    cfg = config or LossConfig()
    frame = model.compile() if isinstance(model, Model) else model
    x = _normalize(trajectory)
    if x.shape[-1] != len(frame.var_order):
        raise ValueError(
            f"trajectory has {x.shape[-1]} columns but the model has "
            f"{len(frame.var_order)} variables"
        )
    col = {n: i for i, n in enumerate(frame.var_order)}
    scale = _scale(x, cfg.scale_floor)  # (1, 1, V)
    T = x.shape[1]

    def v(name: str) -> torch.Tensor:
        return x[..., col[name]]

    def s(name: str) -> torch.Tensor:
        return scale[..., col[name]]

    def step(name: str) -> torch.Tensor:  # normalized forward increments
        c = v(name)
        return (c[:, 1:] - c[:, :-1]) / s(name)

    def zero(name: str) -> float:
        space = frame.spaces[col[name]]
        try:
            lm = space.landmark("0")
        except KeyError:
            return 0.0
        return float(lm.value) if lm.value is not None else 0.0

    out: list[ConstraintLoss] = []
    for cc in frame.constraints:
        kind = cc.kind
        c = cc.source
        m = cfg.margin
        if kind in ("mplus", "mminus"):
            # step-agreement hinge: same sign (M+) or opposite (M-)
            if T < 2:
                pen = x.new_zeros(())
            else:
                prod = step(c.x) * step(c.y)
                wrong = -prod if kind == "mplus" else prod
                pen = torch.relu(wrong + m)
        elif kind == "minus":  # y = -x : a value identity
            r = (v(c.x) - zero(c.x)) + (v(c.y) - zero(c.y))
            pen = (r / (s(c.x) + s(c.y))) ** 2
        elif kind == "add":  # x + y = z
            r = v(c.x) + v(c.y) - v(c.z)
            pen = (r / (s(c.x) + s(c.y) + s(c.z))) ** 2
        elif kind == "mult":  # z = x * y
            r = v(c.x) * v(c.y) - v(c.z)
            pen = (r / (s(c.x) * s(c.y) + s(c.z))) ** 2
        elif kind == "deriv":  # sign of dx/dt matches the rate y
            if T < 2:
                pen = x.new_zeros(())
            else:
                dx = step(c.x)  # forward increment over [t, t+1]
                y_left = (v(c.y)[:, :-1] - zero(c.y)) / s(c.y)  # rate at t
                pen = torch.relu(-(dx * y_left) + m)
        elif kind == "constant":  # flat over time
            if T < 2:
                pen = x.new_zeros(())
            else:
                pen = step(c.x) ** 2
        elif kind == "at":  # x pinned at the landmark value
            try:
                target = frame.spaces[col[c.x]].landmark(c.landmark).value
            except KeyError:
                target = None
            if target is None:  # infinite/valueless landmark: nothing to pin
                pen = x.new_zeros(())
            else:
                pen = ((v(c.x) - float(target)) / s(c.x)) ** 2
        elif kind == "negligible":  # |small| < |large| (compared on one scale)
            sc = torch.maximum(s(c.small), s(c.large))
            small = (v(c.small) - zero(c.small)).abs() / sc
            large = (v(c.large) - zero(c.large)).abs() / sc
            pen = torch.relu(small - large + m)
        else:
            raise ValueError(f"no differentiable loss for constraint kind {kind!r}")
        out.append(ConstraintLoss(c, kind, _reduce(pen, cfg.reduction)))
    return out


def constraint_loss(
    trajectory: torch.Tensor,
    model: Model | CompiledModel,
    *,
    config: LossConfig | None = None,
) -> torch.Tensor:
    """Total differentiable constraint loss: the sum of the per-constraint
    penalties (each reduced per ``config.reduction``). A scalar tensor with
    a gradient to ``trajectory`` (and anything upstream of it)."""
    cfg = config or LossConfig()
    parts = per_constraint_loss(trajectory, model, config=cfg)
    if not parts:
        x = _normalize(trajectory)
        return x.new_zeros(())
    total = parts[0].loss
    for p in parts[1:]:
        total = total + p.loss
    return total
