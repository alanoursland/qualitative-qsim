# Qualitative analysis of piecewise-affine systems

A technique note. Piecewise-affine (PWA) systems are a class of dynamical
system whose qualitative behavior can be read off from the *ordering* of a
few numeric quantities rather than from exact trajectories — which makes
them a natural, and unusually *exact*, target for a qualitative engine. This
note describes the method in domain-neutral terms, maps it onto the
abstractions qrlib already has, and assesses what it would take to support
it directly and what that would buy.

Nothing here depends on a particular application domain; the lineage is the
control-theory / hybrid-systems treatment of discontinuous vector fields
(see references).

## 1. What a PWA system is

The state space `x ∈ R^n` is partitioned by **threshold hyperplanes**: each
variable `x_i` has an ordered set of threshold values
`θ_i^1 < θ_i^2 < … < θ_i^{k_i}`. The thresholds cut the space into
axis-aligned rectangular boxes. Inside each box `R` the dynamics are affine
with box-constant coefficients:

```
ẋ = A_R · x + b_R          (A_R, b_R constant on R)
```

The important and cleanest special case is **decoupled** dynamics, where
each variable relaxes toward a box-constant target:

```
ẋ_i = f_i(R) − γ_i · x_i          γ_i > 0
```

Here the **focal point** of box `R` is `φ_i(R) = f_i(R) / γ_i` — the value
`x_i` would converge to if the box's dynamics held everywhere. In a box the
field points monotonically toward `φ(R)`, so

```
sign(ẋ_i) = sign(φ_i(R) − x_i)
```

and *that sign depends only on where `φ_i(R)` sits relative to the box's
bounding thresholds* — a purely ordinal fact. That is the hook the
qualitative analysis hangs on.

(The coupled case `ẋ = A_R x + b_R` with non-diagonal `A_R` needs
eigenstructure to get per-component directions; qualitatively it reduces to
"the sign of each `ẋ_i` on box `R`", i.e. a per-region sign matrix — see §4.
The decoupled case is where ordering alone suffices.)

## 2. The qualitative analysis

Two kinds of region:

- **Regular domains** — open boxes, no variable sitting exactly on a
  threshold. The field is single-valued and its component signs are fixed
  by the focal point's position, so a regular domain has one qualitative
  direction vector.
- **Switching domains** — the threshold hyperplanes themselves
  (lower-dimensional faces where one or more variables equal a threshold).
  The field is *discontinuous* across a threshold, so on a switching domain
  it is set-valued and needs the resolution in §3.

From these you build a **domain transition graph**: nodes are domains
(regular and switching), edges are the admissible transitions, determined
by comparing each boundary's outward normal with the field direction on
either side. The reachable part of this graph is an envisionment over the
rectangular partition. From it — using only the ordering of focal points
and thresholds, no exact parameter values — you can read:

- **equilibria**: a box whose focal point `φ(R)` lies *inside* `R` is a
  genuine equilibrium, and (decoupled, `γ_i > 0`) an **attracting** one —
  every component points inward. Stability comes for free from
  focal-point-in-box.
- **limit cycles**: cycles in the domain transition graph.
- **reachability / basins**: ordinary graph reachability over domains.

The qualitative behavior is therefore a function of *which threshold
interval each focal point falls into* — a discrete, ordering-only
description, exactly the QR posture, but here it is **exact** rather than
sound-but-spurious, because the PWA structure pins the field direction in
each box instead of only bounding its monotonicity.

## 3. Discontinuities: Filippov / sliding resolution

On a switching domain `x_j = θ`, the field just below (`x_j = θ⁻`) and just
above (`x_j = θ⁺`) can disagree in their `j`-component. Three cases, decided
by those two signs:

- **transparent (crossing)**: both point the same way through `θ` — the
  trajectory crosses; the switching domain is transient.
- **sliding**: the lower field pushes up and the upper field pushes down
  (both point *at* `θ`) — the trajectory is trapped on the surface and
  **slides** along it. The sliding dynamics are the Filippov convex
  combination of the two fields that is tangent to the surface; `x_j` holds
  at `θ` (steady) while the remaining variables evolve under the reduced
  field.
- **repelling**: both point away from `θ` — the surface is unreachable from
  itself (relevant only as a separatrix).

Sliding is the genuinely new semantics relative to a plain box-transition
graph: a switching domain can be an ongoing regime, not just an instant, and
can host its own equilibria and cycles in the reduced dynamics.

## 4. How it maps onto qrlib

PWA qualitative analysis is close kin to the phase-4 operating-regions
machinery, under a direct dictionary:

| PWA concept | qrlib construct |
|---|---|
| Threshold values `θ_i^k` of variable `x_i` | Landmark **values** in `x_i`'s quantity space |
| Rectangular box (regular domain) | A qualitative state: every variable in an open interval (odd magnitude rank) |
| Switching domain `x_j = θ` | A state with `x_j` **at** a landmark (even rank) |
| Focal point `φ_i(R)` inside/outside box | The sign `sign(φ_i − x_i)`, i.e. the **qdir** of `x_i` in that box |
| Domain transition graph | The behavior graph / attainable envisionment |
| Focal point inside its own box | A `QUIESCENT` state — and demonstrably stable |
| Box `A_R`, `b_R` (coupled case) | A per-region **sign matrix** (`bridge.signs`) |

So a large part of PWA analysis is *already expressible*: model each
variable's thresholds as landmark values, and each box's directions as a
per-region sign structure; the engine's region machinery + envisionment
then produces the domain transition graph. The pieces qrlib does **not** yet
have are:

1. **Focal-point front-end.** A model form where you give thresholds +
   focal points (or the decoupled `f_i(R), γ_i`), and the per-box direction
   signs are *derived* from the ordering `φ_i(R)` vs. the box — instead of
   being asserted constraint-by-constraint. This is a compilation step onto
   the existing region + sign-matrix representation. Because the number of
   boxes is exponential in `n`, it must build lazily/reachably (the
   attainable-envisionment mode already explores reachable-only).
2. **Sliding-mode derivation.** Detecting, at a switching domain, that the
   two adjacent regular domains' fields both point at the shared threshold,
   and then generating a *sliding successor* in which `x_j` is held steady
   (qdir STD, magnitude at the landmark) while the reduced field drives the
   rest. qrlib's region transitions re-derive directions on entry but do not
   currently derive this two-sided-agreement → stay-on-surface semantics.
   This is the one real new engine capability.
3. **Focal-point stability tagging.** Classifying a `QUIESCENT` box whose
   focal point lies strictly inside it as *attracting*, sharpening the
   terminal taxonomy with stability information the monotonic-constraint
   path cannot supply.

## 5. Incorporation assessment

- **Effort.** Moderate. The focal-point front-end (item 1) is mostly a
  compiler onto existing primitives — landmarks with numeric values already
  exist, per-region sign matrices already exist (`bridge.signs`), and
  reachable envisionment already exists; the new code is the derivation of
  box direction from focal-point ordering and the lazy box enumeration.
  Sliding-mode derivation (item 2) is the substantive addition: detect
  two-sided threshold agreement between adjacent boxes and emit a
  constrained sliding successor. Stability tagging (item 3) is small.
- **Dependencies.** None beyond what exists. All-ordinal for the decoupled
  case; the coupled case optionally uses the semi-quantitative layer to
  bound focal points from interval data.
- **What it buys.** A regime where qualitative analysis is **exact, not
  merely sound** — the PWA structure removes the spurious-behavior problem
  because box directions are pinned rather than only monotonicity-bounded.
  For any numeric system that is piecewise-affine, or that a host can
  abstract to PWA form (threshold partition + per-box affine fit), the
  engine then delivers an exact behavior graph, equilibria with stability,
  and limit cycles from ordering data alone. It is the natural meeting
  point between the qualitative engine and a numeric hybrid-systems
  representation: the host supplies thresholds and per-box coefficients (or
  focal points); qrlib supplies the exact qualitative phase portrait. This
  slots directly into the existing bridge posture (tensors and structure
  summaries in, serializable behavior graphs out) with no new
  interchange surface.
- **Soundness note.** The exactness claim holds only where the PWA model
  genuinely describes the system. Used as an *abstraction* of a non-PWA
  system, the same machinery reverts to sound-but-approximate, and the
  coverage oracle remains the check that a numeric trajectory is consistent
  with the predicted graph.

## 6. Relationship to existing capabilities

- **Operating regions (phase 4)** are the substrate; PWA boxes are a
  structured, exhaustive family of regions with derived (not authored)
  guards and directions.
- **Sign-matrix intake (`bridge.signs`)** already expresses per-region
  direction signs; the focal-point front-end is a more ergonomic way to
  produce them for the PWA case.
- **Semi-quantitative layer (`semiquant`)** composes: interval-valued
  thresholds/focal points propagate to interval-bounded box directions and
  can refute boxes whose direction is undetermined only for parameter
  values ruled out by the bounds.
- **Attainable envisionment** is the enumeration strategy that keeps the
  exponential box count tractable (reachable-only).

## References (domain-neutral)

- A. F. Filippov, *Differential Equations with Discontinuous Righthand
  Sides*, Kluwer (1988) — the sliding-mode / set-valued resolution used in §3.
- E. D. Sontag, *Nonlinear Regulation: The Piecewise Linear Approach*, IEEE
  Transactions on Automatic Control 26(2) (1981) — PWA systems in control.
- A. Bemporad & M. Morari, *Control of Systems Integrating Logic, Dynamics,
  and Constraints*, Automatica 35(3) (1999) — PWA/hybrid modeling and the
  box-partition view.
- M. Johansson & A. Rantzer, *Computation of Piecewise Quadratic Lyapunov
  Functions for Hybrid Systems*, IEEE Transactions on Automatic Control 43(4)
  (1998) — stability analysis over a PWA partition.

> A broader QR literature survey (`docs/literature-survey.md`, in progress)
> will cross-reference this note; the technique is documented separately here
> because it is a concrete, near-term incorporation candidate.
