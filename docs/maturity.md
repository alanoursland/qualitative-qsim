# Maturity and qualification

This document records how much confidence each public surface has earned in
`0.1.0`. Implementation completeness and test count are not maturity
measures. The labels below reflect the kinds of independent evidence available
and the breadth of scenarios exercised.

## Labels

- **Core alpha** — the supported alpha path. Its public contract is
  exercised by literature-derived golden cases, adversarial cases,
  cross-module checks, and concrete numeric trajectories.
- **Experimental** — implemented with semantic tests, but the configuration,
  performance envelope, or specialized assumptions still need broader use.
- **Research preview** — useful research functionality with explicit
  limitations and focused evidence, but narrower independent review. Audit it
  against the intended domain before consequential use.

All labels remain pre-1.0. None promises API stability, numerical
identifiability, completeness beyond the documented qualitative semantics, or
fitness for safety-critical decisions.

## Evidence record

| Surface | Maturity | Qualification evidence | Important boundary |
|---|---|---|---|
| Model, constraints, practical QSIM, behavior graphs, result/model schemas | Core alpha | Bathtub, U-tube, and spring goldens in [`test_qsim_golden.py`](../tests/test_qsim_golden.py); transition algebra and adversarial validation; randomized concrete-ODE coverage in [`test_soundness.py`](../tests/test_soundness.py); provenance round trips | Qualitative simulation is sound but may include spurious behaviors; practical mode leaves intermediate extrema unnamed |
| Trajectory abstraction and coverage oracle | Core alpha | Concrete trajectory coverage and deliberate refutations in [`test_soundness.py`](../tests/test_soundness.py); crossing, truncation, and diagnosis cases in [`test_bridge.py`](../tests/test_bridge.py) | Guarantees depend on sampling and the recorded abstraction tolerances |
| Classic landmark discovery and chatter controls | Experimental | Per-branch landmark/correspondence cases and legacy-profile equivalence in [`test_qsim_phase2.py`](../tests/test_qsim_phase2.py); automatic/manual chatter equivalence in [`test_chatter.py`](../tests/test_chatter.py) | Classic discovery can grow without termination; caps are per variable per branch, not global |
| Energy, Lyapunov, and phase-space filters | Experimental | Conservative and damped oscillator cases, invalid-assumption checks, and concrete-trajectory coverage in [`test_energy.py`](../tests/test_energy.py) and [`test_phase.py`](../tests/test_phase.py) | Each filter is an explicit model assumption; an unjustified certificate can remove real behavior |
| Semi-quantitative/Q2 refinement | Experimental | Literature-style time/value bounds, empty-interval refutations, and reference/tensor parity in [`test_semiquant.py`](../tests/test_semiquant.py) and [`test_interval.py`](../tests/test_interval.py) | Bounds are only as informative as landmark values and monotone envelopes |
| Tensor execution and differentiable losses | Experimental | Reference/tensor graph and filtering equivalence in [`test_tensor.py`](../tests/test_tensor.py), interval parity, gradient/optimization checks in [`test_losses.py`](../tests/test_losses.py) | GPU benefit is workload-dependent; hardware-specific coverage is gated by available devices |
| Diagnosis | Research preview | Nominal, single-fault, double-fault, budget, truncation, and numeric-observation scenarios in [`test_diagnosis.py`](../tests/test_diagnosis.py) | A returned diagnosis is not refuted, not uniquely proven; fault modes are fixed over the observation window |
| Decomposition/DecSIM | Research preview | Independent and coupled systems compared with monolithic QSIM, interface guidance, cyclic cuts, and stream algebra in [`test_decsim.py`](../tests/test_decsim.py) | Operating-region models and user-guided decomposed simulation are not supported |
| Model induction | Research preview | Known linear decay and oscillator structures, multiple trajectories, explicit derivatives, tolerance sensitivity, and forced refutation in [`test_induce.py`](../tests/test_induce.py) | The result is a ranked, unrefuted structure at a tolerance, not proof of the true or unique model |
| QPT and device front ends | Research preview | Equivalence to handwritten QDEs, activation regions, composition, serialization, and simulation in [`test_frontends.py`](../tests/test_frontends.py); device-to-decomposition integration in [`test_research_integration.py`](../tests/test_research_integration.py) | The QPT front end intentionally supports only its documented activation-condition subset |
| Envisionment and temporal guidance | Research preview | Attainability, recurrence, specification progress, pruning, and validation in [`test_envision.py`](../tests/test_envision.py) and [`test_guide.py`](../tests/test_guide.py) | Path-dependent phase and Lyapunov filters are incompatible with merged envisionments |
| Monotonicity, comparison, and causal analysis | Research preview | Positive certificates, concrete negative-cycle witnesses, region separation, and intervention/comparison cases in [`test_monotonicity.py`](../tests/test_monotonicity.py), [`test_compare.py`](../tests/test_compare.py), and [`test_causal.py`](../tests/test_causal.py) | These are structural qualitative conclusions, not effect-size or statistical-causal estimates |

## Release qualification

A release candidate should pass all of the following:

1. The complete library suite.
2. The external stress suite against the candidate source.
3. Tutorial example execution.
4. Wheel and source-distribution construction plus strict metadata checking.
5. Installation of the wheel without the source tree on `sys.path`, followed
   by the installed-package smoke test in
   [`tests/wheel_smoke.py`](../tests/wheel_smoke.py).

Promoting a surface requires new evidence, not merely more tests: at least one
domain-reference or independently derived golden scenario, one adversarial or
refutation case, and one integration or concrete-trajectory check relevant to
the surface's claims.
