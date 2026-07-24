---
title: "qualitative-qsim: Qualitative reasoning about dynamical systems in Python and PyTorch"
tags:
  - Python
  - qualitative reasoning
  - qualitative simulation
  - QSIM
  - dynamical systems
  - PyTorch
authors:
  - name: Alan Oursland
    affiliation: "1"
affiliations:
  - index: 1
    name: Independent researcher, United States
date: 24 July 2026
bibliography: paper.bib
---

# Summary

`qualitative-qsim` is a Python library for reasoning about continuous and
hybrid dynamical systems when equations, functions, or parameter values are
only partly known. Instead of predicting one numerical trajectory, a
qualitative model uses ordered landmark values, signs, directions of change,
monotonic relationships, and algebraic constraints to describe a family of
possible systems. The QSIM algorithm then predicts a graph containing every
qualitatively distinct behavior consistent with that description
[@Kuipers1986; @Kuipers1994].

The library provides an executable symbolic core, a behavior-graph API, and a
bridge to ordinary numeric workflows. Users can construct qualitative
differential equations, simulate from an initial state, refine predictions
with incomplete numeric bounds, abstract sampled trajectories into
qualitative behavior, and check whether an observed trajectory is covered by
the predicted graph. Additional modules support temporal-logic guidance,
fault diagnosis, model decomposition, model induction, causal and comparative
analysis, process- and device-centered authoring, and explanation and
visualization.

# Statement of need

Numerical simulation assumes a sufficiently specific model. In early-stage
science and engineering, that assumption is often unavailable: the direction
of an influence may be known while its functional form is not, a threshold
may be known only as an interval, and several operating modes may remain
possible. QSIM was designed for precisely this setting. Its guaranteed-
coverage result says that every behavior of every ordinary differential
equation represented by a qualitative differential equation is included in
the qualitative prediction, although the prediction may also contain
spurious behaviors [@Kuipers1986; @Kuipers2001].

`qualitative-qsim` makes that style of reasoning available as a tested,
embeddable Python API. It also addresses a modern integration problem:
qualitative reasoning should complement, rather than replace, numeric
dynamical-systems software. Numeric trajectories can therefore be abstracted
and checked against QSIM predictions with witness paths or localized
refutations. This supports model validation, behavior mining, diagnosis, and
soundness testing of qualitative algorithms. Q2-style bounds add incomplete
quantitative information without abandoning the qualitative behavior graph
[@KuipersBerleant1988].

# State of the field

The original University of Texas QSIM implementations remain the historical
reference, but their principal distributions target legacy Lisp environments
or a C++ core and retain restricted commercial terms [@UTAustinQSIM].
Garp3 is a mature graphical workbench emphasizing process-centered model
construction, simulation, inspection, and education
[@BredewegEtAl2009]. The `qualitative-reasoning` Python distribution provides
process and causal abstractions but uses a custom non-commercial license and
does not target the same QSIM coverage, numeric-bridge, or tensor-execution
surface [@Chen2025].

`qualitative-qsim` is distinguished by combining a QSIM-centered reference
semantics with portable model and result schemas, per-run provenance,
trajectory abstraction and coverage, PyTorch batching, and a collection of
later QR methods behind one model representation. These include DecSIM-style
decomposition [@ClancyKuipers1997], TeQSIM-style temporal guidance
[@BrajnikClancy1998], QDOCS-style diagnosis [@SubramanianMooney1996],
qualitative phase-space filtering [@LeeKuipers1988], process-centered
authoring [@Forbus1984], and device composition and total envisionment
[@deKleerBrown1984].

# Software design

The central design decision is to keep the model as pure, serializable data
and make every reasoning engine a consumer of that model. A readable
reference engine defines the semantics. Constraint predicates and transition
rules also compile into dense lookup tables, allowing PyTorch to batch the
combinatorial filtering work without giving the tensor implementation
different semantics [@PaszkeEtAl2019]. Property tests compare reference and
tensor results.

Behavior graphs retain terminal classifications, truncation, filter
statistics, the simulation configuration, and model identity. This is
important because a qualitative result is meaningful only together with the
assumptions and resource limits that produced it. Numeric abstraction likewise
records tolerances, direction hysteresis, debounce settings, sample spans, and
optional hybrid-mode and event information.

Soundness determines API design. A missing coverage path can refute a model
under the declared abstraction; a matching path establishes compatibility,
not physical existence, because the QSIM path may be spurious. Temporal and
diagnostic APIs preserve the same asymmetry
[@ShultsKuipers1997]. Stronger filters are opt-in and carry explicit premises:
energy arguments [@FoucheKuipers1992], phase-plane non-intersection, and
signed-graph consistency [@Harary1953] never silently become universal
assumptions.

# Research impact statement

The current evidence is reproducible capability and performance rather than
external adoption. The repository includes a fifteen-part tutorial, golden
QSIM examples, randomized numeric soundness checks, reference/tensor
equivalence tests, portable schema round trips, and hardware-gated CUDA
parity tests. Production-shaped benchmark profiles cover interactive
trajectories, service batches, one million timesteps, 64-variable inputs,
high run density, and homogeneous model ensembles. On the documented RTX
3080 Ti qualification system, an already-resident million-timestep,
eight-variable workload completed in 178 ms on CUDA versus 374 ms on CPU;
the committed machine-readable result preserves the environment and
measurement assumptions.

These materials support credible near-term use in research on incomplete
dynamical models, qualitative validation of learned systems, behavior-level
comparison of numerical ensembles, and interpretable fault diagnosis. The
library is feature-complete for its stated open-source scope; general numeric
reachability and domain-specific monotone or hybrid conformance theory are
deliberately left to host systems.

# AI usage disclosure

OpenAI Codex (GPT-5) assisted with code generation, refactoring, test
scaffolding, documentation, bibliographic normalization, and drafting this
paper. The author made the core modeling and architectural decisions and
reviewed, edited, tested, and validated AI-assisted outputs. Bibliographic
records were checked against publisher pages, official proceedings, or author
archives.

# Acknowledgements

The software builds on the qualitative-reasoning research community,
especially the QSIM program established by Benjamin Kuipers. No external
funding is declared for this software release.

# References
