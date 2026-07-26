# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Licensed the project under the MIT License and added matching package and
  citation metadata.
- Made the sound bounded `SimConfig.practical()` profile the default and
  added explicit `SimConfig.classic()` textbook semantics, profile
  provenance, distinct-frame telemetry, and actionable truncation diagnoses.
- Renamed the per-branch landmark cap to
  `max_landmarks_per_variable`; the former `max_landmarks` constructor
  keyword remains available with a deprecation warning for the 0.1 series.
- Added an explicit maturity/qualification record, a cross-surface
  frontend-to-DecSIM test, portable README links for PyPI, and clean-wheel
  smoke tests in CI and the publishing workflow.
- Made an explicitly supplied `EnergyFilter` enable the landmark detail it
  needs, record that effective configuration, and reject infinite amplitude
  directly instead of becoming a silent no-op under practical defaults.
- Raised the bounded default state budget from 500 to 512 so the canonical
  502-node energy-filtered spring completes, while preserving every explicit
  smaller budget and improving auto-discovery truncation advice.
- Replaced the deprecated `max_landmarks` `InitVar` sentinel with a real
  warning read alias and constructor shim; dataclass equality, replacement,
  representation, and serialization continue to use only
  `max_landmarks_per_variable`.
- Made installed public docstrings self-contained by removing references to
  repository-only Markdown files.
- Made relative trajectory-direction thresholds share a trajectory time scale
  and project threshold-steady directions through active model constraints.
- Added structural cycle queries and bounded representative behavior
  enumeration for attainable-envisionment graphs, including streaming and
  explicit result limits.
- Rejected non-finite numeric landmark values and bounds at quantity-space
  construction.
- Added deterministic semantic model hashes to simulation results.
- Added stable built-in successor-filter provenance and explicit opaque
  descriptors for non-replayable user callables.
- Advanced the result export schema to `qrlib.result/v3`.
- Added workload-aware QSIM backend selection with explicit reference/tensor
  overrides and per-result dispatch telemetry.
- Added solver-refined crossing events and physical-time bounds to trajectory
  abstraction while preserving original sample-index provenance.
- Added deterministic bootstrap sign estimation with bounded sign-agreement
  confidence and explicit `UNKNOWN` thresholding.
- Reworked tensor trajectory abstraction to gather a compact ragged run stream
  on-device and transfer it to Python in bulk, eliminating per-trajectory and
  per-run CUDA synchronization.
- Added production-shaped scale and model-ensemble qualification profiles,
  machine-readable benchmark results, and measured CPU/CUDA device guidance.
- Added safe compact constraint parsing/formatting and support for passing
  expressions such as `"M+(level, outflow)"` to `Model.constrain`.
- Added region-aware signed-graph consistency certificates for
  `M+`/`M-`/`Minus` relationships, including deterministic orthant
  polarities and contradictory-cycle witnesses.
