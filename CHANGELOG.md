# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Added deterministic semantic model hashes to simulation results.
- Added stable built-in successor-filter provenance and explicit opaque
  descriptors for non-replayable user callables.
- Advanced the result export schema to `qrlib.result/v2`.
- Added workload-aware QSIM backend selection with explicit reference/tensor
  overrides and per-result dispatch telemetry.
- Added solver-refined crossing events and physical-time bounds to trajectory
  abstraction while preserving original sample-index provenance.
- Added deterministic bootstrap sign estimation with bounded sign-agreement
  confidence and explicit `UNKNOWN` thresholding.
