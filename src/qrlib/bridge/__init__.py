"""Numeric <-> qualitative bridge.

The reference (pure-Python, per-trajectory) implementations of the bridge
surfaces (docs/numeric-bridge.md, docs/host-integration.md):

- ``abstraction`` — sampled trajectories -> QSIM-style qualitative
  behaviors, with explicit abstraction parameters.
- ``coverage`` — the coverage oracle: witness-carrying consistency checks
  of observed behaviors against predicted behavior graphs, plus the
  aggregate qualitative-consistency score.
- ``harvest`` — landmark intake by numeric value (with conflict reporting)
  and data-driven landmark proposal from steady stretches.

- ``signs`` — interaction sign-matrix intake -> Model, sign estimation
  from data with confidences, and the downward per-constraint consistency
  checker.

Planned: tensorized batched implementations (phase 5). Inputs accept any
array-like, including numpy arrays and torch tensors (anything with
``tolist()``); no simulator or CAS dependencies.
"""

from . import abstraction, coverage, harvest, signs

__all__ = ["abstraction", "coverage", "harvest", "signs"]
