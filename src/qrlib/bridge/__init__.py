"""Numeric <-> qualitative bridge (phases 3-4 — not yet implemented).

Planned contents per docs/numeric-bridge.md and docs/host-integration.md:
``abstraction`` (batched trajectory ``(B, T, V)`` -> qualitative behaviors),
``coverage`` (the coverage oracle: witness-carrying consistency results),
``signs`` (interaction sign-matrix intake -> Model; sign estimation from
data), and ``harvest`` (landmark intake/dedup + data-driven proposals).
Consumes plain tensors and structure summaries only; no dependency on any
particular simulator or CAS.
"""
