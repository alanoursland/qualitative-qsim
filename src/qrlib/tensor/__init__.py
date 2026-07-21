"""Tensorized execution — the only subpackage that imports torch.

Contents (docs/gpu-tensorization.md):

- ``encoding`` — qcode packing, canonical ``(B, 2V)`` frontier codecs, and
  per-frame dense constraint tables built from the reference predicates
  (agreement by construction).
- ``engine`` — tensorized prune + interpretation filtering, single-state
  and batched-frontier forms, with a reference fallback for oversized
  products. Activated inside the QSIM engine via
  ``SimConfig(use_tensor=True)``; results are identical to the reference
  path by contract (equivalence-tested on whole behavior graphs).
- ``abstraction`` — batched quantization/direction estimation for
  ``(B, T, V)`` trajectory tensors, mirroring the reference arithmetic
  exactly; segmentation reuses the reference tail.

Everything is device-neutral (tensors stay on whatever device the caller
provides; tables build on CPU and move on demand); the core loops are
all-integer, so CPU and GPU results are identical. Benchmarks live in
``benchmarks/bench_tensor.py``.
"""
