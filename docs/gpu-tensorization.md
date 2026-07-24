# Tensorization and GPU strategy

Where PyTorch actually helps in a symbolic algorithm, and the encoding that
makes it possible. Guiding rule: **the reference engine defines semantics;
the tensor engine is an optimization** that must agree bit-for-bit on
results (up to documented ordering).

## 1. Where the parallelism is

Qualitative simulation offers four independent batching axes:

| Axis | Workload | When it's large |
|---|---|---|
| **Frontier** | all open states of one behavior tree expanded simultaneously | bushy trees (chattery models, envisionments) |
| **Interpretations** | candidate successor tuples of one state, filtered en masse | many weakly-constrained variables |
| **Model ensemble** | same structure, different quantity spaces / corresponding values / toggled constraints | comparative analysis, model search / QDE induction |
| **Trajectory batch** | numeric trajectories being abstracted (`numeric-bridge.md`) | always — this is the truly embarrassingly-parallel case |

Textbook single models with small trees get **no** benefit from GPU — the
reference engine stays the right tool there. The tensor engine targets
trajectory abstraction (the strongest case), large interpretation/frontier
batches, and homogeneous jobs sharing one compiled frame. Measured
heterogeneous ensembles remain independently scheduled; see
`docs/scale-profiles.md`.

## 2. State encoding

Given a compiled model with `V` variables and per-variable landmark counts
`L_v`:

- **Magnitude rank:** integer in `[0, 2*L_v - 2]`; even = at landmark
  `rank/2`, odd = in the open interval above landmark `(rank-1)/2`.
- **Direction:** integer in `{0,1,2}` = `{DEC, STD, INC}`; `sign = dir - 1`.
- **State tensor:** `int16` (or `int8` for small spaces) tensor of shape
  `(2*V,)`, interleaved `[mag_0, dir_0, mag_1, dir_1, ...]`. A frontier is
  `(B, 2*V)`. Point/interval time-tag and branch bookkeeping live in side
  tensors `(B,)`.
- Landmark identity → rank mapping is frozen at compile time; new-landmark
  introduction re-freezes for the affected branch (rare event, handled by
  re-encoding that branch — see `qsim.md` §4).

Properties this buys: qmag ordering/adjacency are integer comparisons; sign
of a magnitude is `sign(rank - zero_rank_v)`; dedup of states is a unique
over rows; hashing is cheap.

## 3. Tensorized inner loops

- **Transition tables** → per-variable gather: `(B, V)` current values index
  into a padded table `(V, max_rank, K, 2)` of candidate next values + a
  validity mask. Output: candidate sets per variable, `(B, V, K)`.
- **Constraint filtering** → each constraint type compiles to either (a) a
  dense boolean table indexed by its variables' encoded values (small
  spaces), or (b) a closed-form integer/sign predicate evaluated on gathered
  candidate columns (always available). Either way: build `(B, K^arity)`
  candidate grids per constraint (arity ≤ 3, K small — the grids are
  modest), evaluate, reduce to per-variable keep-masks.
- **Waltz propagation** → iterate constraint keep-mask reduction to fixpoint;
  each sweep is a handful of gathers/any-reductions; fixpoint in a few
  sweeps because domains are tiny.
- **Interpretation assembly** — the one genuinely combinatorial step. Plan:
  batched cross-product with early masking (build partial assignments
  variable-by-variable in tensor form, pruning after each join). Escape
  hatch: states whose surviving candidate product exceeds a threshold fall
  back to the reference backtracker; correctness never depends on the fast
  path.
- **Global filters** → row-wise comparisons (no-change), all-STD reductions
  (quiescence), hash-join against ancestor sets (cycle detection).

## 4. Practical rules

- **Precision/dtype:** all-integer; no float anywhere in the core loops, so
  no tolerance questions and CPU/GPU results are identical.
- **Device neutrality:** engine follows the device of its input tensors;
  every kernel-shaped function is exercised on CPU in CI, and
  hardware-gated CUDA tests check abstraction parity, time placement,
  one-step trajectories, and error behavior.
- **Padding over raggedness:** per-variable candidate counts vary; use
  fixed-K padding + masks rather than nested tensors until profiling says
  otherwise.
- **torch.compile later, not first:** get the vectorized eager version
  correct against the oracle, then measure; avoid premature graph-mode
  cleverness.
- **Benchmarks as tests:** a small benchmark suite (frontier sizes 10³–10⁶,
  synthetic chattery models; trajectory batches for the bridge) lives in
  the repo so "GPU when helpful" is a measured claim, with the reference
  engine as the baseline.
- **Scale contracts:** `benchmarks/bench_scale_profiles.py` qualifies
  interactive, service-batch, million-timestep, V=64, high-run-density, and
  model-ensemble shapes. `docs/scale-profiles.md` records the supported
  target, device policy, memory model, and ensemble-axis decision.

## 5. As built (phase 5)

`qrlib.tensor` implements §2–§3 with one deliberate simplification:
constraint tables are built by **exhaustively evaluating the reference
predicates** over each frame's (tiny) value spaces rather than by
re-deriving the sign algebra in tensor form — agreement with the
reference engine holds by construction, tables build in milliseconds and
cache per content-hashed frame. `torch.cartesian_prod` enumerates in
`itertools.product` order, so tensor results are not merely equal but
identically ordered, and whole-graph equivalence is asserted in tests.

Measured on CPU (see `benchmarks/bench_tensor.py`): trajectory abstraction
uses tensor-native boundary detection and gathers a packed `O(actual runs)`
stream on-device. One bulk host transfer replaces per-trajectory start
copies, per-run code copies, and full timestamp copies; Python touches only
the compact run stream and final result objects. Batched frontier filtering
measured ×1.5 at B=2048; single-state
expansion ×0.25 for the small chattery model. QSIM therefore defaults to
`SimConfig(backend="auto")`: constrained interpretation products below 2,048
use the reference path, products at or above 2,048 use tensor tables, and
unconstrained products stay on reference. Explicit `backend="reference"` and
`backend="tensor"` modes, plus the legacy `use_tensor` override, remain
available. Every result reports requested mode, actual per-backend call
counts, selection reasons, and fallbacks. The CUDA path is device-neutral and
hardware-gated tests assert exact CPU/reference parity.

The benchmark uses five post-warm-up CUDA samples with explicit
synchronization before and after every timed region, reporting the median,
minimum, and raw samples. It separates dense quantization/direction work,
dense work plus the packed host transfer, and end-to-end abstraction
(including final Python result construction). Host-to-device input transfer
is excluded and stated in the output. Each CUDA run records the Torch and
CUDA runtime versions, GPU model and compute capability, dtype, batch shape,
packed run count/bytes, and free/process memory. Frontier and engine
measurements are explicitly labeled CPU-only.

### CUDA qualification result (2026-07-23)

Validated with PyTorch 2.3.0, CUDA runtime 11.8, NVIDIA driver 591.86, and an
NVIDIA GeForce RTX 3080 Ti (compute capability 8.6), using float64 input of
shape `(8, 50000, 3)`. The host-to-device input copy was excluded.

| Stage | Median | Minimum | Throughput |
|---|---:|---:|---:|
| Dense CUDA quantization + directions | 0.002969 s | 0.002501 s | — |
| Dense CUDA + packed host transfer | 0.003741 s | 0.003476 s | — |
| End-to-end CUDA abstraction | 0.145893 s | 0.134814 s | 2.742 M samples/s |
| End-to-end tensor CPU abstraction | 0.20 s | — | 2.03 M samples/s |
| Reference Python abstraction | 5.67 s | — | 0.07 M samples/s |

The workload produced 8,624 actual runs. Its packed host payload was 0.724
MiB; it is proportional to actual runs and carries only two timestamps per
run, with no `B × max_runs` padding. The packed transfer added less than 1 ms
over dense device work. End-to-end CUDA abstraction was 38.86× faster than
the reference and 4.68× faster than the previous per-run-transfer
qualification (0.682635 s). Final Python `QState`/behavior construction now
dominates the remaining time. During the run, 10.71 GiB of 12.00 GiB was
free; the process had 0.01 GiB allocated and 0.06 GiB reserved.

### Native-extension boundary

qrlib-owned C, C++, CUDA, Cython, or pybind11 extensions are deferred
indefinitely. The optimization strategy is to compose PyTorch's existing
compiled kernels and keep representations tensor-native or compact for as
long as practical. Native code should be reconsidered only after
production-representative profiling, exhaustion of clean tensor approaches,
and an explicit owner decision that the gain justifies toolchain, ABI,
packaging, and maintenance costs.

### Automatic backend qualification (2026-07-23)

The single-state policy was checked on constrained M+ chains with three
candidate values per variable:

| Variables | Interpretation product | Reference | Tensor | Auto choice |
|---:|---:|---:|---:|---|
| 6 | 729 | 1.04 ms | 1.43 ms | reference |
| 8 | 6,561 | 9.93 ms | 3.48 ms | tensor |

On the 400-state chattery spring, reference took 78.2 ms, forced tensor took
537.6 ms, and auto took 71.1 ms while selecting reference for all 267 filter
calls. The 2,048 threshold is deliberately between the measured losing and
winning products. Unconstrained products always remain on reference because
tensor lookup tables have no predicates to accelerate.

## 6. Differentiable constraint losses (`tensor/losses.py`)

A soft, autograd-friendly rendering of the boolean predicates over numeric
trajectories: each constraint becomes a smooth penalty that is zero exactly
when the data qualitatively satisfies it (at margin 0). Value identities are
scale-normalized squared residuals; monotone/derivative relations are
step-agreement hinges; constancy is squared increments; order-of-magnitude
a same-scale hinge. Because it is pure torch, gradients flow back to
whatever produced the trajectory, turning a qualitative model into a
training signal (parameter fitting, structural regularization, soft
consistency scoring). It layers strictly above the exact integer core — it
never feeds the sound boolean engine, and adopts SIMGEN's lesson (when
numbers are available, use them) without touching the qualitative path.

## 7. Interval extension (semi-quantitative)

Q2-style interval annotations ride along as `(B, V)` `lo`/`hi` float
tensors; the **per-state algebraic narrowing** (ADD/MINUS/MULT/At) is now
built in `tensor/interval.py`, vectorized over the batch to a fixpoint,
with a `feasible_mask` screen that prunes states whose bounds cross. The
primitives mirror `semiquant.Interval` exactly and narrowing is confluent,
so the batched fixpoint equals the reference's (parity-tested). Monotone
envelopes (arbitrary callables) and the cross-state couplings (continuity,
mean-value time bounds, CONSTANT intersection along a behavior) are
sequential and stay in `semiquant.refine`; the batched path is the screen
that runs first or over many states at once. This is the first place
floats enter, and it stays strictly layered above the exact integer core.
