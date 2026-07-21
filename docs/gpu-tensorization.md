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
reference engine stays the right tool there. The tensor engine targets the
bottom three rows first (trajectory abstraction is the strongest case and the
one adjacent to numeric dynamical systems work), and total-envisionment-style
enumeration where the candidate space is a huge cross-product.

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
  every kernel-shaped function is exercised on CPU in CI (GPU runs are a
  perf concern, not a correctness concern).
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

## 5. As built (phase 5)

`qrlib.tensor` implements §2–§3 with one deliberate simplification:
constraint tables are built by **exhaustively evaluating the reference
predicates** over each frame's (tiny) value spaces rather than by
re-deriving the sign algebra in tensor form — agreement with the
reference engine holds by construction, tables build in milliseconds and
cache per content-hashed frame. `torch.cartesian_prod` enumerates in
`itertools.product` order, so tensor results are not merely equal but
identically ordered, and whole-graph equivalence is asserted in tests.

Measured on CPU (see `benchmarks/bench_tensor.py`): trajectory
abstraction ~×22 (run boundaries detected in tensor land; Python touches
O(runs)); batched frontier filtering ×1.5 at B=2048; single-state
expansion ×0.25 — confirming §1's table: one small model at a time gains
nothing, which is why `SimConfig.use_tensor` defaults off. GPU
measurements pend a CUDA environment; the code is device-neutral and the
integer core makes CPU/GPU results identical.

## 6. Interval extension (semi-quantitative, later)

Q2-style interval annotations ride along as a float tensor `(B, V, 2)` of
`(lo, hi)` bounds; interval arithmetic for ADD/MULT/monotonic envelopes is
straightforwardly vectorizable, and a behavior is pruned when any bound pair
crosses. This is the first place floats enter, and it stays strictly layered
above the exact integer core.
