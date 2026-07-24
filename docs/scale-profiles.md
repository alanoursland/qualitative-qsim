# Production-shaped scale profiles

This document defines qrlib's repeatable workload contracts for tensor
trajectory abstraction and qualitative-model ensembles. They are
**production-shaped profiles**, not external-user telemetry. Replace or extend
them when real deployments provide stronger evidence.

The executable qualification is
`benchmarks/bench_scale_profiles.py`; the complete machine-readable result is
`benchmarks/scale-profile-2026-07-23.json`.

## Workload contracts

| Profile | Shape `(B,T,V)` | Purpose |
|---|---:|---|
| Interactive | `(1, 2,048, 4)` | One trajectory in an interactive analysis |
| Service batch | `(32, 4,096, 12)` | Moderate multivariate request batch |
| Million timesteps | `(16, 62,500, 8)` | One million total trajectory timesteps |
| Wide V=64 | `(8, 8,192, 64)` | Qualified upper variable-count target |
| High run density | `(4, 4,096, 8)` | Chatter stress: nearly every sample changes code |

Smooth profiles use deterministic multi-frequency trajectories. The
high-density profile alternates across a landmark every sample. This separates
dense tensor cost from the cost of transferring, debouncing, and constructing
ragged qualitative output.

Latency samples are post-warm-up medians with device synchronization. Input
construction and host-to-device transfer are excluded: the input is assumed
to reside where abstraction is requested. Peak CUDA allocation includes the
resident input. Host working-set growth is recorded but is allocator- and
execution-order-sensitive, so it is diagnostic rather than a capacity bound.

## Qualification result

Measured on Python 3.11.7, PyTorch 2.3.0, CUDA 11.8, NVIDIA driver 591.86,
an RTX 3080 Ti (12 GiB), and a 24-logical-CPU AMD64 host:

| Profile | CPU total | CUDA total | CUDA packed prefix | Runs / timestep | Packed payload | CUDA peak |
|---|---:|---:|---:|---:|---:|---:|
| Interactive | 1.8 ms | 3.8 ms | 3.5 ms | 1.7% | 0.003 MiB | 0.3 MiB |
| Service batch | 590 ms | 545 ms | 10.8 ms | 5.2% | 1.521 MiB | 73.0 MiB |
| Million timesteps | 374 ms | 178 ms | 9.6 ms | 0.4% | 0.568 MiB | 376.1 MiB |
| Wide V=64 | 2.04 s | 2.18 s | 28.2 ms | 9.2% | 6.128 MiB | 192.6 MiB |
| High run density | 246 ms | 229 ms | 10.6 ms | 100% | 2.625 MiB | 7.8 MiB |

The packed prefix includes quantization, direction estimation, run detection,
and the compact device-to-host transfer. The difference between that prefix
and total latency is Python debounce and final `QState`/behavior construction.

Two independent sizes predict cost:

- Dense tensor work and peak device memory scale with scalar values `B*T*V`.
  The two largest dense profiles used about 48–49 bytes per scalar value at
  peak with the current float64/int64 implementation.
- Ragged transfer and final-object work scale with actual runs and emitted
  states. The packed wire size is `(5 + 2*V) * 8` bytes per run: batch/start/end,
  ranks, directions, and two endpoint times.

Consequently, "one million timesteps" is a valid target only together with
variable count and run density. The qualified million-timestep/V=8 smooth
profile is comfortable on CPU and CUDA. V=64 is supported, but a wide profile
that emits about ten thousand qualitative states is a seconds-scale batch
operation, not an interactive one.

## Device policy

- Prefer CPU for interactive/small inputs. CUDA launch and synchronization
  overhead made the interactive profile about 2.1× slower.
- CUDA is worthwhile for large, low-run-density inputs already resident on
  the device. It made the million-timestep profile about 2.1× faster.
- Service-sized, wide, or high-run-density inputs are dominated by final
  Python objects. CUDA accelerates their packed prefix but yields little or no
  end-to-end advantage. Prefer CPU when such inputs originate on the host.
- Re-evaluate with the caller's real `B,T,V`, run density, and residency.
  Total timesteps alone is not a sufficient dispatch signal.

## Model ensembles

The real in-library ensemble shapes are modest and heterogeneous:

- comparative analysis usually compares two models;
- QDE induction currently emits at most 12 candidate structures;
- host-side model selection may supply more, but no such production trace is
  currently available.

The qualification filters a 6,561-interpretation, eight-variable constrained
workload:

| Models | Shared-frame sequential | Shared-frame batch | Speedup | Heterogeneous warm | Heterogeneous table build |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.9 ms | 1.8 ms | 1.6× | 2.5 ms | 3.7 ms |
| 8 | 21.1 ms | 3.8 ms | 5.5× | 20.3 ms | 28.2 ms |
| 32 | 80.8 ms | 12.5 ms | 6.5× | 91.0 ms | 103.3 ms |

The existing shared-frame frontier batch is the right optimization for
homogeneous members. Heterogeneous models have different constraint masks,
tables, variable counts, and sometimes quantity spaces; their warm cost is
linear but only about 2.5–2.8 ms per model in this profile. A first-class
padded model axis would add substantial representation and soundness-testing
complexity for ensemble sizes that are currently 2–12.

Decision: do **not** add a first-class heterogeneous model-ensemble axis.
Schedule heterogeneous models independently; when several jobs share the
same compiled frame, group their states and use the existing
`filtered_combos_batch` path. Revisit only with a demonstrated workload of
hundreds of compatible models or an ensemble-latency requirement this policy
cannot meet.

## `int8` and `torch.compile`

Neither is justified by these profiles:

- Peak CUDA allocation is under 400 MiB at the qualified million-timestep
  target. Integer codes are also used as table indices, where PyTorch commonly
  requires wider index types; conversion could trade memory for latency.
- The CUDA packed prefix is only 9.6–28.2 ms on the large profiles, while
  final Python construction is 168 ms to more than two seconds. Compiling the
  dense prefix cannot materially change end-to-end latency for those shapes.

Keep eager tensor code and the readable reference oracle. Reconsider dtype
compression or compilation only when a measured workload is dense-prefix- or
memory-bound.

## Discovered stress limitation

The high-run-density profile transfers 16,384 raw runs but emits only 12
states after debounce. CPU scaling measured 0.107, 0.234, 0.543, and 1.001
seconds for 8,192 through 65,536 raw runs. This is approximately linear, but
it performs avoidable host transfer and Python deletion work. A separate gap
tracks exact pre-transfer/device-side debounce for sustained chattery inputs;
it does not invalidate the smooth million-timestep qualification.
