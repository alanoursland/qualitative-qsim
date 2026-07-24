"""Benchmark: reference vs tensorized paths (docs/gpu-tensorization.md §4
— "GPU when helpful" is a measured claim, with the reference as baseline).

Run:  python benchmarks/bench_tensor.py

Reports CPU numbers everywhere; when CUDA is available the abstraction
stages also run on GPU. Expected shape of results (see the doc): single
small models gain little or nothing from tensorization — the wins are
trajectory batches (per-sample stages) and large/batched frontiers.
"""

import sys
import subprocess
import time
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import torch

import qrlib as qr
from qrlib import SimConfig
from qrlib.bridge import abstraction as rabs
from qrlib.engines import filters
from qrlib.engines.transitions import interval_successors, point_successors
from qrlib.state import TimeTag
from qrlib.tensor import abstraction as tabs
from qrlib.tensor import engine as tengine


def samples(fn, repeat=3, warmup=0, synchronize=None):
    """Return wall-clock samples, synchronizing asynchronous devices."""
    for _ in range(warmup):
        fn()
        if synchronize is not None:
            synchronize()
    observed = []
    for _ in range(repeat):
        if synchronize is not None:
            synchronize()
        t0 = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        observed.append(time.perf_counter() - t0)
    return observed


def clock(fn, repeat=3):
    return median(samples(fn, repeat))


def report_samples(label, observed):
    values = ", ".join(f"{value:.6f}" for value in observed)
    print(
        f"{label} median {median(observed):.6f}s "
        f"(min {min(observed):.6f}s; samples [{values}])"
    )


def cuda_driver_version():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def bench_abstraction(B=8, T=50_000):
    from test_soundness import CFG, spring_instance

    m, _, rows = spring_instance(0)
    # tile the spring trajectory to length T
    reps = T // len(rows) + 1
    long_rows = (rows * reps)[:T]
    batch = [long_rows] * B

    ref = clock(lambda: [rabs.abstract_trajectory(x, m, config=CFG) for x in batch], 1)
    x = torch.tensor(batch, dtype=torch.float64)
    ten = clock(lambda: tabs.abstract_batch_tensor(x, m, config=CFG))
    rows_per_s = B * T
    print(f"abstraction  B={B} T={T}   reference {ref:8.2f}s ({rows_per_s/ref/1e6:5.2f}M samp/s)   tensor(cpu) {ten:8.2f}s ({rows_per_s/ten/1e6:5.2f}M samp/s)   speedup x{ref/ten:.1f}")
    if torch.cuda.is_available():
        xg = x.cuda()
        ts = torch.arange(T, dtype=torch.float64, device=xg.device).expand(B, T)
        frame = m.compile()
        sync = torch.cuda.synchronize
        dense = samples(
            lambda: (
                tabs.quantize_batch(xg, frame, CFG.landmark_atol),
                tabs.directions_batch(xg, ts, CFG),
            ),
            repeat=5,
            warmup=2,
            synchronize=sync,
        )
        end_to_end = samples(
            lambda: tabs.abstract_batch_tensor(xg, m, config=CFG),
            repeat=5,
            warmup=2,
            synchronize=sync,
        )
        report_samples("  tensor(cuda) dense quantize+directions:", dense)
        report_samples("  tensor(cuda) end-to-end abstraction:", end_to_end)
        teng = median(end_to_end)
        print(
            f"  end-to-end throughput {rows_per_s/teng/1e6:.3f}M samples/s; "
            f"speedup vs reference x{ref/teng:.2f}"
        )
        free, total = torch.cuda.mem_get_info()
        print(
            "  transfer policy: input host-to-device copy excluded; dense output "
            "stays on device; end-to-end timing includes ragged result copies "
            "back to Python"
        )
        print(
            f"  memory: {free/2**30:.2f} GiB free / {total/2**30:.2f} GiB total; "
            f"process allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
            f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB"
        )


def bench_engine():
    from test_qsim_phase2 import damped_spring

    m, initial = damped_spring()
    cfg = SimConfig(discover_landmarks=False, max_states=400)
    ref = clock(lambda: qr.qsim(m, initial, config=cfg))
    from dataclasses import replace

    ten = clock(lambda: qr.qsim(m, initial, config=replace(cfg, use_tensor=True)))
    print(f"engine (CPU; chattery damped spring, 400 states)   reference {ref*1e3:7.1f}ms   tensor {ten*1e3:7.1f}ms   ratio x{ref/ten:.2f}")


def bench_frontier(B=2048):
    from test_qsim_phase2 import spring

    m, initial = spring()
    frame = m.compile()
    result = qr.qsim(m, initial, config=SimConfig(discover_landmarks=False))
    active_idx = frame.region_named(frame.initial_region).constraint_idx
    domains_list = []
    for node in result.graph.nodes.values():
        table = point_successors if node.state.time is TimeTag.POINT else interval_successors
        domains_list.append(
            [table(node.state[v], frame.spaces[i]) for i, v in enumerate(frame.var_order)]
        )
    domains_list = (domains_list * (B // len(domains_list) + 1))[:B]
    active = tuple(frame.constraints[i] for i in active_idx)

    def ref():
        for doms in domains_list:
            pruned = filters.prune_domains(frame, [list(d) for d in doms], active)
            if pruned is not None:
                list(filters.assemble(frame, pruned, active))

    tengine.filtered_combos_batch(frame, domains_list[:4], active_idx)  # warm tables
    r = clock(ref)
    t = clock(lambda: tengine.filtered_combos_batch(frame, domains_list, active_idx))
    print(f"frontier expansion (CPU)  B={B} states   reference {r*1e3:7.1f}ms   tensor-batched {t*1e3:7.1f}ms   speedup x{r/t:.1f}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu-only"
    print(f"torch {torch.__version__} ({dev})")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"CUDA runtime {torch.version.cuda}; driver {cuda_driver_version()}; "
            f"GPU {props.name}; "
            f"compute capability {props.major}.{props.minor}; dtype float64"
        )
    bench_abstraction()
    bench_frontier()
    bench_engine()
