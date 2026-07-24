"""Production-shaped scale qualification for tensor abstraction and ensembles.

These profiles are explicit workload contracts, not claims about external
adoption telemetry. They answer three engineering questions:

1. Do the documented V<=64 and million-timestep abstraction targets fit?
2. Where does CUDA beat tensor CPU after final Python materialization?
3. Does a first-class heterogeneous model-ensemble tensor axis pay for its
   representation complexity, or should models be grouped by frame?

Run the default qualification:

    python benchmarks/bench_scale_profiles.py

Emit the complete machine-readable record:

    python benchmarks/bench_scale_profiles.py --json results.json
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
import platform
from pathlib import Path
from statistics import median
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

import qrlib as qr
from qrlib import Landmark, QVal, Qdir
from qrlib.bridge.abstraction import AbstractionConfig
from qrlib.tensor import abstraction as tabs
from qrlib.tensor import encoding, engine as tengine


@dataclass(frozen=True)
class AbstractionProfile:
    name: str
    batch: int
    timesteps: int
    variables: int
    pattern: str
    cycles: float
    purpose: str


ABSTRACTION_PROFILES = (
    AbstractionProfile(
        "interactive",
        1,
        2_048,
        4,
        "smooth",
        2.0,
        "single trajectory used by an interactive analysis",
    ),
    AbstractionProfile(
        "service_batch",
        32,
        4_096,
        12,
        "smooth",
        3.0,
        "moderate request batch with multivariate trajectories",
    ),
    AbstractionProfile(
        "million_timesteps",
        16,
        62_500,
        8,
        "smooth",
        4.0,
        "documented upper target of one million trajectory timesteps",
    ),
    AbstractionProfile(
        "wide_v64",
        8,
        8_192,
        64,
        "smooth",
        2.0,
        "documented upper target for variable count",
    ),
    AbstractionProfile(
        "high_run_density",
        4,
        4_096,
        8,
        "alternating",
        0.0,
        "ragged-tail stress where nearly every sample changes code",
    ),
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_driver_version():
    try:
        completed = subprocess.run(
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
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _samples(fn, device: torch.device, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
        _sync(device)
    observed = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        observed.append(time.perf_counter() - start)
    return observed


def _rss_bytes() -> int | None:
    """Current process resident memory using only the standard library."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        )
        handle = get_process()
        ok = get_info(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        return resident_pages * page_size
    except (AttributeError, OSError, ValueError, IndexError):
        return None


def _host_peak_delta(fn) -> tuple[object, int | None, int | None]:
    """Run once while polling RSS; kept separate from latency samples."""
    baseline = _rss_bytes()
    if baseline is None:
        return fn(), None, None
    peak = [baseline]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.002):
            current = _rss_bytes()
            if current is not None:
                peak[0] = max(peak[0], current)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    final = _rss_bytes()
    if final is not None:
        peak[0] = max(peak[0], final)
    return result, peak[0], max(0, peak[0] - baseline)


def _abstraction_model(variables: int):
    model = qr.Model(f"scale-v{variables}")
    for index in range(variables):
        model.variable(
            f"x{index}",
            landmarks=(Landmark("0", value=0.0),),
            unbounded=True,
        )
    return model.compile()


def _trajectory(profile: AbstractionProfile, device: torch.device):
    B, T, V = profile.batch, profile.timesteps, profile.variables
    if profile.pattern == "alternating":
        sign = (torch.arange(T, dtype=torch.int64) % 2) * 2 - 1
        scale = torch.linspace(0.5, 1.5, V, dtype=torch.float64)
        batch_scale = torch.linspace(0.9, 1.1, B, dtype=torch.float64)
        data = (
            sign.to(torch.float64).reshape(1, T, 1)
            * scale.reshape(1, 1, V)
            * batch_scale.reshape(B, 1, 1)
        )
    else:
        phase = torch.linspace(0.0, math.pi / 3, B, dtype=torch.float64)
        frequency = torch.linspace(1.0, 2.0, V, dtype=torch.float64)
        time_axis = torch.linspace(
            0.0, 2 * math.pi * profile.cycles, T, dtype=torch.float64
        )
        data = torch.sin(
            time_axis.reshape(1, T, 1) * frequency.reshape(1, 1, V)
            + phase.reshape(B, 1, 1)
        )
    return data.to(device=device)


def _packed_stats(x, frame, config):
    B, T, _ = x.shape
    ts = (
        torch.arange(T, dtype=torch.float64, device=x.device)
        .unsqueeze(0)
        .expand(B, T)
    )
    ranks = tabs.quantize_batch(x, frame, config.landmark_atol)
    dirs = tabs.directions_batch(x, ts, config)
    change = (
        (ranks[:, 1:] != ranks[:, :-1])
        | (dirs[:, 1:] != dirs[:, :-1])
    ).any(-1)
    packed, times = tabs._pack_runs_for_host(ranks, dirs, ts, change)
    payload = (
        packed.numel() * packed.element_size()
        + times.numel() * times.element_size()
    )
    return len(packed), payload


def qualify_abstraction(profile: AbstractionProfile, device: torch.device):
    frame = _abstraction_model(profile.variables)
    config = AbstractionConfig(debounce=3)
    x = _trajectory(profile, device)

    def run():
        return tabs.abstract_batch_tensor(x, frame, config=config)

    repeats = 5 if profile.name == "interactive" else 3
    packed_samples = _samples(
        lambda: _packed_stats(x, frame, config),
        device,
        repeats=repeats,
        warmup=1,
    )
    observed = _samples(run, device, repeats=repeats, warmup=1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result, host_working_set_peak, host_peak_delta = _host_peak_delta(run)
    _sync(device)
    cuda_peak = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    run_count, payload_bytes = _packed_stats(x, frame, config)
    qualitative_states = sum(len(item.states) for item in result)
    total_steps = profile.batch * profile.timesteps
    input_bytes = x.numel() * x.element_size()
    return {
        "profile": profile.name,
        "device": device.type,
        "batch": profile.batch,
        "timesteps": profile.timesteps,
        "variables": profile.variables,
        "total_timesteps": total_steps,
        "scalar_values": x.numel(),
        "pattern": profile.pattern,
        "packed_prefix_median_s": median(packed_samples),
        "packed_prefix_min_s": min(packed_samples),
        "latency_median_s": median(observed),
        "latency_min_s": min(observed),
        "samples_s": total_steps / median(observed),
        "scalar_values_s": x.numel() / median(observed),
        "actual_runs": run_count,
        "run_density": run_count / total_steps,
        "qualitative_states": qualitative_states,
        "packed_payload_bytes": payload_bytes,
        "input_bytes": input_bytes,
        "host_working_set_peak_bytes": host_working_set_peak,
        "host_peak_delta_bytes": host_peak_delta,
        "cuda_peak_allocated_bytes": cuda_peak,
        "packed_prefix_samples_s": packed_samples,
        "timing_samples_s": observed,
    }


def _chain_frame(variables: int, variant: int):
    model = qr.Model(f"ensemble-v{variables}-{variant}")
    for index in range(variables):
        model.variable(f"x{index}", landmarks=("0",), unbounded=True)
    for index in range(variables - 1):
        relation = qr.MMinus if (variant >> index) & 1 else qr.MPlus
        model.constrain(relation(f"x{index}", f"x{index + 1}"))
    return model.compile()


def _domains(variables: int):
    domain = [
        QVal(1, Qdir.DEC),
        QVal(1, Qdir.STD),
        QVal(1, Qdir.INC),
    ]
    return [domain[:] for _ in range(variables)]


def qualify_ensembles(counts=(1, 8, 32), variables: int = 8):
    rows = []
    domains = _domains(variables)
    shared = _chain_frame(variables, 0)
    active = shared.regions[0].constraint_idx
    encoding.tables_for(shared)
    for count in counts:
        workload = [[column[:] for column in domains] for _ in range(count)]
        sequential = _samples(
            lambda: [
                tengine.filtered_combos(shared, item, active)
                for item in workload
            ],
            torch.device("cpu"),
            repeats=3,
            warmup=1,
        )
        batched = _samples(
            lambda: tengine.filtered_combos_batch(shared, workload, active),
            torch.device("cpu"),
            repeats=3,
            warmup=1,
        )

        variants = [_chain_frame(variables, index + 1) for index in range(count)]
        cold_start = time.perf_counter()
        for frame in variants:
            encoding.tables_for(frame)
        cold_table_s = time.perf_counter() - cold_start

        def heterogeneous():
            return [
                tengine.filtered_combos(
                    frame, domains, frame.regions[0].constraint_idx
                )
                for frame in variants
            ]

        heterogeneous_samples = _samples(
            heterogeneous,
            torch.device("cpu"),
            repeats=3,
            warmup=1,
        )
        rows.append(
            {
                "models": count,
                "variables": variables,
                "interpretations_per_model": 3**variables,
                "homogeneous_sequential_median_s": median(sequential),
                "homogeneous_batched_median_s": median(batched),
                "homogeneous_batch_speedup": median(sequential) / median(batched),
                "heterogeneous_table_build_s": cold_table_s,
                "heterogeneous_warm_median_s": median(heterogeneous_samples),
                "heterogeneous_warm_per_model_s": (
                    median(heterogeneous_samples) / count
                ),
            }
        )
    return rows


def _mib(value):
    return "n/a" if value is None else f"{value / 2**20:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=Path, help="write the full machine-readable result"
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="skip CUDA even when available"
    )
    args = parser.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append(torch.device("cuda"))

    abstraction = []
    print("abstraction profiles")
    print(
        "profile              dev   B       T   V  prefix  median  Mstep/s "
        "run%  packedMiB hostPeakMiB cudaPeakMiB"
    )
    for profile in ABSTRACTION_PROFILES:
        for device in devices:
            row = qualify_abstraction(profile, device)
            abstraction.append(row)
            print(
                f"{profile.name:20} {device.type:4} "
                f"{profile.batch:3d} {profile.timesteps:7d} "
                f"{profile.variables:3d} "
                f"{row['packed_prefix_median_s']:7.4f} "
                f"{row['latency_median_s']:7.4f} "
                f"{row['samples_s']/1e6:7.3f} "
                f"{100*row['run_density']:5.1f} "
                f"{row['packed_payload_bytes']/2**20:9.3f} "
                f"{_mib(row['host_peak_delta_bytes']):>8} "
                f"{_mib(row['cuda_peak_allocated_bytes']):>11}"
            )

    print("\nmodel-ensemble profiles (CPU tensor filtering)")
    print(
        "models V product  shared-seq shared-batch speedup "
        "hetero-warm per-model table-build"
    )
    ensembles = qualify_ensembles()
    for row in ensembles:
        print(
            f"{row['models']:6d} {row['variables']:1d} "
            f"{row['interpretations_per_model']:7d} "
            f"{row['homogeneous_sequential_median_s']:10.4f} "
            f"{row['homogeneous_batched_median_s']:12.4f} "
            f"{row['homogeneous_batch_speedup']:7.2f} "
            f"{row['heterogeneous_warm_median_s']:11.4f} "
            f"{row['heterogeneous_warm_per_model_s']:9.4f} "
            f"{row['heterogeneous_table_build_s']:11.4f}"
        )

    record = {
        "schema": "qrlib.scale-profile/v1",
        "qualification": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_driver": _cuda_driver_version(),
            "cuda_available": torch.cuda.is_available(),
            "profiles": [asdict(profile) for profile in ABSTRACTION_PROFILES],
            "claim_scope": (
                "production-shaped workload contracts; no external-user "
                "telemetry"
            ),
        },
        "abstraction": abstraction,
        "ensembles": ensembles,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record["qualification"]["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
        }
    if args.json:
        args.json.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
