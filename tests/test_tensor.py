"""Phase-5 equivalence: the tensorized path must reproduce the reference
engine and abstraction pipeline exactly. Skipped when torch is absent."""

import pytest

torch = pytest.importorskip("torch")

import qrlib as qr
from qrlib import Qdir, SimConfig, TimeTag
from qrlib.engines import filters
from qrlib.engines.transitions import interval_successors, point_successors
from qrlib.quantity import QVal
from qrlib.tensor import abstraction as tabs
from qrlib.tensor import encoding, engine as tengine

from test_qsim_golden import bathtub, spring, utube
from test_qsim_phase2 import damped_spring, energy_filter
from test_regions import two_region_bathtub


# --- tables ≡ predicates ---------------------------------------------------


@pytest.mark.parametrize("maker", [bathtub, utube, spring])
def test_tables_equal_reference_predicates(maker):
    m, _ = maker()
    frame = m.compile()
    ts = encoding.tables_for(frame)
    from itertools import product

    concrete = (Qdir.DEC, Qdir.STD, Qdir.INC)
    for ci, cc in enumerate(frame.constraints):
        axes = [
            [QVal(r, d) for r in range(frame.spaces[vi].num_ranks) for d in concrete]
            for vi in cc.vars
        ]
        combos = list(product(*axes))
        cand = torch.tensor(
            [[encoding.qcode(qv) for qv in combo] for combo in combos]
        )
        got = ts.mask(ci, cand).tolist()
        want = [filters.check(cc, combo) for combo in combos]
        assert got == want


def test_frontier_encoding_round_trip():
    m, initial = bathtub()
    frame = m.compile()
    result = qr.qsim(m, initial)
    states = [n.state for n in result.graph.nodes.values() if n.model == frame]
    enc = encoding.encode_frontier(states, frame.var_order)
    assert enc.shape == (len(states), 2 * len(frame.var_order))
    dec = encoding.decode_frontier(enc, frame.var_order, states[0].time)
    assert [s.values for s in dec] == [s.values for s in states]


# --- engine equivalence ----------------------------------------------------

ENGINE_CASES = [
    ("bathtub", bathtub, SimConfig()),
    ("bathtub-envision", bathtub, SimConfig(envisionment=True)),
    ("utube", utube, SimConfig()),
    ("spring-nodiscover", spring, SimConfig(discover_landmarks=False)),
    ("spring-envision", spring, SimConfig(discover_landmarks=False, envisionment=True)),
    ("spring-discover", spring, SimConfig(max_states=200)),
    ("spring-energy", spring, SimConfig(successor_filters=(energy_filter,))),
    ("damped-ignore", damped_spring, SimConfig(discover_landmarks=False, max_states=400, ignore_qdir=("s", "a"))),
    ("damped-chatter", damped_spring, SimConfig(discover_landmarks=False, max_states=150)),
]


@pytest.mark.parametrize("name,maker,cfg", ENGINE_CASES, ids=[c[0] for c in ENGINE_CASES])
def test_engine_equivalence(name, maker, cfg):
    m, initial = maker()
    from dataclasses import replace

    ref = qr.qsim(m, initial, config=replace(cfg, use_tensor=False))
    ten = qr.qsim(m, initial, config=replace(cfg, use_tensor=True))
    assert ref.status == ten.status
    assert ref.stats == ten.stats
    assert ref.graph.export() == ten.graph.export()
    assert len(ref.behaviors()) == len(ten.behaviors())


def test_engine_equivalence_regions():
    m, initial, _ = two_region_bathtub()
    ref = qr.qsim(m, initial)
    ten = qr.qsim(m, initial, config=SimConfig(use_tensor=True))
    assert ref.stats == ten.stats
    assert ref.graph.export() == ten.graph.export()


# --- batched frontier expansion --------------------------------------------


def test_batched_expansion_matches_per_state():
    m, initial = spring()
    frame = m.compile()
    result = qr.qsim(m, initial, config=SimConfig(discover_landmarks=False))
    active_idx = frame.region_named(frame.initial_region).constraint_idx

    domains_list = []
    for node in result.graph.nodes.values():
        table = (
            point_successors
            if node.state.time is TimeTag.POINT
            else interval_successors
        )
        domains_list.append(
            [
                table(node.state[v], frame.spaces[i])
                for i, v in enumerate(frame.var_order)
            ]
        )
    batched = tengine.filtered_combos_batch(frame, domains_list, active_idx)
    single = [
        tengine.filtered_combos(frame, doms, active_idx) for doms in domains_list
    ]
    assert batched == single
    # and the tensor path equals the reference pipeline, order included
    for doms, combos in zip(domains_list, single):
        active = tuple(frame.constraints[i] for i in active_idx)
        pruned = filters.prune_domains(frame, [list(d) for d in doms], active)
        want = [] if pruned is None else list(filters.assemble(frame, pruned, active))
        assert combos == want


# --- abstraction equivalence -----------------------------------------------


def test_abstraction_equivalence_on_harness_trajectories():
    from qrlib.bridge import abstraction as rabs
    from test_soundness import CFG, bathtub_instance, spring_instance

    m, _, rows, times = bathtub_instance(0, overflow=True)
    ref = rabs.abstract_trajectory(rows, m, times=times, config=CFG)
    (ten,) = tabs.abstract_batch_tensor(rows, m, times=times, config=CFG)
    assert ten == ref

    m, _, rows = spring_instance(1)
    ref = rabs.abstract_trajectory(rows, m, config=CFG)
    (ten,) = tabs.abstract_batch_tensor(rows, m, config=CFG)
    assert ten == ref


def test_abstraction_batch_with_modes_matches_reference():
    from qrlib.bridge import abstraction as rabs
    from qrlib.bridge.abstraction import AbstractionConfig
    from test_regions import two_region_bathtub

    m, _, (full, inflow, f1, f2) = two_region_bathtub(values=True)
    dt = 0.005
    A, t = 0.0, 0.0
    rows, times, modes = [], [], []
    while A < full:
        rows.append([A, f1(A), f2(f1(A)), inflow, inflow - f2(f1(A))])
        times.append(t)
        modes.append("filling")
        prev = A
        A += dt * (inflow - f2(f1(A)))
        t += dt
        if A >= full:
            frac = (full - prev) / (A - prev)
            rows.append([full, f1(full), f2(f1(full)), inflow, inflow - f2(f1(full))])
            times.append(t - dt + frac * dt)
            modes.append("filling")
            break
    for _ in range(40):
        rows.append([full, f1(full), f2(f1(full)), inflow, inflow - f2(f1(full))])
        times.append(times[-1] + dt)
        modes.append("overflowing")

    cfg = AbstractionConfig(landmark_atol=1e-9, direction_eps=1e-4, debounce=3)
    ref = rabs.abstract_trajectory(rows, m, times=times, modes=modes, config=cfg)
    (ten,) = tabs.abstract_batch_tensor(
        rows, m, times=times, modes=[modes], config=cfg
    )
    assert ten == ref


def test_quantize_batch_out_of_space_raises():
    from test_qsim_golden import bathtub

    m, _ = bathtub()
    frame = m.compile()
    # bathtub landmarks carry no numeric values -> informative error
    with pytest.raises(ValueError, match="without"):
        tabs.quantize_batch(torch.zeros((1, 2, 5), dtype=torch.float64), frame, 1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_abstraction_default_and_explicit_times_match_cpu():
    from test_soundness import CFG, spring_instance

    m, _, rows = spring_instance(0)
    rows = rows[:32]
    x_cpu = torch.tensor(rows, dtype=torch.float64)
    x_cuda = x_cpu.cuda()

    (cpu_default,) = tabs.abstract_batch_tensor(x_cpu, m, config=CFG)
    (cuda_default,) = tabs.abstract_batch_tensor(x_cuda, m, config=CFG)
    assert cuda_default == cpu_default

    times = torch.linspace(0.0, 0.31, len(rows), dtype=torch.float64)
    (cpu_timed,) = tabs.abstract_batch_tensor(x_cpu, m, times=times, config=CFG)
    # CPU times are accepted and placed with the CUDA trajectory.
    (cuda_timed,) = tabs.abstract_batch_tensor(x_cuda, m, times=times, config=CFG)
    assert cuda_timed == cpu_timed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_one_step_and_out_of_space_behavior():
    from test_soundness import CFG, bathtub_instance, spring_instance

    m, _, rows = spring_instance(0)
    one_cpu = torch.tensor([rows[0]], dtype=torch.float64)
    one_cuda = one_cpu.cuda()
    (cpu_result,) = tabs.abstract_batch_tensor(one_cpu, m, config=CFG)
    (cuda_result,) = tabs.abstract_batch_tensor(one_cuda, m, config=CFG)
    assert cuda_result == cpu_result

    bounded, _, bounded_rows, _ = bathtub_instance(0, overflow=True)
    bad = torch.tensor(
        [[bounded_rows[0]]], dtype=torch.float64, device="cuda"
    )
    bad[..., 0] = 1e9
    with pytest.raises(ValueError, match="space"):
        tabs.quantize_batch(bad, bounded.compile(), CFG.landmark_atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_full_abstraction_matches_reference():
    from qrlib.bridge import abstraction as rabs
    from test_soundness import CFG, bathtub_instance

    m, _, rows, times = bathtub_instance(0, overflow=True)
    reference = rabs.abstract_trajectory(rows, m, times=times, config=CFG)
    x = torch.tensor(rows, dtype=torch.float64, device="cuda")
    (actual,) = tabs.abstract_batch_tensor(x, m, times=times, config=CFG)
    assert actual == reference
