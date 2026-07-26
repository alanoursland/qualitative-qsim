"""Phase-5 equivalence: the tensorized path must reproduce the reference
engine and abstraction pipeline exactly. Skipped when torch is absent."""

import random

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
    ("spring-discover", spring, SimConfig.classic(max_states=200)),
    (
        "spring-energy",
        spring,
        SimConfig.classic(successor_filters=(energy_filter,)),
    ),
    ("damped-ignore", damped_spring, SimConfig(discover_landmarks=False, max_states=400, ignore_qdir=("s", "a"))),
    ("damped-chatter", damped_spring, SimConfig(discover_landmarks=False, max_states=150)),
]


def semantic_stats(result):
    return {key: value for key, value in result.stats.items() if key != "backend"}


@pytest.mark.parametrize("name,maker,cfg", ENGINE_CASES, ids=[c[0] for c in ENGINE_CASES])
def test_engine_equivalence(name, maker, cfg):
    m, initial = maker()
    from dataclasses import replace

    ref = qr.qsim(m, initial, config=replace(cfg, use_tensor=False))
    ten = qr.qsim(m, initial, config=replace(cfg, use_tensor=True))
    assert ref.status == ten.status
    assert semantic_stats(ref) == semantic_stats(ten)
    assert ref.stats["backend"]["reference_filter_calls"] > 0
    assert ten.stats["backend"]["tensor_filter_calls"] > 0
    assert ref.graph.export() == ten.graph.export()
    assert len(ref.behaviors()) == len(ten.behaviors())


def test_engine_equivalence_regions():
    m, initial, _ = two_region_bathtub()
    ref = qr.qsim(m, initial)
    ten = qr.qsim(m, initial, config=SimConfig(use_tensor=True))
    assert semantic_stats(ref) == semantic_stats(ten)
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


def _random_filter_model(seed):
    rng = random.Random(seed)
    count = rng.randint(2, 4)
    names = [f"v{i}" for i in range(count)]
    model = qr.Model(f"random-filter-{seed}")
    for name in names:
        style = rng.choice(("unbounded", "upper", "bounded"))
        if style == "unbounded":
            model.variable(name, landmarks=("0",), unbounded=True)
        elif style == "upper":
            model.variable(name, landmarks=("0",), upper_unbounded=True)
        else:
            model.variable(name, landmarks=("0", "top"))

    left, right = rng.sample(names, 2)
    model.constrain(qr.Deriv(left, right))
    for _ in range(rng.randint(0, 3)):
        kind = rng.choice(("mplus", "mminus", "deriv", "constant", "add"))
        if kind == "constant":
            model.constrain(qr.Constant(rng.choice(names)))
        elif kind == "add" and count >= 3:
            a, b, total = rng.sample(names, 3)
            model.constrain(qr.Add(a, b, total))
        else:
            a, b = rng.sample(names, 2)
            relation = {
                "mplus": qr.MPlus,
                "mminus": qr.MMinus,
                "deriv": qr.Deriv,
                "add": qr.MPlus,
            }[kind]
            model.constrain(relation(a, b))
    return model


NONEMPTY_FILTER_SEEDS = (
    0, 2, 3, 4, 5, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 20, 21, 22,
    24, 25, 26, 27, 28, 29,
)


@pytest.mark.parametrize("seed", NONEMPTY_FILTER_SEEDS)
def test_random_filtered_batches_match_single_state_filtering(seed):
    """Per-state tensor filtering is identical, including result order."""
    rng = random.Random(3000 + seed)
    frame = _random_filter_model(seed).compile()
    active = tuple(range(len(frame.constraints)))
    domains_list = []
    for _ in range(4):
        domains = []
        for space in frame.spaces:
            domains.append(
                [
                    QVal(rng.randrange(space.num_ranks), direction)
                    for direction in (Qdir.DEC, Qdir.STD, Qdir.INC)
                ]
            )
        domains_list.append(domains)

    batched = tengine.filtered_combos_batch(frame, domains_list, active)
    assert any(batched), "the generated case must exercise accepted combinations"
    assert batched == [
        tengine.filtered_combos(frame, domains, active)
        for domains in domains_list
    ]


@pytest.mark.parametrize("maker", (bathtub, utube, spring))
def test_named_backend_selector_matches_legacy_override(maker):
    """Reference and tensor backends produce identical results by contract."""
    model, initial = maker()
    config = dict(max_states=100, max_depth=12)
    reference = qr.qsim(
        model,
        initial,
        config=SimConfig(backend="reference", **config),
    )
    legacy = qr.qsim(
        model,
        initial,
        config=SimConfig(use_tensor=False, **config),
    )
    tensor = qr.qsim(
        model,
        initial,
        config=SimConfig(backend="tensor", **config),
    )

    assert reference.graph.export() == legacy.graph.export()
    assert reference.graph.export() == tensor.graph.export()
    assert reference.status is legacy.status is tensor.status


def test_unknown_backend_is_rejected():
    """An unknown backend name must not silently fall back."""
    with pytest.raises(ValueError, match="backend"):
        SimConfig(backend="tensorflow")


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


def test_ragged_tail_packs_only_actual_runs_and_endpoint_times():
    ranks = torch.tensor(
        [
            [[1], [1], [2], [2], [3]],
            [[4], [4], [4], [4], [4]],
        ],
        dtype=torch.long,
    )
    dirs = torch.full_like(ranks, int(Qdir.STD))
    times = torch.tensor(
        [[0.0, 0.5, 2.0, 3.0, 5.0], [10.0, 11.0, 13.0, 16.0, 20.0]],
        dtype=torch.float64,
    )
    change = ((ranks[:, 1:] != ranks[:, :-1]) | (
        dirs[:, 1:] != dirs[:, :-1]
    )).any(-1)
    packed, endpoints = tabs._pack_runs_for_host(
        ranks, dirs, times, change
    )
    # Three actual runs in batch 0 and one in batch 1: no B*max_runs padding,
    # and only two timestamps per run cross into the host view.
    assert packed.device.type == "cpu"
    assert endpoints.device.type == "cpu"
    assert packed.shape == (4, 5)  # batch/start/end + one rank + one direction
    assert endpoints.shape == (4, 2)
    assert packed[:, :3].tolist() == [
        [0, 0, 2],
        [0, 2, 4],
        [0, 4, 5],
        [1, 0, 5],
    ]
    assert endpoints.tolist() == [
        [0.0, 0.5],
        [2.0, 3.0],
        [5.0, 5.0],
        [10.0, 20.0],
    ]


def test_compact_control_debounce_matches_reference_fixpoint():
    from qrlib.bridge import abstraction as rabs

    rng = random.Random(20260725)
    for _ in range(200):
        count = rng.randint(1, 50)
        debounce = rng.randint(1, 6)
        labels = [rng.randrange(5)]
        for _ in range(1, count):
            choices = [label for label in range(5) if label != labels[-1]]
            labels.append(rng.choice(choices))
        lengths = [rng.randint(1, 8) for _ in range(count)]
        starts = []
        cursor = 0
        for length in lengths:
            starts.append(cursor)
            cursor += length

        runs = [
            [(label, None), start, start + length]
            for label, start, length in zip(labels, starts, lengths)
        ]
        expected = rabs._debounce(runs, debounce)
        control = torch.tensor(
            [[0, start, label] for start, label in zip(starts, labels)],
            dtype=torch.long,
        )
        actual = tabs._debounce_run_control(control, debounce, cursor)
        decoded = [
            [(labels[representative], None), start, end]
            for _batch, start, end, representative in actual.tolist()
        ]
        assert decoded == expected


def test_high_density_debounce_compacts_full_host_payload():
    B, T, V = 4, 4096, 8
    sample = torch.arange(T) % 2
    ranks = sample.reshape(1, T, 1).expand(B, T, V).to(torch.long)
    dirs = torch.full_like(ranks, int(Qdir.STD))
    times = (
        torch.arange(T, dtype=torch.float64)
        .reshape(1, T)
        .expand(B, T)
    )
    change = torch.ones((B, T - 1), dtype=torch.bool)

    raw, raw_times = tabs._pack_runs_for_host(ranks, dirs, times, change)
    telemetry = {}
    packed, packed_times = tabs._pack_debounced_runs_for_host(
        ranks, dirs, times, change, debounce=3, telemetry=telemetry
    )

    assert len(raw) == B * T
    assert len(packed) == 2 * B
    for batch in range(B):
        assert packed[2 * batch : 2 * batch + 2, :3].tolist() == [
            [batch, 0, T - 1],
            [batch, T - 1, T],
        ]
    raw_payload = (
        raw.numel() * raw.element_size()
        + raw_times.numel() * raw_times.element_size()
    )
    compact_payload = (
        telemetry["control_payload_bytes"]
        + telemetry["survivor_payload_bytes"]
    )
    assert telemetry["raw_runs"] == B * T
    assert telemetry["surviving_runs"] == 2 * B
    assert telemetry["strategy"] == "debounced_dense"
    assert compact_payload < raw_payload / 5


def test_low_density_uses_original_packed_tail_without_overhead():
    T = 100
    ranks = torch.ones((1, T, 2), dtype=torch.long)
    ranks[:, 50:] = 3
    dirs = torch.full_like(ranks, int(Qdir.STD))
    times = torch.arange(T, dtype=torch.float64).reshape(1, T)
    change = ((ranks[:, 1:] != ranks[:, :-1]) | (
        dirs[:, 1:] != dirs[:, :-1]
    )).any(-1)

    expected = tabs._pack_runs_for_host(ranks, dirs, times, change)
    telemetry = {}
    actual = tabs._pack_debounced_runs_for_host(
        ranks, dirs, times, change, debounce=3, telemetry=telemetry
    )
    assert actual[0].equal(expected[0])
    assert actual[1].equal(expected[1])
    assert telemetry["strategy"] == "raw_sparse"
    assert telemetry["control_payload_bytes"] == 0


def test_high_density_end_to_end_matches_reference():
    from qrlib.bridge import abstraction as rabs

    model = qr.Model("dense-debounce")
    for index in range(3):
        model.variable(
            f"x{index}",
            landmarks=(qr.Landmark("0", value=0.0),),
            unbounded=True,
        )
    T = 128
    sign = (torch.arange(T, dtype=torch.float64) % 2) * 2 - 1
    rows = torch.stack((sign, 2 * sign, 3 * sign), dim=1)
    cfg = rabs.AbstractionConfig(
        debounce=3,
        direction_eps=0.0,
        eps_relative=False,
    )
    reference = rabs.abstract_trajectory(rows, model, config=cfg)
    (actual,) = tabs.abstract_batch_tensor(rows, model, config=cfg)
    assert actual == reference


def test_tensor_abstraction_validates_mode_shape():
    from test_soundness import CFG, spring_instance

    m, _, rows = spring_instance(0)
    with pytest.raises(ValueError, match="modes must have shape"):
        tabs.abstract_batch_tensor(
            rows[:5], m, modes=[["only-one"]], config=CFG
        )


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


def test_endpoint_extremum_policy_matches_reference():
    import math

    from qrlib.bridge import abstraction as rabs

    landmarks = (
        qr.Landmark("NEG", value=-1.0),
        qr.Landmark("0", value=0.0),
        qr.Landmark("POS", value=1.0),
    )
    m = qr.Model("oscillator-endpoint")
    for name in ("x", "v", "a"):
        m.variable(name, landmarks=landmarks)
    times = torch.linspace(0.0, 2.0 * math.pi, 17, dtype=torch.float64)
    x = torch.stack(
        (torch.cos(times), -torch.sin(times), -torch.cos(times)), dim=1
    )
    cfg = rabs.AbstractionConfig(
        debounce=1,
        direction_eps=1e-6,
        eps_relative=False,
    )
    reference = rabs.abstract_trajectory(x, m, times=times, config=cfg)
    (actual,) = tabs.abstract_batch_tensor(x, m, times=times, config=cfg)
    assert actual == reference
