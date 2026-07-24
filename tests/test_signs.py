"""Phase-4: sign-structure intake, estimation from data, the structure
export, and the downward consistency checker."""

import random

import pytest

import qrlib as qr
from qrlib import Landmark, Qdir, TimeTag
from qrlib.bridge import signs
from qrlib.constraints import Add, Constant, Deriv, MMinus, MPlus


def test_model_from_signs_damped_spring_structure():
    # dx/dt depends positively on v; dv/dt negatively on both x and v
    m = signs.model_from_signs(["x", "v"], [[0, 1], [-1, -1]])
    assert set(m.variables) == {"x", "v", "d_x", "t_v_x", "t_v_v", "d_v"}
    kinds = [type(c).__name__ for c in m.constraints]
    assert kinds.count("MPlus") == 1  # v -> d_x
    assert kinds.count("MMinus") == 2  # x -> t_v_x, v -> t_v_v
    assert kinds.count("Add") == 1  # t_v_x + t_v_v = d_v
    assert kinds.count("Deriv") == 2
    m.compile()  # compiles cleanly


def test_model_from_signs_zero_row_and_unknown():
    m = signs.model_from_signs(["x", "u"], [[0, 0], [signs.UNKNOWN, 0]])
    # x has no influences at all -> constant
    assert any(isinstance(c, Constant) and c.x == "x" for c in m.constraints)
    # u depends on x in an unknown direction: derivative variable exists,
    # Deriv ties it, but no monotonic constraint is asserted
    assert "d_u" in m.variables
    assert any(isinstance(c, Deriv) and c.x == "u" for c in m.constraints)
    assert not any(isinstance(c, (MPlus, MMinus)) for c in m.constraints)


def test_model_from_signs_runs_in_the_engine():
    m = signs.model_from_signs(["x", "v"], [[0, 1], [-1, -1]])
    initial = m.state(
        time=TimeTag.POINT,
        x=("0", Qdir.INC),
        v=(("0", "+inf"), Qdir.DEC),
        d_x=(("0", "+inf"), Qdir.DEC),
        t_v_x=(("-inf", "0"), Qdir.DEC),
        t_v_v=(("-inf", "0"), Qdir.INC),
        d_v=(("-inf", "0"), Qdir.DEC),
    )
    result = qr.qsim(m, initial, config=qr.SimConfig(max_states=80))
    assert result.stats["nodes"] > 1  # weak model, but it simulates


def test_estimate_signs_recovers_linear_system():
    rng = random.Random(0)
    A = [[-1.0, -2.0, 0.0], [3.0, -4.0, 0.5], [0.0, 1.5, -0.7]]
    xs, dxs = [], []
    for _ in range(200):
        x = [rng.uniform(-2, 2) for _ in range(3)]
        xs.append(x)
        dxs.append([sum(A[i][j] * x[j] for j in range(3)) for i in range(3)])
    est, conf = signs.estimate_signs(xs, dxs)
    S = signs.signs_with_threshold(est, conf, threshold=5.0)
    for i in range(3):
        for j in range(3):
            true = (A[i][j] > 0) - (A[i][j] < 0)
            if true == 0:
                assert S[i][j] is signs.UNKNOWN  # never a confident zero
            else:
                assert S[i][j] == true, (i, j)


def test_estimate_signs_average_monotonicity_for_nonlinear():
    rng = random.Random(1)
    xs, dxs = [], []
    for _ in range(300):
        a, b = rng.uniform(0.1, 2), rng.uniform(0.1, 2)
        xs.append([a, b])
        # da/dt = -a^1.3 (monotone dec in a), db/dt = +sqrt(a) (inc in a)
        dxs.append([-(a**1.3), a**0.5])
    est, conf = signs.estimate_signs(xs, dxs)
    S = signs.signs_with_threshold(est, conf, threshold=5.0)
    assert S[0][0] == -1
    assert S[1][0] == 1
    assert S[0][1] is signs.UNKNOWN
    assert S[1][1] is signs.UNKNOWN


def test_calibrated_signs_recover_stable_noisy_nonlinear_effects():
    rng = random.Random(11)
    xs, dxs = [], []
    for _ in range(240):
        a, b = rng.uniform(0.1, 2.0), rng.uniform(-1.5, 1.5)
        xs.append([a, b])
        dxs.append(
            [
                -(a**1.3) + 0.03 * rng.gauss(0.0, 1.0),
                a**0.5 + 0.03 * rng.gauss(0.0, 1.0),
            ]
        )
    estimate = signs.estimate_signs_calibrated(
        xs, dxs, resamples=120, seed=7
    )
    matrix = estimate.threshold(0.95)
    assert matrix[0][0] == -1
    assert matrix[1][0] == 1
    assert matrix[0][1] is signs.UNKNOWN
    assert matrix[1][1] is signs.UNKNOWN
    assert 0.0 <= estimate.confidence[0][0] <= 1.0
    assert estimate.to_dict()["method"] == "bootstrap-sign-agreement"


def test_calibrated_signs_leave_weak_noisy_effect_unknown():
    rng = random.Random(19)
    xs, dxs = [], []
    for _ in range(180):
        a, b = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        xs.append([a, b])
        dxs.append([2.0 * a + rng.gauss(0.0, 0.1), rng.gauss(0.0, 1.0)])
    estimate = signs.estimate_signs_calibrated(
        xs, dxs, resamples=160, seed=3
    )
    matrix = estimate.threshold(0.99)
    assert matrix[0][0] == 1
    assert matrix[1][0] is signs.UNKNOWN
    assert matrix[1][1] is signs.UNKNOWN


def test_calibrated_signs_are_reproducible_and_legacy_api_is_unchanged():
    xs = [[float(i), float((i * 7) % 11)] for i in range(20)]
    dxs = [[2.0 * row[0], -3.0 * row[1]] for row in xs]
    first = signs.estimate_signs_calibrated(xs, dxs, resamples=20, seed=42)
    second = signs.estimate_signs_calibrated(xs, dxs, resamples=20, seed=42)
    assert first == second
    legacy_signs, legacy_confidence = signs.estimate_signs(xs, dxs)
    assert isinstance(legacy_signs, list)
    assert isinstance(legacy_confidence, list)


def test_calibrated_sign_parameter_validation():
    xs = [[float(i)] for i in range(4)]
    dxs = [[-row[0]] for row in xs]
    with pytest.raises(ValueError, match="positive integer"):
        signs.estimate_signs_calibrated(xs, dxs, resamples=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        signs.estimate_signs_calibrated(xs, dxs, resamples=2).threshold(1.1)


def test_sign_structure_export():
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=("0", "FULL"))
    m.variable("level", landmarks=("0", "TOP"))
    m.variable("outflow", landmarks=("0", "OMAX"))
    m.variable("inflow", landmarks=("0", "IF*"))
    m.variable("netflow", landmarks=("0",), unbounded=True)
    m.constrain(qr.MPlus("amount", "level", cvals=(("0", "0"),)))
    m.constrain(qr.MMinus("level", "netflow"))
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    m.constrain(qr.Deriv("amount", "netflow"))
    m.constrain(qr.Constant("inflow"))

    ss = m.sign_structure()
    assert ("amount", "level", 1) in ss.monotone
    assert ("level", "netflow", -1) in ss.monotone
    assert ("netflow", "outflow", "inflow") in ss.sums
    assert ("amount", "netflow") in ss.derivatives
    assert "inflow" in ss.constants
    assert (("amount", "level"), (("0", "0"),)) in ss.corresponding


def bathtub_rows(negate_level=False):
    f1 = lambda A: 1.2 * A
    f2 = lambda L: 0.4 * L
    inflow = 1.0
    rows = []
    A = 0.0
    for _ in range(400):
        level = -f1(A) if negate_level else f1(A)
        rows.append([A, level, f2(f1(A)), inflow, inflow - f2(f1(A))])
        A += 0.005 * (inflow - f2(f1(A)))
    return rows


def consistency_model():
    m = qr.Model("bathtub")
    m.variable("amount", landmarks=(Landmark("0", 0.0),), upper_unbounded=True)
    m.variable("level", landmarks=(Landmark("0", 0.0),), unbounded=True)
    m.variable("outflow", landmarks=(Landmark("0", 0.0),), upper_unbounded=True)
    m.variable("inflow", landmarks=(Landmark("0", 0.0),), upper_unbounded=True)
    m.variable("netflow", landmarks=(Landmark("0", 0.0),), unbounded=True)
    m.constrain(qr.MPlus("amount", "level"))
    m.constrain(qr.MPlus("level", "outflow"))
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    m.constrain(qr.Deriv("amount", "netflow"))
    m.constrain(qr.Constant("inflow"))
    return m


def test_check_consistency_accepts_true_instance():
    m = consistency_model()
    records = signs.check_consistency(bathtub_rows(), m)
    assert all(r.violation < 0.02 for r in records), [
        (r.kind, r.violation) for r in records
    ]


def test_check_consistency_localizes_corruption():
    m = consistency_model()
    records = signs.check_consistency(bathtub_rows(negate_level=True), m)
    by_kind = {}
    for r in records:
        by_kind.setdefault(type(r.constraint).__name__ + ":" + r.constraint.variables[0], r)
    # amount-level monotonicity is broken...
    broken = next(
        r for r in records if isinstance(r.constraint, qr.MPlus) and r.constraint.x == "amount"
    )
    assert broken.violation > 0.9
    # ...while unrelated structure still checks out
    intact = next(r for r in records if isinstance(r.constraint, qr.Add))
    assert intact.violation < 0.02


def test_estimate_needs_enough_samples():
    with pytest.raises(ValueError, match="at least"):
        signs.estimate_signs([[1.0, 2.0]], [[0.1, 0.2]])
