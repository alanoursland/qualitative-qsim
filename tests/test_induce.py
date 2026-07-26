"""QDE induction: recovering model structure from trajectory data."""

import json

import pytest

from qrlib import induce
from qrlib.constraints import Constant, Deriv

from test_soundness import rk4


def influences(candidate):
    return set(candidate.influences)


# --- structure recovery on systems with known signs ------------------------


def test_first_order_decay():
    # dx = -x : a single self-limiting influence
    data = rk4(lambda y: [-y[0]], [1.0], 0.05, 200)
    res = induce.induce(data, ["x"])
    assert res.best is not None
    assert influences(res.best) == {("x", "x", -1)}


def test_decoupled_decay_recovers_independence():
    # dx = -x, dy = -2y : no cross-influence
    data = rk4(lambda y: [-y[0], -2 * y[1]], [1.0, 1.0], 0.05, 200)
    res = induce.induce(data, ["x", "y"])
    # the sparsest consistent structure is the true decoupled pair; denser
    # candidates (spurious cross-terms that still fit within noise) are also
    # consistent but rank lower on parsimony
    assert influences(res.best) == {("x", "x", -1), ("y", "y", -1)}


def test_undamped_oscillator():
    # dx = v, dv = -x
    data = rk4(lambda y: [y[1], -y[0]], [1.0, 0.0], 0.02, 700)
    res = induce.induce(data, ["x", "v"])
    assert influences(res.best) == {("x", "v", 1), ("v", "x", -1)}


def test_damped_oscillator_recovers_the_damping_term():
    # dx = v, dv = -x - 0.3 v : the damping influence v -> v (-) is the one a
    # too-sparse structure drops, and its Deriv row is what refutes dropping it
    data = rk4(lambda y: [y[1], -y[0] - 0.3 * y[1]], [1.0, 0.0], 0.05, 400)
    res = induce.induce(data, ["x", "v"])
    assert influences(res.best) == {("x", "v", 1), ("v", "x", -1), ("v", "v", -1)}
    # the 2-influence structure (no damping) is present but inconsistent at
    # this tolerance: its derivative row disagrees with the data near the
    # turning points
    two = next(c for c in res.candidates if c.n_influences == 2)
    assert not two.consistent
    assert two.worst == "deriv"


def test_constant_acceleration_has_no_spurious_state_influence():
    # dy = v, dv = -9.8: the second rate is constant, not state-dependent.
    data = rk4(lambda y: [y[1], -9.8], [0.0, 10.0], 0.01, 200)
    res = induce.induce(data, ["y", "v"])

    assert influences(res.best) == {("y", "v", 1)}
    constraints = res.best.model.constraints
    assert any(
        isinstance(c, Constant) and c.x == "d_v"
        for c in constraints
    )
    assert any(
        isinstance(c, Deriv) and c.x == "v" and c.y == "d_v"
        for c in constraints
    )
    assert not any(
        isinstance(c, Constant) and c.x == "v"
        for c in constraints
    )


def test_tolerance_trades_sparsity_for_fidelity():
    data = rk4(lambda y: [y[1], -y[0] - 0.3 * y[1]], [1.0, 0.0], 0.05, 400)
    # a looser tolerance accepts dropping the weak damping influence, giving
    # the sparser undamped structure as best
    loose = induce.induce(data, ["x", "v"], tol=0.15)
    assert influences(loose.best) == {("x", "v", 1), ("v", "x", -1)}
    strict = induce.induce(data, ["x", "v"], tol=0.02)
    assert loose.best.n_influences < strict.best.n_influences


# --- inputs ----------------------------------------------------------------


def test_multiple_trajectories_are_pooled():
    field = lambda y: [y[1], -y[0] - 0.3 * y[1]]  # noqa: E731
    t1 = rk4(field, [1.0, 0.0], 0.05, 250)
    t2 = rk4(field, [0.0, 1.0], 0.05, 250)
    res = induce.induce([t1, t2], ["x", "v"])
    assert influences(res.best) == {("x", "v", 1), ("v", "x", -1), ("v", "v", -1)}


def test_explicit_derivatives():
    field = lambda y: [y[1], -y[0] - 0.3 * y[1]]  # noqa: E731
    states = rk4(field, [1.0, 0.0], 0.05, 300)
    derivs = [field(s) for s in states]
    res = induce.induce(states, ["x", "v"], derivatives=derivs)
    assert influences(res.best) == {("x", "v", 1), ("v", "x", -1), ("v", "v", -1)}


def test_input_validation():
    with pytest.raises(ValueError, match="at least 2 samples"):
        induce.induce([[1.0, 2.0]], ["x", "v"])
    states = rk4(lambda y: [y[1], -y[0]], [1.0, 0.0], 0.05, 100)
    with pytest.raises(ValueError, match="match its trajectory length"):
        induce.induce(states, ["x", "v"], derivatives=[[[0.0, 0.0]]])
    two = [states, states]
    with pytest.raises(ValueError, match="derivatives must match the number"):
        induce.induce(two, ["x", "v"], derivatives=[[[0.0, 0.0]] * len(states)])
    with pytest.raises(ValueError, match="expected 2 columns"):
        induce.induce([[1.0], [2.0], [3.0], [4.0]], ["x", "v"])


# --- soundness of refutation, non-uniqueness of consistency ----------------


def test_ranked_set_and_note():
    data = rk4(lambda y: [y[1], -y[0] - 0.3 * y[1]], [1.0, 0.0], 0.05, 300)
    res = induce.induce(data, ["x", "v"])
    # candidates are ranked consistent-first, then by parsimony
    kinds = [(c.consistent, c.n_influences) for c in res.candidates]
    assert kinds == sorted(kinds, key=lambda k: (0 if k[0] else 1, k[1]))
    # forcing state-independent rates on an oscillator is soundly refuted
    refuted = induce.induce(data, ["x", "v"], thresholds=[1e30])
    assert refuted.best is None
    assert refuted.note is not None
    assert refuted.candidates[0].worst == "constant"


def test_result_exports_plain_data():
    data = rk4(lambda y: [-y[0]], [1.0], 0.05, 100)
    res = induce.induce(data, ["x"])
    d = res.to_dict()
    json.dumps(d)
    assert d["variables"] == ["x"]
    assert d["candidates"][0]["influences"] == [["x", "x", -1]]
    assert d["candidates"][0]["consistent"] is True
