"""Differentiable constraint losses (torch autograd, above the exact core)."""

import math

import pytest

torch = pytest.importorskip("torch")

import qrlib as qr
from qrlib.tensor import losses
from qrlib.tensor.losses import LossConfig, constraint_loss, per_constraint_loss


def _f64(rows):
    return torch.tensor(rows, dtype=torch.float64)


# --- zero iff satisfied, per constraint kind -------------------------------


def test_value_identities():
    # Add: x + y = z ; Minus: y = -x ; Mult: z = x*y ; At: x = landmark
    m = qr.Model("alg")
    for v in ("x", "y", "z", "w", "p", "q", "r"):
        m.variable(v, landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.variable("pinned", landmarks=(qr.Landmark("0", value=0.0), qr.Landmark("L", value=3.0)))
    m.constrain(qr.Add("x", "y", "z"))
    m.constrain(qr.Minus("w", "p"))     # p = -w
    m.constrain(qr.Mult("q", "r", "z"))  # z = q*r (shares z; keep consistent)
    m.constrain(qr.At("pinned", "L"))

    good = _f64([
        # x   y   z    w    p    q    r   pinned
        [1.0, 2.0, 3.0, 1.0, -1.0, 3.0, 1.0, 3.0],
        [2.0, 1.0, 3.0, -2.0, 2.0, 1.0, 3.0, 3.0],
    ])
    for cl in per_constraint_loss(good, m):
        assert float(cl.loss) < 1e-18, (cl.kind, float(cl.loss))
    assert float(constraint_loss(good, m)) < 1e-15

    bad = good.clone()
    bad[0, 2] = 99.0        # breaks Add and Mult (z wrong) and nothing else
    bad[0, 7] = 0.0         # breaks At (pinned != L)
    per = {cl.kind: float(cl.loss) for cl in per_constraint_loss(bad, m)}
    assert per["add"] > 0 and per["mult"] > 0 and per["at"] > 0
    assert per["minus"] < 1e-18  # untouched


def test_monotone_and_deriv_and_constant():
    m = qr.Model("dyn")
    for v in ("a", "b", "c"):
        m.variable(v, landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.MPlus("a", "b"))     # b moves with a
    m.constrain(qr.MMinus("a", "c"))    # c moves against a
    m.variable("k", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Constant("k"))
    m.variable("rate", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Deriv("a", "rate"))  # sign(da) == sign(rate)

    T = 20
    a = [math.sin(0.3 * t) for t in range(T)]
    rows = []
    for t in range(T):
        da = (a[t + 1] - a[t]) if t + 1 < T else (a[t] - a[t - 1])
        rows.append([a[t], a[t], -a[t], 5.0, da])  # b=a, c=-a, k const, rate=da
    good = _f64(rows)
    per = {cl.kind: float(cl.loss) for cl in per_constraint_loss(good, m)}
    for kind in ("mplus", "mminus", "constant", "deriv"):
        assert per[kind] < 1e-9, (kind, per[kind])

    # break each: b anti-correlated, k drifting, rate wrong-signed
    bad = good.clone()
    bad[:, 1] = -bad[:, 0]           # b now anti-correlates with a -> M+ broken
    bad[:, 3] = torch.linspace(0, 1, T)  # k drifts -> Constant broken
    bad[:, 4] = -bad[:, 4]           # rate sign flipped -> Deriv broken
    perb = {cl.kind: float(cl.loss) for cl in per_constraint_loss(bad, m)}
    assert perb["mplus"] > 0 and perb["constant"] > 0 and perb["deriv"] > 0


def test_negligible():
    m = qr.Model("oom")
    m.variable("small", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.variable("large", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Negligible("small", "large"))
    ok = _f64([[0.1, 1.0], [0.2, 2.0], [-0.1, 3.0]])
    assert float(constraint_loss(ok, m)) < 1e-18
    bad = _f64([[1.0, 1.0], [3.0, 2.0]])  # |small| >= |large|
    assert float(constraint_loss(bad, m)) > 0


# --- autograd: the point of the whole module -------------------------------


def test_gradient_fit_recovers_a_parameter():
    # learn the constant inflow that balances Add(netflow, outflow, inflow)
    m = qr.Model("balance")
    for v in ("netflow", "outflow", "inflow"):
        m.variable(v, landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Add("netflow", "outflow", "inflow"))
    T = 30
    outflow = torch.linspace(0.0, 1.0, T, dtype=torch.float64)
    netflow = torch.zeros(T, dtype=torch.float64)  # observed equilibrium: net = 0
    inflow = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([inflow], lr=0.05)
    for _ in range(500):
        x = torch.stack([netflow, outflow, inflow.expand(T)], dim=1)
        loss = constraint_loss(x, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # net = 0 and net + outflow = inflow forces inflow = outflow everywhere,
    # only consistent if inflow tracks... outflow varies, so the least-squares
    # optimum is the mean of outflow (0.5)
    assert abs(float(inflow.detach()) - 0.5) < 1e-2


def test_gradient_pushes_a_trajectory_monotone():
    # regularize a trajectory toward M+ by descending the loss on it
    m = qr.Model("mono")
    m.variable("x", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.variable("y", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.MPlus("x", "y"))
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 12, dtype=torch.float64)
    y = torch.randn(12, dtype=torch.float64).requires_grad_(True)
    opt = torch.optim.Adam([y], lr=0.1)
    start = float(constraint_loss(torch.stack([x, y], 1), m).detach())
    for _ in range(300):
        loss = constraint_loss(torch.stack([x, y], 1), m)
        opt.zero_grad()
        loss.backward()
        opt.step()
    end = float(constraint_loss(torch.stack([x, y], 1), m).detach())
    assert end < start and end < 1e-6


# --- shapes, reduction, validation -----------------------------------------


def test_batch_and_reduction():
    m = qr.Model("b")
    m.variable("x", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.variable("y", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Add("x", "y", "y"))  # x + y = y  => x must be 0
    batch = _f64([[[1.0, 5.0]], [[0.0, 5.0]]])  # (B=2, T=1, V=2)
    none = per_constraint_loss(batch, m, config=LossConfig(reduction="none"))[0].loss
    assert none.shape == (2, 1)
    assert float(none[0]) > 0 and float(none[1]) < 1e-18
    summed = per_constraint_loss(batch, m, config=LossConfig(reduction="sum"))[0].loss
    assert summed.dim() == 0


def test_validation():
    m = qr.Model("v")
    m.variable("x", landmarks=(qr.Landmark("0", value=0.0),), unbounded=True)
    m.constrain(qr.Constant("x"))
    with pytest.raises(ValueError, match="columns but the model"):
        constraint_loss(_f64([[1.0, 2.0]]), m)
    with pytest.raises(ValueError, match="reduction must be"):
        constraint_loss(_f64([[1.0], [1.0]]), m, config=LossConfig(reduction="max"))
    with pytest.raises(ValueError, match="must be"):
        constraint_loss(torch.zeros(2, 2, 2, 2), m)
