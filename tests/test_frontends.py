"""QPT and device-composition authoring front-ends."""

import json
from collections import Counter

import pytest

import qrlib as qr
from qrlib import Qdir, SimStatus, TerminalClass, TimeTag
from qrlib.frontends import devices, qpt


# --- QPT: process-centric authoring ----------------------------------------


def equalization_system():
    """Two containers exchange through a flow driven by their difference."""
    s = qpt.System("equalization")
    s.quantity("a", landmarks=("0",), upper_unbounded=True)
    s.quantity("b", landmarks=("0",), upper_unbounded=True)
    s.quantity("diff", landmarks=("0",), unbounded=True)
    s.quantity("flow", landmarks=("0",), unbounded=True)
    s.constrain(qr.Add("diff", "b", "a"))  # diff = a - b
    s.proportional("flow", "diff", +1, cvals=(("0", "0"),))
    return s


def transfer_initial(m):
    return m.state(
        time=TimeTag.POINT,
        a=(("0", "+inf"), Qdir.DEC),
        b=(("0", "+inf"), Qdir.INC),
        diff=(("0", "+inf"), Qdir.DEC),
        flow=(("0", "+inf"), Qdir.DEC),
        **{"a'": (("-inf", "0"), Qdir.INC), "b'": (("0", "+inf"), Qdir.DEC)},
    )


def test_unconditioned_process_matches_handwritten_qde():
    s = equalization_system()
    p = s.process("transfer")  # always active
    p.influence("a", -1, "flow")
    p.influence("b", +1, "flow")
    m = s.build()
    assert not m.regions  # no conditions: a plain single-region model

    h = qr.Model("equalization")
    h.variable("a", landmarks=("0",), upper_unbounded=True)
    h.variable("b", landmarks=("0",), upper_unbounded=True)
    h.variable("diff", landmarks=("0",), unbounded=True)
    h.variable("flow", landmarks=("0",), unbounded=True)
    h.constrain(qr.Add("diff", "b", "a"))
    h.constrain(qr.MPlus("diff", "flow", cvals=(("0", "0"),)))
    h.variable("a'", landmarks=("0",), unbounded=True)
    h.constrain(qr.Deriv("a", "a'"))
    h.variable("b'", landmarks=("0",), unbounded=True)
    h.constrain(qr.Deriv("b", "b'"))
    h.constrain(qr.MMinus("flow", "a'", cvals=(("0", "0"),)))
    h.constrain(qr.MPlus("flow", "b'", cvals=(("0", "0"),)))

    init = transfer_initial(m)
    assert (
        qr.qsim(m, init).graph.export() == qr.qsim(h, init).graph.export()
    )


def test_conditioned_process_builds_activation_regions():
    s = equalization_system()
    p = s.process("transfer", when=(("diff", ">", "0"),))  # one-way valve
    p.influence("a", -1, "flow")
    p.influence("b", +1, "flow")
    m = s.build()
    assert list(m.regions) == ["transfer", "!transfer"]
    assert {(t.source, t.target) for t in m.region_transitions} == {
        ("transfer", "!transfer"),
        ("!transfer", "transfer"),
    }

    init = transfer_initial(m)
    m.initial_region = s.active_region(init)
    assert m.initial_region == "transfer"
    r = qr.qsim(m, init)
    assert r.status is SimStatus.COMPLETE
    # a single behavior: drain toward equality, mandatory switch to the
    # process-off region at the diff == 0 boundary, quiesce there — the
    # engine's Zeno guard keeps the mutual boundary transitions from
    # ping-ponging at the crossing point
    (b,) = r.behaviors()
    assert b.terminal is TerminalClass.QUIESCENT
    assert b.regions[-1] == "!transfer" and b.regions[0] == "transfer"
    assert len(b.states) == 4


def test_sole_mechanism_pins_uninfluenced_rates():
    # in the process-off region the rate is pinned to zero: QPT's
    # sole-mechanism assumption, not "anything goes"
    s = equalization_system()
    p = s.process("transfer", when=(("diff", ">", "0"),))
    p.influence("a", -1, "flow")
    p.influence("b", +1, "flow")
    m = s.build()
    off = dict(m.regions)["!transfer"]
    kinds = Counter(type(c).__name__ for c in off)
    assert kinds["At"] == 2 and kinds["Constant"] == 2  # a' = 0, b' = 0


def test_qpt_validation():
    s = equalization_system()
    with pytest.raises(ValueError, match="multi-condition"):
        s.process("p", when=(("a", ">", "0"), ("b", ">", "0")))
    with pytest.raises(ValueError, match="condition op"):
        s.process("p", when=(("a", "==", "0"),))
    with pytest.raises(ValueError, match="unknown quantity"):
        s.process("p", when=(("nope", ">", "0"),))
    with pytest.raises(ValueError, match="not a landmark"):
        s.process("p", when=(("a", ">", "NOPE"),))
    with pytest.raises(ValueError, match="already has a proportionality"):
        s.proportional("flow", "a", +1)
    with pytest.raises(ValueError, match="sign"):
        s.proportional("a", "b", 2)
    p = s.process("ok")
    with pytest.raises(ValueError, match="influence sign"):
        p.influence("a", 0, "flow")
    p.influence("a", -1, "flow")
    p.influence("a", -1, "flow")  # two negative influences on one quantity
    with pytest.raises(ValueError, match="not both negative"):
        s.build()

    s2 = qpt.System("bad-rate")
    s2.quantity("q", landmarks=("0",))
    s2.process("p").influence("q", 1, "missing")
    with pytest.raises(ValueError, match="unknown quantity 'missing'"):
        s2.build()

    with pytest.raises(ValueError, match="build"):
        qpt.System("unbuilt").active_region(None)


# --- devices: compositional authoring --------------------------------------


def tank_library():
    lib = devices.Library()
    lib.component(
        "tank",
        terminals={
            "in": devices.VarSpec(("0", "QMAX")),
            "out": devices.VarSpec(("0", "QMAX")),
        },
        internals={
            "amt": devices.VarSpec(("0", "CAP")),
            "net": devices.VarSpec.unbounded(),
        },
        law=lambda v: [
            qr.MPlus(v("amt"), v("out"), cvals=(("0", "0"), ("CAP", "QMAX"))),
            qr.Add(v("net"), v("out"), v("in")),  # net = in - out
            qr.Deriv(v("amt"), v("net")),
        ],
    )
    return lib


def cascade_device():
    d = devices.Device("cascade", tank_library())
    d.add("A", "tank")
    d.add("B", "tank")
    d.connect(("A", "out"), ("B", "in"))  # A drains into B
    return d


def test_composition_matches_handwritten_model():
    dm = cascade_device().build()
    # the connected pair (A.out, B.in) is one shared variable
    assert list(dm.variables) == [
        "A.in", "A.out", "B.out", "A.amt", "A.net", "B.amt", "B.net",
    ]

    h = qr.Model("cascade")
    h.variable("A.in", landmarks=("0", "QMAX"))
    h.variable("A.out", landmarks=("0", "QMAX"))
    h.variable("B.out", landmarks=("0", "QMAX"))
    h.variable("A.amt", landmarks=("0", "CAP"))
    h.variable("A.net", landmarks=("0",), unbounded=True)
    h.variable("B.amt", landmarks=("0", "CAP"))
    h.variable("B.net", landmarks=("0",), unbounded=True)
    h.constrain(qr.MPlus("A.amt", "A.out", cvals=(("0", "0"), ("CAP", "QMAX"))))
    h.constrain(qr.Add("A.net", "A.out", "A.in"))
    h.constrain(qr.Deriv("A.amt", "A.net"))
    h.constrain(qr.MPlus("B.amt", "B.out", cvals=(("0", "0"), ("CAP", "QMAX"))))
    h.constrain(qr.Add("B.net", "B.out", "A.out"))  # B's inflow IS A's outflow
    h.constrain(qr.Deriv("B.amt", "B.net"))
    assert dm.to_dict() == h.to_dict()


def test_composed_device_simulates():
    dm = cascade_device().build()
    dm.constrain(qr.Constant("A.in"))  # boundary condition: no external feed
    init = dm.state(
        time=TimeTag.POINT,
        **{
            "A.in": ("0", Qdir.STD),
            "A.out": (("0", "QMAX"), Qdir.DEC),
            "B.out": ("0", Qdir.INC),
            "A.amt": (("0", "CAP"), Qdir.DEC),
            "A.net": (("-inf", "0"), Qdir.INC),
            "B.amt": ("0", Qdir.INC),
            "B.net": (("0", "+inf"), Qdir.DEC),
        },
    )
    r = qr.qsim(dm, init, config=qr.SimConfig(dynamic_chatter=True))
    assert r.status is SimStatus.COMPLETE
    census = Counter(b.terminal for b in r.behaviors())
    assert census == {TerminalClass.QUIESCENT: 4, TerminalClass.DOMAIN_EXIT: 1}


def test_device_validation():
    lib = tank_library()
    with pytest.raises(ValueError, match="already defined"):
        lib.component("tank", terminals={}, law=lambda v: [])
    with pytest.raises(KeyError, match="unknown component type"):
        devices.Device("d", lib).add("X", "nope")
    d = devices.Device("d", lib)
    d.add("A", "tank")
    with pytest.raises(ValueError, match="already added"):
        d.add("A", "tank")
    with pytest.raises(ValueError, match="unknown instance"):
        d.connect(("A", "out"), ("Z", "in"))
    with pytest.raises(ValueError, match="no terminal"):
        d.connect(("A", "out"), ("A", "side"))
    lib.component(
        "gauge",
        terminals={"tap": devices.VarSpec(("0",), upper_unbounded=True)},
        law=lambda v: [],
    )
    d.add("G", "gauge")
    with pytest.raises(ValueError, match="quantity spaces differ"):
        d.connect(("A", "out"), ("G", "tap"))


def test_built_models_serialize():
    s = equalization_system()
    p = s.process("transfer", when=(("diff", ">", "0"),))
    p.influence("a", -1, "flow")
    p.influence("b", +1, "flow")
    m = s.build()
    data = m.to_dict()
    json.dumps(data)
    assert qr.Model.from_dict(data).to_dict() == data

    dd = cascade_device().build().to_dict()
    json.dumps(dd)
    assert qr.Model.from_dict(dd).to_dict() == dd
