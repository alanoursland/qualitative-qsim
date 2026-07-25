"""Cross-surface qualification for research-preview modules."""

import qrlib as qr
from qrlib import Qdir, SimStatus, TimeTag
from qrlib import decompose as dc

from test_decsim import assert_covers
from test_frontends import cascade_device


def test_device_frontend_decomposition_preserves_monolithic_behaviors():
    """A composed device must retain its behaviors after DecSIM partitioning.

    This crosses the device front end, model compilation, reference QSIM,
    guided component simulation, and DecSIM's behavior join. It complements
    the handwritten-model checks in the individual frontend and decomposition
    suites.
    """
    model = cascade_device().build()
    model.constrain(qr.Constant("A.in"))
    initial = model.state(
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
    config = qr.SimConfig(ignore_qdir=("A.net", "B.net"))

    monolithic = qr.qsim(model, initial, config=config)
    decomposed = dc.decsim(
        model,
        initial,
        {
            "tankA": ("A.in", "A.out", "A.amt", "A.net"),
            "tankB": ("B.out", "B.amt", "B.net"),
        },
        config=config,
    )

    assert monolithic.status is SimStatus.COMPLETE
    assert decomposed.status is SimStatus.COMPLETE
    assert len(monolithic.behaviors()) == len(decomposed.joint_behaviors()) == 5
    assert decomposed.stats["component_nodes"] == {
        "tankA": 3,
        "tankB": 15,
    }
    assert_covers(decomposed, monolithic, model)
