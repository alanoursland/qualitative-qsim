"""qrlib — qualitative reasoning about dynamical systems.

Pre-alpha skeleton. See the repository's ``docs/`` for the design:
``docs/architecture.md`` maps this package layout, ``docs/roadmap.md`` says
what exists vs. what is planned.
"""

from .constraints import Add, Constant, Constraint, Deriv, Minus, MMinus, MPlus, Mult
from .model import Model, Variable
from .quantity import Landmark, Qdir, QuantitySpace, QVal
from .state import QState, TimeTag

__all__ = [
    "Add",
    "Constant",
    "Constraint",
    "Deriv",
    "Minus",
    "MMinus",
    "MPlus",
    "Mult",
    "Model",
    "Variable",
    "Landmark",
    "Qdir",
    "QuantitySpace",
    "QVal",
    "QState",
    "TimeTag",
    "qsim",
]

__version__ = "0.0.1a0"


def qsim(model, initial, **kwargs):
    """Run QSIM. Placeholder — lands in phase 1 (docs/roadmap.md)."""
    from .engines.qsim import qsim as _qsim

    return _qsim(model, initial, **kwargs)
