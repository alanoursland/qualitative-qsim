"""qrlib — qualitative reasoning about dynamical systems.

Pre-alpha. See the repository's ``docs/`` for the design:
``docs/architecture.md`` maps this package layout, ``docs/roadmap.md`` says
what exists vs. what is planned. The reference QSIM engine is available as
:func:`qrlib.qsim`.
"""

from .behavior import (
    Behavior,
    BehaviorGraph,
    SimConfig,
    SimResult,
    SimStatus,
    TerminalClass,
)
from .constraints import Add, Constant, Constraint, Deriv, Minus, MMinus, MPlus, Mult
from .engines.qsim import qsim
from .model import CompiledModel, Model, Variable
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
    "CompiledModel",
    "Variable",
    "Landmark",
    "Qdir",
    "QuantitySpace",
    "QVal",
    "QState",
    "TimeTag",
    "Behavior",
    "BehaviorGraph",
    "SimConfig",
    "SimResult",
    "SimStatus",
    "TerminalClass",
    "qsim",
]

__version__ = "0.0.1a0"
