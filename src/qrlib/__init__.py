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
from .constraints import (
    Add,
    At,
    Constant,
    Constraint,
    Deriv,
    Minus,
    MMinus,
    MPlus,
    Mult,
    Negligible,
)
from . import decompose, diagnosis, frontends, guide, semiquant, viz
from .engines.envision import Envisionment, envision
from .engines.qsim import qsim
from .model import (
    CompiledModel,
    Guard,
    Model,
    RegionTransition,
    SignStructure,
    Variable,
)
from .quantity import Landmark, Qdir, QuantitySpace, QVal
from .state import QState, TimeTag

__all__ = [
    "Add",
    "At",
    "Constant",
    "Constraint",
    "Deriv",
    "Minus",
    "MMinus",
    "MPlus",
    "Mult",
    "Negligible",
    "Model",
    "CompiledModel",
    "Variable",
    "Guard",
    "RegionTransition",
    "SignStructure",
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
    "envision",
    "Envisionment",
    "decompose",
    "diagnosis",
    "frontends",
    "guide",
    "semiquant",
    "viz",
]

__version__ = "0.1.0a0"
