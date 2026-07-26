"""qrlib — qualitative reasoning about dynamical systems.

Pre-alpha. The reference QSIM engine is available as :func:`qrlib.qsim`;
the package also provides behavior analysis, numeric-trajectory bridges,
model frontends, and tensorized execution.
"""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _version,
)

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
from .energy import EnergyFilter, LyapunovCertificate, Trend
from .constraint_syntax import (
    ConstraintSyntaxError,
    format_constraint,
    parse_constraint,
)
from . import decompose, diagnosis, frontends, guide, induce, semiquant, viz
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
    "EnergyFilter",
    "LyapunovCertificate",
    "Trend",
    "ConstraintSyntaxError",
    "parse_constraint",
    "format_constraint",
    "qsim",
    "envision",
    "Envisionment",
    "decompose",
    "diagnosis",
    "frontends",
    "induce",
    "guide",
    "semiquant",
    "viz",
]

try:
    __version__ = _version("qualitative-qsim")
except _PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0.dev0"
