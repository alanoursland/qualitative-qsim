"""Device-centric compositional authoring (after de Kleer & Brown's
ENVISION).

Research lineage: [deKleerBrown1984] in ``docs/references.md``.

Models are built by wiring instances of reusable **component types**
rather than by writing one flat constraint list. A :class:`ComponentType`
declares *terminals* (the variables a component exposes for connection),
*internal* variables, and a constraint factory that states the
component's law over its own local names ("no function in structure":
the type knows nothing about what it will be wired to). A
:class:`Device` instantiates types under instance names and ``connect``s
terminals; connected terminals are unified into one shared model
variable, and every instance's constraints are emitted against the
resolved global names — yielding an ordinary :class:`~qrlib.model.Model`.

Local variables compile to ``instance.name``; a group of connected
terminals takes the name of its first-declared member. Connected
terminals must agree on their declared quantity space. Unconnected
terminals stay as free boundary variables — constrain them afterwards on
the built model (e.g. a ``Constant`` source), exactly as ENVISION's
boundary conditions do.

Component modes (a valve's open/closed) are not modeled here: per-mode
constraint subsets are what operating regions and
``qrlib.diagnosis.Component`` already provide on the built model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..constraints import Constraint
from ..model import Model
from ..quantity import Landmark

__all__ = ["VarSpec", "ComponentType", "Library", "Device"]


@dataclass(frozen=True)
class VarSpec:
    """Quantity-space declaration for a terminal or internal variable."""

    landmarks: tuple[Landmark | str, ...] = ("0",)
    lower_unbounded: bool = False
    upper_unbounded: bool = False

    @classmethod
    def unbounded(cls, landmarks: tuple[Landmark | str, ...] = ("0",)) -> "VarSpec":
        return cls(landmarks, True, True)


@dataclass(frozen=True)
class ComponentType:
    """A reusable component: terminals + internals + its law.

    ``law`` receives a resolver mapping local names (terminals and
    internals) to global model variable names and returns the instance's
    constraints."""

    name: str
    terminals: tuple[tuple[str, VarSpec], ...]
    internals: tuple[tuple[str, VarSpec], ...]
    law: Callable[[Callable[[str], str]], Sequence[Constraint]]

    def spec_of(self, local: str) -> VarSpec:
        for n, spec in self.terminals + self.internals:
            if n == local:
                return spec
        raise KeyError(local)


class Library:
    """A catalog of component types."""

    def __init__(self) -> None:
        self._types: dict[str, ComponentType] = {}

    def component(
        self,
        name: str,
        *,
        terminals: dict[str, VarSpec],
        internals: dict[str, VarSpec] | None = None,
        law: Callable[[Callable[[str], str]], Sequence[Constraint]],
    ) -> ComponentType:
        if name in self._types:
            raise ValueError(f"component type {name!r} already defined")
        ct = ComponentType(
            name,
            tuple(terminals.items()),
            tuple((internals or {}).items()),
            law,
        )
        self._types[name] = ct
        return ct

    def type_named(self, name: str) -> ComponentType:
        if name not in self._types:
            raise KeyError(f"unknown component type {name!r}")
        return self._types[name]


class Device:
    """A netlist: named instances of library types + terminal connections."""

    def __init__(self, name: str, library: Library):
        self.name = name
        self.library = library
        self._instances: dict[str, ComponentType] = {}
        self._pins: list[tuple[str, str]] = []  # declaration order
        self._parent: dict[tuple[str, str], tuple[str, str]] = {}

    def add(self, instance: str, type_name: str) -> None:
        if instance in self._instances:
            raise ValueError(f"instance {instance!r} already added")
        ct = self.library.type_named(type_name)
        self._instances[instance] = ct
        for term, _spec in ct.terminals:
            pin = (instance, term)
            self._pins.append(pin)
            self._parent[pin] = pin

    def _find(self, pin: tuple[str, str]) -> tuple[str, str]:
        while self._parent[pin] != pin:
            self._parent[pin] = self._parent[self._parent[pin]]
            pin = self._parent[pin]
        return pin

    def connect(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        for inst, term in (a, b):
            if inst not in self._instances:
                raise ValueError(f"unknown instance {inst!r}")
            if (inst, term) not in self._parent:
                raise ValueError(f"{inst!r} has no terminal {term!r}")
        sa = self._instances[a[0]].spec_of(a[1])
        sb = self._instances[b[0]].spec_of(b[1])
        if sa != sb:
            raise ValueError(
                f"cannot connect {a} to {b}: quantity spaces differ "
                f"({sa} vs {sb})"
            )
        self._parent[self._find(a)] = self._find(b)

    def build(self) -> Model:
        """Unify connected terminals and emit the composed Model."""
        m = Model(self.name)
        # canonical pin of each connection group: its first-declared member
        canonical: dict[tuple[str, str], tuple[str, str]] = {}
        for pin in self._pins:
            root = self._find(pin)
            if root not in canonical:
                canonical[root] = pin

        def global_name(pin: tuple[str, str]) -> str:
            inst, term = canonical[self._find(pin)]
            return f"{inst}.{term}"

        declared: set[str] = set()
        for pin in self._pins:  # declaration order keeps names deterministic
            name = global_name(pin)
            if name not in declared:
                declared.add(name)
                spec = self._instances[pin[0]].spec_of(pin[1])
                m.variable(
                    name,
                    landmarks=spec.landmarks,
                    lower_unbounded=spec.lower_unbounded,
                    upper_unbounded=spec.upper_unbounded,
                )
        for inst, ct in self._instances.items():
            for local, spec in ct.internals:
                m.variable(
                    f"{inst}.{local}",
                    landmarks=spec.landmarks,
                    lower_unbounded=spec.lower_unbounded,
                    upper_unbounded=spec.upper_unbounded,
                )
        for inst, ct in self._instances.items():
            terminal_names = {t for t, _ in ct.terminals}

            def resolve(local: str, inst=inst, terminal_names=terminal_names) -> str:
                if local in terminal_names:
                    return global_name((inst, local))
                return f"{inst}.{local}"

            for c in ct.law(resolve):
                m.constrain(c)
        return m
