"""Declarative QDE model descriptions.

A :class:`Model` is pure data — variables (each owning a
:class:`~qrlib.quantity.QuantitySpace`), constraints, and optional
**operating regions** — consumed by every engine. :meth:`Model.compile`
freezes the name -> index and landmark -> rank mappings and resolves
constraints into :class:`CompiledConstraint` records. The compiled
artifacts are inert data; the consistency predicates that interpret them
live with the engines (``qrlib.engines.filters``).

Operating regions (docs/host-integration.md, Surface 5): a region names a
subset of the model's constraints that are active while the system is in
it, and **transitions** declare when the system crosses into another
region — guard conditions are conjunctions of landmark predicates
(``amount == FULL``, ``netflow > 0``) evaluated on qualitative magnitudes.
A model with no declared regions has one implicit region (``"default"``)
containing every constraint. Region entry re-derives directions: the
vector field may change discontinuously at the boundary, so magnitudes
carry over and directions are re-enumerated under the new region's
constraints.

Models serialize to a versioned JSON-able schema (:meth:`Model.to_dict` /
:meth:`Model.from_dict`) — the interchange format hosts author against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

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
from .quantity import Landmark, Qdir, QuantitySpace, QVal
from .state import QState, TimeTag

__all__ = [
    "Variable",
    "Guard",
    "RegionTransition",
    "Model",
    "CompiledConstraint",
    "CompiledRegion",
    "CompiledTransition",
    "CompiledModel",
    "SignStructure",
    "MODEL_SCHEMA",
]

MODEL_SCHEMA = "qrlib.model/v1"

_KIND: dict[type, str] = {
    MPlus: "mplus",
    MMinus: "mminus",
    Add: "add",
    Mult: "mult",
    Minus: "minus",
    Deriv: "deriv",
    Constant: "constant",
    At: "at",
    Negligible: "negligible",
}
_CLASS: dict[str, type] = {v: k for k, v in _KIND.items()}

ZERO = "0"
DEFAULT_REGION = "default"
_GUARD_OPS = ("==", "<", ">", "<=", ">=")


@dataclass(frozen=True)
class Variable:
    name: str
    space: QuantitySpace


@dataclass(frozen=True)
class Guard:
    """One landmark predicate: ``variable <op> landmark`` on magnitudes."""

    var: str
    op: str
    landmark: str

    def __post_init__(self) -> None:
        if self.op not in _GUARD_OPS:
            raise ValueError(f"guard op must be one of {_GUARD_OPS}, got {self.op!r}")


@dataclass(frozen=True)
class RegionTransition:
    """When every guard holds at a point state in ``source``, the behavior
    crosses into ``target`` (magnitudes carry over, directions re-derive)."""

    source: str
    target: str
    guards: tuple[Guard, ...]


@dataclass(frozen=True)
class CompiledConstraint:
    """A constraint resolved against frozen variable/landmark mappings.

    ``vars`` are indices into the compiled variable order; ``cvals`` are
    corresponding-value tuples as magnitude ranks (including any implicit
    zero tuples added at compile time); ``zeros`` is the rank of the ``0``
    landmark per constrained variable (or None); ``infs`` is the pair
    (rank of -inf, rank of +inf) per constrained variable (None = bounded).
    ``dominant`` (ADD only) marks the operand (0 or 1) that a
    ``Negligible`` declaration proves strictly larger in magnitude: the
    sum's zero-referenced sign is then that operand's sign exactly.
    """

    kind: str
    vars: tuple[int, ...]
    cvals: tuple[tuple[int, ...], ...]
    zeros: tuple[int | None, ...]
    infs: tuple[tuple[int | None, int | None], ...]
    source: Constraint
    dominant: int | None = None


@dataclass(frozen=True)
class CompiledTransition:
    """Guards as (variable index, op, landmark *name*) — names, not ranks,
    because discovered landmarks shift ranks per branch frame."""

    target: str
    guards: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class CompiledRegion:
    name: str
    constraint_idx: tuple[int, ...]
    transitions: tuple[CompiledTransition, ...]


@dataclass(frozen=True)
class CompiledModel:
    """Frozen, engine-consumable form of a :class:`Model`."""

    name: str
    var_order: tuple[str, ...]
    spaces: tuple[QuantitySpace, ...]
    constraints: tuple[CompiledConstraint, ...]
    model_hash: str
    regions: tuple[CompiledRegion, ...] = ()
    initial_region: str = DEFAULT_REGION

    def index(self, var: str) -> int:
        return self.var_order.index(var)

    def inf_ranks(self, vi: int) -> tuple[int | None, int | None]:
        space = self.spaces[vi]
        lo = 0 if space.lower_unbounded else None
        hi = space.num_ranks - 1 if space.upper_unbounded else None
        return (lo, hi)

    def region_named(self, name: str) -> CompiledRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def constraints_of(self, region: str) -> tuple[CompiledConstraint, ...]:
        idx = self.region_named(region).constraint_idx
        return tuple(self.constraints[i] for i in idx)


@dataclass(frozen=True)
class SignStructure:
    """The sign/monotonicity closure a model's constraints imply, as plain
    data (docs/host-integration.md, Surface 6): what a host maps onto its
    own regression parameterization or checks against its own analyses.

    - ``monotone``: (x, y, s) with s = sign of dy/dx (M+, M-, MINUS).
    - ``derivatives``: (x, y) with dx/dt = y.
    - ``sums``: (x, y, z) with x + y = z; ``products``: z = x * y.
    - ``constants``: variables pinned constant.
    - ``corresponding``: (variable names, landmark-name tuples) pinned to
      co-occur, per constraint that carries corresponding values.
    - ``negligible``: (small, large) order-of-magnitude declarations.
    """

    monotone: tuple[tuple[str, str, int], ...]
    derivatives: tuple[tuple[str, str], ...]
    sums: tuple[tuple[str, str, str], ...]
    products: tuple[tuple[str, str, str], ...]
    constants: tuple[str, ...]
    corresponding: tuple[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]], ...]
    negligible: tuple[tuple[str, str], ...] = ()


@dataclass
class Model:
    """A qualitative differential equation: variables + constraints
    (+ optional operating regions)."""

    name: str = "model"
    variables: dict[str, Variable] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    regions: dict[str, tuple[Constraint, ...] | None] = field(default_factory=dict)
    region_transitions: list[RegionTransition] = field(default_factory=list)
    initial_region: str | None = None

    # --- authoring ---------------------------------------------------------

    def variable(
        self,
        name: str,
        *,
        landmarks: tuple[str | Landmark, ...] = ("0",),
        lower_unbounded: bool = False,
        upper_unbounded: bool = False,
        unbounded: bool = False,
    ) -> Variable:
        """Declare a variable and return its handle."""
        if name in self.variables:
            raise ValueError(f"variable {name!r} already declared")
        space = QuantitySpace(
            landmarks,
            lower_unbounded=lower_unbounded or unbounded,
            upper_unbounded=upper_unbounded or unbounded,
        )
        var = Variable(name, space)
        self.variables[name] = var
        return var

    def constrain(self, constraint: Constraint) -> Constraint:
        for ref in constraint.variables:
            if ref not in self.variables:
                raise ValueError(
                    f"constraint {constraint!r} references undeclared variable {ref!r}"
                )
        for cv in constraint.corresponding_values:
            for ref, landmark in zip(constraint.variables, cv):
                if landmark not in self.variables[ref].space.effective_landmarks:
                    raise ValueError(
                        f"corresponding value {landmark!r} is not a landmark of {ref!r}"
                    )
        self.constraints.append(constraint)
        return constraint

    def region(
        self, name: str, *, constraints: tuple[Constraint, ...] | None = None
    ) -> str:
        """Declare an operating region. ``constraints=None`` means all of
        the model's constraints are active in it. The first declared region
        becomes the initial region unless ``initial_region`` is set."""
        if name in self.regions:
            raise ValueError(f"region {name!r} already declared")
        if constraints is not None:
            for c in constraints:
                if not any(c is existing for existing in self.constraints):
                    raise ValueError(
                        f"region {name!r} references a constraint not added "
                        f"to the model: {c!r}"
                    )
        self.regions[name] = tuple(constraints) if constraints is not None else None
        if self.initial_region is None:
            self.initial_region = name
        return name

    def transition(
        self,
        source: str,
        target: str,
        *,
        when: tuple[tuple[str, str, str], ...],
    ) -> RegionTransition:
        """Declare a region transition guarded by landmark predicates
        (conjunction of ``(variable, op, landmark)`` atoms)."""
        for region in (source, target):
            if region not in self.regions:
                raise ValueError(f"undeclared region {region!r}")
        guards = []
        for var, op, landmark in when:
            if var not in self.variables:
                raise ValueError(f"guard references undeclared variable {var!r}")
            if landmark not in self.variables[var].space.effective_landmarks:
                raise ValueError(
                    f"guard landmark {landmark!r} is not a landmark of {var!r}"
                )
            guards.append(Guard(var, op, landmark))
        tr = RegionTransition(source, target, tuple(guards))
        self.region_transitions.append(tr)
        return tr

    def state(
        self,
        time: TimeTag = TimeTag.POINT,
        **values: tuple[str | tuple[str, str], Qdir] | QVal,
    ) -> QState:
        """Build a state from keyword args.

        Each value is either a ``QVal``, or a ``(magnitude, qdir)`` pair
        where magnitude is a landmark name (``"0"``) for an at-landmark
        value or a pair of adjacent landmark names (``("0", "FULL")``) for
        an open-interval value.
        """
        out: dict[str, QVal] = {}
        for name, v in values.items():
            if name not in self.variables:
                raise ValueError(f"unknown variable {name!r}")
            if isinstance(v, QVal):
                out[name] = v
                continue
            mag_spec, qdir = v
            space = self.variables[name].space
            if isinstance(mag_spec, tuple):
                rank = space.rank_between(*mag_spec)
            else:
                rank = space.rank_of(mag_spec)
            out[name] = QVal(rank, qdir)
        missing = set(self.variables) - set(out)
        if missing:
            raise ValueError(f"state is missing variables: {sorted(missing)}")
        return QState.from_dict(out, time)

    # --- analysis exports --------------------------------------------------

    def sign_structure(self) -> SignStructure:
        """Export the constraint closure as plain data (Surface 6)."""
        monotone: list[tuple[str, str, int]] = []
        derivatives: list[tuple[str, str]] = []
        sums: list[tuple[str, str, str]] = []
        products: list[tuple[str, str, str]] = []
        constants: list[str] = []
        corresponding: list = []
        negligible: list[tuple[str, str]] = []
        for c in self.constraints:
            kind = _KIND[type(c)]
            if kind == "mplus":
                monotone.append((c.x, c.y, 1))
            elif kind in ("mminus", "minus"):
                monotone.append((c.x, c.y, -1))
            elif kind == "deriv":
                derivatives.append((c.x, c.y))
            elif kind == "add":
                sums.append((c.x, c.y, c.z))
            elif kind == "mult":
                products.append((c.x, c.y, c.z))
            elif kind == "constant":
                constants.append(c.x)
            elif kind == "negligible":
                negligible.append((c.small, c.large))
            if c.corresponding_values:
                corresponding.append((c.variables, c.corresponding_values))
        return SignStructure(
            tuple(monotone),
            tuple(derivatives),
            tuple(sums),
            tuple(products),
            tuple(constants),
            tuple(corresponding),
            tuple(negligible),
        )

    # --- serialization (versioned schema) ----------------------------------

    def to_dict(self) -> dict:
        def landmark_dict(lm: Landmark) -> dict:
            out: dict = {"name": lm.name}
            for k in ("value", "lo", "hi"):
                if getattr(lm, k) is not None:
                    out[k] = getattr(lm, k)
            return out

        def constraint_dict(c: Constraint) -> dict:
            if isinstance(c, At):  # the landmark is an argument, not a cval
                return {"kind": "at", "args": [c.x, c.landmark]}
            out = {"kind": _KIND[type(c)], "args": list(c.variables)}
            if c.corresponding_values:
                out["cvals"] = [list(cv) for cv in c.corresponding_values]
            return out

        def index_of(c: Constraint) -> int:
            for i, existing in enumerate(self.constraints):
                if existing is c:
                    return i
            raise ValueError(f"constraint {c!r} not in model")

        data: dict = {
            "schema": MODEL_SCHEMA,
            "name": self.name,
            "variables": [
                {
                    "name": v.name,
                    "landmarks": [landmark_dict(lm) for lm in v.space.landmarks],
                    "lower_unbounded": v.space.lower_unbounded,
                    "upper_unbounded": v.space.upper_unbounded,
                }
                for v in self.variables.values()
            ],
            "constraints": [constraint_dict(c) for c in self.constraints],
        }
        if self.regions:
            data["regions"] = [
                {
                    "name": name,
                    "constraints": (
                        None if subset is None else [index_of(c) for c in subset]
                    ),
                }
                for name, subset in self.regions.items()
            ]
            data["transitions"] = [
                {
                    "source": t.source,
                    "target": t.target,
                    "when": [[g.var, g.op, g.landmark] for g in t.guards],
                }
                for t in self.region_transitions
            ]
            data["initial_region"] = self.initial_region
        return data

    def content_hash(self) -> str:
        """Deterministic semantic identity for this model.

        The hash is computed from a canonical form of ``qrlib.model/v1``:
        dictionary and authoring order do not matter for variables,
        constraints, regions, transitions, guards, or corresponding-value
        sets. Landmark order and constraint argument order remain significant
        because they define quantity spaces and equations respectively.
        """
        payload = _canonical_model_payload(self.to_dict())
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        if data.get("schema") != MODEL_SCHEMA:
            raise ValueError(
                f"unsupported model schema {data.get('schema')!r} "
                f"(expected {MODEL_SCHEMA!r})"
            )
        m = cls(data["name"])
        for v in data["variables"]:
            m.variable(
                v["name"],
                landmarks=tuple(
                    Landmark(
                        lm["name"],
                        value=lm.get("value"),
                        lo=lm.get("lo"),
                        hi=lm.get("hi"),
                    )
                    for lm in v["landmarks"]
                ),
                lower_unbounded=v.get("lower_unbounded", False),
                upper_unbounded=v.get("upper_unbounded", False),
            )
        for c in data["constraints"]:
            if c["kind"] == "at":
                m.constrain(At(c["args"][0], c["args"][1]))
                continue
            cls_ = _CLASS[c["kind"]]
            args = c["args"]
            kwargs: dict = {}
            if "cvals" in c:
                kwargs["cvals"] = tuple(tuple(cv) for cv in c["cvals"])
            m.constrain(cls_(*args, **kwargs))
        for r in data.get("regions", []):
            subset = r["constraints"]
            m.region(
                r["name"],
                constraints=(
                    None
                    if subset is None
                    else tuple(m.constraints[i] for i in subset)
                ),
            )
        for t in data.get("transitions", []):
            m.transition(
                t["source"],
                t["target"],
                when=tuple((g[0], g[1], g[2]) for g in t["when"]),
            )
        if "initial_region" in data:
            m.initial_region = data["initial_region"]
        return m

    # --- compilation -------------------------------------------------------

    def compile(self) -> CompiledModel:
        """Freeze mappings and resolve constraints/regions."""
        model_hash = self.content_hash()
        var_order = tuple(self.variables)
        spaces = tuple(self.variables[v].space for v in var_order)

        def zero_rank(space: QuantitySpace) -> int | None:
            eff = space.effective_landmarks
            return 2 * eff.index(ZERO) if ZERO in eff else None

        def inf_pair(space: QuantitySpace) -> tuple[int | None, int | None]:
            lo = 0 if space.lower_unbounded else None
            hi = space.num_ranks - 1 if space.upper_unbounded else None
            return (lo, hi)

        compiled: list[CompiledConstraint] = []
        for c in self.constraints:
            kind = _KIND[type(c)]
            idx = tuple(var_order.index(v) for v in c.variables)
            csp = [spaces[i] for i in idx]
            zeros = tuple(zero_rank(s) for s in csp)
            cvals = [
                tuple(csp[k].rank_of(nm) for k, nm in enumerate(cv))
                for cv in c.corresponding_values
            ]
            if kind == "deriv" and zeros[1] is None:
                raise ValueError(
                    f"{c!r}: the derivative variable {c.variables[1]!r} needs a "
                    f"'0' landmark (its sign drives the other's direction)"
                )
            if kind == "mult" and any(z is None for z in zeros):
                raise ValueError(
                    f"{c!r}: MULT requires a '0' landmark in every operand space"
                )
            if kind == "negligible" and any(z is None for z in zeros):
                raise ValueError(
                    f"{c!r}: NEGLIGIBLE requires a '0' landmark in both spaces"
                )
            # Implicit corresponding values at zero: x+y=z pins (0,0,0);
            # y=-x pins (0,0). These carry the sign algebra of the constraint.
            if kind == "add" and all(z is not None for z in zeros):
                implicit3 = (zeros[0], zeros[1], zeros[2])
                if implicit3 not in cvals:
                    cvals.append(implicit3)
            if kind == "minus" and all(z is not None for z in zeros):
                implicit2 = (zeros[0], zeros[1])
                if implicit2 not in cvals:
                    cvals.append(implicit2)
            compiled.append(
                CompiledConstraint(
                    kind, idx, tuple(cvals), zeros, tuple(inf_pair(s) for s in csp), c
                )
            )

        def constraint_index(c: Constraint) -> int:
            for i, existing in enumerate(self.constraints):
                if existing is c:
                    return i
            raise ValueError(f"region references unknown constraint {c!r}")

        if self.regions:
            initial = self.initial_region or next(iter(self.regions))
            if initial not in self.regions:
                raise ValueError(f"initial region {initial!r} is not declared")
            regions = []
            for rname, subset in self.regions.items():
                idxs = (
                    tuple(range(len(self.constraints)))
                    if subset is None
                    else tuple(constraint_index(c) for c in subset)
                )
                transitions = tuple(
                    CompiledTransition(
                        t.target,
                        tuple(
                            (var_order.index(g.var), g.op, g.landmark)
                            for g in t.guards
                        ),
                    )
                    for t in self.region_transitions
                    if t.source == rname
                )
                regions.append(CompiledRegion(rname, idxs, transitions))
            for t in self.region_transitions:
                if t.target not in self.regions:
                    raise ValueError(f"transition targets undeclared region {t.target!r}")
        else:
            initial = DEFAULT_REGION
            regions = [
                CompiledRegion(DEFAULT_REGION, tuple(range(len(self.constraints))), ())
            ]
        # --- order-of-magnitude closure (FOG's Ne) ----------------------
        # Ne is transitive; a cycle is a contradiction. An ADD gains a
        # dominant operand where the closed relation orders its operands —
        # only when every region activating the ADD supports the ordering.
        ne_idx = [i for i, cc in enumerate(compiled) if cc.kind == "negligible"]
        if ne_idx:

            def closed_pairs(active: frozenset[int]) -> set[tuple[int, int]]:
                pairs = {compiled[i].vars for i in ne_idx if i in active}
                changed = True
                while changed:
                    changed = False
                    for a, b in list(pairs):
                        for c2, d2 in list(pairs):
                            if b == c2 and (a, d2) not in pairs:
                                pairs.add((a, d2))
                                changed = True
                return pairs

            per_region: dict[str, set[tuple[int, int]]] = {}
            for reg in regions:
                pr = closed_pairs(frozenset(reg.constraint_idx))
                cyclic = sorted({var_order[a] for a, b in pr if a == b})
                if cyclic:
                    raise ValueError(
                        f"contradictory Negligible cycle involving {cyclic}"
                    )
                per_region[reg.name] = pr
            for i, cc in enumerate(compiled):
                if cc.kind != "add":
                    continue
                x, y = cc.vars[0], cc.vars[1]
                doms = [
                    1 if (x, y) in per_region[reg.name]
                    else 0 if (y, x) in per_region[reg.name]
                    else None
                    for reg in regions
                    if i in reg.constraint_idx
                ]
                if doms and doms[0] is not None and all(d == doms[0] for d in doms):
                    compiled[i] = replace(cc, dominant=doms[0])

        return CompiledModel(
            name=self.name,
            var_order=var_order,
            spaces=spaces,
            constraints=tuple(compiled),
            model_hash=model_hash,
            regions=tuple(regions),
            initial_region=initial,
        )


def _canonical_model_payload(data: dict) -> dict:
    """Canonicalize ``qrlib.model/v1`` for semantic content hashing."""

    def json_key(value: object) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def constraint(c: dict) -> dict:
        out = {"kind": c["kind"], "args": list(c["args"])}
        if "cvals" in c:
            out["cvals"] = sorted(
                (list(cv) for cv in c["cvals"]),
                key=json_key,
            )
        return out

    constraints = [constraint(c) for c in data["constraints"]]
    payload: dict = {
        "schema": data["schema"],
        "name": data["name"],
        "variables": sorted(
            (
                {
                    "name": v["name"],
                    "landmarks": [dict(lm) for lm in v["landmarks"]],
                    "lower_unbounded": v.get("lower_unbounded", False),
                    "upper_unbounded": v.get("upper_unbounded", False),
                }
                for v in data["variables"]
            ),
            key=lambda v: v["name"],
        ),
        "constraints": sorted(constraints, key=json_key),
    }
    if "regions" in data:
        regions = []
        for region in data["regions"]:
            subset = region["constraints"]
            regions.append(
                {
                    "name": region["name"],
                    "constraints": (
                        None
                        if subset is None
                        else sorted(
                            (constraints[i] for i in subset),
                            key=json_key,
                        )
                    ),
                }
            )
        payload["regions"] = sorted(regions, key=lambda r: r["name"])
        payload["transitions"] = sorted(
            (
                {
                    "source": t["source"],
                    "target": t["target"],
                    "when": sorted(
                        (list(guard) for guard in t["when"]),
                        key=json_key,
                    ),
                }
                for t in data.get("transitions", [])
            ),
            key=json_key,
        )
        payload["initial_region"] = data.get("initial_region")
    return payload
