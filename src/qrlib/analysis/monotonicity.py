"""Signed-graph consistency for qualitative monotone relationships.

Research lineage: the classical signed-graph balance characterization of
Harary (1953).

``M+`` requires two variables to have the same order polarity; ``M-`` and
``Minus`` require opposite polarities.  A set of such requirements is
*orthant-consistent* exactly when every signed cycle has positive sign, or
equivalently when each variable can be assigned a polarity ``+1``/``-1``
such that every edge sign is the product of its endpoint polarities.

This is deliberately a structural certificate only.  It does not inspect a
numeric vector field and therefore does not, by itself, prove that a
continuous-time system is monotone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..model import DEFAULT_REGION, CompiledModel, Model, _KIND

__all__ = [
    "SignedRelation",
    "MonotonicityCertificate",
    "check_signed_graph",
]


_SIGNS = {"mplus": 1, "mminus": -1, "minus": -1}
_LABELS = {"mplus": "M+", "mminus": "M-", "minus": "Minus"}


@dataclass(frozen=True)
class SignedRelation:
    """One pairwise order constraint in the signed graph.

    ``sign`` is ``+1`` when the endpoints must have the same polarity and
    ``-1`` when they must have opposite polarities.  ``constraint_index`` is
    the constraint's position in the model's full constraint list, including
    when the check is restricted to an operating region.
    """

    left: str
    right: str
    sign: int
    kind: str
    constraint_index: int

    def to_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "sign": self.sign,
            "kind": self.kind,
            "constraint_index": self.constraint_index,
        }


@dataclass(frozen=True)
class MonotonicityCertificate:
    """Result of checking whether pairwise signs admit an orthant order.

    ``polarities`` maps every model variable to ``+1`` (ordinary order) or
    ``-1`` (reversed order).  The assignment is canonical but not unique:
    flipping every polarity in one connected component gives the same
    certificate.  Isolated variables form singleton components with polarity
    ``+1``.

    When inconsistent, ``conflict_cycle`` contains a cycle whose edge-sign
    product is negative.  That cycle is a concrete witness that no polarity
    assignment can satisfy all of the declared pairwise relationships.
    """

    variable_order: tuple[str, ...]
    relations: tuple[SignedRelation, ...]
    polarities: dict[str, int]
    components: tuple[tuple[str, ...], ...]
    conflict_cycle: tuple[SignedRelation, ...] = ()
    region: str | None = None

    @property
    def is_consistent(self) -> bool:
        return not self.conflict_cycle

    def polarity_of(self, variable: str) -> int:
        try:
            return self.polarities[variable]
        except KeyError:
            raise KeyError(f"unknown variable {variable!r}") from None

    def to_dict(self) -> dict:
        return {
            "consistent": self.is_consistent,
            "region": self.region,
            "variable_order": list(self.variable_order),
            "polarities": dict(self.polarities),
            "components": [list(component) for component in self.components],
            "relations": [relation.to_dict() for relation in self.relations],
            "conflict_cycle": [relation.to_dict() for relation in self.conflict_cycle],
        }


def check_signed_graph(
    model: Model | CompiledModel, *, region: str | None = None
) -> MonotonicityCertificate:
    """Check the orthant consistency of a model's ``M+``/``M-``/``Minus`` graph.

    Other constraints are intentionally ignored because their pairwise signs
    depend on additional values or causal interpretation.  For a model with
    operating regions, pass ``region`` to check only relationships active in
    that region.  With no region, the union of the model's constraints is
    checked; mutually exclusive regional signs may therefore conflict in the
    whole-model result while each region remains consistent.
    """

    variable_order, selected = _selected_constraints(model, region)
    relations = tuple(
        SignedRelation(
            names[0],
            names[1],
            _SIGNS[kind],
            _LABELS[kind],
            constraint_index,
        )
        for constraint_index, kind, names in selected
        if kind in _SIGNS
    )

    adjacency: dict[str, list[int]] = {variable: [] for variable in variable_order}
    for edge_index, relation in enumerate(relations):
        adjacency[relation.left].append(edge_index)
        if relation.right != relation.left:
            adjacency[relation.right].append(edge_index)

    polarities: dict[str, int] = {}
    parent: dict[str, tuple[str, int] | None] = {}
    components: list[tuple[str, ...]] = []
    first_conflict: tuple[SignedRelation, ...] = ()
    order_index = {variable: i for i, variable in enumerate(variable_order)}

    for root in variable_order:
        if root in polarities:
            continue
        polarities[root] = 1
        parent[root] = None
        component: list[str] = []
        queue = deque([root])

        while queue:
            left = queue.popleft()
            component.append(left)
            for edge_index in adjacency[left]:
                relation = relations[edge_index]
                right = relation.right if relation.left == left else relation.left
                expected = polarities[left] * relation.sign
                if right not in polarities:
                    polarities[right] = expected
                    parent[right] = (left, edge_index)
                    queue.append(right)
                elif polarities[right] != expected and not first_conflict:
                    cycle_indices = _tree_path(left, right, parent)
                    cycle_indices.append(edge_index)
                    first_conflict = tuple(relations[i] for i in cycle_indices)

        components.append(tuple(sorted(component, key=order_index.__getitem__)))

    return MonotonicityCertificate(
        variable_order,
        relations,
        polarities,
        tuple(components),
        first_conflict,
        region,
    )


def _selected_constraints(
    model: Model | CompiledModel, region: str | None
) -> tuple[tuple[str, ...], list[tuple[int, str, tuple[str, ...]]]]:
    if isinstance(model, CompiledModel):
        if region is None:
            indices = range(len(model.constraints))
        else:
            indices = model.region_named(region).constraint_idx
        selected = [
            (
                index,
                model.constraints[index].kind,
                tuple(model.var_order[i] for i in model.constraints[index].vars),
            )
            for index in indices
        ]
        return model.var_order, selected

    variable_order = tuple(model.variables)
    if region is None or (region == DEFAULT_REGION and not model.regions):
        constraints = list(enumerate(model.constraints))
    else:
        if region not in model.regions:
            raise KeyError(region)
        active = model.regions[region]
        if active is None:
            constraints = list(enumerate(model.constraints))
        else:
            constraints = [
                (
                    next(
                        i
                        for i, existing in enumerate(model.constraints)
                        if existing is constraint
                    ),
                    constraint,
                )
                for constraint in active
            ]
    selected = [
        (index, _KIND[type(constraint)], tuple(constraint.variables))
        for index, constraint in constraints
    ]
    return variable_order, selected


def _tree_path(
    left: str,
    right: str,
    parent: dict[str, tuple[str, int] | None],
) -> list[int]:
    """Edge indices on the spanning-tree path from ``left`` to ``right``."""

    left_ancestors: dict[str, int] = {}
    left_edges: list[int] = []
    node = left
    while True:
        left_ancestors[node] = len(left_edges)
        step = parent[node]
        if step is None:
            break
        node, edge_index = step
        left_edges.append(edge_index)

    right_edges: list[int] = []
    node = right
    while node not in left_ancestors:
        step = parent[node]
        if step is None:  # defensive: a checked edge always stays in one tree
            raise RuntimeError("signed-graph parent forest is disconnected")
        node, edge_index = step
        right_edges.append(edge_index)

    return left_edges[: left_ancestors[node]] + list(reversed(right_edges))
