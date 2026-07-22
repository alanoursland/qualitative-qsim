"""Causal ordering: which variable determines which.

Derives causal structure from a model's constraints alone — no simulation —
following the Simon / Iwasaki-Simon account (docs/literature-survey.md §4).
The library's explanation layer narrates *what* changes along a behavior;
this narrates *what causes what* in the model's structure.

Method (structural matching + strongly-connected components):

1. Treat each constraint as a structural equation over its variables and
   assign each equation the one variable it "solves for" (a matching of
   equations to variables). A variable's determining equation, together
   with that equation's *other* variables, gives the causal edges
   ``input → variable``.
2. **Integral causality** (Iwasaki & Simon): a ``DERIV(x, y)`` — ``y = ẋ``
   — is read as "``x`` is determined by integrating ``y``". So the state
   variable ``x`` is a *given* at each instant, the integration equation is
   its determiner, and the edge is ``y → x`` (an integration edge). This is
   the physically standard reading; the alternative ("derivative
   causality") is not used.
3. ``CONSTANT(x)`` fixes ``x`` — an exogenous input, a determiner with no
   causal inputs.
4. The remaining (algebraic) equations are matched to the remaining
   variables by augmenting-path bipartite matching — the structural
   equivalent of constraint propagation ("plunking").
5. Splitting integration edges from instantaneous ones leaves an
   instantaneous causal graph that is acyclic for a well-posed model; its
   topological order is the "given the state and inputs, compute forward"
   chain. The integration edges close **feedback loops** over time. Loops
   (and any purely algebraic simultaneity) surface as strongly-connected
   components — variables with *no* causal order among them, exactly as the
   theory requires for simultaneous equations.

A model whose equations do not match its variables one-to-one is
**structurally singular** (over- or under-determined); the unmatched
variables/equations are reported rather than guessed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..graph import tarjan_scc
from ..model import CompiledModel, Model, _KIND

__all__ = ["CausalLink", "CausalOrder", "causal_order", "narrate_causes"]

_RELATION = {
    "mplus": "M+",
    "mminus": "M-",
    "minus": "−",
    "add": "+",
    "mult": "×",
    "deriv": "d/dt",
    "constant": "const",
    "at": "@",
}


@dataclass(frozen=True)
class CausalLink:
    """One determined variable and what determines it."""

    variable: str
    relation: str  # constraint kind label (see _RELATION)
    inputs: tuple[str, ...]  # variables that causally determine it
    integral: bool  # True if determined by integration (a DERIV)


@dataclass(frozen=True)
class CausalOrder:
    """Causal structure of a model, derived from its constraints.

    - ``exogenous``: inputs with no causal determiner (CONSTANT-fixed or
      otherwise free-standing) — the drivers of everything else.
    - ``state_variables``: variables under a derivative (given at each
      instant by their own history).
    - ``links``: the determiner of each determined variable.
    - ``instantaneous_edges`` / ``integration_edges``: causal edges
      ``(cause, effect)``, split by whether they come from integration.
    - ``levels``: instantaneous causal depth (roots at 0), over the
      integration-free graph condensed by algebraic SCCs.
    - ``loops``: strongly-connected components (size > 1) of the *full*
      causal graph — feedback loops / simultaneous variable sets with no
      internal causal order.
    - ``singular``: variables left without a determining equation
      (under-determined); empty when the model is a well-posed square system.
    - ``redundant``: equation labels left without a variable to determine
      (over-determined); empty for a square system.
    """

    var_order: tuple[str, ...]
    exogenous: tuple[str, ...]
    state_variables: tuple[str, ...]
    links: tuple[CausalLink, ...]
    instantaneous_edges: tuple[tuple[str, str], ...]
    integration_edges: tuple[tuple[str, str], ...]
    levels: dict[str, int]
    loops: tuple[tuple[str, ...], ...]
    singular: tuple[str, ...] = ()
    redundant: tuple[str, ...] = ()

    @property
    def is_singular(self) -> bool:
        return bool(self.singular or self.redundant)

    def link_of(self, variable: str) -> CausalLink | None:
        for link in self.links:
            if link.variable == variable:
                return link
        return None

    def to_dict(self) -> dict:
        return {
            "exogenous": list(self.exogenous),
            "state_variables": list(self.state_variables),
            "links": [
                {
                    "variable": lk.variable,
                    "relation": lk.relation,
                    "inputs": list(lk.inputs),
                    "integral": lk.integral,
                }
                for lk in self.links
            ],
            "instantaneous_edges": [list(e) for e in self.instantaneous_edges],
            "integration_edges": [list(e) for e in self.integration_edges],
            "levels": dict(self.levels),
            "loops": [list(c) for c in self.loops],
            "singular": list(self.singular),
            "redundant": list(self.redundant),
        }


def causal_order(
    model: Model | CompiledModel, *, region: str | None = None
) -> CausalOrder:
    """Derive the causal ordering of ``model``.

    For a multi-region compiled model, pass ``region`` to restrict the
    analysis to that region's active constraints. The whole-model default
    unions every region's constraints, which is typically over-determined
    for a multi-region model (a variable integrated in one region and held
    constant in another) — analyze per region instead.
    """
    var_order, eqs = _equations(model, region)

    # determiner[var] = equation index that solves for it
    determiner: dict[str, int] = {}
    consumed_eq: set[int] = set()

    # 1. CONSTANT / AT fix their variable (exogenous).
    for i, (kind, names, _src) in enumerate(eqs):
        if kind in ("constant", "at") and names[0] not in determiner:
            determiner[names[0]] = i
            consumed_eq.add(i)

    # 2. Integral causality: DERIV determines its state variable x.
    for i, (kind, names, _src) in enumerate(eqs):
        if kind != "deriv":
            continue
        x, y = names
        if x not in determiner:
            determiner[x] = i
            consumed_eq.add(i)
        elif y not in determiner:  # x already fixed (e.g. CONSTANT): y = ẋ
            determiner[y] = i
            consumed_eq.add(i)

    # 3. Match the remaining algebraic equations to the remaining variables.
    free_vars = set(var_order) - set(determiner)
    residual = [i for i in range(len(eqs)) if i not in consumed_eq]
    adj = {i: [v for v in eqs[i][1] if v in free_vars] for i in residual}
    match_var: dict[str, int] = {}

    def augment(i: int, seen: set[str]) -> bool:
        for v in adj[i]:
            if v in seen:
                continue
            seen.add(v)
            if v not in match_var or augment(match_var[v], set(seen)):
                match_var[v] = i
                return True
        return False

    for i in residual:
        augment(i, set())
    for v, i in match_var.items():
        determiner[v] = i
        consumed_eq.add(i)

    # 4. Build causal edges from the determiner map.
    inst_edges: list[tuple[str, str]] = []
    integ_edges: list[tuple[str, str]] = []
    links: list[CausalLink] = []
    for v in var_order:
        if v not in determiner:
            continue
        kind, names, _src = eqs[determiner[v]]
        inputs = tuple(u for u in names if u != v)
        integral = kind == "deriv"
        links.append(CausalLink(v, _RELATION[kind], inputs, integral))
        for u in inputs:
            (integ_edges if integral else inst_edges).append((u, v))

    full_edges = inst_edges + integ_edges
    indeg = {v: 0 for v in var_order}
    for _u, w in full_edges:
        indeg[w] += 1
    exogenous = tuple(v for v in var_order if indeg[v] == 0)
    state_vars = tuple(
        names[0] for kind, names, _s in eqs if kind == "deriv"
    )
    # de-dupe state vars preserving order
    seen_sv: set[str] = set()
    state_variables = tuple(
        v for v in state_vars if not (v in seen_sv or seen_sv.add(v))
    )

    levels = _condensation_levels(var_order, inst_edges)
    loops = tuple(
        tuple(sorted(comp))
        for comp in tarjan_scc(_adjacency(var_order, full_edges))
        if len(comp) > 1
    )
    singular = tuple(v for v in var_order if v not in determiner)
    redundant = tuple(
        f"{eqs[i][0]}({', '.join(eqs[i][1])})"
        for i in range(len(eqs))
        if i not in consumed_eq
    )

    return CausalOrder(
        var_order,
        exogenous,
        state_variables,
        tuple(links),
        tuple(inst_edges),
        tuple(integ_edges),
        levels,
        loops,
        singular,
        redundant,
    )


def narrate_causes(order: CausalOrder) -> str:
    """A prose account of a model's causal structure."""
    lines: list[str] = []
    if order.exogenous:
        lines.append(f"exogenous inputs: {', '.join(order.exogenous)}")
    if order.state_variables:
        lines.append(
            f"state variables (given by integration): "
            f"{', '.join(order.state_variables)}"
        )

    inst = [lk for lk in order.links if not lk.integral and lk.inputs]
    if inst:
        lines.append("instantaneous causal order (given the state and inputs):")
        for lk in sorted(inst, key=lambda l: (order.levels[l.variable], l.variable)):
            lines.append(
                f"  {', '.join(lk.inputs)} → {lk.variable}   (via {lk.relation})"
            )
    integ = [lk for lk in order.links if lk.integral]
    if integ:
        lines.append("integration (feedback over time):")
        for lk in integ:
            lines.append(
                f"  {', '.join(lk.inputs)} → {lk.variable}   "
                f"(d({lk.variable})/dt = {', '.join(lk.inputs)})"
            )
    for comp in order.loops:
        lines.append("feedback loop (no causal order within): " + " ↔ ".join(comp))
    if order.singular:
        lines.append(
            f"structurally singular — undetermined variables: "
            f"{', '.join(order.singular)}"
        )
    if order.redundant:
        lines.append(
            f"structurally singular — redundant equations: "
            f"{', '.join(order.redundant)}"
        )
    return "\n".join(lines)


# --- internals -------------------------------------------------------------


def _equations(model: Model | CompiledModel, region: str | None):
    if isinstance(model, Model):
        var_order = tuple(model.variables)
        # Negligible is an inequality annotation, not an equation
        eqs = [
            (_KIND[type(c)], tuple(c.variables), c)
            for c in model.constraints
            if _KIND[type(c)] != "negligible"
        ]
        return var_order, eqs
    # CompiledModel
    var_order = model.var_order
    if region is not None:
        constraints = model.constraints_of(region)
    else:
        constraints = model.constraints
    eqs = [
        (cc.kind, tuple(var_order[i] for i in cc.vars), cc.source)
        for cc in constraints
        if cc.kind != "negligible"
    ]
    return var_order, eqs


def _adjacency(nodes, edges):
    adj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
    return adj


def _condensation_levels(nodes, edges) -> dict[str, int]:
    """Longest-path level of each node over the SCC-condensation of the
    graph (algebraic simultaneity collapses to a single level)."""
    comps = tarjan_scc(_adjacency(nodes, edges))
    comp_of = {n: i for i, comp in enumerate(comps) for n in comp}
    cadj: dict[int, set[int]] = {i: set() for i in range(len(comps))}
    indeg = {i: 0 for i in range(len(comps))}
    for u, v in edges:
        cu, cv = comp_of[u], comp_of[v]
        if cu != cv and cv not in cadj[cu]:
            cadj[cu].add(cv)
            indeg[cv] += 1
    level = {i: 0 for i in range(len(comps))}
    q = deque(i for i in indeg if indeg[i] == 0)
    while q:
        c = q.popleft()
        for d in cadj[c]:
            level[d] = max(level[d], level[c] + 1)
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    return {n: level[comp_of[n]] for n in nodes}
