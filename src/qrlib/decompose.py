"""DecSIM-style decomposed simulation.

Research lineage: [ClancyKuipers1997] in ``docs/references.md``.

Monolithic QSIM branches on every qualitative event of every variable: ``n``
simultaneously moving but mutually unconstrained variables produce up to
``2**n`` successor orderings, so loosely coupled systems drown in
interleavings of events that never interact. Decomposed simulation attacks
the blowup at its source:

1. **Partition** the model's variables into named components — user-chosen,
   or :func:`suggest_partition`'s connected-components split of the
   constraint graph.
2. **Assign** every constraint to exactly one component (the one owning the
   most of its variables); foreign variables a constraint drags along
   become the component's **interface**. Each component induces a
   sub-:class:`~qrlib.model.Model` over its owned + interface variables.
3. **Simulate** components independently, upstream before downstream
   (owners of interface variables first). A downstream run is **guided**:
   each upstream behavior's story for a shared variable — its sequence of
   qualitative-value *episodes* — is compiled into a temporal-logic word
   (nested untils over ``qrlib.guide`` atoms), and the disjunction of the
   upstream stories rides the downstream simulation as a guide spec,
   pruning states no upstream story can produce. This is DecSIM's
   coordination realized through the TeQSIM machinery: the upstream
   component acts as a temporally-specified exogenous input to the
   downstream one. Interface variables that cannot be guided (cyclic
   coupling, or genuinely periodic upstream words) are chatter-abstracted
   (``ignore_qdir``) instead, since the component does not model their
   dynamics.
4. **Join**: two component behaviors are compatible when their shared
   variables tell the same story — the same sequence of magnitude episodes,
   expressed in the parent model's *declared* landmarks so per-branch
   discovered landmarks don't obstruct comparison. Terminal classes give
   the sequences their suffix semantics: a quiescent behavior ends in an
   episode that lasts forever, a cycle repeats its episodes, and a
   truncated / dead-ended / domain-or-region-exit behavior simply ends — agreement
   on the common prefix suffices, because the shorter behavior's time
   stops where the longer one continues. Behaviors ended by the guide
   (``SPEC_PRUNED``) are pruning records, not behaviors, and never join.

The result (:class:`DecResult`) is one behavior graph *per component* plus
pairwise compatibility links — an implicit product whose explicit tuples
:meth:`DecResult.joint_behaviors` enumerates on demand. Guarantees follow
the library's governing caveat: every real behavior of the full system
projects into each component's behavior set and survives the guide and the
join (both prune only provably-inconsistent prefixes), while some linked
tuples may be spurious — including the deliberate loss of cross-component
event *ordering*, which is exactly the information whose omission buys the
exponential savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import reduce
from math import prod

from .behavior import Behavior, BehaviorGraph, SimConfig, SimResult, SimStatus, TerminalClass
from .engines.qsim import qsim
from .model import Model
from .quantity import Qdir, QuantitySpace
from .state import QState

__all__ = [
    "Component",
    "Part",
    "Decomposition",
    "ComponentRun",
    "DecResult",
    "Stream",
    "suggest_partition",
    "decompose",
    "decsim",
    "episode_stream",
    "compatible",
]


# --- partitioning ----------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """A named set of variables the caller declares tightly coupled."""

    name: str
    variables: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if not self.name:
            raise ValueError("a Component needs a non-empty name")
        if not self.variables:
            raise ValueError(f"component {self.name!r} owns no variables")


@dataclass(frozen=True)
class Part:
    """One component resolved against a model: its sub-model + interface."""

    component: Component
    interface: tuple[str, ...]  # foreign variables assigned constraints use
    model: Model  # sub-model over owned + interface variables


@dataclass(frozen=True)
class Decomposition:
    source: Model
    parts: tuple[Part, ...]

    def shared(self, i: int, j: int) -> tuple[str, ...]:
        """Variables both parts simulate, in the parent model's order."""
        vi = set(self.parts[i].model.variables)
        vj = set(self.parts[j].model.variables)
        return tuple(v for v in self.source.variables if v in vi and v in vj)


def suggest_partition(model: Model) -> tuple[Component, ...]:
    """Connected components of the constraint graph (variables linked when
    they co-occur in a constraint), named ``c0, c1, ...`` in declaration
    order. Fully coupled models yield a single component."""
    parent = {v: v for v in model.variables}

    def find(v: str) -> str:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for c in model.constraints:
        first = c.variables[0]
        for other in c.variables[1:]:
            parent[find(other)] = find(first)

    groups: dict[str, list[str]] = {}
    for v in model.variables:  # declaration order
        groups.setdefault(find(v), []).append(v)
    return tuple(
        Component(f"c{i}", tuple(vs)) for i, vs in enumerate(groups.values())
    )


def decompose(
    model: Model,
    components: dict[str, tuple[str, ...]] | tuple[Component, ...] | list[Component],
) -> Decomposition:
    """Resolve a variable partition into per-component sub-models.

    Every constraint is assigned to the component owning the most of its
    variables (ties to the earliest component); the assigned component
    absorbs the constraint's foreign variables as interface. An owned
    variable no assigned constraint mentions stays in its sub-model,
    unconstrained.
    """
    if not isinstance(model, Model):
        raise TypeError("decompose needs an uncompiled Model")
    if model.regions:
        raise ValueError("decomposed simulation does not support operating regions")
    if isinstance(components, dict):
        comps = tuple(Component(k, tuple(v)) for k, v in components.items())
    else:
        comps = tuple(components)
    if not comps:
        raise ValueError("at least one component is required")
    names = [c.name for c in comps]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate component names: {names!r}")

    owner: dict[str, int] = {}
    for i, comp in enumerate(comps):
        unknown = set(comp.variables) - set(model.variables)
        if unknown:
            raise ValueError(
                f"component {comp.name!r} names unknown variables: {sorted(unknown)}"
            )
        for v in comp.variables:
            if v in owner:
                raise ValueError(
                    f"variable {v!r} is owned by both "
                    f"{comps[owner[v]].name!r} and {comp.name!r}"
                )
            owner[v] = i
    unowned = [v for v in model.variables if v not in owner]
    if unowned:
        raise ValueError(f"variables not owned by any component: {unowned}")

    assigned: list[list] = [[] for _ in comps]
    for c in model.constraints:
        votes = [owner[v] for v in c.variables]
        best = max(set(votes), key=lambda i: (votes.count(i), -i))
        assigned[best].append(c)

    parts = []
    for comp, constraints in zip(comps, assigned):
        keep = set(comp.variables)
        for c in constraints:
            keep.update(c.variables)
        sub = Model(f"{model.name}.{comp.name}")
        for v in model.variables:  # parent order, for determinism
            if v in keep:
                sub.variables[v] = model.variables[v]
        sub.constraints = list(constraints)
        interface = tuple(v for v in sub.variables if v not in set(comp.variables))
        parts.append(Part(comp, interface, sub))
    return Decomposition(model, tuple(parts))


# --- shared-variable episode streams ---------------------------------------


@dataclass(frozen=True)
class Stream:
    """A behavior's shared-variable word.

    ``letters`` are per-state value tuples; ``loop`` (cycles) repeats
    forever after them; ``forever`` (quiescence) pins the last letter for
    all future time. Neither set means the word — the behavior's time —
    simply ends.
    """

    letters: tuple[tuple, ...]
    loop: tuple[tuple, ...] | None = None
    forever: bool = False


def _declared_rank(space: QuantitySpace, declared: QuantitySpace, rank: int) -> int:
    """Map a magnitude rank in a (possibly grown) frame space onto the
    declared space: discovered landmarks dissolve into the declared
    interval that contains them."""
    eff = space.effective_landmarks
    deff = declared.effective_landmarks
    i = rank // 2
    if rank % 2 == 0:
        if eff[i] in deff:
            return 2 * deff.index(eff[i])
        upto = i  # at a discovered landmark: strictly inside a declared gap
    else:
        upto = i + 1  # interval (eff[i], eff[i+1]) lies inside a declared gap
    for k in range(upto - 1, -1, -1):
        if eff[k] in deff:
            return 2 * deff.index(eff[k]) + 1
    raise ValueError(f"rank {rank} has no declared landmark below it")


def _stream(
    graph: BehaviorGraph,
    behavior: Behavior,
    variables: tuple[str, ...],
    declared: dict[str, QuantitySpace],
    *,
    with_dirs: bool = False,
) -> Stream:
    def letter(nid: int) -> tuple:
        node = graph.nodes[nid]
        frame = node.model
        out = []
        for v in variables:
            qv = node.state[v]
            rank = _declared_rank(frame.spaces[frame.index(v)], declared[v], qv.mag)
            out.append((rank, int(qv.dir)) if with_dirs else rank)
        return tuple(out)

    if behavior.terminal is TerminalClass.CYCLE and behavior.cycle_target is not None:
        loop_start = behavior.node_ids.index(behavior.cycle_target)
        ids = behavior.node_ids[:-1]  # the closing node repeats the target
        return Stream(
            tuple(letter(n) for n in ids[:loop_start]),
            tuple(letter(n) for n in ids[loop_start:]),
        )
    letters = tuple(letter(n) for n in behavior.node_ids)
    return Stream(letters, None, behavior.terminal is TerminalClass.QUIESCENT)


def episode_stream(
    graph: BehaviorGraph,
    behavior: Behavior,
    variables: tuple[str, ...],
    declared: dict[str, QuantitySpace],
) -> Stream:
    """Project a behavior onto ``variables`` as a magnitude word in the
    declared spaces, with the suffix semantics of its terminal class."""
    return _stream(graph, behavior, variables, declared)


def _episodes(stream: Stream, bound: int) -> tuple[list, str]:
    """Collapse a stream into its episode sequence (adjacent equal letters
    merged). The tail marks what follows the returned episodes: ``"end"``
    (time stops), ``"forever"`` (the last episode persists), or ``"more"``
    (a genuinely periodic word, truncated at ``bound``)."""
    eps: list = []

    def push(letter) -> None:
        if not eps or eps[-1] != letter:
            eps.append(letter)

    for letter in stream.letters:
        push(letter)
        if len(eps) > bound:
            return eps, "more"
    if stream.loop:
        while True:
            grew = False
            for letter in stream.loop:
                n = len(eps)
                push(letter)
                if len(eps) > bound:
                    return eps, "more"
                grew = grew or len(eps) > n
            if not grew:  # a full pass added nothing: eventually constant
                return eps, "forever"
    return eps, ("forever" if stream.forever else "end")


def compatible(a: Stream, b: Stream) -> bool:
    """Do two words tell the same shared-variable story?

    Episode sequences must agree on their common length. Beyond it, a word
    whose time ends ("end") or that is merely truncated ("more") never
    conflicts; only a word pinned constant forever conflicts with a partner
    that provably changes again (episodes are maximal, so the partner's
    next episode differs from the pinned letter).
    """
    bound = (
        len(a.letters) + len(a.loop or ()) + len(b.letters) + len(b.loop or ()) + 2
    ) ** 2
    ea, ta = _episodes(a, bound)
    eb, tb = _episodes(b, bound)
    n = min(len(ea), len(eb))
    if ea[:n] != eb[:n]:
        return False
    if len(ea) == len(eb):
        return {ta, tb} != {"forever", "more"}
    shorter_tail = tb if len(ea) > len(eb) else ta
    return shorter_tail != "forever"


# --- upstream stories as guide specs ---------------------------------------

_DIR_NAME = {int(Qdir.DEC): "dec", int(Qdir.STD): "std", int(Qdir.INC): "inc"}


def _upstream_words(
    run: "ComponentRun", var: str, declared: QuantitySpace
) -> list[tuple[tuple, str]] | None:
    """The deduplicated (episodes, tail) words ``var`` follows across the
    owner's live behaviors, or None when some word is inexpressible as a
    finite until-chain (genuinely periodic)."""
    words: list[tuple[tuple, str]] = []
    seen: set = set()
    for k in run.live:
        b = run.behaviors[k]
        s = _stream(
            run.result.graph, b, (var,), {var: declared}, with_dirs=True
        )
        eps, tail = _episodes(s, len(s.letters) + len(s.loop or ()) + 2)
        if tail == "more":
            return None
        key = (tuple(eps), tail)
        if key not in seen:
            seen.add(key)
            words.append((tuple(eps), tail))
    return words or None


def _word_formula(var: str, eps: tuple, tail: str, declared: QuantitySpace, use_dir: bool):
    """One upstream story as a temporal-logic word: nested untils through
    its episodes, ending in G(last) when the last episode persists."""
    from .guide import G, U, dir_is, mag

    deff = declared.effective_landmarks

    def phi(letter):
        (rank, d) = letter[0]
        if rank % 2 == 0:
            f = mag(var, "==", deff[rank // 2])
        else:
            f = mag(var, ">", deff[rank // 2]) & mag(var, "<", deff[rank // 2 + 1])
        if use_dir and d in _DIR_NAME:
            f = f & dir_is(var, _DIR_NAME[d])
        return f

    phis = [phi(e) for e in eps]
    f = G(phis[-1]) if tail == "forever" else phis[-1]
    for p in reversed(phis[:-1]):
        f = U(p, f)
    return f


# --- decomposed simulation -------------------------------------------------


@dataclass(frozen=True)
class ComponentRun:
    part: Part
    result: SimResult
    behaviors: tuple[Behavior, ...]
    guided: tuple[str, ...] = ()  # interface vars guided by upstream words

    @property
    def name(self) -> str:
        return self.part.component.name

    @property
    def live(self) -> tuple[int, ...]:
        """Indices of behaviors that are not guide-pruning records."""
        return tuple(
            k
            for k, b in enumerate(self.behaviors)
            if b.terminal is not TerminalClass.SPEC_PRUNED
        )


@dataclass(frozen=True)
class DecResult:
    """Per-component behavior graphs + pairwise compatibility links."""

    decomposition: Decomposition
    runs: tuple[ComponentRun, ...]
    links: dict[tuple[int, int], tuple[tuple[int, int], ...]] = field(
        default_factory=dict
    )

    @property
    def status(self) -> SimStatus:
        complete = all(r.result.status is SimStatus.COMPLETE for r in self.runs)
        return SimStatus.COMPLETE if complete else SimStatus.TRUNCATED

    def component(self, name: str) -> ComponentRun:
        for run in self.runs:
            if run.name == name:
                return run
        raise KeyError(name)

    def joint_behaviors(self, *, limit: int | None = None) -> tuple[tuple[int, ...], ...]:
        """Tuples of per-component behavior indices that are pairwise
        compatible on every shared variable — the explicit form of the
        implicit product. Pairs with no shared variables never constrain;
        spec-pruned behaviors never participate."""
        pair_ok = {k: frozenset(v) for k, v in self.links.items()}
        out: list[tuple[int, ...]] = []

        def extend(chosen: list[int]) -> None:
            if limit is not None and len(out) >= limit:
                return
            k = len(chosen)
            if k == len(self.runs):
                out.append(tuple(chosen))
                return
            for b in self.runs[k].live:
                ok = all(
                    (i, k) not in pair_ok or (chosen[i], b) in pair_ok[(i, k)]
                    for i in range(k)
                )
                if ok:
                    chosen.append(b)
                    extend(chosen)
                    chosen.pop()

        extend([])
        return tuple(out)

    @property
    def stats(self) -> dict:
        return {
            "component_nodes": {
                r.name: r.result.stats["nodes"] for r in self.runs
            },
            "total_nodes": sum(r.result.stats["nodes"] for r in self.runs),
            "component_behaviors": {r.name: len(r.live) for r in self.runs},
            "behavior_product": prod(len(r.live) for r in self.runs),
        }

    def to_dict(self) -> dict:
        return {
            "schema": "qrlib.decsim/v1",
            "status": self.status.value,
            "components": [
                {
                    "name": r.name,
                    "variables": list(r.part.component.variables),
                    "interface": list(r.part.interface),
                    "guided": list(r.guided),
                    "status": r.result.status.value,
                    "nodes": r.result.stats["nodes"],
                    "behaviors": len(r.live),
                    "spec_pruned": len(r.behaviors) - len(r.live),
                }
                for r in self.runs
            ],
            "links": [
                [i, j, [list(p) for p in pairs]]
                for (i, j), pairs in sorted(self.links.items())
            ],
            "joint_behaviors": len(self.joint_behaviors()),
            "stats": self.stats,
        }


def decsim(
    model: Model,
    initial: QState,
    components: dict[str, tuple[str, ...]] | tuple[Component, ...] | list[Component] | None = None,
    *,
    config: SimConfig | None = None,
    max_states: int | None = None,
    max_depth: int | None = None,
) -> DecResult:
    """Simulate ``model`` component-by-component — upstream components
    guiding downstream ones — and join on shared variables.
    ``components=None`` partitions by :func:`suggest_partition` (a fully
    coupled model then degenerates to one monolithic component)."""
    cfg = config or SimConfig()
    overrides = {
        k: v
        for k, v in (("max_states", max_states), ("max_depth", max_depth))
        if v is not None
    }
    if overrides:
        cfg = replace(cfg, **overrides)
    if cfg.guide is not None:
        raise ValueError(
            "guided decomposed simulation is not supported; guide per "
            "component or run qrlib.guide.guided on the whole model"
        )
    dec = decompose(
        model, components if components is not None else suggest_partition(model)
    )

    owner: dict[str, int] = {}
    for k, part in enumerate(dec.parts):
        for v in part.component.variables:
            owner[v] = k
    deps = [{owner[v] for v in part.interface} for part in dec.parts]

    runs: dict[int, ComponentRun] = {}
    remaining = list(range(len(dec.parts)))
    while remaining:
        # next part whose upstream owners all ran; cyclic coupling breaks
        # the tie by running the earliest part unguided on the cycle vars
        pick = next((k for k in remaining if deps[k] <= set(runs)), remaining[0])
        remaining.remove(pick)
        part = dec.parts[pick]
        missing = [v for v in part.model.variables if v not in initial.variables]
        if missing:
            raise ValueError(f"initial state is missing variables: {missing}")

        words: dict[str, list] = {}
        for v in part.interface:
            if owner[v] in runs:
                w = _upstream_words(runs[owner[v]], v, model.variables[v].space)
                if w is not None:
                    words[v] = w
        ignore = [x for x in cfg.ignore_qdir if x in part.model.variables]
        ignore += [v for v in part.interface if v not in words and v not in ignore]
        spec = None
        for v, wlist in words.items():
            use_dir = v not in ignore
            formula = reduce(
                lambda a, b: a | b,
                (
                    _word_formula(v, eps, tail, model.variables[v].space, use_dir)
                    for eps, tail in wlist
                ),
            )
            spec = formula if spec is None else spec & formula
        # guided interface vars carry direction conjuncts in their words:
        # keep their directions tracked even under dynamic chatter
        tracked = tuple(
            dict.fromkeys(
                list(cfg.track_qdir) + [v for v in words if v not in ignore]
            )
        )
        sub_cfg = replace(
            cfg, ignore_qdir=tuple(ignore), guide=spec, track_qdir=tracked
        )
        sub_initial = QState.from_dict(
            {v: initial[v] for v in part.model.variables}, initial.time
        )
        result = qsim(part.model, sub_initial, config=sub_cfg)
        runs[pick] = ComponentRun(
            part, result, result.behaviors(), tuple(words)
        )
    ordered = tuple(runs[k] for k in range(len(dec.parts)))

    links: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            shared = dec.shared(i, j)
            if not shared:
                continue
            declared = {v: model.variables[v].space for v in shared}
            si = {
                a: episode_stream(
                    ordered[i].result.graph, ordered[i].behaviors[a], shared, declared
                )
                for a in ordered[i].live
            }
            sj = {
                b: episode_stream(
                    ordered[j].result.graph, ordered[j].behaviors[b], shared, declared
                )
                for b in ordered[j].live
            }
            links[(i, j)] = tuple(
                (a, b)
                for a, sa in si.items()
                for b, sb in sj.items()
                if compatible(sa, sb)
            )
    return DecResult(dec, ordered, links)
