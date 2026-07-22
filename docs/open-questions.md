# Open questions

Updated after the host-integration requirements landed
(`docs/host-integration.md`). Resolved items are kept (struck through in
spirit) with their resolutions; new questions raised by that design follow.

## Resolved

1. **Which bridge direction matters most?** → Both upward abstraction *and*
   validation are primary: the coverage oracle (Surface 3) is the flagship
   host-facing product and the exit criterion of phase 3; abstraction is its
   prerequisite. Identification priors are served as data exports
   (Surface 6); active "guidance" of numeric search stays out of scope.
2. **Time semantics of the numeric side?** → Continuous-time, possibly
   irregularly sampled, **with hybrid/mode-switching support required
   early** — operating regions moved to phase 4 and the abstraction
   pipeline reserves a mode channel from day one.
3. **Differentiability?** → Not near-term. Hosts consume qualitative
   structure as *data* (sign structure for constrained regression, coverage
   scores for model selection); soft losses remain a possible phase-7 layer
   strictly above the exact core.
4. **Model interchange format?** → Yes, needed: hosts author and ship QDEs
   programmatically. Versioned JSON schema for models and results; frozen at
   the end of phase 4, marked unstable before that.
5. **Symbolic values on landmarks?** → No CAS dependency, ever, in core.
   Landmarks carry optional numeric `value`/bounds; symbolic identities stay
   host-side keyed by landmark name (Surface 1).

## Open

6. **Scale profile.** Typical variable counts, trajectory batch sizes, and
   whether model ensembles are a real workload — decides how hard to push
   the tensor engine (`int8` encodings, `torch.compile`).
   *Provisional:* design for `V ≤ ~64`, batches to ~10⁶ total timesteps,
   ensembles as a first-class batch axis.
7. **Analytic filter semantics.** The pluggable global-filter hook accepts
   user predicates; the classic use is an energy argument (a declared
   variable that must be non-increasing along behaviors). Should qrlib ship
   a first-class `EnergyFilter` (declare variable + monotonicity) or leave
   it to host predicates? *Provisional:* ship it — it is the canonical
   spurious-behavior killer and needs careful point/interval semantics that
   shouldn't be reinvented per host.
8. **Crossing refinement at the seam.** Abstraction tolerates sample-level
   landmark crossings; hosts with event-accurate solvers can pre-refine.
   Should the seam optionally accept per-crossing refined times to tighten
   behaviors, or is sample-level always enough? *Provisional:* accept an
   optional refined-events input later; not in the phase-3 pipeline.
9. **String constraint syntax.** `"M+(level, outflow)"` parsing is cheap and
   host-ergonomic, but a second authoring path to maintain.
   *Provisional:* add in phase 4 alongside the schema freeze, as a thin
   layer over the schema only.
10. **Confidence semantics for estimated signs.** `bridge.signs` estimation
    returns per-entry confidence; what statistic (sign agreement rate?
    bootstrap?) and what threshold feeds model construction?
    Decide during phase 4 with real data in hand.
11. **License and distribution.** Deliberately deferred by the owner —
    do not add a license file until they choose one. Until then the
    default applies (all rights reserved; not publishable/distributable).
    PyPI name (`qualitative-reasoning-lib`? `qrlib` availability) to
    check alongside that decision.
12. **Order of the phase-7 extras.** *Partially resolved:* explanation
    and viz are built; the rest (total envisionment, induction,
    comparative analysis, temporal-logic queries, soft losses) now live
    as the demand-driven backlog in `docs/roadmap.md` — pick up when the
    first host adapter asks.
