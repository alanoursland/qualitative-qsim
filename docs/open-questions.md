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
3. **Differentiability?** → Implemented as `qrlib.tensor.losses`, a soft
   autograd layer strictly above the exact core. Hosts may still consume
   qualitative structure as data when differentiation is unnecessary.
4. **Model interchange format?** → Yes, needed: hosts author and ship QDEs
   programmatically. Versioned JSON schema for models and results; frozen at
   the end of phase 4, marked unstable before that.
5. **Symbolic values on landmarks?** → No CAS dependency, ever, in core.
   Landmarks carry optional numeric `value`/bounds; symbolic identities stay
   host-side keyed by landmark name (Surface 1).
6. **Scale profile and model ensembles.** → Resolved with explicit
   production-shaped profiles (`docs/scale-profiles.md`). The qualified
   target is V ≤ 64 and up to one million total timesteps, conditioned on
   scalar volume and run density. CUDA is recommended for large,
   low-run-density, device-resident inputs; small/output-bound work stays on
   CPU. Existing shared-frame batching serves homogeneous ensembles.
   Heterogeneous ensembles are currently 2–12 models in library workflows and
   remain independently scheduled; no first-class padded model axis, int8
   conversion, or `torch.compile` work is justified by the measurements.
8. **Crossing refinement at the seam.** → Resolved. The sample-only path
   remains the default. Event-aware hosts can supply `CrossingEvent` records
   containing the exact time, declared landmark, and complete solver state.
   Events are protected from debounce, original sample spans are retained,
   and physical-time bounds distinguish exact events from inferred brackets.
10. **Confidence semantics for estimated signs.** → Resolved. The calibrated
    estimator uses deterministic bootstrap sign agreement in `[0, 1]`, with
    explicit seed and resample metadata. It is documented as stability under
    the observed sample distribution rather than a truth probability;
    thresholding maps unstable and fitted-zero effects to `UNKNOWN`. The
    legacy t-like score remains available for compatibility.
12. **Order of the phase-7 extras.** → Resolved. Explanation,
    visualization, total envisionment, induction, comparative analysis,
    temporal-logic queries, and soft losses are implemented. Their completed
    entries remain in `docs/roadmap.md` as a development record.

## Open

7. **Analytic filter semantics.** ~~Should qrlib ship a first-class
   `EnergyFilter`?~~ **Resolved: shipped** (`qrlib.EnergyFilter`). A
   declarative `SuccessorFilter` (amplitude-contributing variables +
   `Trend` = CONSERVED | NONINCREASING + reference landmark) that enforces
   the energy argument through landmark discovery: turning points must
   coincide with discovered extrema (conserved) or never grow past them
   (dissipative), with the point/interval landmark semantics handled once.
   It reproduces the bespoke frictionless-spring filter byte-for-byte
   (the single 17-node cycle) and keeps the numeric soundness harness
   covered.
9. **String constraint syntax.** `"M+(level, outflow)"` parsing is cheap and
   host-ergonomic, but a second authoring path to maintain.
   *Provisional:* add in phase 4 alongside the schema freeze, as a thin
   layer over the schema only.
11. **License and distribution.** Deliberately deferred by the owner —
    do not add a license file until they choose one. Until then the
    default applies (all rights reserved; not publishable/distributable).
    PyPI name (`qualitative-reasoning-lib`? `qrlib` availability) to
    check alongside that decision.
