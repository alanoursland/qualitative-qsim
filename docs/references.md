# Research references

This is the canonical, human-readable guide to the research behind
`qualitative-qsim`. The machine-readable records live in
[`paper.bib`](../paper.bib); the bracketed keys below are stable identifiers
used throughout the code and documentation.

For the code-and-test crosswalk covering references with direct implemented
lineage, see [`implemented-references.md`](implemented-references.md).

The categories matter:

- **Direct lineage** identifies a paper or book whose algorithm, semantics, or
  modeling vocabulary materially shaped an implemented feature.
- **Background** identifies broader context or a closely related approach; it
  does not claim that qrlib reimplements the cited system.
- **Adjacent work** records ideas discussed in design notes but not implemented
  as open-library functionality.

## Core qualitative simulation

### Kuipers1986

Benjamin Kuipers, “Qualitative Simulation,” *Artificial Intelligence* 29(3),
289–338 (1986). [doi:10.1016/0004-3702(86)90073-1](https://doi.org/10.1016/0004-3702(86)90073-1).

**Direct lineage:** QDE semantics, qualitative states, transition generation,
the guaranteed-coverage theorem, and the sound-but-potentially-spurious
interpretation of QSIM results.

### Kuipers1994

Benjamin J. Kuipers, *Qualitative Reasoning: Modeling and Simulation with
Incomplete Knowledge*, MIT Press (1994), ISBN 978-0-262-11190-4.
[Author’s book page](https://web.eecs.umich.edu/~kuipers/research/qsim/QR-book.html).

**Direct lineage:** the authoritative specification used for the reference
engine, quantity spaces, corresponding values, landmark discovery, filtering,
worked examples, semi-quantitative reasoning, and compositional modeling.

### Kuipers2001

Benjamin Kuipers, “Qualitative Simulation,” in *Encyclopedia of Physical
Science and Technology*, 3rd ed., 287–300, Academic Press (2001).
[Author’s record and manuscript](https://web.eecs.umich.edu/~kuipers/research/pubs/Kuipers-epst-01.html).

**Background:** compact retrospective used for the governing coverage caveat
and for relating QSIM to its later extensions.

## Alternative model vocabularies

### Forbus1984

Kenneth D. Forbus, “Qualitative Process Theory,” *Artificial Intelligence*
24(1–3), 85–168 (1984).
[doi:10.1016/0004-3702(84)90038-9](https://doi.org/10.1016/0004-3702(84)90038-9).

**Direct lineage:** `qrlib.frontends.qpt`, including process activation,
qualitative proportionalities, direct influences, and influence resolution.

### deKleerBrown1984

Johan de Kleer and John Seely Brown, “A Qualitative Physics Based on
Confluences,” *Artificial Intelligence* 24(1–3), 7–83 (1984).
[doi:10.1016/0004-3702(84)90037-7](https://doi.org/10.1016/0004-3702(84)90037-7).

**Direct lineage:** total envisionment and device/component composition in
`qrlib.envision` and `qrlib.frontends.devices`.

## Semi-quantitative reasoning

### KuipersBerleant1988

Benjamin J. Kuipers and Daniel Berleant, “Using Incomplete Quantitative
Knowledge in Qualitative Reasoning,” *Proceedings of the Seventh National
Conference on Artificial Intelligence*, 324–329 (1988).
[Manuscript](https://www.qrg.northwestern.edu/papers/Files/qr-workshops/QP88/Kuipers_1988_Using_Incomplete_Qualitative_Knowledge.pdf).

**Direct lineage:** Q2-style landmark intervals, constraint propagation,
behavior refutation, and time-bound refinement in `qrlib.semiquant`.

### BerleantKuipers1997

Daniel Berleant and Benjamin J. Kuipers, “Qualitative and Quantitative
Simulation: Bridging the Gap,” *Artificial Intelligence* 95(2), 215–255
(1997).
[doi:10.1016/S0004-3702(97)00050-7](https://doi.org/10.1016/S0004-3702(97)00050-7).

**Background:** Q3 and the larger semi-quantitative simulation lineage.

## Scaling and abstraction

### ClancyKuipers1997

Daniel J. Clancy and Benjamin Kuipers, “Model Decomposition and Simulation: A
Component Based Qualitative Simulation Algorithm,” *Proceedings of the
Fourteenth National Conference on Artificial Intelligence*, 118–124 (1997).
[AAAI paper](https://cdn.aaai.org/AAAI/1997/AAAI97-019.pdf).

**Direct lineage:** DecSIM-style partitioning, component simulation,
interface-variable guidance, and compatible-history joining in
`qrlib.decompose`.

### ClancyKuipers1997Chatter

Daniel J. Clancy and Benjamin Kuipers, “Static and Dynamic Abstraction Solves
the Problem of Chatter in Qualitative Simulation,” *Proceedings of the
Fourteenth National Conference on Artificial Intelligence*, 125–131 (1997).
[AAAI paper](https://cdn.aaai.org/AAAI/1997/AAAI97-020.pdf).

**Direct lineage:** structural chatter detection and dynamic direction
abstraction in `qrlib.engines.chatter`.

### ForbusFalkenhainer1995

Kenneth D. Forbus and Brian Falkenhainer, “Scaling up Self-Explanatory
Simulators: Polynomial-time Compilation,” *Proceedings of the Fourteenth
International Joint Conference on Artificial Intelligence* (1995).
[Author manuscript](https://www.qrg.northwestern.edu/papers/Files/Forbus_Falkenhainer95.pdf).

**Background:** the SIMGEN Mk3 scaling lesson: avoid global qualitative
analysis when a numerically instantiated task does not require it.

## Diagnosis and temporal reasoning

### SubramanianMooney1996

Siddarth Subramanian and Raymond J. Mooney, “Qualitative Multiple-Fault
Diagnosis of Continuous Dynamic Systems Using Behavioral Modes,”
*Proceedings of the Thirteenth National Conference on Artificial
Intelligence*, 965–970 (1996).
[AAAI paper](https://cdn.aaai.org/AAAI/1996/AAAI96-143.pdf).

**Direct lineage:** QDOCS-style behavioral modes, consistency checking, and
minimal qualitative diagnoses in `qrlib.diagnosis`.

### BrajnikClancy1998

Giorgio Brajnik and Daniel J. Clancy, “Focusing Qualitative Simulation Using
Temporal Logic: Theoretical Foundations,” *Annals of Mathematics and
Artificial Intelligence* 22(1–2), 59–86 (1998).
[doi:10.1023/A:1018990024350](https://doi.org/10.1023/A:1018990024350).

**Direct lineage:** TeQSIM-style temporal constraints and guided simulation
in `qrlib.guide`.

### ShultsKuipers1997

Benjamin Shults and Benjamin J. Kuipers, “Proving Properties of Continuous
Systems: Qualitative Simulation and Temporal Logic,” *Artificial
Intelligence* 92(1–2), 91–129 (1997).
[doi:10.1016/S0004-3702(96)00050-1](https://doi.org/10.1016/S0004-3702(96)00050-1).

**Direct lineage:** the sound interpretation of universal temporal
conclusions over a QSIM behavior graph and the standalone temporal
classification API.

## Structural and comparative analysis

### IwasakiSimon1986

Yumi Iwasaki and Herbert A. Simon, “Causality in Device Behavior,”
*Artificial Intelligence* 29(1), 3–32 (1986).
[doi:10.1016/0004-3702(86)90089-5](https://doi.org/10.1016/0004-3702(86)90089-5).

**Direct lineage:** equation-based causal ordering and integral causality in
`qrlib.analysis.causal`.

### deKleerBrown1986

Johan de Kleer and John Seely Brown, “Theories of Causal Ordering,”
*Artificial Intelligence* 29(1), 33–61 (1986).
[doi:10.1016/0004-3702(86)90090-1](https://doi.org/10.1016/0004-3702(86)90090-1).

**Background:** the relation between equation-based causal ordering,
component topology, feedback, and constraint propagation.

### ChiuKuipers1992

Charles Chiu and Benjamin J. Kuipers, “Comparative Analysis and Qualitative
Integral Representations,” in *Recent Advances in Qualitative Physics*, MIT
Press (1992).
[Author bibliography](https://web.eecs.umich.edu/~kuipers/research/qsim/papers.html).

**Direct lineage:** qualitative comparative analysis; qrlib implements the
narrower equilibrium comparative-statics case in `qrlib.analysis.compare`.

### Harary1953

Frank Harary, “On the Notion of Balance of a Signed Graph,” *Michigan
Mathematical Journal* 2, 143–146 (1953).
[doi:10.1307/mmj/1028989917](https://doi.org/10.1307/mmj/1028989917).

**Direct lineage:** the positive-cycle characterization used by
`qrlib.analysis.monotonicity` to certify signed-graph consistency. This is a
graph-theoretic certificate, not a proof that an external vector field is a
monotone dynamical system.

## Global behavior filters

### Raiman1986

Olivier Raiman, “Order of Magnitude Reasoning,” *Proceedings of the Fifth
National Conference on Artificial Intelligence*, 100–104 (1986).
[AAAI paper](https://cdn.aaai.org/AAAI/1986/AAAI86-016.pdf).

**Direct lineage:** FOG’s negligible relation, implemented in its sound
instantaneous form as `qrlib.Negligible`.

### LeeKuipers1988

Wood Wai Lee and Benjamin J. Kuipers, “Non-Intersection of Trajectories in
Qualitative Phase Space: A Global Constraint for Qualitative Simulation,”
*Proceedings of the Seventh National Conference on Artificial Intelligence*,
286–290 (1988).
[AAAI paper](https://cdn.aaai.org/AAAI/1988/AAAI88-051.pdf).

**Direct lineage:** the path-dependent phase-plane non-intersection filter in
`qrlib.engines.phase`.

### LeeKuipers1993

Wood Wai Lee and Benjamin J. Kuipers, “A Qualitative Method to Construct
Phase Portraits,” *Proceedings of the Eleventh National Conference on
Artificial Intelligence*, 614–619 (1993).
[AAAI paper](https://cdn.aaai.org/AAAI/1993/AAAI93-092.pdf).

**Background:** QPORTRAIT and qualitative phase-portrait construction; the
full method is not implemented.

### FoucheKuipers1992

Pierre Fouché and Benjamin J. Kuipers, “Reasoning about Energy in Qualitative
Simulation,” *IEEE Transactions on Systems, Man, and Cybernetics* 22(1),
47–63 (1992).
[doi:10.1109/21.141310](https://doi.org/10.1109/21.141310).

**Direct lineage:** energy-based pruning of spurious oscillatory behaviors in
`qrlib.EnergyFilter`.

## Model induction

### RichardsKraanKuipers1992

Bradley L. Richards, Ina Kraan, and Benjamin J. Kuipers, “Automatic
Abduction of Qualitative Models,” *Proceedings of the Fifth International
Workshop on Qualitative Reasoning about Physical Systems*, 295–301 (1992).
[UT Austin record](https://www.cs.utexas.edu/~ai-lab/pub-view.php?PubID=51381).

**Direct lineage:** the MISQ/QDE-abduction lineage behind `qrlib.induce`.
qrlib uses its own parsimony-ranked, data-consistency-checked procedure
rather than claiming a reproduction of MISQ.

### RamachandranMooneyKuipers1994

Sowmya Ramachandran, Raymond J. Mooney, and Benjamin J. Kuipers, “Learning
Qualitative Models for Systems with Multiple Operating Regions,”
*Proceedings of the Eighth International Workshop on Qualitative Reasoning
about Physical Systems* (1994).
[UT Austin publication list](https://www.cs.utexas.edu/~ml/publications/area/120/qualitative_modeling_and_diagnosis).

**Background:** learning region-dependent qualitative models.

## Piecewise-affine and hybrid background

These works support the separate technique note
[`piecewise-affine.md`](piecewise-affine.md). They are **adjacent work**:
qrlib does not claim a general Filippov or hybrid-systems solver.

- **Filippov1988** — Aleksei F. Filippov, *Differential Equations with
  Discontinuous Righthand Sides*, Kluwer (1988).
  [doi:10.1007/978-94-015-7793-9](https://doi.org/10.1007/978-94-015-7793-9).
- **Sontag1981** — Eduardo D. Sontag, “Nonlinear Regulation: The Piecewise
  Linear Approach,” *IEEE Transactions on Automatic Control* 26(2), 346–358
  (1981).
  [doi:10.1109/TAC.1981.1102596](https://doi.org/10.1109/TAC.1981.1102596).
- **BemporadMorari1999** — Alberto Bemporad and Manfred Morari, “Control of
  Systems Integrating Logic, Dynamics, and Constraints,” *Automatica* 35(3),
  407–427 (1999).
  [doi:10.1016/S0005-1098(98)00178-2](https://doi.org/10.1016/S0005-1098(98)00178-2).
- **JohanssonRantzer1998** — Mikael Johansson and Anders Rantzer,
  “Computation of Piecewise Quadratic Lyapunov Functions for Hybrid Systems,”
  *IEEE Transactions on Automatic Control* 43(4), 555–559 (1998).
  [doi:10.1109/9.664157](https://doi.org/10.1109/9.664157).

## Related software

### UTAustinQSIM

The University of Texas at Austin Qualitative Reasoning Group’s original
QSIM distributions include the most complete Lisp implementation and the CQ
C++ core implementation.
[Software overview](https://web.eecs.umich.edu/~kuipers/research/qsim/qsim-overview.html).

**Related software:** historically authoritative QSIM implementation. Its
distribution terms and legacy environments differ from qrlib’s planned
open-source, pip-installable Python/PyTorch delivery.

### BredewegEtAl2009

Bert Bredeweg, Floris Linnebank, Anders Bouwer, and Jochem Liem,
“Garp3—Workbench for Qualitative Modelling and Simulation,” *Ecological
Informatics* 4(5–6), 263–281 (2009).
[doi:10.1016/j.ecoinf.2009.09.009](https://doi.org/10.1016/j.ecoinf.2009.09.009).

**Related software:** a graphical, process-centered qualitative-modeling
workbench. qrlib instead emphasizes an embeddable Python API, QSIM behavior
coverage, numeric-system interchange, and tensor workloads.

### Chen2025

Benedict Chen, `qualitative-reasoning` 1.4.0 (2025).
[PyPI project](https://pypi.org/project/qualitative-reasoning/).

**Related software:** a Python package centered on qualitative process and
causal reasoning. It is distributed under a custom non-commercial license;
qrlib has a different QSIM-centered semantic and interoperability scope.

### PaszkeEtAl2019

Adam Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep
Learning Library,” *Advances in Neural Information Processing Systems 32*,
8024–8035 (2019).
[Proceedings record](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).

**Software dependency:** tensor encoding, batched filtering, interval
propagation, differentiable losses, and CUDA execution.
