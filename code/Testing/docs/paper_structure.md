# Paper structure — L5 apical dendritic calcium (calcium-only)

Framing: **L5 apical dendritic calcium activity is high-dimensional, sparse, mostly local, and
compartmentalized across mice, with rare state-dependent transitions into coordinated global
recruitment.** No propagation / bAP / dendritic-spike / apical-nexus / branch-order / topology claims.

---

## A. Final Results summary paragraph

> Across 8 fields of view from 4 mice and two acquisition modes, L5 apical dendritic calcium activity
> was high-dimensional and weakly correlated among segmented dendritic compartments, indicating that
> dendritic segments behave as largely independent functional units. This independence was not
> attributable to low signal-to-noise, periods of inactivity, temporal drift, spatial separation, or
> acquisition mode, and stood in sharp contrast to a contamination-control field in which a single
> field-wide mode dominated (pairwise r ≈ 0.9; first principal component ≈ 90% of variance).
> Mask-level analysis showed that activity was not uniformly distributed: a minority of compartments
> accounted for a disproportionate share of events (event-count Gini 0.41–0.59; the most active 10% of
> masks carried 26–47% of events), forming moderately stable functional hotspots within and across
> sessions. Event-footprint analysis revealed that most events recruited only one or two compartments,
> whereas rare multi-compartment motifs recurred above shuffled-mask nulls in every field
> (p = 0.003), indicating a modest but reproducible co-activation structure superimposed on a
> predominantly local regime. In a behavior-rich field, locomotion transiently reorganized this
> grammar: event rate rose steeply and the balance shifted from sparse local activity toward
> coordinated global recruitment around movement onset, with global recruitment frequently preceding
> or coinciding with the detected onset. Together these results support a model in which L5 apical
> dendritic calcium activity is primarily local and compartmentalized, punctuated by rare,
> state-dependent transitions into coordinated global recruitment.

---

## B. Main figures (essential panels) — and what moves to supplement

**Main Fig 1 — Dendritic segments are independent functional units** (from `fig6_independence`)
- ESSENTIAL: (1) effective dimensionality/N per FOV (mouse-colored, contaminated hatched);
  (2) pairwise-r distribution, real vs contaminated; (3) PCA cumulative variance, real vs contaminated;
  (4) effective dimensionality/N grouped by mouse (the cross-mouse anchor).
- → SUPPLEMENT: mean pairwise r per FOV and % |r|>0.3 per FOV (redundant with 1 & 2).

**Main Fig 2 — Activity is sparse and hotspot-concentrated** (from `fig8_hotspots`)
- ESSENTIAL: (1) Lorenz curves of event counts (all FOVs); (2) event-count Gini per FOV;
  (3) event share of top 5/10/20% masks; (4) hotspot rank stability (within-session + across-run).
- → SUPPLEMENT: per-FOV rank curves (redundant with Lorenz); hotspot spatial map (optional, weak).
- Recommended layout: 2×2; Lorenz + Gini on top, top-% share + stability on bottom.

**Main Fig 3 — Events are mostly local but include reproducible multi-mask motifs** (from `fig10_footprints` + new panel)
- ESSENTIAL: (1) footprint-size histogram (mostly 1–2 masks); (2) event-class fractions per FOV
  (local/multi/global); (3) **observed recurrence vs shuffled-mask null per FOV (p = 0.003)** — *this
  panel must be added (see notes)*; (4) example 0508 multi-mask footprint matrix.

**Main Fig 4 — Movement transiently shifts the event grammar (FOV-specific, 0508)** (from `fig11` + `fig1`)
- ESSENTIAL: (1) global Ca aligned to movement onset (event-locked, r=+0.24, p=0.002);
  (2) global-event fraction by behavioral state (quiet→pre→onset→post); (3) local vs global event
  rate by state; (4) **global-event probability in fine time bins relative to onset** — *to be added*,
  to show the pre-onset rise honestly.
- → SUPPLEMENT: raw vs residual per-mask coupling + per-mask null (fig1_controls); coupling-sign map.

---

## C. Supplement figure groups
- **S1 Validation of independence:** high-SNR-mask and high-amplitude-frame metrics; first/second-half
  stability; per-FOV table (from `validation_independence.csv`).
- **S2 Spatial dependence:** pairwise r and coactivation vs Euclidean distance (r≈−0.06/−0.11) and
  vs |Δdepth| (r≈−0.02) (`fig7_spatial`); note family comparison excluded (reconstruction over-merges).
- **S3 Acquisition-mode controls:** independence metrics by mouse/mode (descriptive; mouse↔mode confounded).
- **S4 Functional fingerprints/clusters:** 2-cluster solution + feature means (local- vs
  global-participating), cluster↔FOV AMI=0.30, modest depth difference only (`fig9_funcclusters`).
- **S5 Movement-coupling sign (0508):** pos/neg/uncoupled map + features; emphasize sign is *not*
  run-to-run stable (41% agreement) — not a stable identity.
- **S6 Tested-and-not-supported (negative results):** branch-order null, distance-to-trunk null,
  tortuosity null, family-reconstruction unreliability, and the heterogeneous/condition-dependent depth
  gradient (negative in 5/6 132/139 FOVs, flat/reversed in the two new mice). Including these
  *strengthens* the independence claim by showing structure was sought and not found.

---

## D. Claims

**Headline (one sentence):**
> L5 apical dendritic calcium activity is high-dimensional, sparse, and predominantly local across
> mice, with rare state-dependent transitions into coordinated global recruitment.

**Three major claims (robust):**
1. Segmented L5 apical dendritic compartments are weakly correlated and high-dimensional across mice
   and acquisition modes — they behave as largely independent functional units (validated against SNR,
   amplitude, time, space, and a contamination control).
2. Local events are non-uniformly distributed: a minority of compartments form moderately stable
   functional hotspots (Gini 0.41–0.59; top 10% = 26–47% of events).
3. Events are predominantly local (≤2 compartments), yet rare multi-compartment co-activation motifs
   recur above chance in every field — modest reproducible structure on a local backbone.

**Three minor/cautious claims:**
1. Compartments span a local-preferring → global-participating functional continuum that is mostly
   independent of morphology (clustering is weak-to-moderate; partly FOV-influenced).
2. In a behavior-rich field, locomotion transiently shifts the grammar toward high-rate global
   recruitment; this is FOV/condition-specific, not general.
3. Global recruitment in that field frequently preceded or coincided with detected movement onset
   (descriptive; see §E — not interpreted as preparatory).

**Explicitly NOT claimed:**
- No propagation velocity.
- No backpropagating action potentials (bAPs).
- No dendritic-spike identification.
- No apical-nexus identification.
- No reliable branch-order or distance-to-trunk effect.
- No general depth rule (depth–activity is heterogeneous/condition-dependent).
- No general global movement-activation across all FOVs (it is FOV-specific).

---

## E. Pre-onset global-event finding (careful wording)

> In the behavior-rich 0508 field, the fraction of global (multi-compartment) events was near zero
> during quiet periods, rose markedly in the pre-onset window (~0.48) and remained elevated at detected
> movement onset (~0.34), before returning to near-quiet levels afterward (~0.05). The apparent
> elevation in the pre-onset window indicates that coordinated global recruitment often **preceded or
> coincided with** the accelerometer-defined movement onset. We interpret this cautiously and do not
> claim preparatory or predictive activity. Several non-exclusive factors could contribute: the
> accelerometer threshold may lag an earlier behavioral or neural state transition; sub-threshold body,
> whisker, or eye movements may occur before onset is detected; and the small number of frames in the
> pre-onset window can inflate fractional estimates. The effect is specific to this field and is not
> generalized across the dataset, and the sustained-movement state was underpowered because bouts were
> brief. Resolving the timing will require finer alignment — to the acceleration peak, to alternative
> movement thresholds, and to video/whisker/eye onset — together with event-probability curves in fine
> time bins around onset, and, in future dual-channel recordings, alignment to the cholinergic (ACh)
> signal.

(Avoid: "predictive", "preparatory", "motor planning" unless explicitly flagged speculative.)

---

## F. Title options
1. *Sparse local and rare state-dependent global calcium recruitment in L5 apical dendrites*
2. *Functional compartmentalization of L5 apical dendritic calcium activity across mice*
3. *L5 apical dendritic calcium events are high-dimensional, sparse, and state-dependent*

---

## Notes — two main-figure panels still to generate
- **Fig 3 panel:** observed-vs-shuffled recurrence per FOV (values exist: recur 0.02–0.18 vs ~0,
  p=0.003) — add as a bar/CI panel.
- **Fig 4 panel:** global-event probability in fine time bins relative to movement onset (0508),
  to display the pre-onset rise honestly.
