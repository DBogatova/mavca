# NMDA-spike vs bAP "vertical-streak" — amplitude/morphology testing (results)

Campaign: 2026-06. Code + figures + tables: `code/Testing/` (see its README for usage).
Status key matches experiment_log.md: ROBUST · NULL · ARTIFACT · EXPLORATORY.

## Hypothesis
NMDA-spike Ca²⁺ transients (localized to a branch) have **lower amplitude** than bAP
"vertical-streak" transients (signal extended along the apical trunk / cortical depth).
Operationalized "bAP/streak" = spatially **extended** (whole-tree / trunk / both halves);
"NMDA" = **localized** (single branch / one segment). Tested whether amplitude separates them.

## BOTTOM LINE
**Amplitude is NOT a robust discriminator of NMDA-like vs bAP-like activity at 5 Hz.** The
hypothesis was probed four independent ways and only appears under non-robust conditions.
Kinetics don't separate them either. The only real handle is **behavior coupling (0508)**.
The trunk/branch morphology classifier is a *structural* tool, not a functional NMDA/bAP call.

## Tests & outcomes
| # | test | scale | result | status |
|---|---|---|---|---|
| 3 | static mask morphology → peak ΔF/F | per mask | length ρ≈+0.09, verticality≈0, linearity+0.12; vertical-streak vs localized masks p=0.48 | **NULL** |
| 1 | event amplitude vs co-activation extent (# co-active *masks*) | per event | isolated (NMDA-like) z=−0.35 vs whole-tree (bAP-like) z=+0.03, pooled MWU **p=3e-6**; amp~participation +ve in 10/11 FOVs (median ρ+0.19); strong only where whole-tree events exist (0416_r123 +0.37, 0508 +0.27) | **EXPLORATORY** (FOV-specific) |
| 1c | coupling control: within-mask (does a mask fire bigger when others co-fire?) | per mask | whole-tree>isolated 7/10 masks, +2%, Wilcoxon p=0.024; demeaned p=0.003 — modest, underpowered (57 isolated vs 273 whole-tree events); residual-field caveat | **EXPLORATORY** |
| 2a | within-dendrite split, naïve top/bottom, **noise-threshold** classifier | per event/segment | extended>localized: 0508 1.23× (p=2e-3), 0512_r9 1.45× (3e-4), 0416_r1 0.96× (n.s.); pooled p=9e-4 | **ARTIFACT** (see 2b) |
| 2b | within-dendrite split, branch/bend seg, **SNR-robust relative-amplitude** classifier | per event/segment | effect **vanishes**: pooled extended vs localized p=0.70; per-FOV 0.73–1.19×. REL_FRAC sweep: significant only at extreme cutoff ≤0.33. The 2a effect was detection-threshold coupling (bigger events cross threshold in more segments) | **NULL / ARTIFACT** |
| – | event **kinetics** (FWHM, decay, rise, area) isolated vs whole-tree | per event | no difference (FWHM 0.60 s both, p=0.46; decay 0.40 s, p=0.41; area p=0.065). NMDA plateau not resolvable at 5 Hz | **NULL** |
| – | active-region **morphology** (vertical streak vs tilted) + amplitude | per event | morphology gives 2 real shape populations (Ca: 20° from vertical, 58 µm extent; NMDA: 46°, 46 µm) but **amplitude is flat** (pooled p=0.35; per-FOV 0.92–1.10×) | **NULL** (amplitude) |
| – | whole-mask trunk/branch class + **per-segment amplitude** (6 FOVs) | per segment | trunk>branch only in **0508/run5** (3.55 vs 3.03 %, p=0.02) and **0320/run3** (2.48 vs 2.30, p=0.006); null in 0416_r567, 0512_r5, 0512_r9, 0320_r1. Small (1.08–1.17×) | **EXPLORATORY** (FOV-specific) |

## Methods findings (reusable; add to constraints)
- **Illumination gradient** (dim-left → bright-right, ~1.1–1.23× across X) is removed by per-voxel ΔF/F.
  Not a confound: trunk/branch event classes are matched on X-centroid (p=0.99) and baseline brightness (p=0.32).
- **Pial surface is tilted** (X-tilt up to ~12°; surface Y spans ~55 µm across the FOV in 0508).
  → **Depth MUST be slope-corrected**; **orientation must NOT** — dendrite long-axes align about equally to
  image-Y vs the fitted surface-normal (differ 1–4°, split across FOVs), i.e. the tilt is small vs the
  ~25–34° spread of dendrite orientations. (`fov_geometry_diagnostics.py`, `diag_orientation.py`.)
- **"Silent" masks are core-shell cancellation, not inactivity**: every silent mask had an active core but a
  shell that brightened identically (core↔shell r≈0.96–1.0) → core−shell ≈ flat. They are field/shared-signal
  dendrites. (`diag_silent_masks.py`, `diag_silent_extract.py`.)
- **Within-segment SNR coupling**: splitting a dendrite into small segments makes the activity call threshold-
  dependent → inflated the 2a effect. Use relative-amplitude (not per-segment MAD) classification.

## Morphology classifier (trunk / branch / combined) — structural only
`code/Testing/nmda_bap_gallery2.py::decompose()`. Skeleton → longest-path backbone → RDP straight runs;
longest straight run (+collinear) = trunk, divergent/off-backbone = candidate branch; a branch is kept only if
it diverges ≥ BRANCH_ANGLE and is ≥ BRANCH_MIN_FRAC of the mask, else folded into trunk; then trunk must be
≥ TRUNK_FRAC and straight (≥ LIN_TRUNK) or the whole mask = branch.
Recommended params: **LIN_TRUNK=0.80, BRANCH_ANGLE=45°, BRANCH_MIN_FRAC=0.10, TRUNK_FRAC=0.30, RDP_EPS=8, TRUNK_MIN=35 µm**.
Matches ~5/8 hand-checked cases. **Residual limit is fundamental**: trunk-vs-fork is topological (branch count),
which PCA/elongation cannot see — a straight trunk and a fork can have near-identical linearity/aspect.
Next step if needed: skeleton endpoint-count (topological) discriminator. Optional `DEPTH_TRUNK_Y` forces deep
voxels to trunk (tried Y=110 on 0508; it then tracks the depth→amplitude gradient, not morphology).

## Recommendation for the paper
- Do **not** claim a within-dendrite (or general) NMDA<bAP amplitude difference — it is not robust.
- If used at all, frame at the **population scale** (whole-tree vs isolated, Test 1) with the FOV-specificity
  and residual-field caveat, and/or lean on **behavior coupling (0508)** which is the defensible discriminator.
- The trunk/branch classifier is publishable as **anatomy/structure**, decoupled from any amplitude claim.

## Files
Figures `code/Testing/figures/`, tables `code/Testing/data/`, scripts `code/Testing/` (see README for the
per-script table and run order).
