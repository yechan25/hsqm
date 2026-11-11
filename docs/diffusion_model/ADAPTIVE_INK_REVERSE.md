# Per-stroke adaptive ink guidance

Run `notebooks/adaptive_ink_diffusion.ipynb` cells 1–4 on L4. This retains local
erase/optional restore pressures, without candidate RGB matching, and adds live
extraction every three additional denoising steps. No model is trained.

## Inputs and identity

The original fit-sweep ZIP is required: four canonical reference/stroke arrays,
the chosen case's `warped_slot_*.npz`, original generated page, candidate and
checkpoint hash. The old fitted/ink reverse ZIP alone does not contain the stroke
bundle or extractor. Colab supports Drive mount/search and manual uploads. A
checkpoint is accepted only if its full SHA256 matches `experiment_info.json`;
the loader then requires `reference_v2_topology_loss_v1` and strict state loading.
The notebook embeds runtime source and uses the existing Swin reference extractor.
Fresh pretrained Swin weights are not downloaded for checkpoint initialization.

Canonical channel k always names the same reference stroke. The original fitted
paths, source main-path protection and calibrated widths stay fixed. Current flow
does not re-fit the guide, because doing so could move it towards new defects.
The model re-estimates *soft membership*, not proven topological identity.

## Bands and protection

For each stroke, skeletonize initial fitted support. Estimate its half-width by
the median distance-transform radius on the skeleton. Source foreground/ownership
samples on this anchored path refine the width, bounded to .75–1.25 times the
initial estimate. Margin defaults to .20 times full observed width, clamped to
.6–1.8 model pixels. Width/margin calibration is frozen, so an extra branch cannot
keep enlarging its own allowance. Tables record model and sheet pixel units.
Local radius comes from the nearest initial skeleton sample.

An original extracted principal path also protects source body pixels when its
coverage, length and proximity pass heuristic checks. Its radius is capped to
1.1 calibrated half-width to avoid protecting an entire side branch. Otherwise
protection falls back to the anchored path. Each stroke records the fallback.
This protects original straight/corner geometry rather than copying smooth
candidate outlines. Initial source-covered centerlines receive an ink-floor
penalty, averaged per stroke. This is a path-coverage guard, not a proof of
connectivity or a strict healthy-stroke quality constraint.

## Refresh and sampling

The frozen extractor runs on decoded clean estimates at scheduled guided steps,
never on raw noisy states. New detached softmax ownership is blended .35 with
previous ownership. Deletion uses membership-weighted distance outside each
stroke's radius plus margin. Any legitimate stroke corridor or trusted original
body protects a pixel even if ownership assigns it to a different stroke near
an intersection. Current foreground seeds are grown one model pixel and weighted
by off-path responsibility. Refresh can expand deletion to remaining/new errors.
New-region seeds require ink > .12 to avoid expanding on faint decoder noise.
Previous deletion weights are retained, so partial erasure cannot make pressure
disappear; newly found pressure is blended at .5. This monotone retention may
also retain a false positive: inspect masks before interpreting results.

Optional restoration marks absent anchored centerline pixels, not a full target
silhouette. It is off by default; source-covered path protection stays on. Masks
are frozen within each gradient/backtracking loop. Three-step refresh, stable
channel order, EMA membership and fixed geometry reduce feedback oscillation.
Gradients go through VAE RGB/ink losses, not extractor ownership.

The existing paired DDIM runner shares initial encoding/noise, prompt, seed and
15 actual steps (60 nominal, strength .25) between both arms. Local gradient
support, RMS step budgets, backtracking, L1/tail deletion pressure and soft source
preservation remain. Unconstrained denoising can still change normal regions.
No RGB candidate copying or final pixel compositing is used.

## Inspection and limits

Inspect `comparison.png`, `adaptive_trajectory.png`, initial and refreshed erase
maps, `ownership_step_*.npz`, and per-stroke history in `report.json`. All final
diagnostic metrics use the *same initial* masks for both arms and input. Dynamic
optimization losses across different refreshes must not be compared as a fixed
quality score. Changes in normal stroke RGB and trusted-path deficits are logged
separately. The report retains model revision and exact extractor hash.

Seven adaptive tests plus prior runner/objective regression tests (17 total) cover differentiated
bands, fixed identities/geometry, live ownership refresh, residual-mask retention,
source-body guards, common evaluation and sampler refresh hooks. A local check
reads all 19 stroke conditions from the supplied sweep ZIP; archived probabilities
can inspect initial masks but cannot validate live re-extraction. Actual L4 output
quality and false-positive behavior remain to be evaluated in Colab.
