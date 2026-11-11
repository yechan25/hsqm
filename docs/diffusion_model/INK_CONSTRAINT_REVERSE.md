# Sparse ink constraints in reverse diffusion

Run `notebooks/ink_constraint_diffusion.ipynb` cells 1–3 on Colab L4. The standalone
notebook accepts the existing sweep ZIP or a fitted-reverse ZIP. The latter uses
`input.png` and `fitted_candidate.png`, not the already guided diffusion result.
No Drive mount or extractor checkpoint is needed. Inspect the colored masks in
cell 2 before running diffusion. The masks are fixed for the whole pass.

The old candidate RGB objective remains available for reproduction. This notebook
selects `guidance_mode='ink_constraints'` instead. The candidate provides binary
support and optional centerline information only, not RGB reconstruction targets:

- Threshold source and candidate luminance ink at .35. Source ink more than five
  image pixels from candidate support seeds deletion. Grow seeds two pixels and
  taper pressure near candidate support. This tolerance excludes ordinary width
  and boundary differences, but can miss short extras or misclassify displaced
  normal endpoints. It is a heuristic, not a learned calibrated error detector.
- Default restoration is OFF. If enabled, missing candidate skeleton pixels more
  than five pixels from source support provide sparse addition constraints. No
  candidate width or exact silhouette is prescribed. Crossing constraints are
  excluded. A very incorrect candidate can still provide incorrect instructions.
- Deletion penalizes positive ink above .015 with L1 plus .25 times the mean of
  the darkest 10% of deletion-mask pixels. This keeps pressure on faint residue.
- Restoration penalizes ink below .7 on absent centerline pixels, with no pressure
  for further darkening once sufficient ink is present.
- RGB source-preservation MSE outside the edit mask and a two-pixel buffer has
  weight .5. It references the original input, never candidate RGB. All region
  losses are area-normalized. This is soft preservation, not a guarantee that
  straight edges or healthy strokes remain pixel-identical.

Clean-latent gradient updates are limited to latent cells intersecting the edit
mask with a one-cell context halo. DDIM still predicts the entire image. The
same noisy starting latent is shared by guided and unguided runs. Default 60
nominal steps and strength .25 yield 15 actual additional denoising steps; clean
updates use RMS ratio .012, four inner iterations, eight in the final step.
No candidate pixel copying, hard output cleanup, or latent clamping to reference
pixels occurs. The ordinary denoiser, VAE and soft preservation can still change
normal regions. In particular a weakly constrained restoration need not recover
the intended thickness; inspect results rather than relying on loss alone.

Inspect comparison and trajectory PNGs, erase/restore/preserve masks, and the
per-step loss components in `report.json`. Evaluate removal, restored stroke
connectivity if enabled, and changes to healthy straight segments separately.
The losses are not HSQM scores or validated quality metrics. There is no automatic
mask refresh or multi-pass deletion/addition feedback loop in this experiment.
CPU tests cover selective removal, restoration saturation, source preservation,
local updates, tolerance and candidate-shade invariance, plus paired DDIM plumbing.
Actual L4 image quality must be checked in Colab.
