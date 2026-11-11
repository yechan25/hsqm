# Restore → settle → erase, with coverage checks

Run `notebooks/cyclic_stroke_repair.ipynb` cells 1–5 on L4. Four-glyph selection,
exact checkpoint SHA256, training-reference recovery, generation seed 100 and
strength .65 remain. Only one generated sheet per selected group is repaired.
Other seeds require generation again. This is reference-assisted img2img, not
prompt-only generation. The old selectable adaptive notebook remains available.

## Expected paths and missing ink

Each case is initially extracted and fitted as before. Fit records now carry
observed masks and fit warnings. Reliable visible canonical/fitted stroke
centroids estimate a bounded glyph affine (or bounded translation). Missing or
unreliable strokes use the transformed original prototypes, not a path fitted
to absent ink. If no reliable anchors exist, canonical placement is retained and
explicitly reported as a fallback. Position accuracy cannot be guaranteed.

Skeleton coverage below .2 marks a potential whole-stroke absence; uncovered
segments mark partial loss. Unreliable fit warnings or coverage < .35 choose the
aligned prototype rather than the observed fitted path. Local thickness is
estimated from visible support, borrowing a median from other strokes when
needed. Missing skeleton segments are widened to a small renderable band, not a
full RGB candidate. These are geometric heuristics; wrong alignment can invent
an incorrect restoration. Crossing ink can also make absent paths appear covered.
For present/partial strokes, endpoint neighborhoods and isolated tiny gaps are
excluded to preserve original caps/serifs rather than forcing prototype tips.

## Cycle controller

1. Restore: decode predicted clean x0. Within missing bands, explicitly raise
   luminance-derived ink to at least .7; encode the corrected clean image using
   frozen fp32 VAE posterior mode. Mix the encoded latent into original x0 only
   on affected latent cells plus one-cell context. Repeat at most three VAE
   projections per timestep. Decode again to measure actual retained paths.
2. Coverage verification: every expected path needs >= .85 of its pixels above
   ink .5. If this fails, continue restoration and keep deletion off. Ordinary
   diffusion steps still advance, but the controller cannot enter erasure.
3. Settle: after successful restoration, allow two diffusion steps with no
   deletion and only restoration/path-presence/source-preservation losses.
4. Erase: re-extract frozen-model soft membership on the restored clean estimate.
   Rebuild deletion from current ink outside complete expected stroke corridors.
   No pre-restoration deletion mask is accumulated across phase changes. Protect
   restoration bands and required paths. Refresh every two erasure steps.
5. Recheck: if path coverage drops, clear deletion and restore again. At most two
   restoration episodes are allowed; unresolved later loss blocks deletion and
   is reported rather than called a successful repair.

Loss gradients go through VAE decoding to clean latent, not through extractor
membership. The explicit structural insertion is intentional and disclosed; this
is not the previous claim of entirely loss-only guidance. Candidate full RGB is
never a target and is not pasted over the final output. Corrected x0 is transferred
to the DDIM previous state using sqrt(alpha_prev) with predicted epsilon held
fixed, preserving that step's noise scale. Original and cyclic arms share exact
initial encoding/noise, prompt, seed and timesteps. Default 60 nominal steps with
strength .25 give 15 actual steps; repair seed remains 2026. No extra unrecorded
re-noising pass or random seed sweep is added.

## Outputs and interpretation

Inspect input, re-denoise-only and cyclic output; `cycle_stages.png` shows phase
changes. The cycle folder includes every clean estimate, inserted estimates,
restore/erase masks, initial missing band, full trace and report. Coverage is
per-stroke and normal-region MSE is reported separately. Restoration status is
not image correctness: `restoration_incomplete` can occur even when some paths
were recovered. Terminal output is decoded from the actual diffusion state;
there is no final cleanup, candidate substitution or hidden original fallback.

Fixed latent-cell insertion has VAE context spill and may change nearby normal
contours. Core coverage does not verify thickness, style, exact connectivity or
correct character identity. Severe generation failures/poor alignment remain
outside any performance guarantee. Phase budget can run out before cleanup.
Sixteen CPU tests (including selection/generation regressions) check full/partial missing detection, local minimum-ink insertion,
VAE round-trip gating, no deletion before restoration, reversion/cycle bounds,
restored-path protection and paired sampler state transfer. Actual L4 results
must be inspected before claiming generalization or a completed final model.
