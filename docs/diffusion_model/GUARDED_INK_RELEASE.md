# Guarded ink release

This is an engineering stabilization of the reference-assisted four-glyph repair
experiment, not a new trained extractor or a universally validated final model.
Run `notebooks/guarded_ink_release.ipynb` cells 1–4 on L4. Use the original sweep
ZIP or the adaptive-result ZIP containing `stroke_bundle/`. The latter is read
from its original `input.png`, not its previously damaged guided output. Drive
checkpoint search, manual upload, exact recorded SHA256 verification and strict
state loading remain. Default parameters are fixed; restoration is off.

## Guidance and proposals

The successful fixed sparse mask is the starting constraint. Candidate RGB is
never an image target. Per-stroke geometry/width and periodic detached softmax
extraction remain as observers. New deletion regions require two consecutive
refreshes. If evidence disappears, new pressure decays and can be withdrawn;
healthy-region masks never shrink to accommodate a proposed deletion. Initial
fixed deletion support remains to suppress faint residuals.

For a straight canonical prototype, an additional *proposal* detects side bulges
within the middle 80% of the initial fitted path. Radius is estimated from a
lower quartile of original owned-body thickness on the main path, bounded relative
to calibrated width. Caps and curved prototypes are excluded from this rule.
These proposals also require repeated soft ownership/ink evidence. This is a
heuristic for bulges, not a general branch classifier; ambiguous fonts can fail.

Normal source ink is partitioned into soft ownership groups, frozen from the
original input. Fixed edit/proposal regions plus a three-pixel repair collar are
excluded so an intended junction repair is not automatically rejected. Other
body/edge/cap pixels stay protected. Source per-stroke mean MSE and worst local
MSE receive additional differentiable preservation penalties, avoiding dilution
by white background. This is original-source preservation, not candidate matching.

## Acceptance and final selection

The unguided arm records healthy-group diagnostics at every clean prediction.
At the same timestep, a proposed clean-latent update must lower the optimization
loss AND pass every healthy group's three guards relative to matched baseline:

- mean RGB MSE: baseline + .001;
- worst 5x5 local RGB MSE: baseline + .025;
- mean positive ink drop: baseline + .02.

If a current anchor is already outside a limit, a proposal may not worsen that
diagnostic. Rejected proposals are backtracked and counted. These limits constrain
guidance corrections; they cannot prevent an ordinary U-Net step from drifting.
The terminal decoded output is therefore checked again against terminal baseline.

Final acceptance requires all terminal healthy-group guards plus improvement over
both input and unguided output on fixed initial deletion/straight-bulge proposal
regions and optional missing-centerline deficits. If checks fail, selected.png
is the original input; otherwise it is the entire guided image. No per-pixel
compositing, candidate copying or hidden success-case filtering is performed.
Raw baseline/guided images, masks, traces, and explicit selection reasons remain
in the ZIP. `original_retained` means abstention, not successful repair.

The default comparison has four panels: original, re-denoise only, raw guarded
reverse, selected output. Inspect `selection` and `rejected_guard_attempts` in
report.json. Fixed base diagnostics are comparable across arms; dynamic losses
at different refreshes are not a calibrated quality score. Final selection error
also includes fixed straight-bulge proposals and is reported separately.

## Validation and limits

Twenty-six CPU tests exercise consensus/revocation, static foundation, coupled decoder
leakage and backtracking rejection, whole-output fallback, clean-input abstention,
matched-baseline sampler hooks, ZIP bundles and notebook-source consistency,
alongside prior per-stroke/objective/paired DDIM tests. An offline replay using
archived ownership from the supplied GPU run rejects the visibly damaged adaptive
output under terminal normal-stroke guards. This is policy verification on saved
images, not a fresh GPU run of this guidance.

No finite collection of internal guards proves visual correctness. Missed errors,
incorrect fixed masks, unprotected proposal areas, crop layout changes or severe
generation failures can remain. Some inputs will intentionally be retained.
Actual output quality and acceptance rate require Colab GPU evaluation. The
method remains reference-assisted and does not establish prompt-only Hangul
generation improvement or calibrated HSQM-score optimization.
