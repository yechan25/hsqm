# Previous adaptive runtime, selectable four-character experiments

User requested return to the adaptive version immediately before guarded release.
The four repair/extractor/fit runtime sources are embedded byte-for-byte from
commit `d74e26e19d1e13360fdecde5d9d80d889344d990`. SHA256 values are checked in
the notebook. Main is not reset and other work is preserved. There is no release
rejection, consensus-mask policy, selected.png, or original-output substitution.
Raw adaptive behavior, including its limitations, is deliberately reproduced.

Run `notebooks/selectable_adaptive_sweep.ipynb` on L4, cells 1–5:

1. Install dependencies and write an isolated runtime with the frozen core.
2. Find the exact existing checkpoint by SHA256 in shared/My Drive. Restore the
   reference catalog using the checkpoint folder's run_config.json,
   selected_paths.csv, conversion_report.json and original reference archives.
   Only reference/target-stroke archives are needed; no training is performed.
   Existing usable manifest can also be supplied. A four-glyph debug ZIP alone
   is not sufficient to choose arbitrary other glyphs. Missing archive metadata
   must be supplied through MANIFEST/DATA_ROOT/CSV_PATH explicitly.
3. Browse a paginated image gallery, select four ordered row IDs with dropdowns,
   and inspect the 2x2 preview. For multiple groups, register each using the add
   button; otherwise cell 4 uses the currently selected four. Numeric file IDs
   are not guessed to be Hangul character names. Unknown labels stay unknown.
4. Generate one whole four-glyph sheet per selected group with the original
   condition: seed 100, strength .65. No strength/seed sweep or control repair is
   run by default. To try a different seed/strength, change the single values in
   cell 4 and rerun generation, then cell 5. This is reference-initialized img2img,
   not prompt-only generation.
   Safety-filtered, missing, displaced or very damaged outputs are not removed.
5. For every case: use the extractor, fit canonical strokes using original hints,
   construct its candidate, then run paired re-denoising/adaptive repair with the
   frozen historical runtime. Defaults remain 60 nominal steps, strength .25,
   seed 2026, four inner iterations, .012 step ratio, refresh every three steps,
   margin ratio .20, restoration off. All groups use the same repair seed.

The whole sheet is generated jointly, with fixed slot crops. Each case gets its
own fitted strokes; no previous case's candidate/ownership is reused. Candidate
RGB is not the repair objective. Checkpoint hash, original selected row IDs,
generation and repair parameters, model revision and frozen runtime hashes are
recorded. Re-running cell 5 creates a new attempt subfolder while reusing the
same generated inputs. It does not silently overwrite prior repair outputs.

REPAIR_CASE_IDS=None processes the one generated case per selected character group.
Cell 5 checks the recorded generation conditions; changing seed/strength cannot
silently reuse an old generated image. Repair noise remains seed 2026, strength
.25, separately from stage-1 generation seed 100/strength .65.
An explicit ID subset is allowed but is user selection and not automatic success
filtering. Failed fits/repairs remain in all_results.json and the comparison grid.
Partial generation failures have generation_failure.json and retain pages already
saved. Final ZIP includes the chosen canonical stroke bundle, all source images,
fit diagnostics, per-case traces/masks/intermediates, comparisons and failures.

This is a robustness inspection across selected references, noise seeds and
corruption strengths. Training-catalog selection is not held-out generalization
evidence. The known normal-stroke erosion of this adaptive algorithm is not claimed
fixed. CPU tests cover selection ordering, all combinations, independent starting
images, failure retention, per-case fitting/repair and exact historical sources.
Actual diffusion quality and total runtime require Colab GPU execution.
