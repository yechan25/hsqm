# Extractor-fitted candidate → paired reverse diffusion

Run `notebooks/fitted_candidate_diffusion.ipynb` on Colab L4, cells 1–3.
Upload an existing `stroke_reference_fit_sweep` ZIP. No Drive mount or checkpoint
is required: the extractor has already produced the fitted candidate in that ZIP.
The default is `case_003_strength_0p65_seed_100`. Change `CASE_ID` in cell 2 to
inspect other cases, including the clean-input control. All cases remain in the
original archive; this notebook runs one explicitly selected case at a time.

The source is the saved generated page. The guide is its `normal_candidate.png`,
which was constructed using extractor masks and learned registration hints.
Neither the unfitted reference nor `geometry_candidate.png` is used as the target.
This is a reference-assisted controlled experiment, not reference-free correction.

Both arms use exactly the same initial VAE encoding and noisy latent, prompt,
DDIM scheduler, noise seed and number of steps. Default 60 nominal steps and
strength .35 give 21 actual additional denoising steps. These are distinct from
the original img2img generation strength .65. The VAE is frozen and decoded in
fp32; the U-Net runs frozen in fp16 without gradient tracking to fit L4 memory.

The fixed edit mask is the source/candidate RGB difference > .10, dilated by two
pixels. The loss is region-normalized candidate RGB MSE within this mask plus
.25 times region-normalized source RGB MSE outside it. These are candidate
agreement and preservation objectives, not the discrete HSQM score. Both removal
and restoration are supported across the whole sheet. Alignment errors can also
enter the edit mask; inspect `edit_region.png` and the candidate first.

After the first 20% of steps, optimize the predicted clean latent through VAE
decoding, with four gradient iterations (eight on the last step), backtracking
until the same-step loss decreases, and a total per-step RMS correction budget
of .08 times the clean latent RMS. Translate the clean correction into the DDIM
previous state while holding the predicted noise residual fixed. There is no
pixel pasting, threshold cleanup or final candidate replacement. Final outputs
are decoded terminal diffusion latents. Saved intermediate PNGs are predicted
clean images, not noisy states. A decrease within one step does not imply a
monotonic decrease over the trajectory or improved visual correctness.

Inspect `comparison.png`, both trajectory images, `edit_region.png`, and
`report.json`. Compare the guided arm against re-denoising only for unwanted
stroke removal, intact intended strokes, style preservation and background
artifacts. Candidate agreement alone is not accuracy: the candidate is not GT.
`report.json` records the fixed model revision, paired latent hash, parameters,
per-step acceptance counts and losses. Local tests cover removal/restoration,
preservation, bounded updates, archive selection and embedded notebook runtime.
Actual SD1.5/L4 output quality requires a Colab GPU run.
