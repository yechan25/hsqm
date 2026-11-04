# Cross-font Topo testing

Open `notebooks/test_topology_fonts.ipynb`, execute setup and configuration, select
a reference thumbnail and type the matching character, select fonts, then execute
the last cell. Re-run the last cell after changing the selection. Latest V2-based
Topo best.pt is selected by default; an explicit checkpoint path is preferable
for reproducible comparisons. No training or checkpoint modification occurs.

Setup installs Ubuntu `fonts-nanum` and fontTools. Defaults select Gothic/Myeongjo
regular and bold if available. Additional TTF/OTF/TTC files can be supplied via
`EXTRA_FONT_DIR`; collections use face index 0. Fonts are checked for character
coverage with fontTools cmap before rendering, so missing glyph boxes fail rather
than silently entering the model. Font binaries are never included in Git.

Reference geometry and ordered stroke masks always come from the selected CSV
row, never from segmentation of the new font. Numeric image IDs are not converted
to character code points. A single Hangul syllable in the reference filename or
an explicit single-character CSV column may prefill the text, but the user must
verify the visible thumbnail. With opaque filenames, text is supplied manually.
The experiment does not verify semantic agreement through OCR.

Pillow renders the chosen font at 4x resolution, tightly crops its visible glyph,
uniformly scales it into the reference's foreground bounding box, and centers it
there. Aspect ratio is preserved; individual components and stroke widths are not
matched. This whole-glyph placement reduces size/position confounding but is not
a test of native typography placement. `PLACEMENT='centered'` instead fits to the
central 75% of the canvas without reference-dependent placement.

A reference-as-input control precedes the other fonts. Every case saves raw input,
soft probabilities, original-width soft masks, exclusive masks, reference/stroke
arrays, overview and per-stroke figures. A final comparison grid has fonts as
rows and extracted strokes as columns. JSON records font path/hash, placement,
checkpoint, epoch, reference row, and coverage diagnostics; notebook also saves
the repository commit. Save directories are unique beside the checkpoint.

No input stroke annotations are read. The notebook calls
`run_scored_topology_fonts`, which extracts once and passes the saved original
font ink snapshots to the existing `score_topology_results` source scorer.
It displays source scores beside Soft/Exclusive predictions and a font score
table (total plus missing/split/junction/extra caps). `font_scores.csv`, per-stroke
scores, JSON, and a downloadable diagnosis ZIP are saved in a unique score
directory; raw font renders remain in the separate extraction directory.
Numeric scores do not use extractor masks. They are existing uncalibrated
structural indicators, not font segmentation accuracy; style/placement differences
can trigger penalties. Scores/settings/formulas are unchanged. Additional source
registration and measurement take time. `SCORE_CONFIG`/`SOURCE_CONFIG` expose
existing scorer overrides. There are no new-font per-stroke ground truths,
so neither ink coverage nor the visual table is an accuracy measurement. Coverage
is trivially complete for V2 partition output. Fonts are not checked against the
training font list, so this is not a guaranteed held-out-font benchmark. Cleaner
font images do not establish quality on diffusion-generated or handwritten input.
If the same-reference control fails, the issue is broader than cross-font transfer.
If the control succeeds and other fonts fail, domain/geometry transfer merits
attention. Evaluate actual diffusion outputs separately before using scores as
generation guidance.

Rendering helpers live in `font_test/render.py`; model orchestration stays in
`stroke_model/font_topology.py`. Unit tests exercise rendering, glyph rejection,
reference mapping, immutable checkpoints, untouched reference geometry, and
saved predictions using a lightweight mock extractor. The real user's model and
Colab apt/Drive execution have not been run locally.

Sources: [Ubuntu fonts-nanum](https://packages.ubuntu.com/jammy/fonts/fonts-nanum),
[fontTools TTFont](https://fonttools.readthedocs.io/en/latest/ttLib/ttFont.html).
