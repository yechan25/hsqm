# Repository instructions

- Keep stroke extraction model, training, inference, postprocessing and supporting runtime code in `stroke_model/`.
- Keep font experiments in `font_test/`; use separate top-level directories for other research components.
- Keep model documentation in `docs/stroke_model/`, execution notebooks in `notebooks/`, and verification in `tests/`.
- Update relevant documentation when changing architecture, execution steps or folder paths.
- Do not commit datasets, checkpoints, generated bulk outputs, credentials or local dependency directories.
- Check Git status and preserve existing user changes. Run checks appropriate to the change.
- Treat GitHub commit history as the source of model update history. When asked to upload changes, verify the remote commit after pushing; do not report local-only commits as uploaded.
- Distinguish implemented architecture from trained checkpoints and validated performance.
