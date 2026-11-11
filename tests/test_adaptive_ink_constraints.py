import ast
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from diffusion_model.fitted_candidate_diffusion import optimize_clean
from stroke_model.adaptive_ink_constraints import AdaptiveInkConstraints


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.))
        self.calls = 0
    def forward(self, image, reference, strokes):
        self.calls += 1
        # Ownership responds to current image but stroke channels never change.
        logits = 3*strokes+torch.cat([image, -image], 1)
        return dict(probabilities=logits.softmax(1))


def example():
    strokes = np.zeros((2, 32, 32), np.float32)
    strokes[0, 4:28, 8:11] = 1
    strokes[1, 18:25, 18:29] = 1
    record = dict(reference=strokes.max(0)[None], strokes=strokes, fitted=strokes.copy())
    source = torch.ones(1, 3, 128, 128)
    ink = torch.from_numpy(strokes.max(0))[None, None]
    ink = F.interpolate(ink, (64, 64), mode='nearest')
    ink[..., 26:32, 2:18] = 1  # A branch attached to the thin vertical stroke.
    for y, x in [(0, 0), (0, 64), (64, 0), (64, 64)]:
        source[..., y:y+64, x:x+64] = 1-ink
    return source, [record for _ in range(4)]


class AdaptiveTests(unittest.TestCase):
    def test_width_tolerances_differ_and_normal_body_is_protected_from_wrong_ownership(self):
        source, records = example()
        objective = AdaptiveInkConstraints(source, Extractor(), records)
        thin, thick = objective.bands[:2]
        self.assertLess(thin['tolerance_model'], thick['tolerance_model'])
        self.assertGreater(float(objective.erase[0, 0, 28, 5]), 0)
        self.assertEqual(float(objective.erase[0, 0, 20, 18]), 0)
        for r in objective.records:
            r['probabilities'] = r['probabilities'][::-1].copy()
        objective.refresh(source, step=3)
        self.assertEqual(float(objective.erase[0, 0, 20, 18]), 0)

    def test_refresh_runs_extractor_on_new_image_with_fixed_geometry_and_stroke_ids(self):
        source, records = example()
        model = Extractor()
        objective = AdaptiveInkConstraints(source, model, records, refresh_every=3)
        count = model.calls
        before = [r['probabilities'].copy() for r in objective.records]
        geometry = [r['distances'].copy() for r in objective.records]
        self.assertIsNone(objective.maybe_refresh(source*.9, 2))
        objective.maybe_refresh(source*.9, 3)
        self.assertEqual(model.calls, count+4)
        self.assertTrue(any(not np.allclose(p, r['probabilities']) for p, r in zip(before, objective.records)))
        for a, r in zip(geometry, objective.records):np.testing.assert_array_equal(a, r['distances'])
        self.assertEqual([(b['slot'], b['stroke']) for b in objective.bands],
                         [(i, k) for i in range(1, 5) for k in (1, 2)])

    def test_partial_erasure_retains_pressure_and_new_errors_expand_mask(self):
        source, records = example()
        objective = AdaptiveInkConstraints(source, Extractor(), records)
        old = objective.erase.clone()
        changed = source.clone()
        changed[..., 26:32, 2:18] = .99  # Nearly erased original branch.
        changed[..., 2:8, 54:60] = 0  # New detached error, outside all normal bodies.
        objective.refresh(changed, step=3)
        self.assertTrue(bool((objective.erase >= old).all()))
        self.assertGreater(float(objective.erase[..., 2:8, 54:60].sum()), 0)
        self.assertGreater(float(objective.erase[..., 26:32, 2:18].sum()), 0)

    def test_metrics_use_fixed_initial_regions_after_refresh(self):
        source, records = example()
        objective = AdaptiveInkConstraints(source, Extractor(), records)
        probe = source*.95
        before = float(objective.evaluate_fixed(probe)[0])
        changed = source.clone();changed[..., 2:8, 54:60] = 0
        objective.refresh(changed, step=3)
        self.assertAlmostEqual(before, float(objective.evaluate_fixed(probe)[0]), places=7)

    def test_optional_restore_missing_path_without_conflicting_deletion(self):
        source, records = example()
        source[..., 36:50, 36:58] = 1  # Remove the thick stroke from slot 1.
        objective = AdaptiveInkConstraints(source, Extractor(), records, enable_restore=True)
        self.assertGreater(float(objective.restore.sum()), 0)
        self.assertFalse(bool(((objective.erase > 0) & (objective.restore > 0)).any()))
        self.assertGreater(objective(source)[1]['restore_deficit'], 0)

    def test_inner_updates_do_not_reextract_and_guard_present_paths(self):
        source, records = example()
        model = Extractor()
        objective = AdaptiveInkConstraints(source, model, records)
        count = model.calls
        result, row = optimize_clean(source, lambda z: z, objective, iterations=2, step_ratio=.005)
        self.assertEqual(model.calls, count)
        self.assertIsNone(model.anchor.grad)
        self.assertFalse(model.anchor.requires_grad)
        self.assertLess(row['loss_after'], row['loss_before'])
        self.assertTrue(torch.equal(result[..., 14:22, 17:21], source[..., 14:22, 17:21]))
        broken = source.clone();broken[..., 14:22, 17:21] = 1
        self.assertGreater(objective(broken)[1]['trusted_path_deficit'], 0)

    def test_notebook_embeds_all_runtime_sources_and_requires_exact_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/adaptive_ink_diffusion.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 4)
        for source in code:ast.parse(source)
        nodes = [n for n in ast.parse(code[0]).body if isinstance(n, ast.Assign)]
        helpers = next(ast.literal_eval(n.value) for n in nodes
                       if any(isinstance(t, ast.Name) and t.id == 'RUNTIME_HELPERS' for t in n.targets))
        for path, helper in helpers.items():
            self.assertEqual(helper, (root/path).read_text(encoding='utf-8'))
        self.assertIn('checkpoint_hash(CHECKPOINT)!=expected_sha', code[1])
        self.assertIn('ENABLE_RESTORE=False', code[2])
        self.assertIn("guidance_mode='adaptive_ink'", code[3])


if __name__ == '__main__':unittest.main()
