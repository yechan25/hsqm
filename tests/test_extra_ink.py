import unittest
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from stroke_model.extra_ink import ExtraInkMetric
from stroke_model.structural_indicator import IndicatorConfig, StructuralIndicator, score_structure_report


class ExtraInkTests(unittest.TestCase):
    def setUp(self):
        self.ref = torch.zeros(1, 32, 32)
        self.ref[0, 4:28, 12] = 1
        self.normal = torch.zeros(1, 32, 32)
        self.normal[0, 4:28, 11:14] = 1
        self.metric = ExtraInkMetric(self.ref, radius=3, transition=.25)

    def test_attached_and_detached_added_strokes_are_penalized(self):
        normal = self.metric(self.normal)['extra'].item()
        attached = self.normal.clone()
        attached[0, 19:22, 13:26] = 1
        detached = self.normal.clone()
        detached[0, 19:22, 20:28] = 1
        self.assertLess(normal, .02)
        self.assertGreater(self.metric(attached)['extra'].item(), .1)
        self.assertGreater(self.metric(detached)['extra'].item(), .1)

    def test_score_falls_even_when_all_three_old_errors_are_zero(self):
        scorer = StructuralIndicator()
        perfect = scorer([0.], [0.], [0.], extra=[0.])
        added = scorer([0.], [0.], [0.], extra=[.1])
        self.assertEqual(perfect['score'].item(), 100)
        self.assertLess(added['score'].item(), 25)
        self.assertEqual(added['stroke_scores'][0, 0].item(), 100)

    def test_absent_measurement_is_not_treated_as_no_extra(self):
        with self.assertRaisesRegex(ValueError, 'Added ink'):
            StructuralIndicator()([0.], [0.], [0.])
        self.assertEqual(StructuralIndicator(IndicatorConfig(extra_weight=0))([0.], [0.], [0.])['score'].item(), 100)

    def test_input_gradient_penalizes_added_pixels(self):
        image = self.normal.clone()
        image[0, 20, 25] = .5
        image.requires_grad_(True)
        extra = self.metric(image)['extra']
        gradient, = torch.autograd.grad(extra.sum(), image)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[0, 20, 25].item(), 0)
        score_extra = torch.tensor([.08], requires_grad=True)
        output = StructuralIndicator()([0.], [0.], [0.], extra=score_extra)
        grad, = torch.autograd.grad(output['penalty'].sum(), score_extra)
        self.assertGreater(grad.item(), 0)

    def test_blank_and_small_position_variation(self):
        result = self.metric(torch.zeros_like(self.normal))
        self.assertEqual(result['extra'].item(), 0)
        self.assertTrue(torch.isfinite(result['extra']).all())
        shifted = torch.roll(self.normal, 1, -1)
        self.assertLess(self.metric(shifted)['extra'].item(), .02)

    def test_old_npz_report_is_remeasured_and_visualized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = self.normal.clone()
            image[0, 19:22, 14:26] = 1
            np.savez_compressed(root/'old.npz', image=image.numpy(), reference_strokes=self.ref.numpy())
            report = {'results': [{'npz': str(root/'old.npz'), 'strokes': [
                {'missing': 0, 'split': 0, 'junction': 0}]}]}
            result = score_structure_report(report, config=IndicatorConfig(extra_radius=3, extra_transition=.25),
                                             save_dir=root/'result', show_images=False)
            sample = result['results'][0]
            self.assertTrue(sample['extra_assessed'])
            self.assertGreater(sample['extra'], .1)
            self.assertLess(sample['structural_score'], 25)
            self.assertTrue(Path(sample['extra_overlay']).is_file())
            self.assertIn('extra', result['uses_only'])

    def test_missing_arrays_fail_instead_of_silently_scoring_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = {'results': [{'strokes': [{'missing': 0, 'split': 0, 'junction': 0}]}]}
            with self.assertRaisesRegex(ValueError, 'Cannot assess added ink'):
                score_structure_report(report, save_dir=Path(tmp)/'result', show_images=False)

    def test_full_target_image_precedes_incomplete_stroke_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self.normal.clone()
            target[0, 19:22, 13:26] = 1
            np.savez_compressed(root/'new.npz', image=target.numpy(),
                                reference_image=target.numpy(), reference_geometry=self.ref.numpy())
            report = {'results': [{'npz': str(root/'new.npz'), 'strokes': [
                {'missing': 0, 'split': 0, 'junction': 0}]}]}
            result = score_structure_report(report, config=IndicatorConfig(extra_radius=3, extra_transition=.25),
                                            save_dir=root/'result', show_images=False)
            sample = result['results'][0]
            self.assertLess(sample['extra'], .02)
            self.assertAlmostEqual(sample['structural_score'], 100)
            self.assertTrue(sample['extra_source'].endswith('::reference_image'))

    def test_invalid_extra_config_is_rejected(self):
        for values in ({'extra_scale': 0}, {'extra_scale': 1e-50},
                       {'extra_radius': -1}, {'extra_allowance': 1}, {'extra_transition': 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                IndicatorConfig(**values)


if __name__ == '__main__':
    unittest.main()
