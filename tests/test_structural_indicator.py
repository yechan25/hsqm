import json
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import numpy as np
from PIL import Image
import torch
from stroke_model.structural_indicator import IndicatorConfig, StructuralIndicator, score_structure_report, _image_for_sample


# Compatibility checks for historical stored extractor measurements.
from functools import partial
score_structure_report = partial(score_structure_report, measurement_mode="stored")

def three_term_indicator(config=None):
    return StructuralIndicator(replace(config or IndicatorConfig(), extra_weight=0, priority_mode=False))


class IndicatorTests(unittest.TestCase):
    def test_perfect_and_complete_missing_strokes(self):
        model = three_term_indicator()
        result = model([0., 0., 0.], [0., 0., 0.], [0., 0., 0.])
        self.assertEqual(result['score'].item(), 100)
        result = model([0., 1., 0., 0.], [0.]*4, [0.]*4)
        self.assertEqual(result['stroke_scores'][0, 1].item(), 0)
        self.assertEqual(result['mean_score'].item(), 75)
        self.assertEqual(result['score'].item(), 37.5)

    def test_formula_has_separate_three_component_effects(self):
        cfg = IndicatorConfig(worst_stroke_weight=0)
        model = three_term_indicator(cfg)
        result = model([.809, 0, 0], [0, cfg.split_scale, 0], [0, 0, cfg.junction_scale])
        self.assertAlmostEqual(result['stroke_scores'][0, 0].item(), 19.1, places=4)
        self.assertAlmostEqual(result['stroke_scores'][0, 1].item(), 100/math.e, places=4)
        self.assertAlmostEqual(result['stroke_scores'][0, 2].item(), 100/math.e, places=4)

    def test_errors_monotonically_lower_score(self):
        model = three_term_indicator()
        for component in range(3):
            features = [torch.zeros(4, 1) for _ in range(3)]
            features[component][:, 0] = torch.tensor([0., .1, .3, .8])
            scores = model(*features)['score']
            self.assertTrue(bool((scores[:-1] > scores[1:]).all()))

    def test_finite_gradients_including_missing_endpoint(self):
        for weight in (0., .5, 1., 2.):
            model = three_term_indicator(IndicatorConfig(missing_weight=weight))
            features = [torch.tensor([0., .3, 1.], requires_grad=True),
                        torch.tensor([0., .05, .1], requires_grad=True),
                        torch.tensor([0., .001, .01], requires_grad=True)]
            output = model(*features)
            gradients = torch.autograd.grad(output['penalty'].sum(), features)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(all(bool((g >= -1e-6).all()) for g in gradients))

    def test_active_channels_and_batch_aggregation(self):
        model = three_term_indicator()
        missing = torch.tensor([[0., 1.], [.5, 1.]])
        active = torch.tensor([[True, False], [True, True]])
        result = model(missing, torch.zeros_like(missing), torch.zeros_like(missing), active)
        torch.testing.assert_close(result['score'], torch.tensor([100., 12.5]))

    def test_worst_and_mean_controls(self):
        values = ([0., .8], [0., 0.], [0., 0.])
        average = three_term_indicator(IndicatorConfig(worst_stroke_weight=0))(*values)['score'].item()
        worst = three_term_indicator(IndicatorConfig(worst_stroke_weight=1))(*values)['score'].item()
        self.assertAlmostEqual(average, 60, places=4)
        self.assertAlmostEqual(worst, 20, places=4)
        disabled = three_term_indicator(IndicatorConfig(missing_weight=0, split_weight=0, junction_weight=0))
        self.assertEqual(disabled([1.], [10.], [10.])['score'].item(), 100)

    def test_directional_gradient_matches_finite_difference(self):
        model = three_term_indicator()
        m = torch.tensor([.2, .4], dtype=torch.float64, requires_grad=True)
        d = torch.tensor([.03, .07], dtype=torch.float64)
        j = torch.tensor([.001, .002], dtype=torch.float64)
        gradient, = torch.autograd.grad(model(m, d, j)['penalty'].sum(), m)
        direction = torch.tensor([.7, .3], dtype=torch.float64)
        eps = 1e-5
        finite = (model(m.detach()+eps*direction, d, j)['penalty']
                  - model(m.detach()-eps*direction, d, j)['penalty'])/(2*eps)
        self.assertAlmostEqual(float((gradient*direction).sum()), finite.item(), places=6)

    def test_complete_missing_keeps_recovery_gradient(self):
        missing = torch.tensor([1.], requires_grad=True)
        result = three_term_indicator()(missing, [0.], [0.])
        gradient, = torch.autograd.grad(result['penalty'].sum(), missing)
        self.assertEqual(result['score'].item(), 0)
        self.assertGreater(gradient.item(), .9)

    def test_invalid_input_is_not_silently_scored(self):
        model = three_term_indicator()
        for values in (([1.1], [0.], [0.]), ([-.1], [0.], [0.]),
                       ([0.], [float('nan')], [0.]), ([0.], [0., 0.], [0.])):
            with self.assertRaises(ValueError):
                model(*values)
        with self.assertRaises(ValueError):
            model([0.], [0.], [0.], torch.tensor([False]))
        for values in ({'split_scale': 0}, {'junction_weight': -1}, {'worst_stroke_weight': 2}):
            with self.assertRaises(ValueError):
                IndicatorConfig(**values)

    def test_report_uses_only_m_d_j_and_exports_audit(self):
        base = {'results': [{'csv_row': 43, 'input': 'sample.png', 'strokes': [
            {'stroke': 1, 'missing': .1, 'split': .05, 'junction': .001,
             'width': 999, 'branch_regions_max': 1000}]}]}
        other = json.loads(json.dumps(base))
        other['results'][0]['strokes'][0].update(width=0, branch_regions_max=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = score_structure_report(base, save_dir=root/'first', show_images=False, config=IndicatorConfig(extra_weight=0, priority_mode=False))
            second = score_structure_report(other, save_dir=root/'second', show_images=False, config=IndicatorConfig(extra_weight=0, priority_mode=False))
            self.assertEqual(first['results'][0]['structural_score'], second['results'][0]['structural_score'])
            self.assertEqual(first['uses_only'], ['missing', 'split', 'junction'])
            self.assertFalse(first['trained'])
            self.assertTrue((root/'first/glyph_scores.csv').is_file())
            self.assertTrue((root/'first/stroke_scores.csv').is_file())
            saved = json.loads((root/'first/indicator_report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['config'], first['config'])

    def test_original_images_are_paired_with_their_own_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (12, 12), (255, 0, 0)).save(root/'first.png')
            Image.new('RGB', (12, 12), (0, 0, 255)).save(root/'second.png')
            report = {'results': [{'input': str(root/f'{name}.png'), 'strokes': [
                {'missing': missing, 'split': 0, 'junction': 0}]} for name, missing in [('first', 0), ('second', .5)]]}
            result = score_structure_report(report, save_dir=root/'scores', show_images=False, config=IndicatorConfig(extra_weight=0, priority_mode=False))
            self.assertEqual(result['results'][0]['structural_score'], 100)
            self.assertAlmostEqual(result['results'][1]['structural_score'], 50)
            self.assertEqual(Path(result['results'][0]['image_source']).name, 'first.png')
            self.assertEqual(Path(result['results'][1]['image_source']).name, 'second.png')
            self.assertTrue(all(r['image_available'] for r in result['results']))
            gallery = Path(result['gallery_html']).read_text(encoding='utf-8')
            self.assertEqual(gallery.count('data:image/png;base64,'), 2)
            self.assertIn('100.0', gallery)
            self.assertIn('50.0', gallery)
            self.assertTrue(Path(result['gallery_images'][0]).is_file())

    def test_saved_input_array_fallback_has_correct_ink_polarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.npz'
            ink = np.zeros((1, 8, 8), dtype=np.float32)
            ink[0, 2:6, 3] = 1
            np.savez_compressed(path, image=ink, reference_strokes=np.ones((1, 8, 8)))
            image, source, kind = _image_for_sample({'input': 'unavailable.png', 'npz': str(path)}, None)
            self.assertEqual(kind, 'scored_input_array')
            self.assertEqual(image.getpixel((3, 3)), (0, 0, 0))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_scored_snapshot_precedes_a_changed_original_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (8, 8), 'red').save(root/'changed.png')
            ink = np.zeros((1, 8, 8), dtype=np.float32)
            np.savez_compressed(root/'snapshot.npz', image=ink)
            image, _, kind = _image_for_sample({'input': str(root/'changed.png'),
                                                'npz': str(root/'snapshot.npz')}, None)
            self.assertEqual(kind, 'scored_input_array')
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_reference_array_is_never_presented_as_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reference_only.npz'
            np.savez_compressed(path, reference_strokes=np.ones((1, 8, 8)))
            image, _, kind = _image_for_sample({'npz': str(path)}, None)
            self.assertIsNone(image)
            self.assertEqual(kind, 'unavailable')

    def test_report_relative_image_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (8, 8), 'white').save(root/'input.png')
            image, source, kind = _image_for_sample({'input': 'input.png'}, root/'structure_report.json')
            self.assertEqual(kind, 'original_file')
            self.assertEqual(Path(source), root/'input.png')


if __name__ == '__main__':
    unittest.main()
