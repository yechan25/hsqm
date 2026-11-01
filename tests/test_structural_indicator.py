import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from stroke_model.structural_indicator import IndicatorConfig, StructuralIndicator, score_structure_report


class IndicatorTests(unittest.TestCase):
    def test_perfect_and_complete_missing_strokes(self):
        model = StructuralIndicator()
        result = model([0., 0., 0.], [0., 0., 0.], [0., 0., 0.])
        self.assertEqual(result['score'].item(), 100)
        result = model([0., 1., 0., 0.], [0.]*4, [0.]*4)
        self.assertEqual(result['stroke_scores'][0, 1].item(), 0)
        self.assertEqual(result['mean_score'].item(), 75)
        self.assertEqual(result['score'].item(), 37.5)

    def test_formula_has_separate_three_component_effects(self):
        cfg = IndicatorConfig(worst_stroke_weight=0)
        model = StructuralIndicator(cfg)
        result = model([.809, 0, 0], [0, cfg.split_scale, 0], [0, 0, cfg.junction_scale])
        self.assertAlmostEqual(result['stroke_scores'][0, 0].item(), 19.1, places=4)
        self.assertAlmostEqual(result['stroke_scores'][0, 1].item(), 100/math.e, places=4)
        self.assertAlmostEqual(result['stroke_scores'][0, 2].item(), 100/math.e, places=4)

    def test_errors_monotonically_lower_score(self):
        model = StructuralIndicator()
        for component in range(3):
            features = [torch.zeros(4, 1) for _ in range(3)]
            features[component][:, 0] = torch.tensor([0., .1, .3, .8])
            scores = model(*features)['score']
            self.assertTrue(bool((scores[:-1] > scores[1:]).all()))

    def test_finite_gradients_including_missing_endpoint(self):
        for weight in (0., .5, 1., 2.):
            model = StructuralIndicator(IndicatorConfig(missing_weight=weight))
            features = [torch.tensor([0., .3, 1.], requires_grad=True),
                        torch.tensor([0., .05, .1], requires_grad=True),
                        torch.tensor([0., .001, .01], requires_grad=True)]
            output = model(*features)
            gradients = torch.autograd.grad(output['penalty'].sum(), features)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(all(bool((g >= -1e-6).all()) for g in gradients))

    def test_active_channels_and_batch_aggregation(self):
        model = StructuralIndicator()
        missing = torch.tensor([[0., 1.], [.5, 1.]])
        active = torch.tensor([[True, False], [True, True]])
        result = model(missing, torch.zeros_like(missing), torch.zeros_like(missing), active)
        torch.testing.assert_close(result['score'], torch.tensor([100., 12.5]))

    def test_worst_and_mean_controls(self):
        values = ([0., .8], [0., 0.], [0., 0.])
        average = StructuralIndicator(IndicatorConfig(worst_stroke_weight=0))(*values)['score'].item()
        worst = StructuralIndicator(IndicatorConfig(worst_stroke_weight=1))(*values)['score'].item()
        self.assertAlmostEqual(average, 60, places=4)
        self.assertAlmostEqual(worst, 20, places=4)
        disabled = StructuralIndicator(IndicatorConfig(missing_weight=0, split_weight=0, junction_weight=0))
        self.assertEqual(disabled([1.], [10.], [10.])['score'].item(), 100)

    def test_directional_gradient_matches_finite_difference(self):
        model = StructuralIndicator()
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
        result = StructuralIndicator()(missing, [0.], [0.])
        gradient, = torch.autograd.grad(result['penalty'].sum(), missing)
        self.assertEqual(result['score'].item(), 0)
        self.assertGreater(gradient.item(), .9)

    def test_invalid_input_is_not_silently_scored(self):
        model = StructuralIndicator()
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
            first = score_structure_report(base, save_dir=root/'first')
            second = score_structure_report(other, save_dir=root/'second')
            self.assertEqual(first['results'][0]['structural_score'], second['results'][0]['structural_score'])
            self.assertEqual(first['uses_only'], ['missing', 'split', 'junction'])
            self.assertFalse(first['trained'])
            self.assertTrue((root/'first/glyph_scores.csv').is_file())
            self.assertTrue((root/'first/stroke_scores.csv').is_file())
            saved = json.loads((root/'first/indicator_report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['config'], first['config'])


if __name__ == '__main__':
    unittest.main()
