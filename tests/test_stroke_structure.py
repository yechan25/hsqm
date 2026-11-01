"""Behavioral and gradient checks on known stroke errors, without a checkpoint."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.draw import circle_perimeter

from stroke_model.stroke_structure import StructureConfig, StrokeStructureLoss, reconstruct
from stroke_model.score_stroke_structure import score_saved_strokes, score_structure_csv
from stroke_model.junction_diagnostics import diagnose_junction
from stroke_model.stroke_structure import soft_occupancy


def line_cases(size=64):
    reference = torch.zeros(1, size, size)
    reference[0, 32, 8:56] = 1
    normal = torch.zeros_like(reference)
    normal[0, 30:35, 8:56] = 1
    cases = {'normal': normal, 'missing': torch.zeros_like(normal)}
    cases['gap'] = normal.clone()
    cases['gap'][0, :, 30:34] = 0
    cases['partial'] = normal.clone()
    cases['partial'][0, :, 40:] = 0
    cases['branch'] = normal.clone()
    cases['branch'][0, 15:32, 30:35] = 1
    cases['noise'] = normal.clone()
    cases['noise'][0, 8:11, 44:47] = 1
    cases['notch'] = normal.clone()
    cases['notch'][0, 30:32, 22:30] = 0
    return reference, cases


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reference, cls.cases = line_cases()
        cls.loss = StrokeStructureLoss(cls.reference, StructureConfig(width_weight=1))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def values(self, mask):
        return {name: v.item() for name, v in self.loss(mask)['per_stroke'].items()}

    def test_error_types_and_detached_noise(self):
        v = {name: self.values(mask) for name, mask in self.cases.items()}
        self.assertLess(v['normal']['extraction_error'], .02)
        self.assertGreater(v['missing']['missing'], .99)
        self.assertEqual(v['missing']['split'], 0)
        self.assertGreater(v['gap']['split'], .05)
        self.assertGreater(v['partial']['missing'], .15)
        self.assertGreater(v['branch']['junction'], v['normal']['junction'] + .005)
        self.assertGreater(v['notch']['width'], v['normal']['width'] + .0001)
        for name in ('missing', 'split', 'junction', 'width'):
            self.assertAlmostEqual(v['noise'][name], v['normal'][name], places=6)

    def test_closed_ring_cut_is_detected(self):
        reference = np.zeros((64, 64), dtype=np.float32)
        yy, xx = circle_perimeter(32, 32, 17)
        reference[yy, xx] = 1
        yy, xx = np.indices(reference.shape)
        normal = torch.tensor((np.abs(np.sqrt((yy-32)**2 + (xx-32)**2) - 17) <= 2).astype(np.float32))[None]
        loss = StrokeStructureLoss(reference[None], StructureConfig())
        cut = normal.clone()
        cut[0, 30:35, 46:] = 0
        before, after = loss(normal), loss(cut)
        self.assertTrue(before['convergence'][0]['closed'])
        self.assertLess(before['per_stroke']['split'].item(), 1e-5)
        self.assertGreater(after['per_stroke']['split'].item(), .01)
        self.assertLess(before['per_stroke']['junction'].item(), .02)
        hard = diagnose_junction(normal[0].numpy(), loss.strokes[0].seeds[0, 0].numpy(),
                                 loss.config.junction_thresholds)
        self.assertEqual(hard['branch_regions_max'], 0)

    def test_coverage_allows_small_translation(self):
        shifted = torch.roll(self.cases['normal'], 1, dims=1)
        self.assertLess(self.values(shifted)['missing'], 1e-6)
        self.assertLess(self.values(shifted)['split'], 1e-6)

    def test_live_graph_gradients_and_maps(self):
        mask = self.cases['gap'].clone()
        mask[:, 30:35, 30:34] = .2
        mask.requires_grad_(True)
        result = self.loss(mask, return_maps=True)
        self.assertTrue(result['loss'].requires_grad)
        gradient, = torch.autograd.grad(result['loss'], mask)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        # Filling the known gap must lower the connectivity error.
        split_grad, = torch.autograd.grad(self.loss(mask)['per_stroke']['split'].sum(), mask)
        self.assertLess(split_grad[:, 30:35, 30:34].sum().item(), 0)
        repaired = mask.detach().clone()
        repaired[:, 30:35, 30:34] = .6
        self.assertLess(self.loss(repaired)['per_stroke']['split'].item(),
                        result['per_stroke']['split'].item())
        self.assertEqual(tuple(result['maps'][0]['missing'].shape), (1, 1, 64, 64))
        self.assertTrue(all(c['pairs'] and c['attachment'] for c in result['convergence']))

    def test_missing_gradient_encourages_ink(self):
        mask = torch.zeros_like(self.cases['normal'], requires_grad=True)
        grad, = torch.autograd.grad(self.loss(mask)['per_stroke']['missing'].sum(), mask)
        self.assertLess(grad.sum().item(), 0)

    def test_source_quality_is_not_inferred_from_extraction(self):
        result = self.loss(self.cases['branch'])
        self.assertFalse(result['source_quality']['assessed'])
        self.assertNotIn('total', result['per_stroke'])
        self.assertGreater(result['per_stroke']['extraction_error'].item(), 0)
        empty = self.loss(self.cases['missing'])
        self.assertEqual(empty['loss'].item(), 0)
        self.assertGreater(empty['reference_loss'].item(), .99)

    def test_short_attached_branch_is_not_exempted_as_font_style(self):
        short = self.cases['normal'].clone()
        short[:, 27:30, 30:33] = 1
        self.assertGreater(self.loss(short)['per_stroke']['junction'].item(), 0)

    def test_weak_t_branch_is_not_rounded_down_to_two_arms(self):
        scores = []
        for strength in (1., .8, .6, .4):
            mask = self.cases['branch'] * strength
            result = self.loss(mask)
            scores.append(result['per_stroke']['junction'].item())
            hard = diagnose_junction(mask[0].numpy(), self.loss.strokes[0].seeds[0, 0].numpy(),
                                     self.loss.config.junction_thresholds)
            self.assertGreater(hard['branch_regions_max'], 0)
            self.assertGreater(scores[-1], .005)
        # Averaging threshold evidence intentionally reflects strength, but must
        # not eliminate the clear .6 T as the old raw-arm calculation did.
        self.assertGreater(scores[2], scores[0] * .4)

    def test_weak_side_branch_on_strong_trunk(self):
        for strength in (.25, .4, .6):
            mask = self.cases['normal'].clone()
            mask[0, 15:30, 30:35] = strength
            result = self.loss(mask)
            hard = diagnose_junction(mask[0].numpy(), self.loss.strokes[0].seeds[0, 0].numpy(),
                                     self.loss.config.junction_thresholds)
            # A .25 arm exists at only one default level: report it as
            # unresolved, rather than treating it as confidently branch-free.
            if strength < .3:
                self.assertGreater(hard['candidate_regions_max'], 0)
                self.assertGreater(hard['unresolved_regions_max'], 0)
            else:
                self.assertGreater(hard['branch_regions_max'], 0)
            self.assertGreater(result['per_stroke']['junction'].item(), 1e-4)

    def test_blurred_and_short_weak_branches(self):
        from scipy.ndimage import gaussian_filter
        seed = self.loss.strokes[0].seeds[0, 0].numpy()
        cases = []
        short = self.cases['normal'].clone()
        short[0, 27:30, 30:33] = .4
        cases.append(short)
        blurred = torch.tensor(gaussian_filter(self.cases['branch'][0].numpy(), .7))[None] * .6
        cases.append(blurred)
        for mask in cases:
            result = self.loss(mask)
            hard = diagnose_junction(mask[0].numpy(), seed, self.loss.config.junction_thresholds)
            self.assertGreater(result['per_stroke']['junction'].item(), 1e-5)
            self.assertGreater(hard['branch_regions_max'], 0)

    def test_hard_check_ignores_detached_noise_and_normal_shapes(self):
        seed = self.loss.strokes[0].seeds[0, 0].numpy()
        for name in ('normal', 'noise', 'missing', 'gap'):
            result = diagnose_junction(self.cases[name][0].numpy(), seed, self.loss.config.junction_thresholds)
            self.assertEqual(result['branch_regions_max'], 0, name)

    def test_short_digitized_tip_is_unresolved_not_confirmed_or_silently_clean(self):
        ink = self.reference[0].numpy().copy()
        ink[30:32, 30] = 1
        seed = self.loss.strokes[0].seeds[0, 0].numpy()
        result = diagnose_junction(ink, seed, self.loss.config.junction_thresholds)
        self.assertEqual(result['branch_regions_max'], 0)
        self.assertGreater(result['unresolved_regions_max'], 0)
        # Extending that same thin branch beyond the junction core resolves it.
        ink[26:32, 30] = 1
        result = diagnose_junction(ink, seed, self.loss.config.junction_thresholds)
        self.assertEqual(result['branch_regions_max'], 1)

    def test_reference_corner_and_short_residue_have_no_soft_junction(self):
        residue = torch.zeros_like(self.reference)
        residue[0, 31:34, 25:29] = .6
        self.assertLess(self.loss(residue)['per_stroke']['junction'].item(), 1e-6)
        corner = torch.zeros_like(self.reference)
        corner[0, 30:35, 12:36] = 1
        corner[0, 30:53, 31:36] = 1
        ref = torch.zeros_like(corner)
        ref[0, 32, 12:34] = 1
        ref[0, 32:53, 33] = 1
        criterion = StrokeStructureLoss(ref)
        result = criterion(corner)
        self.assertLess(result['per_stroke']['junction'].item(), 1e-5)

    def test_absolute_occupancy_does_not_amplify_missing_strokes(self):
        ink = torch.tensor([0., .01, .1, .6, 1.], requires_grad=True)
        occupancy = soft_occupancy(ink, .15, .025)
        self.assertEqual(occupancy[0].item(), 0)
        self.assertEqual(occupancy[-1].item(), 1)
        self.assertLess(occupancy[1].item(), .01)
        occupancy.sum().backward()
        self.assertTrue(torch.isfinite(ink.grad).all())

    def test_multilevel_config_rejects_unordered_or_invalid_levels(self):
        for kwargs in ({'junction_thresholds': ()}, {'junction_thresholds': (.4, .2)},
                       {'junction_thresholds': (.2, .2)}, {'junction_thresholds': (0.,)},
                       {'junction_temperature': 0}, {'junction_min_levels': 0},
                       {'junction_min_levels': 6}, {'junction_arm_scale': 0},
                       {'junction_angular_samples': 4}):
            with self.assertRaises(ValueError):
                StructureConfig(**kwargs)

    def test_split_directional_finite_difference(self):
        mask = self.cases['normal'].clone() * .85
        mask[:, 30:35, 30:34] = .23
        mask.requires_grad_(True)
        direction = torch.zeros_like(mask)
        direction[:, 30:35, 30:34] = 1
        value = self.loss(mask)['per_stroke']['split'].sum()
        grad, = torch.autograd.grad(value, mask)
        delta = 1e-3
        plus = self.loss(mask.detach() + delta*direction)['per_stroke']['split'].sum()
        minus = self.loss(mask.detach() - delta*direction)['per_stroke']['split'].sum()
        self.assertAlmostEqual((grad*direction).sum().item(), ((plus-minus)/(2*delta)).item(), places=3)

    def test_junction_gradient_and_branch_shortening(self):
        # Nonuniform positive ink avoids nondifferentiable equal-value pooling
        # ties, where a central finite difference need not equal a subgradient.
        generator = torch.Generator().manual_seed(42)
        ink = .8 + .15 * torch.rand(self.cases['branch'].shape, generator=generator)
        mask = (self.cases['branch'] * ink).requires_grad_(True)
        value = self.loss(mask)['per_stroke']['junction'].sum()
        gradient, = torch.autograd.grad(value, mask)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        # Weakening the added branch must reduce the junction response.
        direction = torch.zeros_like(mask)
        direction[:, 15:30, 30:35] = 1
        self.assertGreater((gradient*direction).sum().item(), 0)
        delta = 1e-5
        plus = self.loss(mask.detach()+delta*direction)['per_stroke']['junction'].sum()
        minus = self.loss(mask.detach()-delta*direction)['per_stroke']['junction'].sum()
        self.assertAlmostEqual((gradient*direction).sum().item(), ((plus-minus)/(2*delta)).item(), delta=.001)

    def test_batch_and_short_caps_reported(self):
        batch = torch.stack([self.cases['normal'], self.cases['gap']])
        result = self.loss(batch)
        self.assertEqual(tuple(result['per_stroke']['missing'].shape), (2, 1))
        capped = StrokeStructureLoss(self.reference, StructureConfig(connectivity_steps=1, attachment_steps=1))
        status = capped(self.cases['normal'])['convergence'][0]
        self.assertFalse(status['pairs'])

    def test_invalid_reference_and_config(self):
        with self.assertRaisesRegex(ValueError, 'Reference stroke 1'):
            StrokeStructureLoss(torch.zeros_like(self.reference))
        for kwargs in ({'required_ink': 0}, {'anchor_radius': -1}, {'connectivity_steps': 0}, {'pair_spans': ()}):
            with self.assertRaises(ValueError):
                StructureConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.loss(self.cases['normal'] * 2)

    def test_reconstruction_respects_barrier(self):
        m = torch.ones(1, 1, 7, 12)
        m[..., 5] = 0
        seed = torch.zeros_like(m)
        seed[..., 0] = 1
        reached, converged = reconstruct(m, seed)
        self.assertTrue(converged)
        self.assertEqual(reached[..., 6:].sum().item(), 0)

    def test_saved_inference_report_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prediction.npz'
            np.savez_compressed(path, soft_masks=self.cases['gap'].numpy(),
                                reference_strokes=self.reference.numpy())
            report = score_saved_strokes({'results': [{'npz': str(path)}]},
                save_dir=Path(tmp) / 'scores', show=False)
            saved = json.loads((Path(tmp) / 'scores/structure_report.json').read_text())
            row = saved['results'][0]['strokes'][0]
            self.assertGreater(row['split'], .05)
            self.assertEqual(report['config']['width_weight'], 0)
            for name in ('csv', 'npz'):
                self.assertTrue(Path(saved['results'][0][name]).is_file())
            self.assertTrue(Path(saved['results'][0]['figures'][0]).is_file())
            self.assertIn('junction_checks', saved['results'][0])

    def test_csv_filename_integration_ignores_annotations(self):
        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, image, reference, strokes):
                return {'masks': image.expand_as(strokes), 'hints': strokes}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in [('reference', self.cases['normal']), ('gap', self.cases['gap'])]:
                Image.fromarray((255*(1-value[0].numpy())).astype(np.uint8)).save(root / (name + '.png'))
            pd.DataFrame([{'input_path': 'gap.png', 'target_path': 'reference.png',
                           'target_stroke_paths': 'reference.png',
                           'input_stroke_paths': 'nonexistent_annotations.png'}]).to_csv(root / 'paths.csv', index=False)
            torch.save({'model': {}, 'image_size': 64, 'width': 4}, root / 'best.pt')
            with patch('stroke_model.reference_model.ReferenceStrokeModel', FakeModel):
                report = score_structure_csv(root / 'best.pt', root / 'paths.csv',
                    input_files=['gap.png'], save_dir=root / 'scores', show=False)
            self.assertFalse(report['uses_input_stroke_annotations'])
            self.assertEqual(report['results'][0]['csv_row'], 1)
            self.assertGreater(report['results'][0]['strokes'][0]['split'], .05)


if __name__ == '__main__':
    unittest.main()
