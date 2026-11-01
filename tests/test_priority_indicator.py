"""Rubric regressions; no sample filenames participate in score calculation."""
import unittest
import torch
from stroke_model.extra_ink import ExtraInkMetric
from stroke_model.structural_indicator import StructuralIndicator


class PriorityTests(unittest.TestCase):
    def test_comparable_single_defects_follow_priority(self):
        metric = StructuralIndicator()
        scores = []
        for missing, split, junction, extra in [
                ([0, .81, 0], [0]*3, [0]*3, 0),
                ([0]*3, [0]*3, [0]*3, .07),
                ([0]*3, [0]*3, [0, .029, 0], 0),
                ([0]*3, [0, .22, 0], [0]*3, 0),
                ([0]*3, [0]*3, [0]*3, 0)]:
            scores.append(metric(missing, split, junction, extra=extra)['score'].item())
        self.assertLess(max(scores[:2]), scores[2])
        self.assertLess(scores[2], scores[3])
        self.assertLess(scores[3], scores[4])

    def test_missing_stroke_cannot_be_diluted_by_many_good_strokes(self):
        metric = StructuralIndicator()
        for count in (2, 5, 30):
            result = metric([.81]+[0.]*(count-1), [0.]*count, [0.]*count, extra=0)
            self.assertAlmostEqual(result['score'].item(), 19, places=4)
        missing = torch.tensor([1., 0., 0.], requires_grad=True)
        result = metric(missing, [0.]*3, [0.]*3, extra=0)
        gradient, = torch.autograd.grad(result['penalty'].sum(), missing)
        self.assertEqual(result['score'].item(), 0)
        self.assertGreater(gradient[0].item(), 0)

    def test_isolated_topology_penalties_saturate_and_remain_monotone(self):
        metric = StructuralIndicator()
        values = torch.tensor([[0.], [.1], [1.], [1e38]])
        zero = torch.zeros_like(values)
        for index, floor in ((1, 70), (2, 35)):
            args = [zero, zero, zero]
            args[index] = values
            scores = metric(*args, extra=torch.zeros(4))['score']
            self.assertTrue(torch.isfinite(scores).all())
            self.assertTrue((scores[:-1] > scores[1:]).all())
            self.assertGreaterEqual(scores[-1].item(), floor-1e-5)

    def test_padding_is_excluded_from_missing_caps(self):
        result = StructuralIndicator()([0., 1.], [0., 2.], [0., 3.],
                                       active=torch.tensor([True, False]), extra=0)
        self.assertEqual(result['score'].item(), 100)

    def test_alignment_handles_small_shift_but_keeps_added_bar(self):
        ref = torch.zeros(1, 32, 32)
        ref[0, 4:28, 10] = 1
        image = torch.zeros_like(ref)
        image[0, 4:28, 14:17] = 1
        fixed = ExtraInkMetric(ref, radius=3, transition=.25, alignment=0)(image)['extra']
        aligned_metric = ExtraInkMetric(ref, radius=3, transition=.25)
        aligned = aligned_metric(image)['extra']
        added = image.clone()
        added[0, 19:22, 17:29] = 1
        self.assertLess(aligned.item(), .05)
        self.assertGreater(fixed.item(), .8)
        self.assertGreater(aligned_metric(added)['extra'].item(), .1)

    def test_alignment_gradient_matches_finite_difference(self):
        ref = torch.zeros(1, 20, 20)
        ref[0, 3:17, 8] = 1
        metric = ExtraInkMetric(ref, radius=2, alignment=2).double()
        image = torch.full((1, 20, 20), .05, dtype=torch.float64)
        image[0, 3:17, 8:11] = .8
        image.requires_grad_()
        direction = torch.zeros_like(image)
        direction[0, 12, 15] = 1
        value = metric(image)['extra'].sum()
        gradient, = torch.autograd.grad(value, image)
        eps = 1e-5
        fd = (metric(image.detach()+eps*direction)['extra']
              - metric(image.detach()-eps*direction)['extra'])/(2*eps)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual((gradient*direction).sum().item(), fd.item(), places=7)


if __name__ == '__main__':
    unittest.main()
