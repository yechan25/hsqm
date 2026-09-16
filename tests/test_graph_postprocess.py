"""Run: python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import torch

from stroke_model.graph_postprocess import refine_strokes_graph, postprocess_batch_predictions_graph


class GraphPostprocessTests(unittest.TestCase):
    def test_crossing_can_belong_to_both_strokes(self):
        target = np.zeros((2, 25, 25), np.float32)
        target[0, 11:14, 2:23] = 1
        target[1, 2:23, 11:14] = 1
        result = refine_strokes_graph(target, target.max(0), reference_strokes=target)
        np.testing.assert_array_equal(result > 0.25, target > 0)
        self.assertTrue((result[:, 12, 12] > 0.9).all())

    def test_short_stroke_is_not_absorbed_by_large_stroke(self):
        target = np.zeros((2, 25, 25), np.float32)
        target[0, 12, 2:23] = 1
        target[1, 10:13, 12] = 1
        result = refine_strokes_graph(target, target.max(0), reference_strokes=target)
        np.testing.assert_array_equal(result > 0.25, target > 0)

    def test_diagonal_and_disconnected_parts_preserved(self):
        target = np.zeros((2, 25, 25), np.float32)
        for i in list(range(2, 9)) + list(range(11, 19)):
            target[0, i, i] = 1
        target[1, 23, 2:20] = 1
        result = refine_strokes_graph(target, target.max(0), reference_strokes=target)
        np.testing.assert_array_equal(result > 0.25, target > 0)

    def test_smoothing_reduces_low_amplitude_noise(self):
        rng = np.random.default_rng(7)
        pred = (0.65 + rng.uniform(-0.10, 0.10, (1, 20, 20))).astype(np.float32)
        result = refine_strokes_graph(pred, np.ones((20, 20)), active_channels=[0])
        self.assertLess(np.mean((result - 0.65) ** 2), np.mean((pred - 0.65) ** 2))

    def test_no_diffusion_across_background_gap(self):
        ink = np.zeros((9, 9), np.float32)
        ink[2:7, 1:3] = 1
        ink[2:7, 6:8] = 1
        pred = ink[None].copy()
        pred[:, :, 6:8] = 0.1 * ink[:, 6:8]
        result = refine_strokes_graph(pred, ink, active_channels=[0], smoothness=100)
        np.testing.assert_allclose(result, pred, atol=1e-6)

    def test_no_per_channel_peak_normalization_or_forced_fill(self):
        pred = np.stack([np.full((9, 9), 0.8), np.full((9, 9), 0.02)])
        result = refine_strokes_graph(pred, np.ones((9, 9)), active_channels=[0, 1])
        np.testing.assert_allclose(result, pred, atol=1e-6)

    def test_padding_cannot_compete(self):
        pred = np.ones((3, 9, 9), np.float32)
        ref = np.zeros_like(pred)
        ref[1] = 1
        result = refine_strokes_graph(pred, np.ones((9, 9)), reference_strokes=ref)
        self.assertEqual(result[0].sum() + result[2].sum(), 0)
        self.assertEqual(result[1].sum(), 81)

    def test_reference_is_not_an_unaligned_spatial_prior(self):
        pred = np.full((1, 9, 9), 0.8)
        ref = np.zeros_like(pred)
        ref[0, 0, 0] = 1
        result = refine_strokes_graph(pred, np.ones((9, 9)), reference_strokes=ref, smoothness=0)
        np.testing.assert_allclose(result, pred)

    def test_zero_regularization_and_optional_aligned_hint(self):
        pred = np.full((1, 9, 9), 0.2)
        hints = np.full_like(pred, 0.8)
        result = refine_strokes_graph(pred, np.ones((9, 9)), active_channels=[0],
                                      warped_hints=hints, smoothness=0, hint_weight=1)
        np.testing.assert_allclose(result, 0.5)

    def test_empty_input_and_isolated_pixel(self):
        pred = np.ones((1, 3, 3))
        empty = refine_strokes_graph(pred, np.zeros((3, 3)), active_channels=[])
        self.assertEqual(empty.sum(), 0)
        ink = np.eye(3)
        result = refine_strokes_graph(pred, ink, active_channels=[0])
        np.testing.assert_array_equal(result[0], ink)

    def test_invalid_inputs_fail_explicitly(self):
        pred, ink = np.ones((1, 3, 3)), np.ones((3, 3))
        for kwargs in [{}, {"active_channels": []}, {"active_channels": [2]},
                       {"active_channels": [0, 0]}, {"active_channels": [0], "edge_scale": 0}]:
            with self.assertRaises(ValueError):
                refine_strokes_graph(pred, ink, **kwargs)
        pred[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            refine_strokes_graph(pred, ink, active_channels=[0])

    def test_exclusive_covers_low_confidence_pixels_and_ties(self):
        pred = torch.zeros(1, 3, 9, 9)
        refs = torch.zeros_like(pred)
        refs[:, 1:, 4, 4] = 1
        ink = torch.zeros(1, 1, 9, 9)
        ink[:, :, 2:7, 2:7] = 1
        # Even zero evidence must be assigned to one active channel;
        # inactive channel 0 must not win the tie.
        exclusive = postprocess_batch_predictions_graph(pred, ink, refs, mode="exclusive")
        overlap = postprocess_batch_predictions_graph(pred, ink, refs, mode="overlap")
        torch.testing.assert_close(exclusive.sum(1, keepdim=True), ink)
        self.assertEqual(float(exclusive[:, 0].sum()), 0)
        self.assertEqual(float(overlap.sum()), 0)

    def test_batch_modes_and_no_ink_leak(self):
        ref = torch.zeros(2, 2, 5, 5)
        ref[:, :, 2, 2] = 1
        ink = ref.amax(1, keepdim=True)
        for mode in ["soft", "overlap", "exclusive"]:
            result = postprocess_batch_predictions_graph(ref, ink, ref, mode=mode)
            self.assertEqual(result.device, ref.device)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.shape, ref.shape)
            self.assertEqual(float((result * (1 - ink)).sum()), 0)
            self.assertEqual(float(result.sum()), 2 if mode == "exclusive" else 4)


if __name__ == "__main__":
    unittest.main()
