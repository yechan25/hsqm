import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from stroke_model.preview_reference import soft_diagnostics, show_soft_strokes


class SoftPreviewTests(unittest.TestCase):
    def test_probabilities_masks_entropy_and_export(self):
        image = torch.full((1, 8, 8), .8)
        probabilities = torch.full((1, 3, 8, 8), 0.)
        probabilities[:, :2] = .5
        output = {'probabilities': probabilities, 'masks': probabilities * image,
                  'active': torch.tensor([[True, True, False]])}
        payload = soft_diagnostics(output)
        np.testing.assert_allclose(payload['probabilities'][:2], .5)
        np.testing.assert_allclose(payload['soft_masks'][:2], .4)
        np.testing.assert_allclose(payload['entropy'], 1)
        with tempfile.TemporaryDirectory() as tmp, patch('matplotlib.pyplot.show'):
            prefix = Path(tmp) / 'example'
            show_soft_strokes(image, torch.ones(3, 8, 8), output, prefix)
            with np.load(str(prefix) + '_soft.npz') as saved:
                np.testing.assert_allclose(saved['soft_masks'].sum(0), image[0])
            self.assertTrue(Path(str(prefix) + '_soft_1.png').is_file())
            self.assertTrue(Path(str(prefix) + '_entropy.png').is_file())

    def test_single_stroke_entropy_is_zero(self):
        output = {'probabilities': torch.ones(1, 1, 3, 3), 'masks': torch.ones(1, 1, 3, 3),
                  'active': torch.tensor([[True]])}
        np.testing.assert_array_equal(soft_diagnostics(output)['entropy'], 0)
