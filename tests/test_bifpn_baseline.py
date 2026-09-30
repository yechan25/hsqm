import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import torch
from stroke_model.easy_inference import show_bifpn_baseline


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.linspace(-3, 3, 192).reshape(1, 3, 8, 8)
        item = {'I_prep': torch.zeros(1, 8, 8), 'I_g': torch.ones(1, 8, 8),
                'I_G_strokes': torch.ones(3, 8, 8), 'target_strokes': torch.ones(3, 8, 8), 'char_id': 'test'}
        self.session = {'architecture': 'BiFPN', 'checkpoint_path': 'test.pt', 'dataset': [item],
                        'device': 'cpu', 'model': lambda *args: {'logits': self.logits}}

    def test_raw_is_exact_sigmoid_even_outside_input_foreground(self):
        with (patch('matplotlib.pyplot.show'),
             patch('stroke_model.graph_postprocess.postprocess_batch_predictions_graph', side_effect=AssertionError('no graph')),
             patch('stroke_model.postprocess.postprocess_batch_predictions_seeded_partition', side_effect=AssertionError('no legacy'))):
            result = show_bifpn_baseline(self.session)
        torch.testing.assert_close(result['raw_masks'], self.logits.sigmoid())
        self.assertIsNone(result['legacy_masks'])

    def test_rejects_v2(self):
        self.session['architecture'] = 'reference_v2'
        with self.assertRaises(ValueError):
            show_bifpn_baseline(self.session)
