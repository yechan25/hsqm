import unittest
import torch
from stroke_model.diagnose_reference import sample_metrics, select_examples


class DiagnosisTests(unittest.TestCase):
    def test_unknown_pixels_do_not_hide_known_errors_in_report(self):
        image = torch.ones(1, 2, 2)
        labels = torch.tensor([[0, 1], [-1, -1]])
        p = torch.tensor([[[[.99, .99], [.99, .99]], [[.01, .01], [.01, .01]]]])
        metrics = sample_metrics(image, labels, {'probabilities': p})
        self.assertEqual(metrics['annotation_coverage'], .5)
        self.assertEqual(metrics['known_error_pixels'], 1)
        self.assertEqual(metrics['confident_known_error_pixels'], 1)
        self.assertAlmostEqual(metrics['annotated_macro_dice'], 1/3)

    def test_selection_includes_bad_and_good_without_duplicates(self):
        rows = [{'index': 0, 'annotated_macro_dice': .4, 'annotation_coverage': .9},
                {'index': 1, 'annotated_macro_dice': .8, 'annotation_coverage': .2},
                {'index': 2, 'annotated_macro_dice': 1., 'annotation_coverage': .95}]
        self.assertEqual([r['index'] for r in select_examples(rows)], [0, 1, 2])
        self.assertEqual(len(select_examples(rows[:1])), 1)
