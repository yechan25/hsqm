"""Exercise the notebook selection flow without a Colab/network dependency."""
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import torch
from notebooks.build_structural_indicator import RUN, SETTINGS
import stroke_model.structural_indicator as scoring


class CsvNotebookTests(unittest.TestCase):
    def namespace(self, root):
        ns = {}
        exec(SETTINGS, ns)
        ns.update(MODE='csv', CSV_PATH=str(root/'paths.csv'), SHOW_IMAGES=False,
                  SAVE_DIR=str(root/'scores'), structure_results={'stale': True})
        return ns

    def test_explicit_png_is_newly_inferred_and_shown_with_its_score(self):
        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, image, reference, strokes):
                return {'masks': image.expand_as(strokes), 'hints': strokes}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ink = np.zeros((64, 64), dtype=np.float32)
            ink[8:56, 29:34] = 1
            for name in ('reference.png', 'first.png', 'other.png'):
                Image.fromarray((255*(1-ink)).astype(np.uint8)).save(root/name)
            pd.DataFrame([dict(input_path=name, target_path='reference.png',
                               target_stroke_paths='reference.png')
                          for name in ('first.png', 'other.png')]).to_csv(root/'paths.csv', index=False)
            torch.save({'model': {}, 'image_size': 64, 'width': 4}, root/'best.pt')
            ns = self.namespace(root)
            ns.update(CHECKPOINT=str(root/'best.pt'), INPUT_FILES=['other.png'])
            with patch.dict(sys.modules, {'_structural_indicator_runtime': scoring}), \
                 patch('stroke_model.reference_model.ReferenceStrokeModel', FakeModel):
                exec(RUN, ns)
            result = ns['indicator_results']['results']
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['csv_row'], 2)
            self.assertTrue(result[0]['input'].endswith('other.png'))
            self.assertTrue(result[0]['image_available'])
            self.assertEqual(result[0]['image_kind'], 'scored_input_array')
            self.assertTrue(Path(ns['indicator_results']['gallery_html']).is_file())

    def test_random_selection_is_reproducible_and_all_mode_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([dict(input_path=f'{i}.png', target_path='r.png', target_stroke_paths='r.png')
                          for i in range(30)]).to_csv(root/'paths.csv', index=False)
            selections = []
            for seed, count in ((42, 4), (42, 4), (43, 4), (42, None)):
                ns = self.namespace(root)
                ns.update(RANDOM_SEED=seed, SAMPLE_COUNT=count)
                with patch.dict(sys.modules, {'_structural_indicator_runtime': scoring}), \
                     patch('stroke_model.score_stroke_structure.score_structure_csv', return_value={'new': True}) as infer, \
                     patch.object(scoring, 'score_structure_report', return_value={}) as score:
                    exec(RUN, ns)
                selections.append(infer.call_args.kwargs['sample_rows'])
                self.assertEqual(score.call_args.args[0], {'new': True})
            self.assertEqual(selections[0], selections[1])
            self.assertNotEqual(selections[0], selections[2])
            self.assertEqual(len(selections[0]), 4)
            self.assertEqual(selections[3], list(range(1, 31)))

    def test_invalid_filename_does_not_fall_back_to_old_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([dict(input_path='real.png', target_path='r.png',
                               target_stroke_paths='r.png')]).to_csv(root/'paths.csv', index=False)
            ns = self.namespace(root)
            ns['INPUT_FILES'] = ['missing.png']
            with patch.dict(sys.modules, {'_structural_indicator_runtime': scoring}), \
                 self.assertRaisesRegex(ValueError, 'not found'):
                exec(RUN, ns)


if __name__ == '__main__':
    unittest.main()
