import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stroke_model.test_reference import run_test


class OneStrokeModel(torch.nn.Module):
    def forward(self, image, reference, strokes):
        return {'probabilities': torch.ones_like(strokes)}


class TestEvaluationTests(unittest.TestCase):
    def test_evaluation_uses_all_test_rows_and_reuses_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'best.pt'
            torch.save({'model': {}, 'image_size': 16, 'width': 8, 'epoch': 7}, checkpoint)
            csv = root / 'paths.csv'
            csv.write_text('dummy')
            def convert(csv, folder, size, split):
                self.assertEqual(split, 'test')
                folder.mkdir()
                labels = np.zeros((16, 16), np.int64)
                labels[0] = -1
                np.savez(folder / 'sample.npz', image=np.ones((1,16,16), np.float32),
                         reference=np.ones((1,16,16), np.float32), strokes=np.ones((1,16,16), np.float32), labels=labels)
                manifest = folder / 'manifest.json'
                manifest.write_text(json.dumps([{'path': 'sample.npz', 'split': 'test', 'group_id': 'one'}]))
                (folder / 'conversion_report.json').write_text(json.dumps({'rows': [{'foreground_pixels': 256, 'overlap_pixels': 16, 'missing_pixels': 0}]}))
                return manifest
            with patch('stroke_model.test_reference.convert_csv', side_effect=convert) as converter, \
                 patch('stroke_model.test_reference.ReferenceStrokeModel', side_effect=lambda **kw: OneStrokeModel()), \
                 patch('stroke_model.test_reference.preview_reference') as preview:
                result = run_test(checkpoint, csv)
                self.assertEqual(result['test_samples'], 1)
                self.assertEqual(result['test_macro_dice_annotated'], 1.)
                self.assertEqual(result['annotation_coverage'], 240/256)
                self.assertEqual(result['checkpoint_epoch'], 7)
                self.assertEqual(preview.call_args.kwargs['split'], 'test')
                run_test(checkpoint, csv)
                self.assertEqual(converter.call_count, 1)


if __name__ == '__main__':
    unittest.main()
