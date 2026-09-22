import json
import tempfile
import unittest
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from stroke_model.convert_reference_csv import convert_csv, existing_path, Resolver
from stroke_model.train_reference import read_manifest, StrokeDataset


class CSVConversionTests(unittest.TestCase):
    def test_legacy_csv_to_trainable_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for i in range(4):
                horizontal = np.zeros((16, 16), np.uint8)
                vertical = horizontal.copy()
                horizontal[3+i:5+i, 2:14] = 255
                vertical[2:14, 7:9] = 255
                masks = [horizontal, vertical]
                for j, mask in enumerate(masks):
                    Image.fromarray(255 - mask).save(root / f's{i}_{j}.png')
                Image.fromarray(255 - np.maximum(*masks)).save(root / f'i{i}.png')
                rows.append({'I_prep': f'i{i}.png', 'I_g': 'i0.png',
                             'I_G_strokes': str(['s0_0.png', 's0_1.png']),
                             'input_stroke_paths': str([f's{i}_0.png', f's{i}_1.png'])})
            pd.DataFrame(rows).to_csv(root / 'paths.csv', index=False)
            manifest = convert_csv(root / 'paths.csv', root / 'out', size=16)
            converted = read_manifest(manifest)
            self.assertEqual({r['split'] for r in converted}, {'train', 'val'})
            dataset = StrokeDataset(converted, size=16)
            _, _, _, labels = dataset[0]
            self.assertEqual(labels[3, 7].item(), -1)  # Crossing remains unknown.
            self.assertEqual(labels[3, 3].item(), 0)
            self.assertEqual(labels[10, 7].item(), 1)
            report = json.loads((root / 'out/conversion_report.json').read_text(encoding='utf-8'))
            self.assertGreater(report['rows'][0]['overlap_pixels'], 0)
            pd.DataFrame(rows[:1]).to_csv(root / 'single.csv', index=False)
            test_manifest = convert_csv(root / 'single.csv', root / 'test_out', size=16, split='test')
            test_rows = read_manifest(test_manifest)
            self.assertEqual(len(test_rows), 1)
            self.assertEqual(test_rows[0]['split'], 'test')

    def test_unicode_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual = Path(tmp) / unicodedata.normalize('NFD', '자율연구')
            actual.mkdir()
            (actual / 'paths.csv').write_text('test')
            requested = Path(tmp) / unicodedata.normalize('NFC', '자율연구') / 'paths.csv'
            self.assertEqual(existing_path(requested), actual / 'paths.csv')

    def test_ambiguous_filename_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('a', 'b'):
                (root / folder).mkdir()
                Image.fromarray(np.zeros((8, 8), np.uint8)).save(root / folder / 'same.png')
            with self.assertRaises(ValueError):
                Resolver(root / 'paths.csv', root)('/old/same.png')


if __name__ == '__main__':
    unittest.main()
