import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from stroke_model.convert_reference_csv import convert_csv
from stroke_model.train_independent import IndependentDataset
from stroke_model.train_reference import read_manifest


class TopologyCsvTests(unittest.TestCase):
    def test_actual_overlap_empty_slots_and_variant_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for group in range(2):
                ref = np.zeros((32, 32), np.uint8)
                ref[5+group:8+group, 4:25-group*3] = 255
                Image.fromarray(255-ref).save(root/f'ref{group}.png')
                for variant in range(2):
                    ink = np.roll(ref, variant, axis=0)
                    Image.fromarray(255-ink).save(root/f'input{group}_{variant}.png')
                    # Two coincident strokes and an absent third stroke: no unique-owner pixels.
                    for k, mask in enumerate((ink, ink, np.zeros_like(ink))):
                        Image.fromarray(255-mask).save(root/f'gt{group}_{variant}_{k}.png')
                    rows.append(dict(input_path=f'input{group}_{variant}.png', target_path=f'ref{group}.png',
                        target_stroke_paths=';'.join([f'ref{group}.png']*3),
                        input_stroke_paths=';'.join(f'gt{group}_{variant}_{k}.png' for k in range(3))))
            Image.fromarray(np.full((32, 32), 255, np.uint8)).save(root/'empty.png')
            bad = dict(rows[0]); bad['target_stroke_paths'] = 'ref0.png;empty.png;ref0.png'
            # An invalid reference must not remove valid rows that follow it.
            rows.insert(1, bad)
            pd.DataFrame(rows).to_csv(root/'path (1).csv', index=False)
            manifest = convert_csv(root/'path (1).csv', root/'prepared', size=32, independent=True)
            prepared = read_manifest(manifest)
            self.assertEqual(len(prepared), 4)
            sample = IndependentDataset(prepared, 32)[0]
            target = sample[3]
            self.assertGreater(float((target[0]*target[1]).sum()), 0)
            self.assertEqual(float(target[2].sum()), 0)
            self.assertTrue(bool((sample[4] == 1).all()))
            self.assertEqual(prepared[0]['split'], prepared[1]['split'])
            self.assertEqual(prepared[2]['split'], prepared[3]['split'])
            self.assertNotEqual(prepared[0]['split'], prepared[2]['split'])
            report = json.loads((manifest.parent/'conversion_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['supervision'], 'actual_input_multihot')
            self.assertEqual(report['skipped_count'], 1)
            self.assertEqual(report['skipped_rows'][0]['csv_row'], 2)
            skipped = pd.read_csv(manifest.parent/'skipped_samples.csv')
            self.assertEqual(skipped.iloc[0]['empty_reference_paths'], 'empty.png')


if __name__ == '__main__':
    unittest.main()
