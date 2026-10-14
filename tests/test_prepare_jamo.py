import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from stroke_model.prepare_jamo import prepare_jamo, scan_jamo


class JamoPreparationTests(unittest.TestCase):
    def make_font(self, source, name, offset=0):
        folder = source / (name + '_unique_images')
        folder.mkdir(parents=True)
        for class_id, char in ((0, '가'), (3, '각')):
            for slot in range(3 if char == '각' else 2):
                ink = np.zeros((24, 24), np.uint8)
                if slot == 0:
                    ink[2:6, 2+offset:12+offset] = 255
                elif slot == 1:
                    ink[3:17, 10+offset:14+offset] = 255
                else:
                    ink[19:23, 3:19] = 255
                Image.fromarray(255 - ink).save(folder / f'{class_id}_{slot}_{char}.png')
        return folder

    def test_union_thickness_overlap_correspondence_and_font_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            for i, font in enumerate(('ref', 'target_a', 'target_b')):
                self.make_font(source, font, i)
            manifest = prepare_jamo(source, root / 'cache', reference_font='ref', size=24)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 4)
            train = {r['font'] for r in rows if r['split'] == 'train'}
            val = {r['font'] for r in rows if r['split'] == 'val'}
            self.assertTrue(train and val and train.isdisjoint(val))
            self.assertNotIn('ref', train | val)
            for row in rows:
                with np.load(row['path']) as sample:
                    k = 3 if row['char_id'] == '각' else 2
                    self.assertEqual(sample['strokes'].shape, (k, 24, 24))
                    self.assertEqual(sample['strokes'][0].sum(), 40)
                    np.testing.assert_array_equal(sample['reference'], sample['strokes'].max(0, keepdims=True))
                    self.assertEqual(set(np.unique(sample['labels'])), {-1, *range(k)})
                    offset = 1 if row['font'] == 'target_a' else 2
                    self.assertEqual(sample['labels'][3, 10+offset], -1)
                    self.assertEqual(sample['labels'][3, 3+offset], 0)
                    self.assertEqual(sample['labels'][12, 11+offset], 1)
            self.assertEqual(manifest, prepare_jamo(source, root / 'cache', reference_font='ref', size=24))
            rebuilt = prepare_jamo(source, root / 'cache', reference_font='ref', size=24, rebuild=True)
            self.assertNotEqual(manifest, rebuilt)

            # Exercise the real loader, mixed K padding, model head and training loss.
            import torch
            from test_reference_model import TinyBackbone
            from stroke_model.train_reference import read_manifest, StrokeDataset, collate
            from stroke_model.reference_model import ReferenceStrokeModel, partition_loss
            torch.set_num_threads(2)
            dataset = StrokeDataset(read_manifest(manifest), size=24)
            image, reference, strokes, labels = collate([dataset[i] for i in range(len(dataset))])
            model = ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16])
            prediction = model(image, reference, strokes)
            loss = partition_loss(prediction, labels)['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None for p in model.head.parameters()))
            torch.testing.assert_close(prediction['masks'].sum(1, keepdim=True), image)

    def test_two_fonts_use_character_split_and_unicode_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            folder = self.make_font(source, 'ref')
            self.make_font(source, 'target', 1)
            for path in list(folder.iterdir()):
                path.rename(path.with_name(unicodedata.normalize('NFD', path.name)))
            manifest = prepare_jamo(source, root / 'cache', reference_font='ref', size=24)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual({r['split'] for r in rows}, {'train', 'val'})
            self.assertTrue(all(r['group_id'].startswith('char:') for r in rows))

    def test_incomplete_bundle_is_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            folder = self.make_font(source, 'ref')
            self.make_font(source, 'target')
            scan_jamo(source)
            self.make_font(source, 'target_b', 2)
            (folder / '3_2_각.png').unlink()
            _, fonts, _ = scan_jamo(source)
            self.assertEqual(fonts['ref'][(3, '각')]['missing_slots'], [2])
            manifest = prepare_jamo(source, source / 'cache', reference_font='target', size=24)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertFalse(any(r['font'] == 'ref' and r['char_id'] == '각' for r in rows))
            report = json.loads((manifest.parent / 'conversion_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['skipped_incomplete'][0]['missing_slots'], [2])

    def test_same_class_has_multiple_characters_and_distinct_bundle_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            for font in ('ref', 'target', 'target_b'):
                folder = self.make_font(source, font)
                for slot in (0, 1):
                    (folder / f'0_{slot}_까.png').write_bytes((folder / f'0_{slot}_가.png').read_bytes())
            manifest = prepare_jamo(source, source / 'cache', reference_font='ref', size=24)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual({(r['class_id'], r['char_id']) for r in rows}, {(0, '가'), (0, '까'), (3, '각')})
            self.assertEqual(len(rows), 6)

    def test_class_range_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            folder = self.make_font(source, 'ref')
            self.make_font(source, 'target')
            (folder / '0_0_가.png').rename(folder / '6_0_가.png')
            with self.assertRaisesRegex(ValueError, '0..5'):
                scan_jamo(source)

    def test_mismatched_canvas_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            folder = self.make_font(source, 'ref')
            self.make_font(source, 'target')
            Image.new('L', (12, 12), 255).save(folder / '0_1_가.png')
            with self.assertRaisesRegex(ValueError, 'canvas'):
                prepare_jamo(source, root / 'cache', reference_font='ref', size=24)


if __name__ == '__main__':
    unittest.main()
