import json
import tempfile
import unittest
import unicodedata
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from stroke_model.prepare_jamo import prepare_jamo, scan_jamo, make_labels, decompose, compose


class JamoPreparationTests(unittest.TestCase):
    def make_font(self, source, name):
        folder = source / (name + '_unique_images')
        folder.mkdir(parents=True)
        # No character has a full set of source files: every slot is shared.
        names = ('0_0_가.png', '0_0_까.png', '0_1_아.png', '0_1_애.png',
                 '3_0_각.png', '3_1_난.png', '3_2_긱.png')
        for name in names:
            slot = int(name.split('_')[1])
            ink = np.zeros((24, 24), np.uint8)
            if slot == 0:
                ink[2:5, 2:8] = 255
            elif slot == 1:
                ink[8:15, 16:19] = 255
            else:
                ink[19:22, 4:13] = 255
            Image.fromarray(255 - ink).convert('RGBA').save(folder / name)
        return folder

    def prepare_fixture(self, root):
        source = root / 'source'
        for font in ('ref', 'target_a', 'target_b'):
            self.make_font(source, font)
        return source

    def build(self, root, source, **kwargs):
        return prepare_jamo(source, root / 'drive_cache', reference_font='ref', size=24,
                            local_root=root / 'local', samples_per_class=4, workers=2, **kwargs)

    def test_shared_semantic_composition_font_split_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.prepare_fixture(root)
            manifest = self.build(root, source)
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 10)
            self.assertEqual({r['char_id'] for r in rows}, {'가', '개', '까', '깨', '각'})
            train = {r['font'] for r in rows if r['split'] == 'train'}
            val = {r['font'] for r in rows if r['split'] == 'val'}
            self.assertTrue(train and val and train.isdisjoint(val))
            for row in rows:
                self.assertEqual(decompose(row['char_id']), tuple(row['jamo_indices']))
                self.assertFalse(Path(row['path']).is_absolute())
                with np.load(manifest.parent / row['path']) as sample:
                    k = 3 if row['char_id'] == '각' else 2
                    self.assertEqual(sample['strokes'].shape, (k, 24, 24))
                    self.assertEqual(sample['strokes'][0].sum(), 18)
                    np.testing.assert_array_equal(sample['reference'], sample['strokes'].max(0, keepdims=True))
                    self.assertEqual(set(np.unique(sample['labels'])), {-1, *range(k)})
            ga = next(r for r in rows if r['char_id'] == '가')
            self.assertEqual(ga['target_components'], ['0_0_가.png', '0_1_아.png'])
            # Verify archive publication and that hot reuse does not rescan/open Drive sources.
            self.assertEqual(len(list((root / 'drive_cache').glob('*.zip'))), 1)
            with patch('stroke_model.prepare_jamo.scan_jamo', side_effect=AssertionError('rescan')):
                self.assertEqual(manifest, self.build(root, source))

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

    def test_zip_restore_relative_paths_and_deterministic_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.prepare_fixture(root)
            first = self.build(root, source)
            # Restore to a different runtime directory without reopening source images.
            with patch('stroke_model.prepare_jamo.scan_jamo', side_effect=AssertionError('rescan')):
                restored = prepare_jamo(source, root / 'drive_cache', reference_font='ref', size=24,
                                        local_root=root / 'new_runtime', samples_per_class=4, workers=2)
            rows = json.loads(restored.read_text(encoding='utf-8'))
            self.assertTrue(all((restored.parent / r['path']).is_file() for r in rows))
            second = self.build(root, source, rebuild=True)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            for row in rows:
                with np.load(first.parent / row['path']) as a, np.load(second.parent / row['path']) as b:
                    np.testing.assert_array_equal(a['image'], b['image'])
                    np.testing.assert_array_equal(a['labels'], b['labels'])

    def test_class_semantic_decomposition_and_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.prepare_fixture(Path(tmp))
            folder = source / 'ref_unique_images'
            for path in list(folder.iterdir()):
                path.rename(path.with_name(unicodedata.normalize('NFD', path.name)))
            _, fonts = scan_jamo(source)
            self.assertEqual(set(fonts['ref'][0][0]), {0, 1})
            self.assertEqual(set(fonts['ref'][0][1]), {0, 1})
            self.assertEqual(set(fonts['ref'][3][2]), {1})
            self.assertEqual(compose(decompose('각')), '각')

    def test_pairwise_overlap_of_three_masks_is_detected(self):
        masks = np.zeros((3, 16, 16), np.float32)
        masks[0, 2:6, 2:6] = 1
        masks[1, 3:7, 3:7] = 1
        masks[2, 10:14, 10:14] = 1
        labels, overlap, valid = make_labels(masks, .001)
        self.assertFalse(valid)
        self.assertGreater(overlap, 0)
        self.assertEqual(labels[3, 3], -1)
        self.assertEqual(labels[11, 11], 2)

    def test_invalid_class_range_and_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.prepare_fixture(Path(tmp))
            folder = source / 'ref_unique_images'
            original = folder / '0_0_가.png'
            bad = folder / '6_0_가.png'
            original.rename(bad)
            with self.assertRaisesRegex(ValueError, '0..5'):
                scan_jamo(source)
            bad.rename(folder / '0_2_가.png')
            with self.assertRaisesRegex(ValueError, 'Class/slot'):
                scan_jamo(source)


if __name__ == '__main__':
    unittest.main()
