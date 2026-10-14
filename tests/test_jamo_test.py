import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

import test_prepare_jamo as preparation_tests
from stroke_model.prepare_jamo import scan_jamo, read_components
from stroke_model.test_jamo import (generate_test, training_inventory, select_test_fonts,
                                   segmentation_metrics, reference_baseline, evaluate_test,
                                   run_jamo_test, save_training_inventory)


class JamoGeneralizationTests(unittest.TestCase):
    def fixture(self, root):
        return preparation_tests.JamoPreparationTests().prepare_fixture(root)

    def banks(self, source):
        _, fonts = scan_jamo(source)
        paths = sorted({p for classes in fonts.values() for slots in classes.values()
                        for bank in slots.values() for candidates in bank.values() for p in candidates})
        return fonts, read_components(paths, size=24, workers=2)

    def test_metrics_ignore_unknown_and_dice_equals_f1(self):
        labels = np.array([[0, 0, 0, 0, -1], [1, 1, 1, 1, -1]])
        pred = np.array([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
        scores, counts = segmentation_metrics(pred, labels, 2)
        np.testing.assert_allclose(scores['dice'], [.75, .75])
        np.testing.assert_allclose(scores['f1'], scores['dice'])
        np.testing.assert_allclose(scores['iou'], [.6, .6])
        self.assertEqual(counts, dict(correct=6, annotated=8, strokes=2))

    def test_validation_fonts_are_not_true_unseen(self):
        provenance = dict(train_fonts=['a', 'b'], validation_fonts=['c'])
        seen, unseen = select_test_fonts({'a': {}, 'b': {}, 'c': {}, 'd': {}}, provenance)
        self.assertEqual(seen, ['a', 'b'])
        self.assertEqual(unseen, ['d'])

    def test_fresh_transforms_reject_identical_input_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            fonts, loaded = self.banks(source)
            first, _ = generate_test(fonts, loaded, ['ref'], root / 'first', 100, 'standard', 4)
            first_rows = json.loads(first.read_text(encoding='utf-8'))
            hashes = {r['image_sha256'] for r in first_rows}
            second, _ = generate_test(fonts, loaded, ['ref'], root / 'second', 100, 'standard', 4, hashes)
            second_rows = json.loads(second.read_text(encoding='utf-8'))
            self.assertTrue(hashes.isdisjoint({r['image_sha256'] for r in second_rows}))
            for row in second_rows:
                self.assertEqual(row['reference_font'], row['font'])
                self.assertEqual(row['split'], 'test')
            # Exercise actual model inference/evaluation using both two and three jamo.
            import torch
            from test_reference_model import TinyBackbone
            from stroke_model.reference_model import ReferenceStrokeModel
            torch.set_num_threads(2)
            model = ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16])
            result = evaluate_test(model, second, torch.device('cpu'), size=24, batch_size=3)
            self.assertEqual(result['overall']['samples'], len(second_rows))
            self.assertTrue(0 <= result['overall']['macro_dice_annotated'] <= 1)
            self.assertEqual(result['overall']['macro_dice_annotated'], result['overall']['macro_f1_annotated'])
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_recover_archive_provenance_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            manifest = preparation_tests.JamoPreparationTests().build(root, source)
            archive = next((root / 'drive_cache').glob('*.zip'))
            checkpoint = root / 'best.pt'
            checkpoint.write_bytes(b'fixture checkpoint; weights unused by provenance')
            provenance = training_inventory(checkpoint, archive, root / 'test_local')
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(set(provenance['train_fonts']) | set(provenance['validation_fonts']), {'ref', 'target_a', 'target_b'})
            self.assertEqual(len(provenance['fingerprints']), len(rows))

    def test_reference_baseline_exact_original_masks(self):
        masks = np.zeros((2, 16, 16), np.float32)
        masks[0, 2:5, 2:5] = 1
        masks[1, 10:14, 10:14] = 1
        labels = np.full((16, 16), -1, np.int64)
        labels[masks[0] > .5] = 0
        labels[masks[1] > .5] = 1
        scores, _ = segmentation_metrics(reference_baseline(masks), labels, 2)
        self.assertEqual(scores['dice'], [1., 1.])

    def test_complete_runner_and_saved_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            import torch
            from test_reference_model import TinyBackbone
            from stroke_model.reference_model import ReferenceStrokeModel
            torch.set_num_threads(2)
            root = Path(tmp)
            source = self.fixture(root)
            manifest = preparation_tests.JamoPreparationTests().build(root, source)
            archive = next((root / 'drive_cache').glob('*.zip'))
            run = root / 'run'
            run.mkdir()
            save_training_inventory(manifest, run, archive, dict(seed=42, source=str(source)))
            self.assertTrue((run / 'training_inventory.json').is_file())
            model = ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16])
            checkpoint = run / 'best.pt'
            torch.save(dict(model=model.state_dict(), width=8, image_size=24, epoch=16), checkpoint)
            extra = root / 'extra'
            shutil.copytree(source / 'ref_unique_images', extra / 'new_unique_images')
            def create(**kwargs):
                return ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16])
            with patch('stroke_model.test_jamo.ReferenceStrokeModel', side_effect=create):
                summary, output = run_jamo_test(checkpoint, source_root=source, new_font_root=extra,
                                                local_root=root / 'test_local', samples_per_class=2,
                                                preview_count=0, workers=2, max_overlap=.05)
            self.assertEqual(set(summary['tests']), {'seen_standard', 'seen_strong', 'unseen_standard', 'unseen_strong'})
            self.assertTrue(all(r['status'] == 'evaluated' for r in summary['tests'].values()))
            self.assertEqual(summary['tests']['unseen_standard']['fonts'], ['new'])
            self.assertTrue((output / 'summary.json').is_file())
            self.assertTrue((output / 'seen_standard' / 'dataset.zip').is_file())


if __name__ == '__main__':
    unittest.main()
