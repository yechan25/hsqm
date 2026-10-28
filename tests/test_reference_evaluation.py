import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stroke_model.test_reference import run_test, run_manifest, run_zip_validation, select_preview_indices


class OneStrokeModel(torch.nn.Module):
    def forward(self, image, reference, strokes):
        return {'probabilities': torch.ones_like(strokes)}


class TestEvaluationTests(unittest.TestCase):
    def test_preview_diversity_and_character_filter(self):
        rows = [{'char_id': '가'} for _ in range(20)] + [{'char_id': '나'}, {'source_image': '/input/다_001.png'}]
        selected = select_preview_indices(rows, count=3)
        self.assertEqual(len(selected), 3)
        self.assertTrue(20 in selected and 21 in selected)
        self.assertEqual(select_preview_indices(rows, count=10, characters=['나']), [20])
        self.assertEqual(selected, select_preview_indices(rows, count=3))
        with self.assertRaises(ValueError):
            select_preview_indices(rows, characters=['한'])
        unknown = [{'source_image': f'{i}.png'} for i in range(100)]
        chosen = select_preview_indices(unknown, count=5)
        self.assertEqual(len(set(chosen)), 5)
        self.assertNotEqual(chosen, list(range(5)))

    def test_validation_filters_training_rows_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'best.pt'
            torch.save({'model': {}, 'image_size': 16, 'width': 8, 'epoch': 2}, checkpoint)
            np.savez(root / 'val.npz', image=np.ones((1,16,16), np.float32),
                     reference=np.ones((1,16,16), np.float32), strokes=np.ones((1,16,16), np.float32),
                     labels=np.zeros((16,16), np.int64))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([
                {'path': 'missing_train.npz', 'split': 'train', 'group_id': 'train'},
                {'path': 'val.npz', 'split': 'val', 'group_id': 'val'}]))
            (root / 'conversion_report.json').write_text(json.dumps({'rows': [
                {'foreground_pixels': 9999, 'overlap_pixels': 9999, 'missing_pixels': 0},
                {'foreground_pixels': 256, 'overlap_pixels': 0, 'missing_pixels': 0}]}))
            with patch('stroke_model.test_reference.ReferenceStrokeModel', side_effect=lambda **kw: OneStrokeModel()), \
                 patch('stroke_model.test_reference.preview_reference') as preview:
                metrics = run_manifest(checkpoint, manifest)
            self.assertEqual(metrics['val_samples'], 1)
            self.assertEqual(metrics['val_macro_dice_annotated'], 1.)
            self.assertEqual(metrics['annotation_coverage'], 1.)
            self.assertEqual(preview.call_args.kwargs['split'], 'val')

    def test_zip_validation_uses_saved_csv_and_seed_and_rejects_changed_archives(self):
        from stroke_model.prepare_stroke_zips import ARCHIVES
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'best.pt'
            checkpoint.touch()
            (root / 'run_config.json').write_text(json.dumps({'seed': 17, 'image_size': 16}))
            saved = []
            for name in ARCHIVES:
                p = root / name
                p.write_bytes(b'archive')
                saved.append([str(p), p.stat().st_size, p.stat().st_mtime_ns])
            (root / 'complete.json').write_text(json.dumps({'archives': saved}))
            with patch('stroke_model.prepare_stroke_zips.prepare_stroke_zips', return_value=root / 'manifest.json') as prepare, \
                 patch('stroke_model.test_reference.run_manifest', return_value={'ok': True}) as evaluate:
                self.assertEqual(run_zip_validation(checkpoint, root), {'ok': True})
                self.assertEqual(prepare.call_args.kwargs['csv_path'], root / 'selected_paths.csv')
                self.assertEqual(prepare.call_args.kwargs['seed'], 17)
                self.assertEqual(evaluate.call_args.kwargs['split'], 'val')
                (root / ARCHIVES[0]).write_bytes(b'changed archive')
                with self.assertRaises(ValueError):
                    run_zip_validation(checkpoint, root)

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
