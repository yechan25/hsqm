import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from stroke_model.reference_cache import prepare_cached, ensure_augmented, digest


def completed(folder, csv):
    folder.mkdir(parents=True)
    (folder / 'sample.npz').write_bytes(b'test')
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps([{'path': 'sample.npz'}]))
    (folder / 'conversion_report.json').write_text(json.dumps({'csv': str(csv), 'rows': [{}]}))
    return manifest


class CacheTests(unittest.TestCase):
    def test_adopt_existing_conversion_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / 'paths.csv'
            csv.write_text('image\na.png\n')
            manifest = completed(root / 'cache/old', csv)
            with patch('stroke_model.reference_cache.convert_csv') as convert:
                self.assertEqual(prepare_cached(csv, root / 'cache'), manifest)
                self.assertEqual(prepare_cached(csv, root / 'cache'), manifest)
                convert.assert_not_called()

    def test_incomplete_cache_not_reused_and_csv_change_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / 'paths.csv'
            csv.write_text('image\na.png\n')
            def fake_convert(csv, folder, **kwargs):
                return completed(folder, csv)
            with patch('stroke_model.reference_cache.convert_csv', side_effect=fake_convert) as convert:
                manifest = prepare_cached(csv, root / 'cache')
                (manifest.parent / 'sample.npz').unlink()
                second = prepare_cached(csv, root / 'cache')
                self.assertNotEqual(manifest, second)
                csv.write_text('image\nb.png\n')
                self.assertNotEqual(second, prepare_cached(csv, root / 'cache'))
                self.assertEqual(convert.call_count, 3)

    def test_augmentation_reuses_only_matching_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            manifest.write_text('[]')
            out = completed(root / 'augmented_old', root / 'paths.csv')
            (out.parent / 'cache.json').write_text(json.dumps({
                'manifest_sha256': digest(manifest), 'variants': 20, 'seed': 42, 'size': 128, 'version': 1}))
            with patch('stroke_model.reference_cache.subprocess.run') as run:
                self.assertEqual(ensure_augmented(manifest), out)
                run.assert_not_called()
            with patch('stroke_model.reference_cache.subprocess.run', side_effect=RuntimeError('regenerate')):
                with self.assertRaisesRegex(RuntimeError, 'regenerate'):
                    ensure_augmented(manifest, variants=5)


if __name__ == '__main__':
    unittest.main()
