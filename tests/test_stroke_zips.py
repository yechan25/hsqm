import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from stroke_model.prepare_stroke_zips import ARCHIVES, extract_checked, prepare_stroke_zips
from stroke_model.train_reference import StrokeDataset, read_manifest


class StrokeZipTests(unittest.TestCase):
    def test_four_zips_old_absolute_csv_paths_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, staging = root / 'drive', root / 'original'
            source.mkdir()
            staging.mkdir()
            for name in ARCHIVES:
                (staging / Path(name).stem).mkdir()
            rows = []
            for i in range(4):
                a = np.zeros((16, 16), dtype=np.uint8)
                b = a.copy()
                a[3+i:5+i, 2:14] = 255
                b[2:14, 8:10] = 255
                for folder in ('input', 'target'):
                    Image.fromarray(255 - np.maximum(a, b)).save(staging / folder / f'{i}.png')
                for folder in ('input_stroke', 'target_stroke'):
                    for k, image in enumerate((a, b)):
                        Image.fromarray(255 - image).save(staging / folder / f'{i}_{k}.png')
                old = '/content/drive/old_dataset/'
                rows.append(dict(input_path=old + f'input/{i}.png', target_path=old + f'target/{i}.png',
                                 input_stroke_paths=str([old + f'input_stroke/{i}_{k}.png' for k in range(2)]),
                                 target_stroke_paths=str([old + f'target_stroke/{i}_{k}.png' for k in range(2)])))
            for name in ARCHIVES:
                folder = staging / Path(name).stem
                with zipfile.ZipFile(source / name, 'w') as z:
                    for p in folder.iterdir():
                        z.write(p, str(p.relative_to(staging)))
            # Archives must survive a missing CSV and be reused after it arrives.
            with self.assertRaises(FileNotFoundError):
                prepare_stroke_zips(source, root / 'cache', size=16)
            pd.DataFrame(rows).to_csv(source / 'path.csv', index=False)
            manifest = prepare_stroke_zips(source, root / 'cache', size=16)
            data = read_manifest(manifest)
            self.assertEqual({r['split'] for r in data}, {'train', 'val'})
            self.assertTrue(all(Path(r['path']).is_relative_to(root / 'cache') for r in data))
            self.assertEqual(StrokeDataset(data, size=16)[0][2].shape, (2, 16, 16))
            self.assertEqual(manifest, prepare_stroke_zips(source, root / 'cache', size=16))
            (source / 'paths.csv').write_text('duplicate')
            with self.assertRaises(ValueError):
                prepare_stroke_zips(source, root / 'cache', size=16)

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / 'bad.zip', 'w') as z:
                z.writestr('../escape.png', b'bad')
            with self.assertRaises(ValueError):
                extract_checked(root / 'bad.zip', root / 'raw')
            self.assertFalse((root / 'escape.png').exists())

    def test_local_resolver_ignores_existing_external_images(self):
        from stroke_model.convert_reference_csv import Resolver
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            external = root / 'old/input/a.png'
            external.parent.mkdir(parents=True)
            external.write_bytes(b'external')
            raw = root / 'raw'
            local = raw / 'input/a.png'
            local.parent.mkdir(parents=True)
            local.write_bytes(b'local')
            self.assertEqual(Resolver(root / 'paths.csv', raw, local_only=True)(external), local)


if __name__ == '__main__':
    unittest.main()
