import contextlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from stroke_model.prepare_stroke_zips import discover_archives, extract_checked, prepare_stroke_zips
from stroke_model.train_independent import IndependentDataset
from stroke_model.train_reference import read_manifest


def png(mask):
    stream = io.BytesIO()
    Image.fromarray(255-mask).save(stream, format='PNG')
    return stream.getvalue()


class StrokeTarTests(unittest.TestCase):
    def test_six_tars_per_folder_to_topology_data_and_cached_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'dataset'
            source.mkdir()
            rows = []
            for index in range(6):
                a = np.zeros((32, 32), np.uint8)
                b = a.copy()
                a[5+index:8+index, 3:28] = 255
                b[3:28, 15:18] = 255
                image, reference = f'I_{index}.png', f'R_{index}.png'
                gt = [f'I_{index}_{k}.png' for k in range(2)]
                strokes = [f'R_{index}_{k}.png' for k in range(2)]
                contents = {'input': [(image, np.maximum(a,b))],
                            'target': [(reference, np.maximum(a,b))],
                            'input_stroke': list(zip(gt, (a,b))),
                            'target_stroke': list(zip(strokes, (a,b)))}
                for role, files in contents.items():
                    folder = source/role
                    folder.mkdir(exist_ok=True)
                    with tarfile.open(folder/f'{role}_({index}).tar', 'w') as archive:
                        for filename, mask in files:
                            data = png(mask)
                            # Support both bare filenames and nested folders inside TARs.
                            member = tarfile.TarInfo(f'{role}/{filename}' if index % 2 else filename)
                            member.size = len(data)
                            archive.addfile(member, io.BytesIO(data))
                rows.append(dict(input_path=image, target_path=reference, total_paths=2,
                    input_stroke_paths=';'.join(gt), target_stroke_paths=';'.join(strokes)))
            csv = source/'path (1).csv'
            pd.DataFrame(rows).to_csv(csv, index=False)
            paths = discover_archives(source)
            self.assertEqual(len(paths), 24)
            self.assertEqual([p.name for p in paths[:6]], [f'input_stroke_({i}).tar' for i in range(6)])
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare_stroke_zips(source, root/'cache', csv_path=csv, size=32, independent=True)
                cached = prepare_stroke_zips(source, root/'cache', csv_path=csv, size=32, independent=True)
            self.assertEqual(manifest, cached)
            data = read_manifest(manifest)
            self.assertEqual(len(data), 6)
            self.assertEqual({r['split'] for r in data}, {'train', 'val'})
            sample = IndependentDataset(data, 32)[0]
            self.assertGreater(float((sample[3][0]*sample[3][1]).sum()), 0)
            self.assertEqual(len(list((manifest.parent.parent/'raw').rglob('.complete.json'))), 24)
            signature = json.loads((manifest.parent/'.complete.json').read_text())
            self.assertEqual(len(signature['archives']), 24)

    def test_missing_number_is_reported_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'input_stroke'
            folder.mkdir()
            for index in (0,2):
                with tarfile.open(folder/f'input_stroke_({index}).tar', 'w'):
                    pass
            with self.assertRaisesRegex(ValueError, 'TAR'):
                discover_archives(root)

    def test_tar_traversal_and_links_rejected_before_extracting(self):
        for unsafe in ('../escape.png', 'C:/escape.png', '/escape.png', 'link'):
            with self.subTest(member=unsafe), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root/'unsafe.tar'
                with tarfile.open(archive, 'w') as tar:
                    good = tarfile.TarInfo('good.png'); good.size = 1
                    tar.addfile(good, io.BytesIO(b'x'))
                    bad = tarfile.TarInfo(unsafe)
                    if unsafe == 'link':
                        bad.type = tarfile.SYMTYPE; bad.linkname = '../outside'
                    tar.addfile(bad)
                with self.assertRaisesRegex(ValueError, 'Unsafe TAR'):
                    extract_checked(archive, root/'raw')
                self.assertFalse((root/'raw/good.png').exists())


if __name__ == '__main__':
    unittest.main()
