import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from diffusion_model.pilot_data import prepare_pilot_data
from diffusion_model.score_pilot import training_catalog
from stroke_model.prepare_stroke_zips import ARCHIVES, prepare_stroke_zips


class PilotDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'training_archives'
        self.source.mkdir()
        original = self.root/'original_images'
        original.mkdir()
        for name in ARCHIVES:
            (original/Path(name).stem).mkdir()
        rows = []
        for i in range(5):
            masks = np.zeros((2, 16, 16), np.uint8)
            masks[0, 2+i:4+i, 2:14] = 255
            masks[1, 2:14, 8:10] = 255
            for folder in ('input', 'target'):
                Image.fromarray(255-masks.max(0)).save(original/folder/f'{i}.png')
            for folder in ('input_stroke', 'target_stroke'):
                for k, mask in enumerate(masks):
                    Image.fromarray(255-mask).save(original/folder/f'{i}_{k}.png')
            old = '/content/deleted_old_session/'
            rows.append(dict(input_path=old+f'input/{i}.png', target_path=old+f'target/{i}.png',
                             input_stroke_paths=str([old+f'input_stroke/{i}_{k}.png' for k in range(2)]),
                             target_stroke_paths=str([old+f'target_stroke/{i}_{k}.png' for k in range(2)])))
        for name in ARCHIVES:
            with zipfile.ZipFile(self.source/name, 'w') as archive:
                for image in (original/Path(name).stem).iterdir():
                    archive.write(image, image.relative_to(original))
        pd.DataFrame(rows).to_csv(self.source/'path (1).csv', index=False)
        self.old_manifest = prepare_stroke_zips(self.source, self.root/'old_cache',
            csv_path=self.source/'path (1).csv', size=16, seed=7, independent=True)
        self.run = self.root/'run'
        self.run.mkdir()
        self.checkpoint = self.run/'best.pt'
        self.checkpoint.write_bytes(b'untouched trained checkpoint placeholder')
        for name in ('conversion_report.json', 'selected_paths.csv'):
            shutil.copyfile(self.old_manifest.parent/name, self.run/name)
        self.config = dict(manifest='/content/deleted_old_session/manifest.json', image_size=16, seed=7)
        (self.run/'run_config.json').write_text(json.dumps(self.config), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_rebuilds_missing_manifest_with_original_split_and_geometry(self):
        before = self.checkpoint.read_bytes()
        result = prepare_pilot_data(self.checkpoint, self.source, local_root=self.root/'new_cache')
        self.assertEqual(result['mode'], 'rebuilt_from_training_archives')
        manifest = Path(result['manifest'])
        old, new = [json.loads(p.read_text()) for p in (self.old_manifest, manifest)]
        self.assertEqual([(r['group_id'], r['split']) for r in old],
                         [(r['group_id'], r['split']) for r in new])
        for left, right in zip(old, new):
            with np.load(left['path']) as a, np.load(right['path']) as b:
                for key in ('reference', 'strokes'):
                    np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(training_catalog(manifest))
        again = prepare_pilot_data(self.checkpoint, self.source, local_root=self.root/'new_cache')
        self.assertEqual(again['manifest'], result['manifest'])
        self.assertEqual(self.checkpoint.read_bytes(), before)

    def test_live_manifest_reuses_data_without_original_archives(self):
        result = prepare_pilot_data(self.checkpoint, self.root/'missing_source', manifest=self.old_manifest)
        self.assertEqual(result['mode'], 'reuse_live_training_data')

    def test_changed_csv_is_rejected_before_rebuild(self):
        changed = self.root/'changed.csv'
        changed.write_text('different CSV')
        with self.assertRaisesRegex(ValueError, 'CSV_PATH'):
            prepare_pilot_data(self.checkpoint, self.source, csv_path=changed,
                               local_root=self.root/'never_created')
        self.assertFalse((self.root/'never_created').exists())


if __name__ == '__main__':
    unittest.main()
