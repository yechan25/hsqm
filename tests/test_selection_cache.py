from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from repair_audit.selection_cache import prepare_selection_cache,selection_thumbnails
from repair_audit.selectable_target_inputs import save_prototype,load_prototype,prepare_selected_sheet
from repair_audit.stage1 import load_conditions


class SelectionPreparationTests(unittest.TestCase):
    def fixture(self,root):
        catalog=[]
        for i in range(2):
            ref=np.zeros((1,8,8),np.float32);ref[0,1+i:3+i,1:6]=1
            path=root/f'{i}.npz';np.savez(path,reference=ref,strokes=ref)
            catalog.append(dict(manifest_row=i+1,path=str(path),char_id='가'))
        return catalog

    def test_local_snapshot_reuse_without_source_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=self.fixture(root);progress=[]
            cached=prepare_selection_cache(catalog,root/'cache',source_fingerprint='v1',
                progress=lambda done,total:progress.append((done,total)))
            self.assertEqual(progress,[(1,2),(2,2)])
            for original,local in zip(catalog,cached):
                with np.load(original['path']) as a,np.load(local['path']) as b:
                    self.assertTrue(np.array_equal(a['reference'],b['reference']))
                    self.assertTrue(np.array_equal(a['strokes'],b['strokes']))
            with patch('repair_audit.selection_cache.load_catalog_reference',side_effect=AssertionError('source read')):
                self.assertEqual(prepare_selection_cache(catalog,root/'cache',source_fingerprint='v1'),cached)
            self.assertEqual(len(selection_thumbnails(cached)),2)
            for e in cached:
                with Image.open(e['thumbnail']) as image:self.assertEqual(image.size,(92,92))

    def test_editable_preset_does_not_change_previous_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=self.fixture(root);bank=root/'bank'
            first=save_prototype(catalog,1,'가',bank)
            before=prepare_selected_sheet(catalog,[2]*4,root/'experiment1',size=32,
                prototype_keys=['가']*4,prototype_bank=bank)[0]
            frozen=load_conditions(before['directory'])[0]['strokes'].copy()
            second=save_prototype(catalog,2,'가',bank)
            self.assertNotEqual(first['sha256'],second['sha256'])
            self.assertEqual(load_prototype(bank,'가')[1]['manifest_row'],2)
            self.assertTrue(np.array_equal(load_conditions(before['directory'])[0]['strokes'],frozen))
            after=prepare_selected_sheet(catalog,[2]*4,root/'experiment2',size=32,
                prototype_keys=['가']*4,prototype_bank=bank)[0]
            self.assertFalse(np.array_equal(load_conditions(after['directory'])[0]['strokes'],frozen))


if __name__=='__main__':unittest.main()
