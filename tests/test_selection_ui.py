"""Real widget callbacks only; no model or diffusion execution."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
try:
    import ipywidgets as widgets
except ImportError:
    widgets=None

from diffusion_model.score_pilot import load_catalog_reference
from repair_audit.selectable_target_inputs import load_reference_image,load_prototype,save_prototype
from repair_audit.selection_cache import prepare_selection_cache,selection_thumbnails,reference_identity

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipIf(widgets is None,'ipywidgets is a Colab UI dependency')
class SelectionUITests(unittest.TestCase):
    def test_thumbnail_selection_paging_and_fixed_prototype_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=[]
            for i in range(12):
                reference=np.zeros((1,16,16),np.float32);reference[0,2+i%8:4+i%8,3:12]=1
                path=root/f'{i}.npz';np.savez(path,reference=reference,strokes=reference)
                catalog.append(dict(manifest_row=i+1,path=str(path),char_id=f'glyph{i+1}'))
            catalog=prepare_selection_cache(catalog,root/'cache',source_fingerprint='fixture')
            env=dict(CATALOG=catalog,SELECTION_THUMBNAILS=selection_thumbnails(catalog),
                PROTOTYPE_RECORDS={},PROTOTYPE_PREVIEWS={},PROTOTYPE_BANK=str(root/'bank'),Path=Path,json=json,
                widgets=widgets,np=np,Image=Image,display=lambda x:None,
                load_reference_image=load_reference_image,load_catalog_reference=load_catalog_reference,
                load_prototype=load_prototype,save_prototype=save_prototype,reference_identity=reference_identity)
            code=(ROOT/'notebooks/repair_audit/selection_ui.py').read_text(encoding='utf-8')
            exec(code,env)
            self.assertEqual(len(env['gallery'].children),8)
            env['slot_picker'].value=2
            # Click the third visible thumbnail's button: no row typing.
            env['gallery'].children[2].children[1].click()
            self.assertEqual(env['selected_rows'](),[1,2,3,4])
            env['gallery'].children[6].children[1].click()
            self.assertEqual(env['selected_rows'](),[1,2,7,4])
            env['next_page'].click()
            self.assertEqual(len(env['gallery'].children),4)
            env['gallery'].children[0].children[1].click()
            self.assertEqual(env['selected_rows'](),[1,2,9,4])
            env['character_keys'][2].value='가'
            env['pick_mode'].value='prototype'
            env['choose_thumbnail'](2)
            env['pin_buttons'][2].click()
            _,record=load_prototype(root/'bank','가')
            self.assertEqual(record['manifest_row'],2)
            env['identity_checks'][2].value=True
            env['pick_mode'].value='input';env['choose_thumbnail'](6)
            self.assertFalse(env['identity_checks'][2].value)
            self.assertEqual(env['selected_rows']()[2],6)
            self.assertEqual(load_prototype(root/'bank','가')[1]['sha256'],record['sha256'])
            env['pick_mode'].value='prototype';env['choose_thumbnail'](3)
            env['pin_buttons'][2].click()
            self.assertEqual(load_prototype(root/'bank','가')[1]['manifest_row'],3)
            before_search=len(env['gallery'].children)
            env['search'].value='row 12 |'
            self.assertEqual(len(env['gallery'].children),before_search)  # no search on typing
            env['search_button'].click()
            self.assertEqual(len(env['gallery'].children),1)
            self.assertIn('12',env['gallery'].children[0].children[1].description)
            # Browsing must not decode source/stroke files or read saved presets.
            def forbidden(*args,**kwargs):raise AssertionError('I/O during browsing')
            env['load_reference_image']=forbidden;env['load_catalog_reference']=forbidden
            env['load_prototype']=forbidden
            env['search'].value='';env['search_button'].click()
            env['slot_picker'].value=1;env['choose_thumbnail'](4)
            env['next_page'].click();env['refresh_selection']()

    def test_named_images_autofill_and_filter_same_character_prototypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=[]
            for i,(character,font) in enumerate([('가','A'),('가','B'),('나','A')],1):
                ref=np.zeros((1,16,16),np.float32);ref[0,2+i:4+i,3:12]=1
                path=root/f'{font}_{character}.npz';np.savez(path,reference=ref,strokes=ref)
                catalog.append(dict(manifest_row=i,path=str(path),reference=f'{font}_{character}.png'))
            cached=prepare_selection_cache(catalog,root/'cache',source_fingerprint='named-fixture')
            env=dict(CATALOG=cached,SELECTION_THUMBNAILS=selection_thumbnails(cached),
                PROTOTYPE_RECORDS={},PROTOTYPE_PREVIEWS={},PROTOTYPE_BANK=str(root/'bank'),
                Path=Path,json=json,widgets=widgets,np=np,Image=Image,display=lambda x:None,
                load_reference_image=load_reference_image,load_catalog_reference=load_catalog_reference,
                load_prototype=load_prototype,save_prototype=save_prototype,reference_identity=reference_identity)
            exec((ROOT/'notebooks/repair_audit/selection_ui.py').read_text(encoding='utf-8'),env)
            self.assertEqual(env['character_keys'][0].value,'가')
            self.assertTrue(env['character_keys'][0].disabled)
            env['pick_mode'].value='prototype'
            self.assertEqual(len(env['gallery'].children),2)
            env['choose_thumbnail'](2);env['pin_buttons'][0].click()
            self.assertTrue(env['identity_checks'][0].value)
            self.assertTrue(env['identity_checks'][0].disabled)
            env['pick_mode'].value='input';env['choose_thumbnail'](3)
            self.assertEqual(env['character_keys'][0].value,'나')
            env['pick_mode'].value='prototype'
            self.assertEqual(len(env['gallery'].children),1)


if __name__=='__main__':unittest.main()
