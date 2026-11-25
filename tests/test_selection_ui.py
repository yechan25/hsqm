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
from repair_audit.selectable_target_inputs import load_reference_image,load_prototype,lock_prototype

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
            env=dict(CATALOG=catalog,PROTOTYPE_BANK=str(root/'bank'),Path=Path,json=json,
                widgets=widgets,np=np,Image=Image,display=lambda x:None,
                load_reference_image=load_reference_image,load_catalog_reference=load_catalog_reference,
                load_prototype=load_prototype,lock_prototype=lock_prototype)
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
            self.assertEqual(load_prototype(root/'bank','가')[1]['manifest_row'],2)
            env['search'].value='row 12 |'
            self.assertEqual(len(env['gallery'].children),1)
            self.assertIn('12',env['gallery'].children[0].children[1].description)


if __name__=='__main__':unittest.main()
