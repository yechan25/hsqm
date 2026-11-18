import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from stroke_model.boundary_guarded_repair import BoundaryGuardConfig, BoundaryGuardedRepair
from stroke_model.restore_settle_erase import RestoreSettleErase
from repair_audit.stage1 import read_archive, load_conditions
from test_repair_stage1_audit import create_archive


def objective(cls, size=32, sheet=32, config=None):
    obj=object.__new__(cls)
    obj.source=torch.ones(1,3,sheet,sheet)
    obj.erase=torch.zeros(1,1,sheet,sheet);obj.restore=obj.erase.clone();obj.path_union=obj.erase.clone()
    path=np.zeros((size,size),bool);path[6:26,12]=True
    class Extractor:
        def __call__(self,*args):return {'probabilities':torch.ones(1,1,size,size)}
    obj.extractor=Extractor()
    obj.records=[dict(box=(0,0,sheet,sheet),size=(size,size),reference=torch.zeros(1,1,size,size),
        strokes=torch.zeros(1,1,size,size),distances=distance_transform_edt(~path)[None],radii=np.array([[[3.]]]))]
    obj.boundary_config=config or BoundaryGuardConfig();obj.boundary_history=[]
    return obj


class BoundaryTests(unittest.TestCase):
    def test_near_boundary_is_protected_while_far_spur_is_still_targeted(self):
        # Both were deletion seeds previously. Only the far one is clear evidence.
        image=torch.ones(1,3,32,32);image[:,:,15,18]=0;image[:,:,15,24]=0
        old=objective(RestoreSettleErase);new=objective(BoundaryGuardedRepair)
        old.refresh_erase(image);new.refresh_erase(image)
        self.assertGreater(float(old.erase[0,0,15,18]),.5)
        self.assertEqual(float(new.erase[0,0,15,18]),0)
        self.assertGreater(float(new.erase[0,0,15,24]),.5)
        self.assertLessEqual(float((new.erase-old.erase).max()),1e-7)

    def test_dilation_rechecks_neighbor_and_upsampling_does_not_leak(self):
        new=objective(BoundaryGuardedRepair,size=32,sheet=64)
        image=torch.ones(1,3,64,64);image[:,:,28:34,38:42]=0
        new.refresh_erase(image)
        r=new.records[0];radius=r['radii']+1
        resp=np.clip((r['distances']-radius-1.5)/1.5,0,1)[0]
        eligible=(resp>.5)&~(r['distances']<=radius+1).any(0)
        gate=F.interpolate(torch.tensor(eligible)[None,None].float(),(64,64),mode='nearest').bool()
        self.assertEqual(float(new.erase[~gate].sum()),0)
        # Recomputed deletion must continue to protect restored paths.
        new.restore.fill_(1);new.refresh_erase(image)
        self.assertEqual(float(new.erase.sum()),0)

    def test_config_rejects_nonfinite_or_invalid_values(self):
        for kwargs in (dict(uncertainty_pixels=-1),dict(uncertainty_pixels=float('nan')),
                       dict(responsibility_floor=1),dict(responsibility_floor=-.1)):
            with self.assertRaises(ValueError):BoundaryGuardConfig(**kwargs)

    def test_reuses_stage1_original_input_not_guided_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);create_archive(root/'original.zip')
            cases,experiment=read_archive(root/'original.zip',root/'stage1')
            # Simulate a stage1 output ZIP with extra result content; no execution.
            with zipfile.ZipFile(root/'stage1.zip','w') as z:
                for p in (root/'stage1').rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(root/'stage1').as_posix())
                z.writestr('group_000_page_001/audit/cyclic_guided.png',b'not an input')
            loaded,exp=read_archive(root/'stage1.zip',root/'stage2')
            self.assertEqual(len(loaded),2)
            self.assertTrue(all(c['reused_stage1_original'] for c in loaded))
            self.assertEqual((root/'stage1/group_000_page_001/input.png').read_bytes(),
                             (root/'stage2/group_000_page_001/input.png').read_bytes())
            self.assertEqual(len(load_conditions(loaded[0]['directory'])),4)

    def test_stage2_notebook_embeds_guard_and_enables_same_start_comparison(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/repair_audit/stage2.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for cell in code:ast.parse(cell)
        nodes=[n for n in ast.parse(code[0]).body if isinstance(n,ast.Assign)]
        helpers=next(ast.literal_eval(n.value) for n in nodes if any(
            isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
        self.assertEqual(helpers['stroke_model/boundary_guarded_repair.py'],
                         (root/'stroke_model/boundary_guarded_repair.py').read_text(encoding='utf-8'))
        self.assertIn('boundary_guard=True',code[3])
        self.assertIn('REPAIR_STRENGTH = .25',code[1])


if __name__=='__main__':unittest.main()
