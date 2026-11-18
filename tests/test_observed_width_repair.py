import ast
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from stroke_model.boundary_guarded_repair import BoundaryGuardedRepair
from stroke_model.observed_width_repair import ObservedWidthConfig, ObservedWidthRepair


def fixture(closed=True, missing=False):
    size=64
    yy,xx=np.indices((size,size)); radial=np.hypot(xx-32,yy-32)
    reference=(abs(radial-15)<1.5).astype(np.float32)
    observed=(abs(radial-15)<6.5).astype(np.float32)
    observed[30:35,46:63]=1  # A real far spur must remain eligible for erasure.
    if missing: observed.fill(0)
    info=dict(method='affine_topology_preserving_fallback' if closed else 'regularized_curved_reference_fit',
              warnings=['fallback'] if closed else [], reference_path=dict(closed=closed))
    records=[dict(reference=reference[None],strokes=reference[None],fitted=reference[None],
                  extracted_masks=observed[None],fit_metadata=[info]) for _ in range(4)]
    sheet=np.tile(observed,(2,2))
    rgb=torch.tensor(1-sheet)[None,None].repeat(1,3,1,1)
    class Extractor(torch.nn.Module):
        def forward(self,*args):return {'probabilities':torch.ones(1,1,size,size)}
    return rgb, Extractor(), records, abs(radial-15)<6.5


class ObservedWidthTests(unittest.TestCase):
    def test_closed_width_protects_body_without_protecting_far_spur(self):
        rgb,extractor,records,ring=fixture()
        old=BoundaryGuardedRepair(rgb,extractor,records)
        new=ObservedWidthRepair(rgb,extractor,records)
        old.refresh_erase(rgb);new.refresh_erase(rgb)
        before=old.erase[0,0,:64,:64].numpy();after=new.erase[0,0,:64,:64].numpy()
        self.assertGreater(before[ring].sum(),0)
        self.assertLess(after[ring].sum(), before[ring].sum()*.2)
        self.assertGreater(after[32,61],.5)
        self.assertTrue(torch.equal(old.restore,new.restore))
        self.assertTrue(torch.equal(old.path_union,new.path_union))
        self.assertTrue(all(r['reason']=='calibrated' for r in new.width_calibration))
        self.assertLessEqual(float((new.erase-old.erase).max()),1e-6)
        radii=[r['radii'].copy() for r in new.records]
        new.refresh_erase(torch.zeros_like(rgb))
        self.assertTrue(all(np.array_equal(a,r['radii']) for a,r in zip(radii,new.records)))

    def test_open_and_insufficient_support_strokes_are_unchanged(self):
        for closed,missing in [(False,False),(True,True)]:
            rgb,extractor,records,_=fixture(closed,missing)
            old=BoundaryGuardedRepair(rgb,extractor,records)
            new=ObservedWidthRepair(rgb,extractor,records)
            old.refresh_erase(rgb);new.refresh_erase(rgb)
            self.assertTrue(torch.equal(old.erase,new.erase))
            self.assertTrue(all(r['old_radius']==r['new_radius'] for r in new.width_calibration))

    def test_config_limits_and_validation(self):
        rgb,extractor,records,_=fixture()
        new=ObservedWidthRepair(rgb,extractor,records,width_config=ObservedWidthConfig(maximum_increase=.5))
        self.assertTrue(all(r['new_radius']<=r['old_radius']+.5 for r in new.width_calibration))
        for kw in [dict(minimum_samples=1),dict(minimum_path_coverage=float('nan')),
                   dict(maximum_increase=-1),dict(maximum_half_width=float('inf'))]:
            with self.assertRaises(ValueError):ObservedWidthConfig(**kw)

    def test_stage4_notebook_uploads_originals_and_enables_width_arm(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/repair_audit/stage4.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),5)
        for s in code: ast.parse(s)
        helpers=next(ast.literal_eval(n.value) for n in ast.parse(code[0]).body
                     if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
        for name in ['stroke_model/observed_width_repair.py','repair_audit/stage1.py','repair_audit/stage3.py']:
            self.assertEqual(helpers[name],(root/name).read_text(encoding='utf-8'))
        self.assertIn('observed_width=True',code[3])
        self.assertIn('read_archive(',code[1])
        self.assertNotIn('generate_cases(', '\n'.join(code))


if __name__=='__main__':unittest.main()
