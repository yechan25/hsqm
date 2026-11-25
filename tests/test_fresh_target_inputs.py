import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from repair_audit.fresh_target_inputs import canonical_sheet, generate_fresh_cases


class FreshInputTests(unittest.TestCase):
    def fixture(self, root):
        seed=root/'seed';seed.mkdir()
        for i in range(1,5):
            reference=np.zeros((1,16,16),np.float32);reference[0,2+i:4+i,3:12]=1
            np.savez(seed/f'condition_{i}.npz',reference=reference,strokes=reference)
        # Deliberately NOT the canonical sheet: fresh generation must reconstruct it.
        Image.new('RGB',(64,64),'red').save(seed/'input.png')
        return dict(case_id='group_000_normal_control',kind='normal_control',directory=str(seed),
                    original_rows=[1,2,3,4],char_ids=['a','b','c','d'])

    def test_joint_generation_grid_seeds_and_unfiltered_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=self.fixture(root);calls=[];saved=[]
            def pipe(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(images=[Image.new('RGB',(64,64),'white')],nsfw_content_detected=[True])
            cases=generate_fresh_cases(pipe,[control],root/'fresh',strengths=[.45,.65],seeds=[100,101],
                                       steps=60,on_saved=lambda item,rows:saved.append(item['case_id']))
            self.assertEqual(len(cases),5);self.assertEqual(len(calls),4);self.assertEqual(len(saved),5)
            self.assertEqual([c['generator'].initial_seed() for c in calls],[100,101,100,101])
            first=np.asarray(calls[0]['image'])
            self.assertFalse(np.array_equal(first,np.asarray(Image.open(Path(control['directory'])/'input.png'))))
            for call in calls:
                self.assertEqual(call['image'].size,(64,64))
                self.assertTrue(np.array_equal(first,np.asarray(call['image'])))
            for y,x in [(0,0),(0,32),(32,0),(32,32)]:
                self.assertLess(first[y:y+32,x:x+32].min(),128)
            self.assertTrue(all(c['generation']['safety_filtered'] for c in cases[1:]))
            self.assertTrue(all(c['kind']=='generated' for c in cases[1:]))
            self.assertEqual(cases[0]['kind'],'normal_control')
            report=json.loads((root/'fresh/generation_report.json').read_text())
            self.assertEqual(len(report['cases']),5)
            for c in cases:
                self.assertTrue((Path(c['directory'])/'condition_4.npz').is_file())

    def test_invalid_settings_and_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);control=self.fixture(root)
            for strengths,seeds in [([0],[1]),([.5],[1,1]),([.5],[]),([float('nan')],[1])]:
                with self.assertRaises(ValueError):
                    generate_fresh_cases(None,[control],root/'invalid',strengths=strengths,seeds=seeds)
                self.assertFalse((root/'invalid').exists())
            with self.assertRaisesRegex(ValueError,'dimensions'):
                generate_fresh_cases(lambda **kw:SimpleNamespace(images=[Image.new('RGB',(32,32))]),
                                     [control],root/'mismatch',seeds=[1])
            report=json.loads((root/'mismatch/generation_report.json').read_text())
            self.assertEqual(len(report['cases']),1)  # normal control was saved before failure


if __name__=='__main__':unittest.main()
