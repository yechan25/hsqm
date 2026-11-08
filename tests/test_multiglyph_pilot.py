import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from diffusion_model.multiglyph_pilot import compose_sheet, generate_multiglyph_pilot
from diffusion_model.score_pilot import scored_ink, training_catalog, score_pilot


class MultiGlyphTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        for i in range(4):
            strokes = np.zeros((2,32,32), np.float32)
            strokes[0,5+i:9+i,5:25] = 1
            strokes[1,5:25,8+i:12+i] = 1
            path = self.root/f'{i}.npz'
            np.savez(path, reference=strokes.max(0)[None], strokes=strokes)
            rows.append(dict(path=path.name, group_id=f'g{i}', split='train', char_id=f'test{i}'))
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(rows))

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_joint_call_per_sheet_and_missing_crop_preserved(self):
        class JointPipeline:
            scheduler = SimpleNamespace(config={'class': 'test double'})
            def __init__(self): self.calls=[]
            def __call__(self, **kwargs):
                image = kwargs['image'].copy()
                self.calls.append(kwargs)
                if '4 Korean' in kwargs['prompt']:
                    ImageDraw.Draw(image).rectangle((32,32,63,63),fill='white')
                return SimpleNamespace(images=[image], nsfw_content_detected=[False])

        pipe=JointPipeline()
        report=generate_multiglyph_pilot(self.manifest,[1,2,3,4],self.root/'out',
            glyph_counts=[1,4],canvas_size=64,strengths=[.5],seeds=[7],pipeline=pipe)
        self.assertEqual(len(pipe.calls),2)  # 1 sheet + 1 sheet, not 1 glyph + 4 glyph calls.
        self.assertEqual([call['image'].size for call in pipe.calls],[(64,64),(64,64)])
        self.assertEqual(len(report['pages']),4)  # input/generated for each count
        self.assertEqual(len(report['results']),10)  # all slots, including missing output
        generated=[s for s in report['results'] if not s['control'] and s['glyph_count']==4]
        self.assertEqual(len(generated),4)
        self.assertEqual(generated[-1]['crop_bbox'],[32,32,64,64])
        for sample in generated:
            with Image.open(sample['page_input']) as sheet:
                expected=scored_ink(sheet.crop(tuple(sample['crop_bbox'])),32)
            with np.load(sample['npz']) as data:
                np.testing.assert_array_equal(data['image'],expected)
                with np.load(self.root/f"{sample['manifest_row']-1}.npz") as original:
                    np.testing.assert_array_equal(data['reference'],original['reference'])
                    np.testing.assert_array_equal(data['reference_strokes'],original['strokes'])
        with np.load(generated[-1]['npz']) as data:
            self.assertEqual(data['image'].sum(),0)
        class Extractor(torch.nn.Module):
            output_mode='partition'
            def forward(self,image,reference,strokes):
                probabilities=(strokes+1).softmax(1)
                return dict(probabilities=probabilities,masks=image*probabilities)
        with patch('stroke_model.infer_topology.load_topology',return_value=(Extractor(),32,1)):
            result,table=score_pilot(self.root/'out/generation_report.json',self.root/'unused.pt',show=False)
        self.assertEqual(len(table),10)
        missing=table[(table['glyph_count']==4)&(~table['control'])&(table['slot']==3)]
        self.assertEqual(float(missing.iloc[0]['structural_score']),0)
        with zipfile.ZipFile(result['debug_zip']) as archive:
            self.assertIn('generated_sheets/003.png',archive.namelist())

    def test_pixel_budget_modes_and_rejecting_invalid_training_selection(self):
        catalog=training_catalog(self.manifest)
        one,slots=compose_sheet(catalog[:1],64,cell_pixels=32)
        four,_=compose_sheet(catalog,64,cell_pixels=32)
        self.assertEqual(one.size,(32,32)); self.assertEqual(four.size,(64,64))
        with self.assertRaisesRegex(ValueError,'enough training'):
            generate_multiglyph_pilot(self.manifest,[1,2,3,4],self.root/'bad',glyph_counts=[9],pipeline=object())
        with self.assertRaisesRegex(ValueError,'training catalog'):
            generate_multiglyph_pilot(self.manifest,[999],self.root/'bad2',glyph_counts=[1],pipeline=object())
        self.assertFalse((self.root/'bad').exists())


if __name__=='__main__':
    unittest.main()
