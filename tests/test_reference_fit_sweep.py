import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import numpy as np
from PIL import Image
import torch
from torch import nn
from diffusion_model.multiglyph_pilot import generate_multiglyph_pilot
from diffusion_model.reference_fit_cases import cases_from_generation

ROOT=Path(__file__).resolve().parents[1]


def conditions(root):
    rows=[]
    for i in range(4):
        strokes=np.zeros((1,16,16),np.float32);strokes[:,3+i:13,7:9]=1
        np.savez(root/f'{i}.npz',reference=strokes.max(0)[None],strokes=strokes)
        rows.append(dict(path=f'{i}.npz',split='train'))
    path=root/'conditions.json';path.write_text(json.dumps(rows));return path


class SweepTests(unittest.TestCase):
    def test_every_combination_uses_same_clean_input_and_filtered_outputs_survive(self):
        workspace=ROOT/'outputs';workspace.mkdir(exist_ok=True)
        class Pipeline:
            scheduler=SimpleNamespace(config={})
            def __init__(self):self.calls=[]
            def __call__(self,**kwargs):
                self.calls.append(kwargs)
                # Different outputs let us detect accidentally chaining generated inputs.
                return SimpleNamespace(images=[Image.new('RGB',(64,64),len(self.calls)*20)],
                    nsfw_content_detected=[len(self.calls)==3])
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            root=Path(directory);manifest=conditions(root);pipe=Pipeline()
            report=generate_multiglyph_pilot(manifest,[1,2,3,4],root/'generation',glyph_counts=(4,),
                canvas_size=64,strengths=(.45,.65,.85),seeds=(100,101),steps=20,pipeline=pipe)
            self.assertEqual(len(pipe.calls),6)
            self.assertTrue(all(c['image'] is pipe.calls[0]['image'] for c in pipe.calls))
            self.assertEqual({(c['strength'],c['generator'].initial_seed()) for c in pipe.calls},
                {(s,k) for s in (.45,.65,.85) for k in (100,101)})
            cases=cases_from_generation(report)
            self.assertEqual(len(cases),7)
            self.assertEqual(sum(c['control'] for c in cases),1)
            self.assertEqual(sum(c['safety_filtered'] is True for c in cases),1)
            self.assertEqual(len({c['case_id'] for c in cases}),7)
            self.assertTrue(all(Path(c['input']).is_file() for c in cases))
            self.assertFalse(report['free_text_generation'])

    def test_case_ids_include_page_number_to_avoid_strength_rounding_collisions(self):
        pages=[dict(page_id=i,glyph_count=4,control=False,strength=s,seed=100,input=f'{i}.png',label='test')
               for i,s in enumerate((.651,.652))]
        cases=cases_from_generation(dict(pages=pages))
        self.assertNotEqual(cases[0]['case_id'],cases[1]['case_id'])

    def test_notebook_has_generation_before_batch_extraction_and_fit(self):
        data=json.loads((ROOT/'notebooks/stroke_reference_fit_sweep.ipynb').read_text(encoding='utf-8'))
        cells=[c for c in data['cells'] if c['cell_type']=='code'];self.assertEqual(len(cells),4)
        code=[''.join(c['source']) for c in cells]
        for source in code:ast.parse(source)
        self.assertIn('generate_multiglyph_pilot(',code[1])
        self.assertIn('for case in CASES:',code[2])
        self.assertIn('for record in CASE_EXTRACTIONS:',code[3])
        self.assertIn('case_dir=CASE_ROOT/',code[2])
        self.assertIn("'all_cases_comparison.png'",code[3])
        self.assertNotIn('generate_guided_pair(',code[3])

    def test_batch_cells_process_every_case_including_empty_extraction(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data=json.loads((ROOT/'notebooks/stroke_reference_fit_sweep.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in data['cells'] if c['cell_type']=='code']
        class Extractor(nn.Module):
            def forward(self,ink,reference,strokes):
                p=torch.ones_like(strokes)/strokes.shape[1]
                return dict(probabilities=p,masks=ink*p,hints=strokes)
        workspace=ROOT/'outputs';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            root=Path(directory);manifest=conditions(root);cases=[]
            for i,color in enumerate(('white','black')):
                image=root/f'input_{i}.png';Image.new('RGB',(64,64),color).save(image)
                cases.append(dict(case_id=f'case_{i}',page_id=i,input=str(image),control=False,
                    seed=100+i,strength=.65,safety_filtered=False,label=f'case {i}'))
            case_root=root/'results';case_root.mkdir()
            env=dict(Path=Path,json=json,np=np,torch=torch,Image=Image,plt=plt,display=lambda value:None,
                CHECKPOINT='mock',demo=dict(manifest=str(manifest),size=64,boxes=[[0,0,32,32],[32,0,64,32],[0,32,32,64],[32,32,64,64]],source=cases[0]['input']),
                CASES=cases,CASE_ROOT=case_root,SHOW_DETAILS_FOR=(),RUN_DIR=case_root,
                runtime_helpers={},commit='mock',HSQM=root,files=SimpleNamespace(download=Mock()))
            import gc,shutil
            env.update(gc=gc,shutil=shutil)
            original_as_tensor=torch.as_tensor
            def cpu_tensor(*args,**kwargs):
                if kwargs.get('device')=='cuda':kwargs['device']='cpu'
                return original_as_tensor(*args,**kwargs)
            with patch('stroke_model.infer_topology.load_topology',return_value=(Extractor(),16,1)), \
                 patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self),patch('torch.as_tensor',side_effect=cpu_tensor):
                exec(code[2],env)
                exec(code[3],env)
            self.assertEqual(len(env['CASE_EXTRACTIONS']),2)
            self.assertEqual(len(env['CASE_SUMMARIES']),2)
            for case in cases:
                self.assertTrue((case_root/case['case_id']/'normal_candidate.png').is_file())
                self.assertTrue((case_root/case['case_id']/'fit_slot_1.json').is_file())
            self.assertGreater(env['CASE_SUMMARIES'][0]['fallback_strokes'],0)
            self.assertTrue((case_root/'all_cases_summary.csv').is_file())
            self.assertTrue((case_root/'all_cases_comparison.png').is_file())


if __name__=='__main__':unittest.main()
