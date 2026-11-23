import ast
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from repair_audit.resume import prepare_resume, complete_arm
from repair_audit.stage1 import run_case
from test_repair_stage1_audit import create_archive, condition_bytes


class ResumeTests(unittest.TestCase):
    def test_interrupted_arm_restarts_but_completed_arms_never_run_again(self):
        for modern in (True, False):
            with self.subTest(modern=modern), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                image=Image.new('RGB',(32,32),'white');image.paste((0,0,0),(4,4,6,6));image.save(root/'input.png')
                for slot in range(1,5):(root/f'condition_{slot}.npz').write_bytes(condition_bytes())
                calls=[];callbacks=[]
                class Unet(torch.nn.Module):
                    def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.))
                    def forward(self,x,*args,**kw):calls.append(x.clone());return (torch.zeros_like(x),)
                class Vae(torch.nn.Module):
                    config=SimpleNamespace(scaling_factor=1.)
                    def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:x))
                    def decode(self,x,**kw):return (x,)
                class Scheduler:
                    config=SimpleNamespace(num_train_timesteps=1000)
                    alphas_cumprod=torch.linspace(.999,.1,1000);final_alpha_cumprod=torch.tensor(1.)
                    def set_timesteps(self,steps,**kw):self.num_inference_steps=steps
                    def add_noise(self,x,*args):return x.clone()
                    def scale_model_input(self,x,t):return x
                    def step(self,noise,t,x,**kw):return SimpleNamespace(pred_original_sample=x.clone(),prev_sample=x.clone())
                class Pipe:
                    def __init__(self):self.unet=Unet();self.vae=Vae();self.scheduler=Scheduler()
                    def encode_prompt(self,*args,**kw):return torch.zeros(1,1),torch.zeros(1,1)
                    def get_timesteps(self,*args):return torch.tensor([750,500,250,0]),4
                class Controller:
                    def __init__(self,source,*args,**kw):
                        self.erase=torch.zeros_like(source[:,:1]);self.restore=self.erase.clone()
                        self.preserve=1-self.erase;self.metadata={};self.boundary_history=[]
                    def refresh_erase(self,*args):pass
                    def begin_step(self,*args):return dict(phase='erase')
                    def finish_step(self,*args):pass
                    def __call__(self,image):return ((image-1).square().mean(),{})
                case=dict(case_id='test',kind='generated',directory=str(root),prompt='test',negative_prompt='')
                kw=dict(steps=8,strength=.5,boundary_guard=True,observed_width=True)
                def interrupt(case_id,arm,step,*args):
                    if arm=='cyclic_guided' and step==2:raise KeyboardInterrupt()
                with patch('repair_audit.stage1.fit_records',return_value=[]), \
                     patch('repair_audit.resume.load_records',return_value=[]), \
                     patch('repair_audit.stage1.make_figures'), \
                     patch('stroke_model.restore_settle_erase.RestoreSettleErase',Controller), \
                     patch('stroke_model.boundary_guarded_repair.BoundaryGuardedRepair',Controller), \
                     patch('stroke_model.observed_width_repair.ObservedWidthRepair',Controller):
                    pipe=Pipe();extractor=torch.nn.Linear(1,1)
                    with self.assertRaises(KeyboardInterrupt):
                        run_case(pipe,extractor,case,resume=modern,progress=interrupt,**kw)
                    self.assertEqual(len(calls),6)
                    before=(root/'audit/redenoise_only.png').read_bytes()
                    rows=run_case(pipe,extractor,case,resume=True,
                        on_arm_complete=lambda c,a:callbacks.append(a),**kw)
                    self.assertEqual(len(calls),18)  # Three remaining arms, four steps each.
                    self.assertEqual(callbacks,['cyclic_guided','boundary_guarded','observed_width'])
                    self.assertEqual(before,(root/'audit/redenoise_only.png').read_bytes())
                    self.assertTrue(list((root/'audit/interrupted_attempts').glob('cyclic_guided_*')))
                    cfg=json.loads((root/'audit/report.json').read_text())['configuration']
                    self.assertEqual(cfg['legacy_initial_unrecorded'],not modern)
                    self.assertEqual(cfg['paired_initial_latent'],modern)
                    self.assertEqual(len(rows),72)
                    run_case(pipe,extractor,case,resume=True,**kw)
                    self.assertEqual(len(calls),18)
                    with self.assertRaisesRegex(ValueError,'settings/input'):
                        run_case(pipe,extractor,case,resume=True,seed=999,**kw)

    def test_archive_recovery_is_data_only_and_completion_requires_full_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'saved.zip';create_archive(archive)
            settings=dict(steps=60,strength=.25,seed=2026,boundary_config={'x':1},width_config={'y':2})
            with zipfile.ZipFile(archive,'a') as z:
                z.writestr('runtime_manifest.json',json.dumps(dict(settings=settings,source_hashes={'stroke_model/model.py':'known'})))
                z.writestr('group_000_page_001/audit/input.png',z.read('group_000/generation/page_001.png'))
                z.writestr('group_000_page_001/audit/redenoise_only.png',z.read('group_000/generation/page_001.png'))
                z.writestr('group_000_page_001/audit/redenoise_only_trace.json',json.dumps([dict(step=i,timestep=t) for i,t in enumerate([3,2,1])]))
                z.writestr('group_000_page_001/audit/../../escape.json','{}')
                z.writestr('group_000_page_001/audit/unsafe.py','raise RuntimeError()')
            cases,_,_=prepare_resume(archive,root/'recovered',{'stroke_model/model.py':'known'})
            self.assertEqual(len(cases),2)
            audit=root/'recovered/group_000_page_001/audit'
            self.assertTrue(complete_arm(audit,'redenoise_only',[3,2,1]))
            self.assertFalse(complete_arm(audit,'redenoise_only',[3,2,0]))
            self.assertFalse((audit/'unsafe.py').exists())
            self.assertFalse((root/'recovered/escape.json').exists())
            with self.assertRaisesRegex(ValueError,'implementation differs'):
                prepare_resume(archive,root/'wrong',{'stroke_model/model.py':'different'})

    def test_notebook_embeds_resumer_and_saves_after_arms(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/repair_audit/stage4_resume.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),5)
        for source in code:ast.parse(source)
        helpers=next(ast.literal_eval(n.value) for n in ast.parse(code[0]).body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
        for name in ['repair_audit/resume.py','repair_audit/stage1.py']:
            self.assertEqual(helpers[name],(root/name).read_text(encoding='utf-8'))
        self.assertIn('prepare_resume(',code[1])
        self.assertIn('on_arm_complete=arm_saved',code[3])
        self.assertIn('except KeyboardInterrupt:',code[3])


if __name__=='__main__':unittest.main()
