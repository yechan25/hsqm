import ast
import json
from pathlib import Path
import unittest
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from stroke_model.restore_settle_erase import RestoreSettleErase, align_prototypes
from tests.test_adaptive_ink_constraints import example, Extractor
from diffusion_model.cyclic_stroke_repair import run_cycle_pair


def missing_example():
    source,records=example()
    source[...,36:50,36:58]=1  # Entire horizontal prototype absent from slot 1.
    return source,records


class CycleTests(unittest.TestCase):
    def test_whole_missing_stroke_is_detected_and_insertion_only_adds_ink_inside_band(self):
        source,records=missing_example()
        controller=RestoreSettleErase(source,Extractor(),records)
        self.assertTrue(any(r['entire_stroke_missing'] for r in controller.initial_missing))
        inserted=controller.insertion_image(source)
        self.assertTrue(torch.equal(inserted[controller.restore.expand_as(source)==0],source[controller.restore.expand_as(source)==0]))
        self.assertGreater(float((source-inserted).sum()),0)
        self.assertTrue(bool((inserted<=source+1e-7).all()))

    def test_partial_gap_is_restored_without_forcing_present_stroke_caps(self):
        source,records=example();source[...,36:42,17:21]=1
        controller=RestoreSettleErase(source,Extractor(),records)
        row=next(r for r in controller.initial_missing if r['slot']==1 and r['stroke']==1)
        self.assertFalse(row['entire_stroke_missing'])
        self.assertGreater(row['missing_path_pixels'],0)
        self.assertGreater(float(controller.restore[...,36:42,17:21].sum()),0)
        self.assertEqual(float(controller.required_paths[0][...,8:11,17:21].sum()),0)

    def test_projection_verifies_actual_decoder_output_before_enabling_erasure(self):
        source,records=missing_example()
        controller=RestoreSettleErase(source,Extractor(),records)
        row=controller.begin_step(source,0)
        restored,info=controller.project_clean(source,lambda z:z,lambda image:image)
        self.assertTrue(info['restoration_verified'])
        controller.finish_step(restored,row)
        self.assertEqual(controller.phase,'settle')
        self.assertEqual(float(controller.erase.sum()),0)
        for i in (1,2):
            row=controller.begin_step(restored,i);controller.finish_step(restored,row)
        self.assertEqual(controller.phase,'erase')

    def test_failed_vae_roundtrip_never_enables_erasure(self):
        source,records=missing_example()
        controller=RestoreSettleErase(source,Extractor(),records)
        for i in range(4):
            row=controller.begin_step(source,i)
            unchanged,info=controller.project_clean(source,lambda z:z,lambda image:source)
            self.assertFalse(info['restoration_verified'])
            controller.finish_step(unchanged,row)
        self.assertEqual(controller.phase,'restore')
        self.assertEqual(float(controller.erase.sum()),0)

    def test_loss_of_restored_path_returns_to_restore_and_cycles_are_bounded(self):
        source,records=missing_example()
        controller=RestoreSettleErase(source,Extractor(),records)
        controller.phase='erase';controller.erase.fill_(1)
        row=controller.begin_step(source,3)
        self.assertEqual(row['phase'],'restore')
        self.assertEqual(controller.cycles,2)
        self.assertEqual(float(controller.erase.sum()),0)
        controller.phase='erase'
        row=controller.begin_step(source,6)
        self.assertEqual(row['phase'],'unresolved')
        self.assertEqual(float(controller.erase.sum()),0)

    def test_reextract_erase_cannot_delete_restoration_bands_or_centerlines(self):
        source,records=missing_example()
        controller=RestoreSettleErase(source,Extractor(),records)
        restored,_=controller.project_clean(source,lambda z:z,lambda image:image)
        controller.refresh_erase(restored)
        self.assertFalse(bool(((controller.erase>0)&(controller.restore>0)).any()))
        self.assertFalse(bool(((controller.erase>0)&(controller.path_union>0)).any()))

    def test_missing_prototype_alignment_uses_visible_anchors_and_reports_fallback(self):
        _,records=example();record=dict(records[0])
        record['fit_metadata']=[dict(method='fallback',warnings=['missing']) for _ in record['strokes']]
        warped,info=align_prototypes(record)
        self.assertEqual(info['visible_anchors'],0)
        np.testing.assert_allclose(warped,record['strokes'])

    def test_notebook_retains_single_generation_and_embeds_new_runtime(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/cyclic_stroke_repair.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),5)
        for source in code:ast.parse(source)
        nodes=[n for n in ast.parse(code[0]).body if isinstance(n,ast.Assign)]
        helpers=next(ast.literal_eval(n.value) for n in nodes
            if any(isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
        for path in ('stroke_model/restore_settle_erase.py','diffusion_model/cyclic_stroke_repair.py'):
            self.assertEqual(helpers[path],(root/path).read_text(encoding='utf-8'))
        self.assertIn('GENERATION_SEED=100',code[3]);self.assertIn('GENERATION_STRENGTH=.65',code[3])
        self.assertIn('SUMMARIES=repair_cycle_cases(',code[4])

    def test_actual_sampler_pairs_initial_state_and_transfers_structural_projection(self):
        calls=[]
        class Unet(torch.nn.Module):
            def forward(self,x,t,**kwargs):calls.append((int(t),x.clone()));return (torch.zeros_like(x),)
        class Vae(torch.nn.Module):
            config=SimpleNamespace(scaling_factor=1.)
            def decode(self,z,**kwargs):return (z,)
            def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:x))
        class Scheduler:
            config=SimpleNamespace(num_train_timesteps=1000)
            alphas_cumprod=torch.linspace(.999,.1,1000);final_alpha_cumprod=torch.tensor(1.)
            @classmethod
            def from_config(cls,*args,**kwargs):return cls()
            def set_timesteps(self,n,**kwargs):self.num_inference_steps=n
            def scale_model_input(self,x,t):return x
            def step(self,noise,t,z,**kwargs):
                clean=(z-(1-self.alphas_cumprod[t]).sqrt()*noise)/self.alphas_cumprod[t].sqrt()
                prev=int(t)-1000//self.num_inference_steps
                a=self.alphas_cumprod[prev] if prev>=0 else self.final_alpha_cumprod
                return SimpleNamespace(pred_original_sample=clean,prev_sample=a.sqrt()*clean+(1-a).sqrt()*noise)
        class Pipeline:
            def __init__(self):
                self.unet=Unet();self.vae=Vae();self.text_encoder=torch.nn.Identity();self.scheduler=Scheduler()
                self.image_processor=SimpleNamespace(preprocess=lambda image:image)
            @classmethod
            def from_pretrained(cls,*args,**kwargs):return cls()
            def to(self,*args):return self
            def enable_attention_slicing(self):pass
            def enable_vae_slicing(self):pass
            def encode_prompt(self,*args,**kwargs):return torch.zeros(1,1),torch.zeros(1,1)
            def get_timesteps(self,*args):return torch.tensor([875,750,625,500,375,250,125,0]),8
            def prepare_latents(self,image,*args):
                return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].float()/127.5-1
        fake=ModuleType('diffusers');fake.StableDiffusionImg2ImgPipeline=Pipeline;fake.DDIMScheduler=Scheduler
        source,records=missing_example();controller=RestoreSettleErase(source,Extractor(),records)
        image=Image.fromarray((source[0].permute(1,2,0).numpy()*255).astype('uint8'))
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',{'diffusers':fake}),\
             patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.empty_cache'):
            out=Path(directory)/'run'
            report=run_cycle_pair(image,image,controller,out,dict(prompt='test'),steps=8,strength=1.)
            self.assertTrue(torch.equal(calls[0][1],calls[8][1]))
            self.assertGreater(report['metrics']['cyclic_guided']['min_coverage'],report['metrics']['redenoise_only']['min_coverage'])
            rows=report['traces']['cyclic_guided']
            self.assertTrue(any(r.get('projection_rounds',0)>0 for r in rows))
            first_erase=next((i for i,r in enumerate(rows) if r['phase']=='erase'),None)
            if first_erase is not None:self.assertTrue(any(r.get('restoration_verified') for r in rows[:first_erase]))
            self.assertTrue((out/'cyclic_guided.png').is_file())


if __name__=='__main__':unittest.main()
