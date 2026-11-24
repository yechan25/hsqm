import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch

from stroke_model.source_structure import affine_warp
from stroke_model.target_stroke_plan import (TargetPlanConfig,build_target_plan,evaluate_stroke,measure_widths)
from repair_audit.target_guided_repair import repair_generated_sheet,save_plan


def sample():
    yy,xx=np.indices((32,32));r=np.hypot(xx-23,yy-23)
    canonical=np.zeros((2,32,32),np.float32);canonical[0,8,3:16]=1;canonical[1,abs(r-5)<.8]=1
    observed=np.zeros_like(canonical);observed[0,6:11,3:16]=1;observed[1,abs(r-5)<2.5]=1
    glyph=1-observed.max(0);sheet=np.tile(glyph,(2,2))
    source=Image.fromarray(np.uint8(sheet*255)).convert('RGB')
    conditions=[dict(strokes=canonical.copy(),reference=canonical.max(0)[None]) for _ in range(4)]
    return source,conditions,[observed.copy() for _ in range(4)]


def identity_registration(image,reference,strokes,config):
    return reference,strokes,dict(parameters=[0,0,0,1,1,0],dice=.9,optimizer_converged=True,fixed_for_gradients=True)


class TargetPlanTests(unittest.TestCase):
    def test_observed_width_is_rendered_for_open_and_closed_strokes(self):
        source,conditions,masks=sample()
        with patch('stroke_model.target_stroke_plan.register_reference',side_effect=identity_registration):
            guide,rendered=build_target_plan(source,conditions,masks,scores=[[0,0]]*4)
        self.assertTrue(all(s['width']['source']=='observed_stroke' for s in guide.report['strokes']))
        for item in rendered:
            # The closed stroke is thickened too, rather than a thin affine fallback.
            self.assertGreater(item['thick_strokes'][1].sum(),conditions[0]['strokes'][1].sum()*1.5)
        self.assertEqual(len(guide.report['strokes']),8)

    def test_one_shared_affine_per_glyph_preserves_stroke_relationships(self):
        source,conditions,masks=sample();parameters=[2,-1,7,1.05,.95,.03]
        def registration(image,reference,strokes,config):
            return reference,strokes,dict(parameters=parameters,dice=.8)
        with patch('stroke_model.target_stroke_plan.register_reference',side_effect=registration) as fn:
            _,rendered=build_target_plan(source,conditions,masks)
        self.assertEqual(fn.call_count,4)
        for item in rendered:
            for original,moved in zip(item['thick_strokes'],item['aligned_strokes']):
                self.assertTrue(np.allclose(affine_warp(original,parameters),moved))

    def test_scores_control_edit_plan_not_observed_width_or_scalar_gradient(self):
        source,conditions,masks=sample()
        with patch('stroke_model.target_stroke_plan.register_reference',side_effect=identity_registration):
            keep,_=build_target_plan(source,conditions,masks,scores=[[100,100]]*4)
            edit,_=build_target_plan(source,conditions,masks,scores=[[100,0]]*4)
        original=torch.tensor(keep.source).permute(2,0,1)[None]
        self.assertTrue(torch.equal(keep.project(original),original))
        self.assertFalse(keep.roi.any());self.assertTrue(edit.roi.any())
        self.assertEqual([s['edit_strength'] for s in edit.report['strokes']],[0.,1.]*4)
        self.assertEqual([s['width'] for s in keep.report['strokes']],[s['width'] for s in edit.report['strokes']])
        result=edit.project(torch.rand_like(original))
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(result.requires_grad)

    def test_dot_features_and_invalid_observed_widths(self):
        dot=np.zeros((16,16),np.float32);dot[8,8]=1
        self.assertAlmostEqual(evaluate_stroke(dot,dot)['score'],100.)
        with self.assertRaisesRegex(ValueError,'No measurable'):
            measure_widths(np.zeros((2,32,32),np.float32),TargetPlanConfig())
        with self.assertRaises(ValueError):TargetPlanConfig(keep_score=30,full_edit_score=40)
        source,conditions,masks=sample()
        with self.assertRaises(ValueError):build_target_plan(source,conditions,masks,scores=[[1]])

    def test_pipeline_uses_generated_source_and_never_finally_pastes_target(self):
        source,conditions,masks=sample()
        with tempfile.TemporaryDirectory() as temp, \
             patch('stroke_model.target_stroke_plan.register_reference',side_effect=identity_registration), \
             patch('diffusion_model.stroke_projected_diffusion.run_projected_pair',return_value={'source_content_used':True}) as run:
            guide,_=repair_generated_sheet(object(),None,source,conditions,Path(temp)/'run',cached_masks=masks,scores=[[0,0]]*4)
            self.assertIs(run.call_args.args[1],source)
            self.assertIs(run.call_args.args[2],guide)
            self.assertFalse(run.call_args.kwargs['start_from_noise'])
            self.assertFalse(run.call_args.kwargs['save_final_projection'])
            data=json.loads((Path(temp)/'run/pipeline.json').read_text())
            self.assertFalse(data['final_pixel_replacement'])
            self.assertTrue((Path(temp)/'run/target_plan/aligned_target.png').exists())

    def test_full_sampler_connection_on_cpu_fixture(self):
        source,conditions,masks=sample();calls=[]
        class Unet(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x,*args,**kw):calls.append(x.clone());return (torch.zeros_like(x),)
        class Vae(torch.nn.Module):
            config=SimpleNamespace(scaling_factor=1.)
            def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:x))
            def decode(self,x,**kw):return (x,)
        class DDIMScheduler:
            config=SimpleNamespace(num_train_timesteps=1000)
            alphas_cumprod=torch.ones(1000);final_alpha_cumprod=torch.tensor(1.)
            def set_timesteps(self,n,**kw):self.num_inference_steps=n
            def add_noise(self,x,*args):return x.clone()
            def scale_model_input(self,x,t):return x
            def step(self,n,t,x,**kw):return SimpleNamespace(pred_original_sample=x,prev_sample=x)
        class Pipe:
            def __init__(self):self.unet=Unet();self.vae=Vae();self.scheduler=DDIMScheduler()
            def encode_prompt(self,*args,**kw):return torch.zeros(1,1),torch.zeros(1,1)
            def get_timesteps(self,*args):return torch.tensor([500,0]),2
        pipe=Pipe()
        with tempfile.TemporaryDirectory() as temp,patch('stroke_model.target_stroke_plan.register_reference',side_effect=identity_registration):
            guide,report=repair_generated_sheet(pipe,None,source,conditions,Path(temp)/'run',cached_masks=masks,
                scores=[[0,0]]*4,steps=4,strength=.5)
            self.assertTrue(torch.equal(calls[0],calls[2]))
            self.assertTrue(report['source_content_used']);self.assertEqual(report['backward_passes'],0)
            self.assertFalse(report['final_pixel_projection_saved'])
            self.assertEqual(report['guide']['kind'],'observed_width_aligned_target_image')
            self.assertFalse((Path(temp)/'run/reverse/stroke_projected_final_pixel_projection.png').exists())
            with Image.open(Path(temp)/'run/reverse/stroke_projected.png') as im:
                output=np.asarray(im).astype(float)/255
            before=np.mean((guide.source-guide.target)**2);after=np.mean((output-guide.target)**2)
            self.assertLess(after,before)


if __name__=='__main__':unittest.main()
