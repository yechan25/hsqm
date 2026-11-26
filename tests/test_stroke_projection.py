from pathlib import Path
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np
from PIL import Image
import torch

from stroke_model.stroke_projection import StrokeProjection,StrokeProjectionConfig
from diffusion_model.stroke_projected_diffusion import run_projected_pair
from repair_audit.stroke_projection_pilot import run as run_pilot
from repair_audit.stage1 import read_archive
from test_repair_stage1_audit import create_archive, condition_bytes


def constraints():
    strokes=np.zeros((2,16,16),np.float32)
    strokes[0,7:10,2:14]=1;strokes[1,2:14,7:10]=1
    return StrokeProjection.from_glyphs([dict(strokes=strokes,bbox=(8,8,24,24))],(32,32))


class StrokeProjectionTests(unittest.TestCase):
    def test_missing_strokes_extra_ink_crossings_free_boundary_and_outside(self):
        c=constraints();rgb=torch.ones(1,3,32,32)
        rgb[:,:,8:24,8:24]=0  # Extreme unwanted fill, including inter-stroke spaces.
        rgb[:,:,15,12:20]=1  # A missing horizontal core.
        original=rgb.clone();after=c.project(rgb);d=c.diagnostics(after)
        self.assertEqual(d['missing_pixels'],0);self.assertEqual(d['extra_pixels'],0)
        self.assertGreater(c.diagnostics(original)['missing_pixels'],0)
        self.assertGreater(c.diagnostics(original)['extra_pixels'],0)
        self.assertTrue(torch.equal(after[0,:,~c.roi],original[0,:,~c.roi]))
        self.assertTrue(torch.equal(after[0,:,c.free],original[0,:,c.free]))
        self.assertFalse(c.background[c.foreground].any())
        self.assertTrue(torch.allclose(c.project(after),after,atol=1e-6))

    def test_union_equivalence_is_explicit_not_claimed_as_a_gain(self):
        c=constraints();rgb=torch.rand(1,3,32,32)
        union=StrokeProjection(c.foreground[None],c.allowed.any(0)[None],c.roi,[dict(stroke=0)])
        self.assertTrue(torch.equal(c.project(rgb),union.project(rgb)))
        self.assertEqual(len(c.diagnostics(rgb)['strokes']),2)

    def test_reject_invalid_geometry_and_configuration(self):
        for cfg in [dict(core_fraction=0),dict(boundary_margin=-1),dict(minimum_ink=.01,maximum_background_ink=.1),dict(minimum_ink=float('nan'))]:
            with self.assertRaises(ValueError):StrokeProjectionConfig(**cfg)
        with self.assertRaises(ValueError):
            StrokeProjection.from_glyphs([dict(strokes=np.zeros((1,8,8)),bbox=(0,0,8,8))],(8,8))
        c=constraints()
        with self.assertRaises(ValueError):c.project(torch.ones(1,3,16,16))

    def test_sampler_has_shared_initial_state_no_gradients_and_separate_final_projection(self):
        calls=[]
        class Unet(torch.nn.Module):
            def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x,*a,**kw):calls.append(x.clone());return (torch.zeros_like(x),)
        class Vae(torch.nn.Module):
            config=SimpleNamespace(scaling_factor=1.)
            def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:x))
            def decode(self,x,**kw):return (x,)
        class DDIMScheduler:
            config=SimpleNamespace(num_train_timesteps=1000)
            alphas_cumprod=torch.ones(1000);final_alpha_cumprod=torch.tensor(1.)
            init_noise_sigma=1.
            def set_timesteps(self,n,**kw):self.num_inference_steps=n;self.timesteps=torch.tensor([500,0])
            def add_noise(self,x,*args):return x.clone()
            def scale_model_input(self,x,t):return x
            def step(self,n,t,x,**kw):return SimpleNamespace(pred_original_sample=x,prev_sample=x)
        class Pipe:
            def __init__(self):self.unet=Unet();self.vae=Vae();self.scheduler=DDIMScheduler()
            def encode_prompt(self,*a,**kw):return torch.zeros(1,1),torch.zeros(1,1)
            def get_timesteps(self,*a):return torch.tensor([500,0]),2
        pipe=Pipe()
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'result'
            report=run_projected_pair(pipe,Image.new('RGB',(32,32),'white'),constraints(),out,steps=4,strength=.5,
                start_from_noise=True,save_final_projection=True,save_step_images=True)
            self.assertTrue(torch.equal(calls[0],calls[2]))
            self.assertIsNone(pipe.unet.p.grad)
            self.assertEqual(report['backward_passes'],0)
            self.assertFalse(report['source_content_used'])
            self.assertTrue(report['start_from_noise'])
            self.assertTrue((out/'stroke_projected.png').is_file())
            self.assertTrue((out/'stroke_projected_final_pixel_projection.png').is_file())
            self.assertEqual([f['timestep'] for f in report['step_images']],[500,0])
            self.assertEqual([f['step'] for f in report['step_images']],[1,2])
            for frame in report['step_images']:
                with Image.open(out/frame['path']) as image:self.assertEqual(image.size,(32,32))
            self.assertTrue(all(row['pixel_projected']['missing_pixels']==0 for row in report['traces']['stroke_projected']))

    def test_offline_pilot_defers_missing_widths_and_validates_observation_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);create_archive(root/'original.zip')
            cases,_=read_archive(root/'original.zip',root/'inputs')
            case_id=cases[0]['case_id']
            with zipfile.ZipFile(root/'audit.zip','w') as z:
                for path in (root/'inputs').rglob('*'):
                    if path.is_file():z.write(path,path.relative_to(root/'inputs').as_posix())
            deferred=run_pilot(root/'audit.zip',root/'deferred',[case_id])
            self.assertEqual(deferred[0]['status'],'deferred_missing_observed_widths')
            for name,wrong in [('observations.zip',False),('wrong.zip',True)]:
                with zipfile.ZipFile(root/name,'w') as z:
                    image=Image.new('RGB',(32,32),'black' if wrong else 'white')
                    stream=io.BytesIO();image.save(stream,format='PNG')
                    z.writestr(case_id+'/input.png',stream.getvalue())
                    with np.load(io.BytesIO(condition_bytes())) as a:
                        masks=a['strokes'].copy()
                    for slot in range(1,5):
                        stream=io.BytesIO();np.savez_compressed(stream,extracted_masks=masks)
                        z.writestr(f'{case_id}/audit/extraction_input/slot_{slot}.npz',stream.getvalue())
            report=run_pilot(root/'audit.zip',root/'accepted',[case_id],root/'observations.zip')[0]
            self.assertEqual(report['status'],'cpu_projection_only')
            self.assertTrue(report['results'][0]['union_equivalent'])
            self.assertFalse(report['diffusion_run'])
            with self.assertRaisesRegex(ValueError,'different source pixels'):
                run_pilot(root/'audit.zip',root/'wrong_output',[case_id],root/'wrong.zip')


if __name__=='__main__':unittest.main()
