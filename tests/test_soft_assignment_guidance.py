import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt
from stroke_model.soft_assignment_guidance import SoftAssignmentGuidance
from diffusion_model.guided_ddim import GuidanceConfig,denoise,format_guidance_record


class Extractor(nn.Module):
    def __init__(self,swap=False):
        super().__init__();self.gain=nn.Parameter(torch.tensor(1.));self.calls=0;self.swap=swap
    def forward(self,ink,reference,strokes):
        self.calls+=1
        distances=torch.stack([torch.as_tensor(distance_transform_edt(s.cpu().numpy()<.1),device=ink.device,dtype=ink.dtype) for s in strokes[0]])[None]
        p=(-distances*self.gain).softmax(1)
        if self.swap:p=p.flip(1)
        return dict(probabilities=p,masks=ink*p)


def example():
    strokes=np.zeros((2,32,32),np.float32)
    strokes[0,4:28,22]=1;strokes[1,16,23:30]=1
    reference=strokes.max(0)[None]
    rgb=torch.ones(1,3,32,32)
    rgb[:,:,4:28,21:24]=0;rgb[:,:,15:18,23:30]=0
    rgb[:,:,15:18,13:21]=0
    return rgb,[dict(reference=reference,strokes=strokes,bbox=[0,0,32,32])]


class SoftAssignmentTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_console_distinguishes_rejected_guidance_from_inactive_schedule(self):
        inactive=format_guidance_record(dict(active=False,accepted=False))
        rejected=format_guidance_record(dict(active=True,accepted=False,edit_loss_before=1.,edit_loss_after=1.,
            attempts=[dict(reason='healthy_guard',guard_failures=[dict(name='slot_2_stroke_1_mass_drop')])],
            anchor_guard_failures=[dict(name='protected_rms')]))
        self.assertIn('state=inactive',inactive)
        self.assertIn('active=True',rejected);self.assertIn('state=rejected',rejected)
        self.assertIn('slot_2_stroke_1_mass_drop',rejected)
        self.assertIn('anchor_already_outside_guard=True',rejected)

    def test_probabilities_do_not_cancel_and_geometry_is_strong_control(self):
        rgb,slots=example()
        normal=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        swapped=SoftAssignmentGuidance(Extractor(True),slots,rgb,allowance=2)
        geometry=SoftAssignmentGuidance(Extractor(),slots,rgb,mode='geometry',allowance=2)
        self.assertGreater((normal.initial_weights-swapped.initial_weights).abs().sum(),1)
        self.assertGreater((normal.initial_weights-geometry.initial_weights).abs().sum(),.01)
        self.assertTrue((normal.initial_weights>=geometry.initial_weights-1e-6).all())

    def test_inner_search_freezes_assignment_and_reduces_actual_edit_loss(self):
        rgb,slots=example();model=Extractor()
        objective=SoftAssignmentGuidance(model,slots,rgb,allowance=2)
        calls=model.calls
        original_weights=model.gain.detach().clone()
        edited,record=objective.optimize_clean(rgb,lambda z:z,iterations=12,step_ratio=.02,
            backtracks=6,budget_ratio=.3)
        self.assertGreater(record['accepted_inner_steps'],1)
        self.assertLess(record['edit_loss_after'],record['edit_loss_before']*.9)
        self.assertEqual(model.calls,calls+1)
        self.assertFalse(objective.weights.requires_grad)
        self.assertIn('fixed_map_edit_after',record)
        self.assertIn('anchor_guard_failures',record)
        self.assertIsNone(model.gain.grad)
        torch.testing.assert_close(model.gain,original_weights)
        self.assertLessEqual(record['healthy_mass_drop'],record['guard_limits'][2]+1e-6)
        self.assertGreater(edited[:,:,15:18,13:18].mean(),rgb[:,:,15:18,13:18].mean())
        for item in record['attempts']:
            if item['reason']=='accepted':self.assertLess(item['edit_after'],item['edit_before'])

    def test_normal_stroke_erasure_violates_guard(self):
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        damaged=rgb.clone();damaged[:,:,4:28,21:24]=1
        self.assertGreater(objective.guard_values(damaged)[2],.5)

    def test_recorded_run_small_vae_crosstalk_has_a_real_tolerance(self):
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        # Actual rejected step 34 from 20261111_043924_692872.zip.
        objective.baseline_guards[34]=torch.tensor([.0398136489,.7086679339,.0723928586,.01])
        limits=objective.guard_limits(rgb,34)
        candidate=torch.tensor([.0398248844,.7086440325,.0724010468,.01])
        self.assertTrue((candidate<=limits).all())
        self.assertLess(limits[0],.043)

    def test_guard_budget_does_not_accumulate_with_inner_edits(self):
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        objective.record_baseline(0,rgb)
        limits=objective.guard_limits(rgb,0)
        damaged=rgb+.4
        torch.testing.assert_close(objective.guard_limits(damaged,0),limits)

    def test_each_stroke_has_its_own_matched_baseline_limit(self):
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        objective.baseline_guards[0]=torch.tensor([.04,.7,.01,.4])
        limits=objective.guard_limits(rgb,0)
        candidate=torch.tensor([.04,.7,.10,.39])
        self.assertTrue(candidate[2]>limits[2])
        self.assertTrue(candidate[3]<limits[3])

    def test_nonfinite_candidates_are_rejected_not_silently_accepted(self):
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        def decode(value):
            if not value.requires_grad and (value-rgb).abs().amax()>1e-7:return value*float('nan')
            return value
        edited,record=objective.optimize_clean(rgb,decode,iterations=2,step_ratio=.02,backtracks=3,budget_ratio=.2)
        self.assertFalse(record['accepted'])
        self.assertTrue(all(a['reason']=='nonfinite_candidate' for a in record['attempts']))
        torch.testing.assert_close(edited,rgb)
        json.dumps(record,allow_nan=False)

    def test_edit_map_refresh_tracks_new_ink_without_forgetting_initial_defect(self):
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        white=torch.ones_like(rgb);objective.refresh(white)
        self.assertTrue((objective.weights>=.25*objective.initial_weights).all())
        modified=rgb.clone();modified[:,:,20:23,10:13]=0
        before=objective.weights.clone();objective.refresh(modified)
        self.assertGreater(objective.weights[:,:,20:23,10:13].sum(),before[:,:,20:23,10:13].sum())

    def test_real_ddim_integration_updates_state_with_iterative_objective(self):
        from diffusers import DDIMScheduler
        class VAE(nn.Module):
            config=SimpleNamespace(scaling_factor=1.)
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.))
            def decode(self,z,return_dict=False):return (z*self.weight,)
        class UNet(nn.Module):
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(0.))
            def forward(self,x,t,**kwargs):return (x*0+self.weight,)
        rgb,slots=example();model=Extractor()
        objective=SoftAssignmentGuidance(model,slots,rgb,allowance=2)
        vae=VAE().requires_grad_(False);unet=UNet().requires_grad_(False)
        scheduler=DDIMScheduler(clip_sample=False,thresholding=False);scheduler.set_timesteps(6)
        pipe=SimpleNamespace(vae=vae,unet=unet,scheduler=scheduler)
        initial=(rgb*2-1)*scheduler.alphas_cumprod[int(scheduler.timesteps[0])].sqrt()
        base,_=denoise(pipe,initial,scheduler.timesteps,torch.zeros(1),objective=objective,config=GuidanceConfig(noise_ratio=0),cfg_scale=1)
        self.assertEqual(len(objective.baseline_guards),len(scheduler.timesteps))
        result,trace=denoise(pipe,initial,scheduler.timesteps,torch.zeros(1),objective=objective,cfg_scale=1,
            config=GuidanceConfig(clean_step_ratio=.01,state_update='clean_residual',every=1,start_fraction=0,
                end_fraction=1,inner_iterations=3,final_inner_iterations=5))
        self.assertGreater(sum(t['accepted'] for t in trace),0)
        self.assertGreater((result-base).abs().sum(),0)
        self.assertTrue(any(t.get('accepted_inner_steps',0)>1 for t in trace))
        self.assertTrue(all(t['guard_reference']=='matched_unguided_step' for t in trace if t['active']))
        self.assertIsNone(vae.weight.grad);self.assertIsNone(unet.weight.grad)
        json.dumps(trace)

    def test_actual_diffusers_vae_connects_loss_to_latent_without_training(self):
        from diffusers import AutoencoderKL
        torch.manual_seed(8)
        vae=AutoencoderKL(in_channels=3,out_channels=3,
            down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D','UpDecoderBlock2D'),
            block_out_channels=(8,16),layers_per_block=1,latent_channels=4,norm_num_groups=4).eval().requires_grad_(False)
        rgb,slots=example();objective=SoftAssignmentGuidance(Extractor(),slots,rgb,allowance=2)
        with torch.no_grad():latent=vae.encode(rgb*2-1).latent_dist.mode()
        leaf=latent.detach().requires_grad_(True)
        decoded=vae.decode(leaf).sample/2+.5
        value,_=objective(decoded)
        gradient,=torch.autograd.grad(value,leaf)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum(),0)
        self.assertTrue(all(p.grad is None for p in vae.parameters()))

    def test_colab_contains_patched_runtime_and_no_drive_mount(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/soft_assignment_repair_no_drive.ipynb'
        n=json.loads(path.read_text(encoding='utf-8'))
        cells=[c for c in n['cells'] if c['cell_type']=='code'];self.assertEqual(len(cells),3)
        source='\n'.join(''.join(c['source']) for c in cells)
        self.assertNotIn('drive.mount(',source)
        self.assertIn('stroke_model/soft_assignment_guidance.py',source)
        self.assertIn('diffusion_model/guided_ddim.py',source)
        self.assertIn('RUN_GEOMETRY_CONTROL=True',source)
        self.assertIn('FOCUS_SLOTS=(0,1,2,3)',source)
        self.assertIn('for row,index in enumerate(FOCUS_SLOTS)',source)


if __name__=='__main__':unittest.main()
