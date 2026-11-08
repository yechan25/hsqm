from types import SimpleNamespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from PIL import Image

from diffusion_model.guided_ddim import denoise,GuidanceConfig
from stroke_model.diffusion_guidance import TopoGuidance
from stroke_model.stroke_structure import StructureConfig


class MaskModel(nn.Module):
    def __init__(self,swap=False):
        super().__init__();self.gain=nn.Parameter(torch.tensor(5.));self.swap=swap
    def forward(self,image,reference,strokes):
        probabilities=(strokes*self.gain).softmax(1)
        if self.swap:probabilities=probabilities.flip(1)
        return dict(probabilities=probabilities,masks=image*probabilities)


class TopoObjectiveTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_bad_stroke_receives_more_weight_and_candidates_keep_it_fixed(self):
        strokes=np.zeros((2,24,24),np.float32)
        strokes[0,4:7,3:20]=1;strokes[1,10:20,15:18]=1
        reference=strokes.max(0)[None]
        slots=[dict(reference=reference,strokes=strokes,bbox=[x,y,x+24,y+24])
               for y in (0,24) for x in (0,24)]
        cfg=StructureConfig(max_pairs=2,skeleton_iterations=2,junction_thresholds=(.5,),
            junction_min_levels=1,junction_distance_steps=2,junction_angular_samples=16)
        objective=TopoGuidance(MaskModel(),slots,cfg,extra_weight=0,
                               focus_strength=2,max_stroke_weight=2)
        ink=strokes[0]*.2+strokes[1]
        crop=torch.from_numpy(1-ink)[None].repeat(3,1,1)
        sheet=torch.cat([torch.cat([crop,crop],2)]*2,1)[None].requires_grad_(True)
        focused,details=objective(sheet)
        for row in details['per_glyph']:
            bad,good=row['per_stroke']
            self.assertLess(bad['score'],good['score'])
            self.assertGreater(bad['weight'],good['weight'])
            self.assertLessEqual(bad['weight'],2)
            self.assertGreaterEqual(good['weight'],1)
            self.assertFalse(bad['weight'].requires_grad)
            self.assertAlmostEqual(float(bad['weight_share']+good['weight_share']),1)
        frozen=objective.freeze_priority(details)
        normal_crop=torch.from_numpy(1-reference).repeat(3,1,1)
        normal=torch.cat([torch.cat([normal_crop,normal_crop],2)]*2,1)[None]
        _,candidate=objective(normal,stroke_weights=frozen)
        self.assertEqual(frozen,objective.freeze_priority(candidate))
        gradient,=torch.autograd.grad(focused,sheet)
        self.assertTrue(torch.isfinite(gradient).all())
        objective.focus_strength=0
        uniform,uniform_details=objective(sheet)
        self.assertGreater(float(focused.detach()),float(uniform.detach()))
        uniform_gradient,=torch.autograd.grad(uniform,sheet)
        # Focus must actually strengthen the bad stroke's image gradient.
        self.assertGreater(float(gradient[...,4:7,3:20].abs().sum()),
                           float(uniform_gradient[...,4:7,3:20].abs().sum()))
        for row in uniform_details['per_glyph']:
            self.assertAlmostEqual(float(row['stroke_loss'].detach()),float(row['uniform_stroke_loss'].detach()))

    def test_all_four_crops_have_gradients_and_model_weights_are_frozen(self):
        strokes=np.zeros((2,24,24),np.float32)
        strokes[0,4:7,3:20]=1;strokes[1,10:20,15:18]=1
        reference=strokes.max(0)[None]
        slots=[dict(reference=reference,strokes=strokes,bbox=[x,y,x+24,y+24])
               for y in (0,24) for x in (0,24)]
        cfg=StructureConfig(max_pairs=2,skeleton_iterations=2,junction_thresholds=(.5,),
            junction_min_levels=1,junction_distance_steps=2,junction_angular_samples=16)
        model=MaskModel();objective=TopoGuidance(model,slots,cfg,extra_weight=0)
        image=torch.full((1,3,48,48),.5,requires_grad=True)
        loss,details=objective(image)
        gradient,=torch.autograd.grad(loss,image)
        for slot in slots:
            x0,y0,x1,y1=slot['bbox']
            self.assertGreater(gradient[...,y0:y1,x0:x1].abs().sum(),0)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertFalse(model.gain.requires_grad);self.assertIsNone(model.gain.grad)
        # Same source pixels, changed Topo ownership: primary loss MUST change.
        good=[]
        for _ in range(2):good.append(torch.from_numpy(1-reference).repeat(3,1,1))
        sheet=torch.cat([torch.cat(good,2),torch.cat(good,2)],1)[None]
        normal,_=objective(sheet)
        model.swap=True
        swapped,_=objective(sheet)
        self.assertGreater(float(swapped),float(normal)+.1)


class GuidedDDIMTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        try:
            from diffusers import DDIMScheduler
        except ImportError as exc:
            self.skipTest('Install diffusers==0.35.1 for the real DDIM verification: '+str(exc))
        scheduler=DDIMScheduler(num_train_timesteps=40,clip_sample=False,prediction_type='epsilon')
        scheduler.set_timesteps(8)
        class UNet(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(.1),requires_grad=False)
            def forward(self,x,t,encoder_hidden_states=None,return_dict=False):return (self.scale*x,)
        class VAE(nn.Module):
            config=SimpleNamespace(scaling_factor=1.)
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.eye(3),requires_grad=False)
            def decode(self,z,return_dict=False):return (torch.einsum('ij,bjhw->bihw',self.weight,z),)
        self.pipe=SimpleNamespace(scheduler=scheduler,unet=UNet(),vae=VAE())
        self.initial=torch.randn((1,3,8,8),generator=torch.Generator().manual_seed(100))*.2
        self.embed=torch.zeros(2,1,1)

    def objective(self,rgb):
        loss=(rgb-.3).square().mean()
        return loss,dict(per_glyph=[dict(slot=0,loss=loss)])

    def test_zero_guidance_is_bit_identical_to_the_real_ddim_loop(self):
        expected=self.initial.clone()
        with torch.no_grad():
            for t in self.pipe.scheduler.timesteps:
                noise=self.pipe.unet(expected,t)[0]
                expected=self.pipe.scheduler.step(noise,t,expected,eta=0).prev_sample
        actual,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=self.objective,config=GuidanceConfig(noise_ratio=0))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertFalse(any(row['accepted'] for row in trace))

    def test_guidance_has_correct_sign_and_each_accepted_step_reduces_loss(self):
        original=self.initial.clone()
        base,_=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            config=GuidanceConfig(noise_ratio=0))
        guided,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=self.objective,config=GuidanceConfig(noise_ratio=.2,start_fraction=0,end_fraction=1,every=1))
        accepted=[row for row in trace if row['accepted']]
        self.assertTrue(accepted)
        for row in accepted:
            self.assertLess(row['loss_after'],row['loss_before'])
            self.assertGreater(row['gradient_rms'],0)
            self.assertGreater(row['state_delta_rms'],0)
        self.assertLess(self.objective(guided/2+.5)[0],self.objective(base/2+.5)[0])
        torch.testing.assert_close(self.initial,original,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in self.pipe.vae.parameters()))
        self.assertTrue(all(p.grad is None for p in self.pipe.unet.parameters()))

    def test_clean_budget_corrects_the_final_step(self):
        output,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=self.objective,config=GuidanceConfig(noise_ratio=0,clean_step_ratio=.01,
                start_fraction=0,end_fraction=1,every=1))
        final=trace[-1]
        self.assertTrue(final['active'])
        self.assertTrue(final['accepted'])
        self.assertEqual(final['correction_budget'],'clean_latent_rms')
        self.assertGreater(final['clean_delta_rms'],0)
        self.assertGreater(final['state_delta_rms'],0)
        self.assertLess(final['loss_after'],final['loss_before'])
        self.assertTrue(torch.isfinite(output).all())

    def test_residual_preserving_update_transfers_the_full_clean_correction(self):
        kwargs=dict(clean_step_ratio=.01,start_fraction=0,end_fraction=1,every=1)
        _,epsilon_trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=self.objective,config=GuidanceConfig(**kwargs,state_update='epsilon'))
        output,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=self.objective,config=GuidanceConfig(**kwargs,state_update='clean_residual'))
        old=next(r for r in epsilon_trace if r['accepted'])
        new=next(r for r in trace if r['accepted'])
        self.assertGreater(new['state_delta_rms'],old['state_delta_rms']*3)
        for r in trace:
            if not r['accepted']:continue
            prev=r['timestep']-self.pipe.scheduler.config.num_train_timesteps//self.pipe.scheduler.num_inference_steps
            alpha=(self.pipe.scheduler.alphas_cumprod[prev] if prev>=0 else self.pipe.scheduler.final_alpha_cumprod)
            self.assertAlmostEqual(r['state_to_clean_transfer_ratio'],float(alpha.sqrt()),places=4)
            self.assertLess(r['loss_after'],r['loss_before'])
        expected,_=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            config=GuidanceConfig(noise_ratio=0))
        disabled,_=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            config=GuidanceConfig(clean_step_ratio=0,state_update='clean_residual'))
        torch.testing.assert_close(expected,disabled,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(output).all())

    def test_half_unet_keeps_sampling_state_and_corrections_in_float32(self):
        self.pipe.unet.half()
        initial=self.initial.half()
        output,trace=denoise(self.pipe,initial,self.pipe.scheduler.timesteps,self.embed,
            objective=self.objective,config=GuidanceConfig(noise_ratio=.05,start_fraction=0,end_fraction=1,every=1))
        self.assertEqual(output.dtype,torch.float32)
        accepted=[r for r in trace if r['accepted']]
        self.assertTrue(accepted)
        self.assertTrue(all(r['state_delta_rms']>0 for r in accepted))

    def test_residual_update_uses_scheduler_final_alpha_with_offset_timesteps(self):
        from diffusers import DDIMScheduler
        self.pipe.scheduler=DDIMScheduler(num_train_timesteps=1000,clip_sample=False,
            steps_offset=1,set_alpha_to_one=False,prediction_type='epsilon')
        self.pipe.scheduler.set_timesteps(30)
        _,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps[-2:],self.embed,
            objective=self.objective,config=GuidanceConfig(clean_step_ratio=.01,start_fraction=0,
                end_fraction=1,every=1,state_update='clean_residual'))
        final=trace[-1]
        self.assertEqual(final['timestep'],1)
        self.assertTrue(final['accepted'])
        self.assertAlmostEqual(final['state_to_clean_transfer_ratio'],
            float(self.pipe.scheduler.final_alpha_cumprod.sqrt()),places=4)

    def test_missing_objective_cannot_silently_enable_guidance(self):
        with self.assertRaises(ValueError):
            denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,config=GuidanceConfig())

    def test_candidate_that_lowers_loss_but_lowers_primary_score_is_rejected(self):
        def conflicting(rgb):
            loss=(rgb-.3).square().mean()
            return loss,dict(per_glyph=[dict(slot=0,extracted_stroke_score_no_extra=100*loss)])
        base,_=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            config=GuidanceConfig(noise_ratio=0))
        output,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=conflicting,config=GuidanceConfig(noise_ratio=.2,start_fraction=0,end_fraction=1,every=1))
        self.assertFalse(any(row['accepted'] for row in trace))
        torch.testing.assert_close(output,base,rtol=0,atol=0)

    def test_healthy_stroke_guard_rejects_even_when_average_score_improves(self):
        def conflict(rgb):
            loss=(rgb-.3).square().mean()
            healthy=95+10*loss;bad=50-30*loss
            return loss,dict(per_glyph=[dict(slot=0,extracted_stroke_score_no_extra=(healthy+bad)/2,
                per_stroke=[dict(stroke=1,score=healthy),dict(stroke=2,score=bad)])])
        base,_=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            config=GuidanceConfig(noise_ratio=0))
        output,trace=denoise(self.pipe,self.initial,self.pipe.scheduler.timesteps,self.embed,
            objective=conflict,config=GuidanceConfig(noise_ratio=.2,start_fraction=0,end_fraction=1,
                every=1,max_healthy_stroke_drop=0))
        self.assertFalse(any(r['accepted'] for r in trace))
        reasons=[a['reason'] for r in trace if r['active'] for a in r['attempts']]
        self.assertIn('healthy_stroke_worse',reasons)
        torch.testing.assert_close(output,base,rtol=0,atol=0)

    def test_actual_tiny_diffusers_unet_and_vae_backward(self):
        from diffusers import UNet2DConditionModel,AutoencoderKL
        torch.manual_seed(17)
        self.pipe.unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
            layers_per_block=1,block_out_channels=(16,32),
            down_block_types=('DownBlock2D','CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D','UpBlock2D'),
            cross_attention_dim=8,attention_head_dim=4,norm_num_groups=4).eval().requires_grad_(False)
        self.pipe.vae=AutoencoderKL(in_channels=3,out_channels=3,
            down_block_types=('DownEncoderBlock2D',),up_block_types=('UpDecoderBlock2D',),
            block_out_channels=(16,),latent_channels=4,norm_num_groups=4,sample_size=8).eval().requires_grad_(False)
        self.pipe.scheduler.set_timesteps(4)
        initial=torch.randn(1,4,8,8)*.2
        output,trace=denoise(self.pipe,initial,self.pipe.scheduler.timesteps,torch.zeros(2,2,8),
            objective=self.objective,config=GuidanceConfig(noise_ratio=.1,start_fraction=0,end_fraction=1,every=1))
        self.assertEqual(output.shape,initial.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(any(row['accepted'] for row in trace))
        self.assertTrue(all(p.grad is None for p in self.pipe.vae.parameters()))

    def test_appearance_signal_survives_ddim_when_structure_is_already_good(self):
        from test_stroke_appearance import IdealSingleStroke
        ref=np.zeros((1,24,24),np.float32);ref[:,10:13,4:20]=1
        slots=[dict(reference=ref,strokes=ref,bbox=[x,y,x+24,y+24]) for y in (0,24) for x in (0,24)]
        small=StructureConfig(max_pairs=2,skeleton_iterations=2,junction_thresholds=(.5,),
            junction_min_levels=1,junction_distance_steps=2,junction_angular_samples=16)
        objective=TopoGuidance(IdealSingleStroke(),slots,small,extra_weight=0,focus_strength=2,appearance_weight=.5)
        ink=ref[0]*.6+.2;ink[9,10:15]=.8
        crop=torch.from_numpy(1-2*ink)[None].repeat(3,1,1)
        initial=torch.cat([torch.cat([crop,crop],2)]*2,1)[None]
        self.pipe.unet.scale.fill_(0.)
        self.pipe.scheduler.set_timesteps(4)
        base,_=denoise(self.pipe,initial,self.pipe.scheduler.timesteps,self.embed,
            config=GuidanceConfig(noise_ratio=0))
        guided,trace=denoise(self.pipe,initial,self.pipe.scheduler.timesteps,self.embed,
            objective=objective,config=GuidanceConfig(clean_step_ratio=.01,start_fraction=0,end_fraction=1,every=1))
        accepted=[r for r in trace if r['accepted']]
        self.assertTrue(accepted)
        self.assertGreater(float((guided-base).abs().sum()),0)
        with torch.no_grad():
            _,baseline_details=objective(base/2+.5)
            _,guided_details=objective(guided/2+.5)
        before=sum(float(r['appearance_loss']) for r in baseline_details['per_glyph'])
        after=sum(float(r['appearance_loss']) for r in guided_details['per_glyph'])
        self.assertLess(after,before)
        for record in accepted:
            for a,b in zip(record['per_glyph_before'],record['per_glyph_after']):
                self.assertLess(b['appearance_loss'],a['appearance_loss'])

    def test_paired_runner_saves_same_start_and_primary_mask_scores(self):
        import matplotlib
        matplotlib.use('Agg')
        from diffusion_model.run_topo_guidance import generate_guided_pair,evaluate_topo_pair
        from diffusers import DDIMScheduler
        pipe=self.pipe
        pipe._execution_device=torch.device('cpu')
        pipe.text_encoder=nn.Linear(1,1)
        pipe.encoder_calls=0
        pipe.to=lambda device:pipe
        pipe.enable_attention_slicing=lambda:None
        pipe.encode_prompt=lambda *args,**kwargs:(torch.zeros(1,1,1),torch.zeros(1,1,1))
        def prepare(image,*args):
            pipe.encoder_calls+=1
            return image*.3
        pipe.prepare_latents=prepare
        pipe.get_timesteps=lambda steps,strength,device:(pipe.scheduler.timesteps[-int(steps*strength):],int(steps*strength))
        pipe.run_safety_checker=lambda image,*args:(image,[False])
        def postprocess(image,**kwargs):
            a=((image[0].detach().permute(1,2,0).numpy()+1)/2).clip(0,1)
            return [Image.fromarray(np.rint(a*255).astype(np.uint8))]
        pipe.image_processor=SimpleNamespace(
            preprocess=lambda image:torch.from_numpy(np.array(image,dtype=np.float32)/255).permute(2,0,1)[None]*2-1,
            postprocess=postprocess)
        small=StructureConfig(max_pairs=2,skeleton_iterations=2,junction_thresholds=(.5,),
            junction_min_levels=1,junction_distance_steps=2,junction_angular_samples=16)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            for i in range(4):
                strokes=np.zeros((2,24,24),np.float32)
                strokes[0,3+i:6+i,3:20]=1;strokes[1,10:20,15:18]=1
                np.savez(root/f'{i}.npz',reference=strokes.max(0)[None],strokes=strokes)
                rows.append(dict(path=f'{i}.npz',split='train',group_id=str(i)))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
            checkpoint=root/'best.pt';checkpoint.write_bytes(b'immutable mock checkpoint')
            model=MaskModel();model.output_mode='partition'
            fake_class=SimpleNamespace(from_pretrained=lambda *args,**kwargs:pipe)
            with patch('torch.cuda.is_available',return_value=True), \
                 patch('diffusers.StableDiffusionImg2ImgPipeline',fake_class), \
                 patch('diffusion_model.run_topo_guidance.load_topology',return_value=(model,24,1)):
                report=generate_guided_pair(manifest,[1,2,3,4],checkpoint,root/'run',steps=8,
                    canvas_size=48,structure_config=asdict(small),
                    guidance=dict(clean_step_ratio=.01,state_update='clean_residual'),compare_uniform=True)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(pipe.encoder_calls,1)
            self.assertEqual(report['seed'],100);self.assertEqual(report['strength'],.65)
            self.assertEqual(len({p['initial_latent_sha256'] for p in report['pages']}),1)
            self.assertEqual(checkpoint.read_bytes(),b'immutable mock checkpoint')
            with patch('diffusion_model.run_topo_guidance.load_topology',return_value=(model,24,1)):
                frame,extraction,zip_path=evaluate_topo_pair(root/'run/generation_report.json',checkpoint,show=False)
            self.assertEqual(len(frame),20)
            self.assertIn('extracted_stroke_score_no_extra',frame)
            self.assertTrue(Path(zip_path).is_file())
            self.assertEqual(len(extraction['results']),20)
            self.assertEqual({p['method'] for p in report['pages']},
                             {'reference','baseline','uniform_guided','structure_guided','guided'})
            self.assertEqual(report['appearance_weight'],.5)
            self.assertEqual(report['appearance_target'],'extractor_reference')
            self.assertTrue(all(r['calibrated_reference_error']==0 for r in report['reference_calibration']))
            self.assertEqual(next(p for p in report['pages'] if p['method']=='structure_guided')['appearance_weight'],0)
            self.assertEqual(report['stroke_focus']['focus_strength'],2.)
            import pandas as pd
            scores=pd.read_csv(Path(zip_path).parent/'stroke_scores.csv')
            self.assertEqual(len(scores),40)
            self.assertTrue((Path(zip_path).parent/'appearance_comparison.csv').is_file())
            self.assertIn('appearance_loss',scores)
            self.assertTrue((Path(zip_path).parent/'stroke_comparison.csv').is_file())
            with np.load(extraction['results'][-1]['npz']) as arrays:
                self.assertEqual(arrays['stroke_rgb_gradients'].shape,(2,24,24))
                self.assertTrue(np.isfinite(arrays['stroke_rgb_gradients']).all())
                self.assertEqual(arrays['missing_maps'].shape,(2,24,24))
                self.assertEqual(arrays['contour_map'].shape,(2,24,24))
            self.assertTrue((frame[frame.method=='reference']['appearance_loss']==0).all())
            accepted=[json.loads(s) for s in (root/'run/trace_guided.jsonl').read_text().splitlines()
                      if json.loads(s)['accepted']]
            self.assertTrue(accepted)
            for record in accepted:
                before=[s['weight'] for r in record['per_glyph_before'] for s in r['per_stroke']]
                after=[s['weight'] for r in record['per_glyph_after'] for s in r['per_stroke']]
                self.assertEqual(before,after)
            from diffusion_model.guidance_diagnostics import diagnose_pair
            summary,pixels=diagnose_pair(root/'run',show=False)
            self.assertEqual(summary['ddim_state_dtype'],'float32')
            self.assertEqual(len(pixels),4)
            self.assertIsNotNone(summary['final_latent_rms_difference'])
            history=pd.read_csv(root/'run/stroke_guidance_trace.csv')
            self.assertEqual(len(history),8)
            self.assertTrue(history['accepted'].all())
            from diffusion_model.inspect_guidance_archive import inspect_archive
            audit=inspect_archive(zip_path,root/'audit')
            self.assertFalse(audit['resampled'])
            self.assertTrue(all(r['reference_response_appearance_loss']==0 for r in audit['appearance']
                                if r['method']=='reference'))


if __name__=='__main__':unittest.main()
