import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image
from diffusion_model.qwen_repair import optimize_clean, guided_flow_steps, repair_qwen_pair


class FlowScheduler:
    def __init__(self):
        self.sigmas=torch.tensor([.6,.3,0.]);self.step_index=None
    def _init_step_index(self,timestep):self.step_index=0
    def step(self,velocity,timestep,sample,return_dict=False):
        if self.step_index is None:self._init_step_index(timestep)
        index=self.step_index
        result=sample+(self.sigmas[index+1]-self.sigmas[index])*velocity
        self.step_index+=1
        return (result,)


def loss(value):return value.square().mean(),{}


class QwenFlowTests(unittest.TestCase):
    def test_real_flow_scheduler_contract(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler=FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(2)
        history=[];sample=torch.ones(1,4,4)
        with guided_flow_steps(scheduler,lambda x:x,loss,history):
            with torch.no_grad():
                for timestep in scheduler.timesteps:
                    sample=scheduler.step(torch.zeros_like(sample),timestep,sample,return_dict=False)[0]
        self.assertEqual(len(history),2)
        self.assertTrue(all(row['accepted'] for row in history))
        self.assertLess(sample.square().mean().item(),1)

    def test_pair_uses_generated_source_and_identical_noise(self):
        class Scheduler(FlowScheduler):
            config={}
            @classmethod
            def from_config(cls,config):return cls()
        class Pipeline:
            scheduler=Scheduler();vae_scale_factor=1
            vae=SimpleNamespace(dtype=torch.float32,
                config=SimpleNamespace(latents_mean=[0,0,0],latents_std=[1,1,1]),
                decode=lambda x,return_dict:(x,))
            @classmethod
            def from_pipe(cls,other):return cls()
            def _unpack_latents(self,value,height,width,scale):return value.reshape(1,3,1,height,width)
            @torch.no_grad()
            def __call__(self,*,image,prompt,negative_prompt,width,height,strength,num_inference_steps,true_cfg_scale,generator):
                self.asserted_source=image.size
                self.scheduler.step_index=None
                sample=torch.rand((1,3*height*width),generator=generator)*.2
                for timestep in (600,300):
                    sample=self.scheduler.step(torch.zeros_like(sample),timestep,sample,return_dict=False)[0]
                pixels=((sample.reshape(3,height,width).permute(1,2,0)/2+.5)*255).numpy().astype('uint8')
                return SimpleNamespace(images=[Image.fromarray(pixels)])
        workspace=Path(__file__).resolve().parents[1]/'outputs';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            source=Path(directory)/'generated.png';Image.new('RGB',(16,16),'white').save(source)
            with patch.dict('sys.modules',{'diffusers':SimpleNamespace(QwenImageImg2ImgPipeline=Pipeline)}):
                report=repair_qwen_pair(Pipeline(),source,Path(directory)/'repair',loss,
                    prompt='가나다라',steps=4,strength=.5)
            self.assertEqual(len(report['paired_initial_latent_sha256']),2)
            self.assertEqual(*report['paired_initial_latent_sha256'])
            self.assertEqual(report['guided_accepted_steps'],2)
            self.assertEqual(report['input_image'],str(source))
            self.assertTrue((Path(directory)/'repair/guided_step_0002_after.png').exists())
            self.assertFalse(report['clean_reference_used'] or report['rgb_painting'])

    def test_guidance_differentiates_clean_not_transformer(self):
        # Called under the same no-grad context as the native diffusion pipeline.
        with torch.no_grad():
            clean=torch.ones(1,4,4)
            edited,record=optimize_clean(clean,lambda x:x,loss,step_size=.2)
        self.assertTrue(record['accepted'])
        self.assertLess(record['after'],record['before'])
        self.assertFalse(edited.requires_grad)

    def test_flow_residual_and_native_scheduler_are_preserved(self):
        scheduler=FlowScheduler();original=scheduler.step;history=[]
        sample=torch.ones(1,2,4);velocity=torch.full_like(sample,.2)
        with guided_flow_steps(scheduler,lambda x:x,loss,history,step_size=.1):
            result=scheduler.step(velocity,torch.tensor(600.),sample,return_dict=False)[0]
        # clean=.88; edit=-.1; native next=.94; clean residual=.7*(-.1)
        torch.testing.assert_close(result,torch.full_like(sample,.87))
        self.assertEqual(scheduler.step,original)
        self.assertTrue(history[0]['accepted'])
        self.assertEqual(scheduler.step_index,1)

    def test_scheduler_restored_on_decode_failure(self):
        scheduler=FlowScheduler();original=scheduler.step
        def broken(value):raise RuntimeError('decode failed')
        with self.assertRaisesRegex(RuntimeError,'decode failed'):
            with guided_flow_steps(scheduler,broken,loss,[]):
                scheduler.step(torch.zeros(1),torch.tensor(600.),torch.ones(1),return_dict=False)
        self.assertEqual(scheduler.step,original)

    def test_rejected_edit_keeps_original_flow(self):
        scheduler=FlowScheduler();history=[]
        def constant_decode(value):return value*0+1
        sample=torch.ones(1);velocity=torch.ones(1)*.2
        with guided_flow_steps(scheduler,constant_decode,loss,history):
            result=scheduler.step(velocity,600,sample,return_dict=False)[0]
        torch.testing.assert_close(result,torch.tensor([.94]))
        self.assertFalse(history[0]['accepted'])


if __name__=='__main__':unittest.main()
