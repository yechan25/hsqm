import ast
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import torch
from PIL import Image

from diffusion_model.fitted_candidate_diffusion import CandidateObjective, InkConstraintObjective, load_case, optimize_clean, run_pair


class FittedCandidateTests(unittest.TestCase):
    def test_candidate_loss_removes_extra_restores_missing_and_preserves_background(self):
        source = torch.ones(1, 3, 16, 16)
        source[:, :, 3:5, 3:5] = 0  # Extra ink to erase.
        target = torch.ones_like(source)
        target[:, :, 10:12, 10:12] = 0  # Missing ink to restore.
        objective = CandidateObjective(source, target, radius=1)
        result, trace = optimize_clean(source, lambda z: z, objective,
            iterations=8, step_ratio=.02, budget_ratio=.3)
        self.assertGreater(trace['accepted_inner_steps'], 0)
        self.assertLess(trace['loss_after'], trace['loss_before'])
        self.assertGreater(float(result[:, :, 3:5, 3:5].mean()), 0)
        self.assertLess(float(result[:, :, 10:12, 10:12].mean()), 1)
        self.assertTrue(torch.equal(result[:, :, :2, :2], source[:, :, :2, :2]))

    def test_per_step_budget_bounds_many_inner_updates(self):
        source = torch.ones(1, 3, 8, 8)
        objective = CandidateObjective(source, torch.zeros_like(source))
        result, trace = optimize_clean(source, lambda z: z, objective,
            iterations=30, step_ratio=.2, budget_ratio=.05)
        self.assertLessEqual(float((result-source).square().mean().sqrt()), .050001)
        self.assertLess(trace['loss_after'], trace['loss_before'])

    def test_archive_uses_saved_page_and_learned_candidate_not_geometry_or_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'test.zip'
            def png(color):
                b = io.BytesIO(); Image.new('RGB', (16, 16), color).save(b, format='PNG'); return b.getvalue()
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('case_manifest.json', json.dumps([dict(case_id='case', page_id=3, input='/missing.png')]))
                z.writestr('generation/generation_report.json', json.dumps(dict(model_id='test')))
                z.writestr('generation/page_003.png', png('black'))
                z.writestr('case/normal_candidate.png', png('white'))
                z.writestr('case/geometry_candidate.png', png('red'))
            source, target, metadata = load_case(archive, 'case')
            self.assertEqual(source.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(target.getpixel((0, 0)), (255, 255, 255))
            self.assertFalse(metadata['candidate_is_ground_truth'])

    def test_notebook_embeds_current_runner_and_code_parses(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/fitted_candidate_diffusion.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 3)
        for source in code:
            ast.parse(source)
        assignments = [n for n in ast.parse(code[0]).body if isinstance(n, ast.Assign)]
        embedded = next(ast.literal_eval(n.value) for n in assignments
                        if any(isinstance(t, ast.Name) and t.id == 'HELPER_SOURCE' for t in n.targets))
        self.assertEqual(embedded, (root/'diffusion_model/fitted_candidate_diffusion.py').read_text(encoding='utf-8'))

    def test_paired_loop_reuses_initial_state_and_guidance_changes_diffusion_state(self):
        # Import native-extension dependencies before patch.dict(sys.modules),
        # which otherwise removes newly imported modules on leaving the mock.
        import scipy.ndimage
        import skimage.morphology
        # A deterministic epsilon/DDIM stand-in exercises the actual runner,
        # including clean-to-previous-state transfer and terminal VAE decoding.
        calls = []
        class Unet(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                calls.append((int(t), x.detach().clone()))
                return (torch.zeros_like(x),)
        class Vae(torch.nn.Module):
            config = SimpleNamespace(scaling_factor=1.)
            def decode(self, z, **kwargs): return (z,)
        class Scheduler:
            config = SimpleNamespace(num_train_timesteps=1000)
            alphas_cumprod = torch.linspace(.999, .1, 1000)
            final_alpha_cumprod = torch.tensor(1.)
            @classmethod
            def from_config(cls, *args, **kwargs): return cls()
            def set_timesteps(self, steps, **kwargs): self.num_inference_steps = steps
            def scale_model_input(self, inputs, t): return inputs
            def step(self, noise, t, latent, **kwargs):
                clean = (latent-(1-self.alphas_cumprod[t]).sqrt()*noise)/self.alphas_cumprod[t].sqrt()
                prev = int(t)-1000//self.num_inference_steps
                a = self.alphas_cumprod[prev] if prev >= 0 else self.final_alpha_cumprod
                return SimpleNamespace(pred_original_sample=clean,
                    prev_sample=a.sqrt()*clean+(1-a).sqrt()*noise)
        class Pipeline:
            def __init__(self):
                self.unet, self.vae, self.text_encoder = Unet(), Vae(), torch.nn.Identity()
                self.scheduler = Scheduler()
                self.image_processor = SimpleNamespace(preprocess=lambda im: im)
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def to(self, *args): return self
            def enable_attention_slicing(self): pass
            def enable_vae_slicing(self): pass
            def encode_prompt(self, *args, **kwargs): return torch.zeros(1, 1), torch.zeros(1, 1)
            def get_timesteps(self, *args): return torch.tensor([250, 0]), 2
            def prepare_latents(self, *args): return torch.full((1, 3, 16, 16), .3)
        fake = ModuleType('diffusers')
        fake.StableDiffusionImg2ImgPipeline, fake.DDIMScheduler = Pipeline, Scheduler
        from diffusion_model.fitted_candidate_diffusion import image_tensor
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('sys.modules', {'diffusers': fake}), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), \
             patch('diffusion_model.fitted_candidate_diffusion.image_tensor',
                   side_effect=lambda image, device: image_tensor(image, 'cpu')):
            class Dynamic(InkConstraintObjective):
                def __init__(self, source, candidate):
                    super().__init__(source, candidate)
                    self.history = []
                    self.evaluations = 0
                def maybe_refresh(self, image, step):
                    row = dict(step=step, erase_pixels=int((self.erase > 0).sum()))
                    self.history.append(row)
                    return row
                def evaluate_fixed(self, image):
                    self.evaluations += 1
                    return self(image)
                def export_state(self, path):
                    import numpy as np
                    np.savez(path, identity=np.arange(2))
            for mode, name in [('candidate', 'candidate_guided'), ('ink_constraints', 'ink_guided'), ('adaptive_ink', 'adaptive_guided')]:
                with self.subTest(mode=mode):
                    calls.clear()
                    out = Path(directory)/mode
                    target = Image.new('RGB', (16, 16), 'white')
                    if mode != 'candidate':
                        target.paste((0, 0, 0), (12, 2, 15, 14))
                    source = Image.new('RGB', (16, 16), 'gray')
                    dynamic = Dynamic(image_tensor(source, 'cpu'), image_tensor(target, 'cpu')) if mode == 'adaptive_ink' else None
                    report = run_pair(source, target, {}, out, steps=4, strength=.5,
                                      guidance_mode=mode, objective_override=dynamic)
                    self.assertTrue(torch.equal(calls[0][1], calls[2][1]))
                    self.assertTrue(report['paired_initial_latent'])
                    self.assertGreater(report['traces'][name][-1]['accepted_inner_steps'], 0)
                    self.assertLess(report['metrics'][name]['loss'], report['metrics']['redenoise_only']['loss'])
                    self.assertNotEqual((out/(name+'.png')).read_bytes(), (out/'fitted_candidate.png').read_bytes())
                    if dynamic is not None:
                        self.assertEqual(dynamic.evaluations, 3)  # Both arms + input, one fixed evaluator.
                        self.assertEqual(len(report['constraint_history']), 1)
                        self.assertTrue((out/'ownership_step_001.npz').is_file())


if __name__ == '__main__':
    unittest.main()
