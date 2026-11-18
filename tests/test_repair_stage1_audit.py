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

from repair_audit.stage1 import (read_archive, measure, run_case,
    width_probe_conditions, write_tables)


def png(image):
    stream = io.BytesIO(); image.save(stream, format='PNG'); return stream.getvalue()


def condition_bytes():
    reference = np.zeros((1, 16, 16), np.float32)
    reference[:, 3:13, 7:10] = 1
    stream = io.BytesIO()
    np.savez_compressed(stream, reference=reference, strokes=reference)
    return stream.getvalue()


def create_archive(path):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('experiment.json', json.dumps(dict(checkpoint_sha256='a'*64)))
        z.writestr('cases.json', json.dumps([dict(case_id='group_000_page_001', group_id='group_000',
            control=False, input='/untrusted/absolute/location/page_001.png')]))
        z.writestr('runtime_helpers.json', 'THIS IS NOT EXECUTABLE DATA')
        for slot in range(1, 5):
            z.writestr(f'group_000/source_conditions/condition_{slot}.npz', condition_bytes())
        pages = [dict(page_id=i, input=f'/old/page_{i:03d}.png', control=(i == 0),
                      glyph_count=4, safety_filtered=False) for i in range(2)]
        z.writestr('group_000/generation/generation_report.json', json.dumps(dict(pages=pages, prompts={'4':'test'})))
        for i in range(2):
            z.writestr(f'group_000/generation/page_{i:03d}.png', png(Image.new('RGB', (32, 32), 'white')))


class AuditTests(unittest.TestCase):
    def test_archive_restores_named_data_and_normal_control_without_executing_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); create_archive(root/'input.zip')
            cases, experiment = read_archive(root/'input.zip', root/'out')
            self.assertEqual([c['kind'] for c in cases], ['generated', 'normal_control'])
            self.assertEqual(experiment['checkpoint_sha256'], 'a'*64)
            self.assertTrue((root/'out/group_000_page_001/condition_4.npz').is_file())
            self.assertFalse((root/'out/runtime_helpers.json').exists())
            with self.assertRaises(ValueError):
                read_archive(root/'input.zip', root/'bad', ['missing'])

    def test_fixed_width_proxy_counts_removed_ink_instead_of_dropping_zero_pixels(self):
        source = np.ones((32, 32, 3), np.uint8)*255
        source[4:28, 10:17] = 0
        thin = np.ones_like(source)*255; thin[4:28, 12:15] = 0
        healthy = source[..., 0] == 0
        center = np.zeros((32, 32), bool); center[8:24, 13] = True
        support = dict(healthy=healthy, center=center)
        original = Image.fromarray(source)
        before = measure(original, original, original, support)
        after = measure(original, Image.fromarray(thin), original, support)
        absent = measure(original, Image.new('RGB', (32,32),'white'), original, support)
        self.assertAlmostEqual(before['fixed_center_width_ratio_proxy'], 1)
        self.assertLess(after['fixed_center_width_ratio_proxy'], 1)
        self.assertGreater(after['healthy_ink_drop'], 0)
        self.assertEqual(absent['fixed_center_width_ratio_proxy'], 0)
        # Reference distance alone prefers the reference, but is never named accuracy.
        self.assertIn('reference_distance_on_fixed_union', after)

    def test_notebook_is_standalone_and_all_cells_parse(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/repair_audit/stage1.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 5)
        for cell in code: ast.parse(cell)
        assignments = [n for n in ast.parse(code[0]).body if isinstance(n, ast.Assign)]
        helpers = next(ast.literal_eval(n.value) for n in assignments if any(
            isinstance(t, ast.Name) and t.id == 'RUNTIME_HELPERS' for t in n.targets))
        self.assertEqual(helpers['repair_audit/stage1.py'],
                         (root/'repair_audit/stage1.py').read_text(encoding='utf-8'))
        self.assertNotIn('git clone', '\n'.join(code))

    def test_paired_sampler_starts_identically_and_final_guided_state_changes(self):
        calls = []
        class Unet(torch.nn.Module):
            def __init__(self): super().__init__(); self.dummy = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, x, t, **kwargs):
                calls.append(x.detach().clone()); return (torch.zeros_like(x),)
        class Vae(torch.nn.Module):
            config = SimpleNamespace(scaling_factor=1.)
            def encode(self, x): return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x))
            def decode(self, x, **kwargs): return (x,)
        class Scheduler:
            config = SimpleNamespace(num_train_timesteps=1000)
            alphas_cumprod = torch.linspace(.999, .1, 1000)
            final_alpha_cumprod = torch.tensor(1.)
            def set_timesteps(self, steps, **kwargs): self.num_inference_steps = steps
            def add_noise(self, x, noise, t): return x.clone()
            def scale_model_input(self, x, t): return x
            def step(self, noise, t, x, **kwargs):
                return SimpleNamespace(pred_original_sample=x.clone(), prev_sample=x.clone())
        class Pipe:
            def __init__(self): self.unet=Unet(); self.vae=Vae(); self.scheduler=Scheduler()
            def encode_prompt(self, *args, **kwargs): return torch.zeros(1,1), torch.zeros(1,1)
            def get_timesteps(self, *args): return torch.tensor([750,500,250,0]), 4
        class Controller:
            def __init__(self, source, extractor, records):
                self.erase = torch.zeros_like(source[:, :1]); self.erase[..., 4:6, 4:6] = 1
                self.restore = torch.zeros_like(self.erase); self.preserve=1-self.erase
            def refresh_erase(self, image): pass
            def begin_step(self, image, index): return dict(phase='erase')
            def finish_step(self, image, row): pass
            def __call__(self, image):
                loss = ((image-1).square()*self.erase).mean()
                return loss, {}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source=Image.new('RGB',(32,32),'white'); source.paste((0,0,0),(4,4,6,6));source.save(directory/'input.png')
            for slot in range(1,5): (directory/f'condition_{slot}.npz').write_bytes(condition_bytes())
            case=dict(directory=str(directory),case_id='test',kind='generated',prompt='test',negative_prompt='')
            extractor=torch.nn.Linear(1,1)
            with patch('repair_audit.stage1.fit_records',return_value=[]), \
                 patch('repair_audit.stage1.make_figures'), \
                 patch('stroke_model.restore_settle_erase.RestoreSettleErase',Controller):
                rows=run_case(Pipe(),extractor,case,steps=8,strength=.5)
            self.assertTrue(torch.equal(calls[0],calls[4]))
            raw=np.asarray(Image.open(directory/'audit/redenoise_only.png')).astype(float)
            guided=np.asarray(Image.open(directory/'audit/cyclic_guided.png')).astype(float)
            self.assertGreater(guided[4:6,4:6].mean(),raw[4:6,4:6].mean())
            report=json.loads((directory/'audit/report.json').read_text())
            self.assertTrue(report['configuration']['paired_initial_latent'])
            self.assertNotIn('status',report)
            self.assertEqual(len(rows),4*4*3)
            write_tables(directory,[case],rows,[])
            self.assertTrue((directory/'human_review.csv').is_file())


if __name__ == '__main__': unittest.main()
