"""Small CPU checks: data isolation, gradients, resume cursor and actual DDPM."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from diffusion_model.data import BatchStream, GlyphDataset, collate, read_manifest
from diffusion_model.core import clean_from_epsilon, EMA


class DataTests(unittest.TestCase):
    def test_group_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root/'a.npz', image=np.zeros((1, 16, 16)))
            (root/'manifest.json').write_text(json.dumps([
                {'path': 'a.npz', 'group_id': 'writer', 'split': 'train'},
                {'path': 'a.npz', 'group_id': 'writer', 'split': 'val'}]))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                list(read_manifest(root/'manifest.json'))

    def test_resume_batch_stream(self):
        stream = BatchStream(7, 3, 42)
        for _ in range(5):
            stream.next_indices()
        restored = BatchStream(7, 3, 42, **stream.state_dict())
        self.assertEqual([stream.next_indices() for _ in range(10)],
                         [restored.next_indices() for _ in range(10)])

    def test_manifest_npz_padding_and_ink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ink = np.zeros((1, 16, 16), dtype=np.float32)
            ink[:, 4:12, 4:12] = 1
            for i in range(2):
                np.savez(root/f'{i}.npz', image=ink, reference=ink, strokes=np.repeat(ink, i+1, 0))
            rows = [{'path': str(root/f'{i}.npz'), 'text': '가'} for i in range(2)]
            dataset = GlyphDataset(rows, 16)
            batch = collate([dataset[0], dataset[1]])
            self.assertEqual(batch['strokes'].shape, (2, 2, 16, 16))
            self.assertEqual(batch['strokes'][0, 1].sum(), 0)

    def test_epsilon_reconstruction_and_gradient(self):
        clean, noise = torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)
        alpha = torch.tensor([.9, .6])
        noisy = alpha.sqrt()[:, None, None, None]*clean + (1-alpha).sqrt()[:, None, None, None]*noise
        prediction = noise.clone().requires_grad_(True)
        reconstructed = clean_from_epsilon(noisy, prediction, torch.tensor([0, 1]), alpha)
        torch.testing.assert_close(reconstructed, clean)
        reconstructed.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum(), 0)

    def test_frozen_ocr_path_has_input_gradient(self):
        from diffusion_model.core import OCRLoss
        # Native recognizer-shaped stub; no checkpoint download in unit tests.
        class Recognizer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 3)
            def forward(self, image, unused):
                return self.linear(image.mean(3).squeeze(1))[:, None, :].expand(-1, 8, -1)
        loss_fn = OCRLoss.__new__(OCRLoss)
        loss_fn.model = Recognizer().eval().requires_grad_(False)
        loss_fn.table = {'가': 1}
        image = torch.rand(2, 1, 16, 16, requires_grad=True)
        value = loss_fn(image, ['가', '가'])
        value.backward()
        self.assertGreater(image.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in loss_fn.model.parameters()))

    def test_ema_restores_after_error(self):
        model = torch.nn.Linear(2, 2)
        ema = EMA(model)
        with torch.no_grad():
            model.weight.add_(1)
        original = model.weight.detach().clone()
        with self.assertRaises(RuntimeError):
            with ema.apply(model):
                raise RuntimeError('preview failure')
        torch.testing.assert_close(model.weight, original)

    def test_frozen_stroke_loss_keeps_generated_image_gradient(self):
        from diffusion_model.core import StrokeLoss
        class Extractor(torch.nn.Module):
            def forward(self, image, reference, strokes):
                p = strokes.softmax(1)
                return {'masks': image*p}
        loss_fn = StrokeLoss.__new__(StrokeLoss)
        loss_fn.model = Extractor().eval().requires_grad_(False)
        predicted = torch.full((1, 1, 16, 16), .4, requires_grad=True)
        target = torch.full_like(predicted, .8)
        masks = torch.ones(1, 2, 16, 16)
        value = loss_fn(predicted, target, target, masks)
        value.backward()
        self.assertGreater(predicted.grad.abs().sum(), 0)
        with self.assertRaises(ValueError):
            loss_fn(predicted, target, target, None)


class DiffusersTests(unittest.TestCase):
    def test_tiny_unet_forward_backward_and_ddim(self):
        try:
            from diffusers import DDPMScheduler
        except ImportError:
            self.skipTest('Install diffusion_model/requirements.txt for real UNet test')
        from diffusion_model.core import build_model, sample
        torch.set_num_threads(1)
        model = build_model(16, small=True)
        reference = torch.rand(1, 1, 16, 16)
        noisy = torch.randn_like(reference)
        prediction = model(torch.cat((noisy, reference*2-1), 1), torch.tensor([100])).sample
        prediction.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        result = sample(model, scheduler, reference, seed=42, steps=2)
        self.assertEqual(result.shape, reference.shape)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
