import unittest
import torch
from torch import nn
from stroke_model.reference_model import ReferenceStrokeModel, exclusive_masks, partition_loss, stroke_moments


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(3, 8, 3, 2, 1), nn.Conv2d(8, 16, 3, 2, 1)])

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
            outputs.append(x)
        return outputs


class ReferenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)
        self.model = ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16])
        self.image = torch.rand(1, 1, 16, 16)
        self.strokes = torch.zeros(1, 3, 16, 16)
        self.strokes[:, 0, 3:6, 2:14] = 1
        self.strokes[:, 1, 3:14, 8:11] = 1
        self.ref = self.strokes.amax(1, keepdim=True)

    def test_conservation_padding_and_exact_export(self):
        out = self.model(self.image, self.ref, self.strokes)
        torch.testing.assert_close(out['masks'].sum(1, keepdim=True), self.image)
        self.assertEqual(out['masks'][:, 2].count_nonzero(), 0)
        fg = self.image > .3
        hard = exclusive_masks(out, fg)
        torch.testing.assert_close(hard.sum(1, keepdim=True), fg.long())

    def test_reference_permutation_equivariance(self):
        out = self.model(self.image, self.ref, self.strokes)
        order = [1, 2, 0]
        perm = self.model(self.image, self.ref, self.strokes[:, order])
        torch.testing.assert_close(perm['masks'], out['masks'][:, order], atol=1e-6, rtol=1e-5)

    def test_guidance_gradient_through_frozen_extractor(self):
        self.model.eval().requires_grad_(False)
        image = self.image.clone().requires_grad_()
        out = self.model(image, self.ref, self.strokes)
        # Use probabilities as well: prevents the trivial image*masks path hiding detach bugs.
        loss = out['probabilities'][:, 0, :8].sum() + stroke_moments(out['masks'])[:, :2].square().sum()
        grad, = torch.autograd.grad(loss, image)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_training_and_backbone_eval(self):
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        labels = torch.full((1, 16, 16), -1, dtype=torch.long)
        labels[:, 3:6, 2:14] = 0
        labels[:, 3:14, 8:11] = 1
        loss = partition_loss(self.model(self.image, self.ref, self.strokes), labels)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in self.model.head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.backbone.parameters()))

    def test_empty_reference_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.image, self.ref, torch.zeros_like(self.strokes))

    def test_invalid_label_rejected(self):
        out = self.model(self.image, self.ref, self.strokes)
        with self.assertRaises(ValueError):
            partition_loss(out, torch.full((1, 16, 16), 2))


if __name__ == '__main__':
    unittest.main()
