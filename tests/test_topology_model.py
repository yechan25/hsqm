import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import torch
from torch import nn
from stroke_model.reference_model import ReferenceStrokeModel
from stroke_model.reference_topology import (TopologyStrokeModel, TopologyLossConfig,
    AlignmentConfig, junction_signature, topology_stroke_loss)
from stroke_model.train_independent import main


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 2, 1)
    def forward(self, x):
        return [self.conv(x)]


def model(**kwargs):
    return TopologyStrokeModel(backbone=TinyBackbone(), channels=[8], **kwargs)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.target = torch.zeros(1, 3, 64, 64)
        self.target[:, 0, 30:33, 8:56] = 1
        self.target[:, 1, 8:56, 30:33] = 1
        self.image = self.target.amax(1, keepdim=True)

    def output(self, masks):
        p = masks.clamp(.001, .999)
        return dict(logits=torch.logit(p), masks=masks,
                    active=torch.tensor([[True, True, False]]))

    def test_identity_alignment_independent_crossings_and_gradients(self):
        m = model(image_size=64, width=8, context_width=4)
        out = m(self.image, self.image, self.target)
        torch.testing.assert_close(out['aligned_reference'], self.image, atol=1e-5, rtol=0)
        self.assertEqual(out['masks'][:, 2].count_nonzero(), 0)
        losses = topology_stroke_loss(out, self.target, self.image)
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(losses['loss']))
        grad = m.global_aligner[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        with torch.no_grad():
            m.head[-1].weight.zero_(); m.head[-1].bias.fill_(10)
        out = m(self.image, self.image, self.target)
        self.assertTrue((out['masks'][0, :2, 31, 31] > .99).all())

    def test_alignment_changes_priors_only_and_stays_in_bounds(self):
        m = model(image_size=64, width=8, context_width=4,
                  alignment_config=AlignmentConfig(translation_fraction=.1))
        with torch.no_grad():
            m.global_aligner[-1].bias[0] = 100
        out = m(self.image, self.image, self.target)
        self.assertAlmostEqual(out['alignment_theta'][0, 0, 2].item(), .2, places=5)
        self.assertFalse(torch.equal(out['aligned_reference'], self.image))
        self.assertEqual(out['masks'].masked_select(self.image.expand_as(out['masks']) == 0).count_nonzero(), 0)

    def test_assignment_split_is_penalized_and_true_boundary_is_preserved(self):
        good = topology_stroke_loss(self.output(self.target), self.target, self.image)
        bad = self.target.clone()
        bad[:, 0, :, 32:] = 0
        bad[:, 1] = torch.maximum(bad[:, 1], self.target[:, 0]*(torch.arange(64)[None, :] >= 32))
        wrong = topology_stroke_loss(self.output(bad), self.target, self.image)
        self.assertEqual(good['consistency'].item(), 0)
        self.assertGreater(wrong['consistency'].item(), .001)

    def test_branch_discrepancy_detects_added_branch_but_preserves_actual_defect(self):
        branch = self.target.clone()
        branch[:, 0, 18:31, 30:33] = 1
        image = branch.amax(1, keepdim=True)
        cfg = TopologyLossConfig()
        self.assertGreater(junction_signature(branch, cfg).sum().item(), 0)
        wrong = topology_stroke_loss(self.output(branch), self.target, image, config=cfg)
        actual = topology_stroke_loss(self.output(branch), branch, image, config=cfg)
        self.assertGreater(wrong['junction'].item(), 0)
        self.assertEqual(actual['junction'].item(), 0)
        self.assertEqual(actual['consistency'].item(), 0)

    def test_unknown_region_does_not_create_structural_supervision(self):
        bad = self.target.clone(); bad[:, 0, 18:31, 30:33] = 1
        valid = torch.ones_like(self.image); valid[:, :, 10:45, 20:45] = 0
        losses = topology_stroke_loss(self.output(bad), self.target, self.image, valid)
        self.assertEqual(losses['junction'].item(), 0)
        self.assertEqual(losses['consistency'].item(), 0)

    def test_v2_initializer_is_not_accepted_as_trained_topology_model(self):
        base = ReferenceStrokeModel(image_size=64, width=8, backbone=TinyBackbone(), channels=[8])
        m = model(image_size=64, width=8, context_width=4)
        m.warm_start(base.state_dict()); m.eval()
        with self.assertRaisesRegex(RuntimeError, 'trained topology'):
            m(self.image, self.image, self.target)
        with self.assertRaises(ValueError):
            m.load_trained(dict(architecture='reference_v2', model=base.state_dict()))

    def test_training_from_imagenet_initialization_without_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split in ('train', 'val'):
                np.savez_compressed(root/f'{split}.npz', image=self.image[0].numpy(),
                    reference=self.image[0].numpy(), strokes=self.target[0, :2].numpy(),
                    target_masks=self.target[0, :2].numpy())
                rows.append(dict(path=f'{split}.npz', split=split, group_id=split))
            (root/'manifest.json').write_text(json.dumps(rows))
            argv = ['train', '--model', 'topology', '--manifest', str(root/'manifest.json'),
                    '--output', str(root/'run'), '--image-size', '64', '--width', '8',
                    '--epochs', '1', '--batch-size', '1', '--context-width', '4', '--no-shuffle-strokes']
            with patch.object(sys, 'argv', argv), patch('stroke_model.train_independent.TopologyStrokeModel', side_effect=model) as factory:
                main()
            self.assertTrue(factory.call_args.kwargs['pretrained'])
            state = torch.load(root/'run/best.pt', weights_only=True)
            self.assertEqual(state['architecture'], TopologyStrokeModel.architecture)
            history = json.loads((root/'run/history.json').read_text())
            self.assertIn('junction', history[0]['train_components'])

    def test_training_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split in ('train', 'val'):
                np.savez_compressed(root/f'{split}.npz', image=self.image[0].numpy(),
                    reference=self.image[0].numpy(), strokes=self.target[0, :2].numpy(),
                    target_masks=self.target[0, :2].numpy())
                rows.append(dict(path=f'{split}.npz', split=split, group_id=split))
            (root/'manifest.json').write_text(json.dumps(rows))
            base = ReferenceStrokeModel(image_size=64, width=8, backbone=TinyBackbone(), channels=[8])
            torch.save(dict(model=base.state_dict(), image_size=64, width=8), root/'old.pt')
            argv = ['train', '--model', 'topology', '--manifest', str(root/'manifest.json'),
                    '--init-checkpoint', str(root/'old.pt'), '--output', str(root/'run'),
                    '--epochs', '1', '--batch-size', '1', '--context-width', '4']
            with patch.object(sys, 'argv', argv), patch('stroke_model.train_independent.TopologyStrokeModel', side_effect=model):
                main()
            state = torch.load(root/'run/best.pt', weights_only=True)
            self.assertEqual(state['architecture'], TopologyStrokeModel.architecture)
            restored = model(image_size=64, width=8, context_width=4,
                             alignment_config=AlignmentConfig(**state['alignment_config']))
            restored.load_trained(state); restored.eval()
            with torch.no_grad():
                out = restored(self.image, self.image, self.target)
            self.assertTrue(torch.isfinite(out['masks']).all())


if __name__ == '__main__':
    unittest.main()
