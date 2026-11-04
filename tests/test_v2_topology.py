from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn

from stroke_model.reference_model import ReferenceStrokeModel, partition_loss, exclusive_masks
from stroke_model.reference_v2_topology import (ARCHITECTURE, JunctionConfig,
    junction_discrepancy, v2_topology_loss)
from stroke_model.infer_topology import ink_diagnostics, load_topology, run_topology_inference
from stroke_model.train_reference import main


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(3, 8, 3, 2, 1)
    def forward(self, x):
        return [self.conv(x)]


def tiny(**kwargs):
    return ReferenceStrokeModel(backbone=TinyBackbone(), channels=[8], **kwargs)


class V2TopologyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.target = torch.zeros(1,2,64,64)
        self.target[:,0,32,10:54] = 1
        self.target[:,1,10:54,32] = 1
        self.image = torch.nn.functional.max_pool2d(self.target.amax(1,keepdim=True),5,1,2)
        self.labels = torch.where(self.target.sum(1)==1,self.target.argmax(1),-1)
        self.valid = torch.ones_like(self.image)

    def test_unchanged_v2_weights_conservation_and_finite_training(self):
        model = tiny(image_size=64,width=8,pretrained=False)
        before = set(model.state_dict())
        output = model(self.image,self.image,self.target)
        torch.testing.assert_close(output['masks'].sum(1,keepdim=True),self.image)
        hard = exclusive_masks(output,self.image>.5)
        self.assertTrue(torch.equal(hard.sum(1,keepdim=True),self.image.long()))
        losses = v2_topology_loss(output,self.labels,self.target,self.valid,JunctionConfig(every=1))
        losses['loss'].backward()
        self.assertTrue(torch.isfinite(losses['loss']))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(set(model.state_dict()),before)
        self.assertFalse(any('context_' in key or 'global_aligner' in key for key in before))

    def test_disabled_loss_exactly_matches_v2_and_unknown_ink_is_not_background(self):
        logits = torch.randn(1,2,64,64,requires_grad=True)
        output = dict(logits=logits,probabilities=logits.softmax(1),flow=torch.zeros(1,2,64,64),
                      masks=self.image*logits.softmax(1),active=torch.ones(1,2,dtype=torch.bool))
        with patch('stroke_model.reference_v2_topology.junction_discrepancy',side_effect=AssertionError):
            loss = v2_topology_loss(output,self.labels,config=JunctionConfig(weight=0))
        torch.testing.assert_close(loss['loss'],partition_loss(output,self.labels)['loss'],atol=0,rtol=0)
        loss['loss'].backward()
        self.assertEqual(logits.grad.permute(0,2,3,1)[self.labels<0].count_nonzero(),0)

    def test_off_interval_does_no_topology_work(self):
        model = tiny(image_size=64,width=8,pretrained=False)
        output = model(self.image,self.image,self.target)
        with patch('stroke_model.reference_v2_topology.junction_discrepancy',side_effect=AssertionError):
            result = v2_topology_loss(output,self.labels,config=JunctionConfig(every=4),step=1)
        self.assertFalse(result['junction_applied'])

    def test_added_branch_penalty_preserves_actual_branch(self):
        branch = self.target.clone(); branch[:,0,20:33,32] = 1
        output = dict(masks=branch.requires_grad_(),active=torch.ones(1,2,dtype=torch.bool))
        cfg = JunctionConfig()
        wrong = junction_discrepancy(output,self.target,self.valid,cfg)
        actual = junction_discrepancy(output,branch.detach(),self.valid,cfg)
        self.assertGreater(wrong.item(),0)
        self.assertEqual(actual.item(),0)
        wrong.backward()
        self.assertTrue(torch.isfinite(branch.grad).all())

    def test_partition_export_covers_ink_even_when_no_channel_exceeds_half(self):
        masks = np.repeat(self.image.numpy()/3,3,axis=1)[0]
        stats = ink_diagnostics(self.image[0].numpy(),masks,output_mode='partition')
        self.assertEqual(stats['covered_ink_fraction'],1.)
        self.assertEqual(stats['overlapping_ink_fraction'],0.)

    def test_one_epoch_and_compatible_inference_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = []
            for split in ('train','val'):
                path = root/f'{split}.npz'
                np.savez_compressed(path,image=self.image[0].numpy(),reference=self.image[0].numpy(),
                    strokes=self.target[0].numpy(),target_masks=self.target[0].numpy(),
                    labels=self.labels[0].numpy(),valid_pixels=self.valid[0].numpy())
                rows.append(dict(path=str(path),split=split,group_id=split))
            (root/'manifest.json').write_text(json.dumps(rows))
            argv = ['train','--manifest',str(root/'manifest.json'),'--output',str(root/'run'),
                    '--epochs','1','--batch-size','1','--num-workers','0','--image-size','64',
                    '--junction-weight','.05','--junction-every','1']
            with patch.object(sys,'argv',argv),patch('stroke_model.train_reference.ReferenceStrokeModel',side_effect=tiny):
                main()
            checkpoint = root/'run/best.pt'
            state = torch.load(checkpoint,weights_only=True)
            self.assertEqual(state['architecture'],ARCHITECTURE)
            with patch('stroke_model.infer_topology.ReferenceStrokeModel',side_effect=tiny):
                model,size,epoch = load_topology(checkpoint,'cpu')
            self.assertEqual((size,epoch),(64,1))
            self.assertEqual(model.output_mode,'partition')
            output = model(self.image,self.image,self.target)
            torch.testing.assert_close(output['masks'].sum(1,keepdim=True),self.image)
            self.assertIn('samples_per_second',json.loads((root/'run/history.json').read_text())[0])
            Image.fromarray(np.uint8(255*(1-self.image[0,0].numpy()))).save(root/'input.png')
            for k in range(2):
                Image.fromarray(np.uint8(255*(1-self.target[0,k].numpy()))).save(root/f's{k}.png')
            pd.DataFrame([dict(input_path='input.png',target_path='input.png',
                target_stroke_paths='s0.png;s1.png')]).to_csv(root/'paths.csv',index=False)
            with patch('stroke_model.infer_topology.ReferenceStrokeModel',side_effect=tiny):
                report = run_topology_inference(checkpoint,root/'paths.csv',show=False)
            self.assertEqual(report['output_mode'],'partition')
            self.assertEqual(report['results'][0]['covered_ink_fraction'],1.)
            with np.load(report['results'][0]['npz']) as arrays:
                np.testing.assert_allclose(arrays['soft_masks'].sum(0),arrays['image'][0],atol=1e-6)
                np.testing.assert_array_equal(arrays['hard_masks'].sum(0),arrays['image'][0]>.5)


if __name__ == '__main__':
    unittest.main()
