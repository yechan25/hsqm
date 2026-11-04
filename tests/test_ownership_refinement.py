import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from stroke_model.ownership_refinement import (expand_ownership, ownership_losses,
    prepare_ownership_manifest)
from stroke_model.reference_model import ReferenceStrokeModel
from stroke_model.train_reference import main


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,8,3,2,1)
    def forward(self,x):return [self.conv(x)]


def tiny(**kwargs):
    return ReferenceStrokeModel(backbone=TinyBackbone(),channels=[8],**kwargs)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.target=np.zeros((2,64,64),np.float32)
        self.target[0,6:57,18]=1;self.target[1,10,24:28]=1
        self.image=np.zeros((1,64,64),np.float32)
        self.image[0,5:58,16:21]=1;self.image[0,9:12,21:29]=1
        self.labels=np.where(self.target.sum(0)==1,self.target.argmax(0),-1)

    def test_neighbour_stroke_is_not_stolen_and_original_labels_are_preserved(self):
        owners,weight=expand_ownership(self.image,self.target,self.labels)
        self.assertEqual(owners[10,20],0)
        self.assertEqual(owners[10,22],1)
        self.assertGreater(weight[10,20],0)
        known=self.labels>=0
        np.testing.assert_array_equal(owners[known],self.labels[known])
        self.assertTrue((owners[self.image[0]==0]==-1).all())

    def test_crossing_stays_unknown(self):
        target=np.zeros((2,64,64),np.float32)
        target[0,32,8:56]=1;target[1,8:56,32]=1
        image=torch.nn.functional.max_pool2d(torch.from_numpy(target).amax(0)[None,None],5,1,2)[0].numpy()
        labels=np.where(target.sum(0)==1,target.argmax(0),-1)
        owners,weight=expand_ownership(image,target,labels)
        self.assertEqual(owners[32,32],-1);self.assertEqual(weight[32,32],0)

    def test_real_disconnected_parts_preserved_and_unannotated_extra_not_invented(self):
        image=np.zeros((1,32,32),np.float32)
        image[0,3:7,3:9]=1;image[0,18:22,3:9]=1;image[0,25:28,25:28]=1
        target=np.zeros((1,32,32),np.float32)
        target[0,5,4:8]=1;target[0,20,4:8]=1
        labels=np.where(target[0]>0,0,-1)
        owners,weight=expand_ownership(image,target,labels)
        self.assertEqual(owners[4,5],0);self.assertEqual(owners[19,5],0)
        self.assertEqual(owners[26,26],-1)
        self.assertTrue((owners[7:18]==-1).all())

    def test_wrong_small_stroke_patch_and_remote_fragments_have_higher_loss(self):
        owners,weight=expand_ownership(self.image,self.target,self.labels)
        known=owners>=0
        p=np.full((1,2,64,64),.5,np.float32)
        for k in range(2):
            p[0,k,known]=np.where(owners[known]==k,.95,.05)
        owner=torch.from_numpy(owners[None]).long();w=torch.from_numpy(weight[None]).float()/255
        good=ownership_losses(dict(probabilities=torch.from_numpy(p)),owner,w)
        bad=p.copy();bad[0,:,10,20]=(.05,.95);bad[0,:,40,18]=(.05,.95)
        bp=torch.tensor(bad,requires_grad=True)
        wrong=ownership_losses(dict(probabilities=bp),owner,w)
        self.assertGreater(wrong['ownership'].item(),good['ownership'].item())
        (wrong['ownership']+wrong['affinity']).backward()
        self.assertTrue(torch.isfinite(bp.grad).all())

    def test_affinity_does_not_cross_labels_gaps_or_unknown(self):
        p=torch.tensor([[[[.9,.1,.2]],[[.1,.9,.8]]]],requires_grad=True)
        owners=torch.tensor([[[0,-1,0]]]);weight=torch.tensor([[[1.,0.,1.]]])
        loss=ownership_losses(dict(probabilities=p),owners,weight)
        self.assertEqual(loss['affinity'].item(),0)
        empty=ownership_losses(dict(probabilities=p),torch.full_like(owners,-1),torch.zeros_like(weight))
        self.assertEqual(empty['ownership'].item(),0)
        self.assertEqual(empty['affinity'].item(),0)

    def test_cached_preparation_keeps_splits_and_training_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for split in ('train','val'):
                path=root/f'{split}.npz'
                np.savez_compressed(path,image=self.image,reference=self.image,strokes=self.target,
                    labels=self.labels,target_masks=self.target,valid_pixels=np.ones_like(self.image))
                rows.append(dict(path=str(path),split=split,group_id=split))
            source=root/'manifest.json';source.write_text(json.dumps(rows))
            prepared=prepare_ownership_manifest(source)
            self.assertEqual(prepared,prepare_ownership_manifest(source))
            prepared_rows=json.loads(prepared.read_text())
            self.assertIn('ownership_path',prepared_rows[0]);self.assertNotIn('ownership_path',prepared_rows[1])
            argv=['train','--manifest',str(source),'--output',str(root/'run'),'--epochs','1',
                '--batch-size','1','--num-workers','0','--image-size','64','--junction-weight','.05',
                '--ownership-weight','.1','--affinity-weight','.02','--epoch-warning-seconds','33',
                '--direction-affinity-weight','.03']
            with patch.object(sys,'argv',argv),patch('stroke_model.train_reference.ReferenceStrokeModel',side_effect=tiny):
                main()
            state=torch.load(root/'run/best.pt',weights_only=True)
            self.assertEqual(state['ownership_weight'],.1)
            model=tiny(image_size=64,pretrained=False);model.load_state_dict(state['model'])
            image=torch.from_numpy(self.image)[None];strokes=torch.from_numpy(self.target)[None]
            output=model(image,image,strokes)
            torch.testing.assert_close(output['masks'].sum(1,keepdim=True),image)
            self.assertFalse(any('ownership' in name or 'affinity' in name for name in state['model']))
            rows[0]['split']='val';rows[1]['split']='train';source.write_text(json.dumps(rows))
            self.assertNotEqual(prepared,prepare_ownership_manifest(source))


if __name__=='__main__':unittest.main()
