import unittest
import torch

from stroke_model.reference_v2_topology import (JunctionConfig,balanced_seed_focal_loss,
    select_junction_strokes)
from stroke_model.train_reference import evaluate


class HardStrokeTests(unittest.TestCase):
    def test_validation_flags_failed_annotation_recall_without_counting_absent_slots(self):
        class Model(torch.nn.Module):
            def forward(self,image,reference,strokes):
                p=torch.zeros_like(strokes);p[:,0]=1
                return dict(probabilities=p)
        image=torch.ones(1,1,1,3);strokes=torch.ones(1,3,1,3)
        labels=torch.tensor([[[0,1,-1]]])
        result=evaluate(Model(),[(image,image,strokes,labels)],'cpu',return_details=True)
        self.assertEqual(result['val_annotated_strokes'],2)
        self.assertEqual(result['val_low_seed_recall_strokes'],1)
        self.assertEqual(result['val_minimum_seed_recall'],0)

    def test_hard_small_stroke_not_diluted_by_many_easy_large_stroke_seeds(self):
        p=torch.tensor([[[[.95]*9+[.9]],[[.05]*9+[.1]]]])
        labels=torch.tensor([[[0]*9+[1]]])
        many=balanced_seed_focal_loss(dict(logits=p.log()),labels)
        few=balanced_seed_focal_loss(dict(logits=p[:,:,:,[0,9]].log()),labels[:,:,[0,9]])
        torch.testing.assert_close(many,few)
        corrected=p.clone();corrected[0,:,0,9]=torch.tensor([.05,.95])
        good=balanced_seed_focal_loss(dict(logits=corrected.log()),labels)
        self.assertGreater(many.item(),good.item()*100)

    def test_unknown_and_inactive_channels_do_not_fabricate_positive_targets(self):
        logits=torch.tensor([[[[-float('inf')]*3],[[2.,0.,2.]],[[0.,2.,0.]]]],requires_grad=True)
        labels=torch.tensor([[[1,2,-1]]])
        loss=balanced_seed_focal_loss(dict(logits=logits),labels)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[:,:,:,2].count_nonzero(),0)
        empty=balanced_seed_focal_loss(dict(logits=logits),torch.full_like(labels,-1))
        self.assertEqual(empty.item(),0)
        perfect=torch.zeros(1,1,1,1,requires_grad=True)
        loss=balanced_seed_focal_loss(dict(logits=perfect),torch.zeros(1,1,1,dtype=torch.long),gamma=.5)
        loss.backward();self.assertTrue(torch.isfinite(perfect.grad).all())

    def test_confused_strokes_prioritized_without_more_skeletons_and_rotation_remains(self):
        target=torch.zeros(1,5,64,64)
        for k in range(5):target[0,k,32,12+k*8]=1
        labels=torch.where(target.sum(1)==1,target.argmax(1),-1)
        p=torch.full_like(target,.2)
        p[0,:,32,12]=.0125;p[0,4,32,12]=.95
        p[0,:,32,20]=.025;p[0,3,32,20]=.9
        output=dict(probabilities=p,active=torch.ones(1,5,dtype=torch.bool))
        valid=torch.ones(1,1,64,64)
        cfg=JunctionConfig(max_strokes=4,hard_fraction=.5)
        selected=select_junction_strokes(output,target,valid,cfg,labels=labels)
        strokes=selected[:,1].tolist()
        self.assertEqual(len(strokes),4);self.assertEqual(len(set(strokes)),4)
        self.assertTrue({3,4}.issubset(set(strokes)))
        visited=set()
        for step in range(0,20,4):
            visited.update(select_junction_strokes(output,target,valid,cfg,step,labels)[:,1].tolist())
        self.assertEqual(visited,set(range(5)))
        legacy=select_junction_strokes(output,target,valid,JunctionConfig(max_strokes=4),labels=labels)
        self.assertEqual(legacy[:,1].tolist(),[0,1,2,3])


if __name__=='__main__':unittest.main()
