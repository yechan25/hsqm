import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from stroke_model.detected_extra_guidance import DetectedExtraGuidance, load_error_head
from stroke_model.error_localizer import StrokeErrorLocalizer, ErrorLocalizerConfig, ARCHITECTURE


class DetectedGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rgb=torch.ones(1,3,32,32);self.rgb[:,:,12:16,8:12]=0
        self.mask=torch.zeros(1,1,32,32);self.mask[:,:,12:16,8:12]=1
        self.slots=[dict(bbox=[0,0,32,32])]

    def test_fixed_map_gives_local_brightening_gradient_without_clean_target(self):
        objective=DetectedExtraGuidance(self.rgb,self.mask,self.slots)
        x=self.rgb.clone().requires_grad_(True)
        loss,_=objective(x);grad,=torch.autograd.grad(loss,x)
        self.assertLess(float(grad[:,:,12:16,8:12].sum()),0)
        self.assertEqual(float(grad[:,:,:8,:].abs().sum()),0)
        after,_=objective((x-.1*grad).detach())
        self.assertLess(float(after),float(loss.detach()))
        self.mask.fill_(0);self.rgb.fill_(0)
        self.assertEqual(float(objective.mask.sum()),16)
        self.assertEqual(float(objective.target[0,:,12:16,8:12].min()),1)

    def test_empty_mask_and_unsupported_background_are_explicit(self):
        with self.assertRaises(ValueError):DetectedExtraGuidance(self.rgb,self.mask*0,self.slots)
        with self.assertRaises(ValueError):DetectedExtraGuidance(self.rgb*.2,self.mask,self.slots)

    def test_trained_checkpoint_and_exact_extractor_are_required(self):
        workspace=Path(__file__).resolve().parents[1]/'outputs'
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as temp:
            root=Path(temp);extractor=root/'extractor.pt';extractor.write_bytes(b'exact trained extractor')
            model=StrokeErrorLocalizer(ErrorLocalizerConfig(width=8))
            state=dict(architecture=ARCHITECTURE,config=dict(width=8),model=model.state_dict(),epoch=1,
                       metadata=dict(extractor_sha256=hashlib.sha256(extractor.read_bytes()).hexdigest()))
            checkpoint=root/'head.pt';torch.save(state,checkpoint)
            head,meta=load_error_head(checkpoint,extractor)
            self.assertFalse(head.training);self.assertFalse(any(p.requires_grad for p in head.parameters()))
            self.assertFalse(meta['threshold_calibrated'])
            extractor.write_bytes(b'wrong weights')
            with self.assertRaises(ValueError):load_error_head(checkpoint,extractor)
            state['epoch']=0;torch.save(state,checkpoint)
            with self.assertRaises(ValueError):load_error_head(checkpoint,extractor)


if __name__=='__main__':unittest.main()
