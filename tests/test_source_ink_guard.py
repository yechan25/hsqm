import unittest
import torch
from stroke_model.source_ink_guard import SourceInkGuard,local_ink_change


class InkGuardTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_normal_junction_is_preserved_but_new_nearby_dot_is_detected(self):
        ink=torch.zeros(1,1,64,64)
        ink[:,:,8:56,28:32]=1;ink[:,:,30:34,30:48]=1
        rgb=(1-ink).repeat(1,3,1,1)
        guard=SourceInkGuard(rgb)
        normal=guard(rgb)
        self.assertEqual(float(normal['loss']),0)
        changed=rgb.clone();changed[:,:,26:28,34:36]=0
        changed.requires_grad_(True)
        bad=guard(changed)
        self.assertGreater(float(bad['loss'].detach()),0)
        gradient,=torch.autograd.grad(bad['loss'],changed)
        self.assertLess(float(gradient[:,:,26:28,34:36].sum()),0)  # brighten the dot
        ok,increase,violation,_=local_ink_change(normal['local_excess'],bad['local_excess'],None,.002)
        self.assertFalse(ok);self.assertGreater(increase,.002);self.assertGreater(violation,0)

    def test_direct_reference_gradient(self):
        ref=torch.ones(1,3,16,16);ref[:,:,5:11,7:9]=0
        guard=SourceInkGuard(ref,support_radius=0,direct=True)
        image=ref.clone();image[:,:,2:4,2:4]=0;image[:,:,6:8,7:9]=1
        image.requires_grad_(True)
        loss=guard(image)['loss']
        grad,=torch.autograd.grad(loss,image)
        self.assertLess(float(grad[:,:,2:4,2:4].sum()),0)
        self.assertGreater(float(grad[:,:,6:8,7:9].sum()),0)
        self.assertLess(float(guard((image-.1*grad).detach())['loss']),float(loss.detach()))
        self.assertEqual(float(guard(ref)['loss']),0)

    def test_small_changes_cannot_accumulate_past_the_anchor(self):
        before=torch.zeros(1,1,3,3)
        ok,_,_,anchor=local_ink_change(before,before+.001,None,.002)
        self.assertTrue(ok)
        ok,_,_,_=local_ink_change(before+.001,before+.0025,anchor,.002)
        self.assertFalse(ok)

    def test_global_error_drop_does_not_hide_one_worse_patch(self):
        before=torch.full((1,1,10,10),.1)
        after=torch.zeros_like(before);after[...,4,4]=.2
        self.assertLess(after.mean(),before.mean())
        ok,_,_,_=local_ink_change(before,after,None,.002)
        self.assertFalse(ok)

    def test_inherited_excess_can_decrease_but_cannot_grow_further(self):
        before=torch.full((1,1,3,3),.1);anchor=torch.zeros_like(before)
        self.assertTrue(local_ink_change(before,before-.01,anchor,.002)[0])
        self.assertFalse(local_ink_change(before,before+.001,anchor,.002)[0])


if __name__=='__main__':unittest.main()
