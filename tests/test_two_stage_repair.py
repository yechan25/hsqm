import unittest
import torch
from torch import nn
from stroke_model.template_structure_guidance import PreserveGeneratedImage
from diffusion_model.guided_ddim import GuidanceConfig

class RepairTests(unittest.TestCase):
    def test_preservation_anchor_is_first_generated_pixels_and_has_gradient(self):
        class Structure(nn.Module):
            def forward(self,x):return (x-.8).square().mean(),dict(per_glyph=[])
        anchor=torch.full((1,3,8,8),.4)
        wrapper=PreserveGeneratedImage(Structure(),anchor,weight=2.)
        anchor.fill_(0)
        self.assertTrue(torch.all(wrapper.generated_rgb==.4))
        x=torch.full((1,3,8,8),.6,requires_grad=True)
        loss,detail=wrapper(x)
        self.assertAlmostEqual(float(detail['preservation_loss'].detach()),.04,places=5)
        gradient,=torch.autograd.grad(loss,x)
        self.assertGreater(float(gradient.sum()),0)  # anchor pulls back an otherwise whitening update
    def test_cap_rejects_unsupported_modes(self):
        for ratio in (-1,0,float('nan')):
            with self.assertRaises(ValueError):GuidanceConfig(max_state_change_to_base_step_ratio=ratio)
        with self.assertRaises(ValueError):GuidanceConfig(max_state_change_to_base_step_ratio=.25)

if __name__=='__main__':unittest.main()
