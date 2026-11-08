import unittest
import torch
from stroke_model.template_structure_guidance import stroke_descriptor

class GeometryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def masks(self):
        x=torch.zeros(1,2,48,48)
        x[:,0,9:35,23:26]=1;x[:,1,20:23,26:37]=1
        return x
    def test_translation_does_not_require_same_pixels(self):
        x=self.masks();y=torch.roll(x,(3,-4),(-2,-1))
        self.assertFalse(torch.equal(x,y))
        torch.testing.assert_close(stroke_descriptor(x),stroke_descriptor(y),atol=2e-5,rtol=2e-5)
    def test_uniform_ink_change_is_not_a_pixel_penalty(self):
        x=self.masks()
        # Contact strength is intentionally ink-sensitive; geometric moments are not.
        a=stroke_descriptor(x);b=stroke_descriptor(x*.8)
        torch.testing.assert_close(a[:,:-5],b[:,:-5],atol=1e-5,rtol=1e-5)
    def test_wrong_side_extension_changes_geometry_and_has_gradient(self):
        x=self.masks();y=x.clone();y[:,1,20:23,15:26]=1;y.requires_grad_(True)
        loss=(stroke_descriptor(y)-stroke_descriptor(x)).square().mean()
        self.assertGreater(float(loss.detach()),0)
        grad,=torch.autograd.grad(loss,y)
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
    def test_empty_masks_remain_finite(self):
        x=torch.zeros(1,3,24,24,requires_grad=True)
        value=stroke_descriptor(x)
        self.assertTrue(torch.isfinite(value).all())
        grad,=torch.autograd.grad(value.sum(),x)
        self.assertTrue(torch.isfinite(grad).all())

if __name__=='__main__':unittest.main()
