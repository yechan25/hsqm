import unittest
import numpy as np
import torch
from torch import nn

from stroke_model.stroke_appearance import StrokeAppearanceLoss,AppearanceConfig
from stroke_model.diffusion_guidance import TopoGuidance
from stroke_model.stroke_structure import StructureConfig


class IdealSingleStroke(nn.Module):
    def forward(self,image,reference,strokes):
        return dict(masks=image,probabilities=torch.ones_like(image))


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.reference=torch.zeros(1,24,24)
        self.reference[:,10:13,4:20]=1
        self.metric=StrokeAppearanceLoss(self.reference)

    def test_matching_stroke_and_matching_serif_have_zero_loss(self):
        serif=self.reference.clone();serif[:,8:15,4:6]=1
        for reference in (self.reference,serif):
            metric=StrokeAppearanceLoss(reference)
            masks=reference[None].clone().requires_grad_(True)
            loss=metric(masks)['loss'].sum()
            gradient,=torch.autograd.grad(loss,masks)
            self.assertEqual(float(loss.detach()),0)
            self.assertEqual(float(gradient.abs().sum()),0)

    def test_thickness_bump_and_notch_have_corrective_gradients(self):
        for name in ('thicker','bump','notch'):
            masks=self.reference[None].clone()
            if name=='thicker':masks[:,:,7:16,4:20]=1;region=(slice(7,10),slice(4,20))
            elif name=='bump':masks[:,:,9,10:15]=1;region=(slice(9,10),slice(10,15))
            else:masks[:,:,10,10:15]=0;region=(slice(10,11),slice(10,15))
            masks.requires_grad_(True)
            initial=self.metric(masks)['loss'].sum()
            gradient,=torch.autograd.grad(initial,masks)
            self.assertGreater(float(initial.detach()),0)
            signed=float(gradient[...,region[0],region[1]].sum())
            self.assertGreater(signed,0) if name!='notch' else self.assertLess(signed,0)
            original_region=masks[...,region[0],region[1]].detach().mean()
            for _ in range(25):
                masks=masks.detach().requires_grad_(True)
                loss=self.metric(masks)['loss'].sum()
                gradient,=torch.autograd.grad(loss,masks)
                masks=(masks-.03*gradient/gradient.abs().max().clamp_min(1e-12)).clamp(0,1)
            final=self.metric(masks)['loss'].sum()
            self.assertLess(float(final.detach()),float(initial.detach())*.5)
            new_region=masks[...,region[0],region[1]].detach().mean()
            self.assertLess(new_region,original_region) if name!='notch' else self.assertGreater(new_region,original_region)

    def test_geometry_is_not_an_impossible_overlapping_reference(self):
        strokes=self.reference.repeat(2,1,1).numpy()
        ref=strokes.max(0)[None]
        slots=[dict(reference=ref,strokes=strokes,bbox=[x,y,x+24,y+24]) for y in (0,24) for x in (0,24)]
        objective=TopoGuidance(IdealSingleStroke(),slots,appearance_weight=.5)
        metric=objective.appearance_metrics[0]
        torch.testing.assert_close(metric.reference.sum(1),torch.from_numpy(ref))
        self.assertEqual(float(metric(metric.reference)['loss'].sum()),0)

    def test_full_topo_path_now_reacts_to_a_high_scoring_contour_defect(self):
        ref=self.reference.numpy()
        slots=[dict(reference=ref,strokes=ref,bbox=[x,y,x+24,y+24]) for y in (0,24) for x in (0,24)]
        cfg=StructureConfig(max_pairs=2,skeleton_iterations=2,junction_thresholds=(.5,),
            junction_min_levels=1,junction_distance_steps=2,junction_angular_samples=16)
        objective=TopoGuidance(IdealSingleStroke(),slots,cfg,extra_weight=0,focus_strength=2)
        ink=self.reference[0].clone();ink[9,10:15]=1
        crop=(1-ink)[None].repeat(3,1,1)
        image=torch.cat([torch.cat([crop,crop],2)]*2,1)[None].requires_grad_(True)
        old,old_details=objective(image)
        old_gradient,=torch.autograd.grad(old,image)
        self.assertEqual(float(old.detach()),0)
        self.assertEqual(float(old_gradient.abs().sum()),0)
        objective.appearance_weight=.5
        loss,details=objective(image)
        gradient,=torch.autograd.grad(loss,image)
        self.assertGreater(float(loss.detach()),0)
        self.assertLess(float(gradient[...,9,10:15].sum()),0)  # Increase RGB -> erase bump.
        for row in details['per_glyph']:
            self.assertEqual(float(row['extracted_stroke_score_no_extra'].detach()),100)
            self.assertGreater(float(row['per_stroke'][0]['weight']),1)
            self.assertGreater(float(row['appearance_loss'].detach()),0)
        frozen=objective.freeze_priority(details)
        _,candidate=objective(image,stroke_weights=frozen)
        self.assertEqual(frozen,objective.freeze_priority(candidate))

    def test_invalid_appearance_settings_are_rejected(self):
        for kwargs in (dict(mass_radii=(0,)),dict(contour_weight=-1),dict(priority_scale=0),
                       dict(mass_allowance=float('nan'))):
            with self.assertRaises(ValueError):AppearanceConfig(**kwargs)


if __name__=='__main__':unittest.main()
