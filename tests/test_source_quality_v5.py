import unittest
import numpy as np
import torch

from stroke_model.source_quality_v5 import (foreground_evidence, contour_diagnosis,
                                          assess_source_quality, separate_unassigned_ink)
from stroke_model.source_structure import PreparedSourceMetric


class QualityV5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.reference=np.zeros((64,64),np.float32)
        self.reference[10:54,28:36]=1
        self.strokes=self.reference[None]

    def test_foreground_gate_is_smooth_and_keeps_dark_ink(self):
        x=torch.tensor([0.,.01,.1,.5,1.],requires_grad=True)
        y=foreground_evidence(x)
        self.assertEqual(float(y[0].detach()),0)
        self.assertEqual(float(y[-1].detach()),1)
        self.assertLess(float(y[1].detach()),.0003)
        self.assertGreater(float(y[3].detach()),.499)
        gradient,=torch.autograd.grad(y.sum(),x)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient>0).all())

    def test_local_bump_and_dent_are_not_lost_when_topology_is_unchanged(self):
        bump=self.reference.copy(); bump[20:24,36:39]=1
        dent=self.reference.copy(); dent[30:34,33:36]=0
        baseline=contour_diagnosis(self.reference,self.reference)
        self.assertEqual(baseline['contour_score_cap'],100)
        self.assertEqual(baseline['contour_excess'],0)
        for image in (bump,dent):
            measured=contour_diagnosis(image,self.reference)
            self.assertGreater(measured['contour_excess'],.01)
            self.assertLess(measured['contour_score_cap'],90)
            self.assertGreater((measured['protrusion_map']+measured['indentation_map']).max(),.1)

    def test_matching_serif_is_not_automatically_an_error(self):
        serif=self.reference.copy(); serif[10:13,24:40]=1
        diagnosis=contour_diagnosis(serif,serif)
        self.assertEqual(diagnosis['contour_excess'],0)
        self.assertEqual(diagnosis['contour_score_cap'],100)

    def test_background_reduced_while_dark_extra_stroke_remains(self):
        background=np.maximum(self.reference,.01)
        pose=dict(parameters=[0,0,0,1,1,0])
        original=PreparedSourceMetric(self.reference,self.strokes)(torch.tensor(background))
        cleaned=assess_source_quality(background,self.reference,self.strokes,registration=pose)
        self.assertLess(cleaned['foreground_extra'],float(original['extra'][0])*.1)
        extra=background.copy();extra[15:35,48:53]=1
        defect=assess_source_quality(extra,self.reference,self.strokes,registration=pose)
        self.assertGreater(defect['foreground_extra'],.1)
        self.assertLess(defect['experimental_score'],cleaned['experimental_score'])

    def test_detached_marks_are_separate_but_still_count_in_whole_crop_score(self):
        reference=np.zeros((64,64),np.float32)
        reference[10:42,12:17]=1
        reference[15:35,38:44]=1  # A legitimate disconnected second stroke.
        strokes=np.stack([reference*(np.indices(reference.shape)[1]<30),
                          reference*(np.indices(reference.shape)[1]>=30)])
        pose=dict(parameters=[0,0,0,1,1,0])
        baseline=assess_source_quality(reference,reference,strokes,registration=pose)
        self.assertEqual(baseline['unassigned_component_count'],0)
        self.assertFalse(baseline['unassigned_ink_review_required'])
        extra=reference.copy();extra[53:57,54:58]=1
        measured=assess_source_quality(extra,reference,strokes,registration=pose)
        self.assertEqual(measured['unassigned_component_count'],1)
        self.assertTrue(measured['unassigned_ink_review_required'])
        self.assertGreater(measured['unassigned_ink_fraction'],.03)
        np.testing.assert_array_equal(measured['body_image'],reference)
        self.assertAlmostEqual(measured['body_diagnostic_score'],baseline['body_diagnostic_score'])
        self.assertLess(measured['experimental_score'],baseline['experimental_score'])
        self.assertEqual(float(measured['source_image'][54,55]),1)  # No original pixel deletion.

    def test_attached_extra_branch_and_missing_stroke_remain_body_errors(self):
        pose=dict(parameters=[0,0,0,1,1,0])
        branch=self.reference.copy();branch[22:27,35:46]=1
        measured=assess_source_quality(branch,self.reference,self.strokes,registration=pose)
        self.assertEqual(measured['unassigned_component_count'],0)
        self.assertLess(measured['body_diagnostic_score'],90)
        missing=assess_source_quality(np.zeros_like(self.reference),self.reference,self.strokes,registration=pose)
        self.assertEqual(missing['body_diagnostic_score'],0)


if __name__=='__main__':
    unittest.main()
