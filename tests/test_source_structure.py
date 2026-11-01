import json
import tempfile
from pathlib import Path
import unittest
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import binary_dilation
import torch
from stroke_model.source_structure import register_reference, affine_warp, PreparedSourceMetric
from stroke_model.structural_indicator import StructuralIndicator, score_structure_report


class SourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.strokes=np.zeros((3,64,64),np.float32)
        self.strokes[0,8:49,24]=1
        self.strokes[1,48,10:51]=1
        self.strokes[2,17,24:44]=1
        self.target=binary_dilation(self.strokes.max(0),iterations=2).astype(np.float32)

    def score(self, image, register=False):
        if register:
            target,strokes,alignment=register_reference(image,self.target,self.strokes)
            metric=PreparedSourceMetric(target,strokes,parameters=alignment['parameters'])
        else:
            metric=PreparedSourceMetric(self.target,self.strokes)
        measured=metric(torch.tensor(image))
        score=StructuralIndicator()(measured['missing'],measured['split'],measured['junction'],extra=measured['extra'])
        return measured,score['score'].item()

    def test_combined_global_affine_variation_is_not_missing_or_extra(self):
        image=affine_warp(self.target,[4,-3,7,1.05,.96,.03])
        measured,score=self.score(image,register=True)
        self.assertGreaterEqual(score,90)
        self.assertLess(measured['missing'].max().item(),.02)
        self.assertLess(measured['extra'].item(),.03)

    def test_target_junction_is_allowed_but_new_arm_is_detected(self):
        normal,normal_score=self.score(self.target)
        branch=self.target.copy()
        branch[49:58,22:27]=1
        measured,score=self.score(branch)
        self.assertEqual(normal_score,100)
        self.assertEqual(normal['junction'].max().item(),0)
        self.assertGreater(measured['junction'].max().item(),.001)
        self.assertLess(score,80)

    def test_real_ink_deletion_addition_and_gap_remain_penalized(self):
        missing=self.target.copy();missing[15:20,28:46]=0
        added=self.target.copy();added[31:36,27:45]=1
        split=self.target.copy();split[45:52,36:39]=0
        for image,term,bound in ((missing,'missing',.65),(added,'extra',.1),(split,'split',.2)):
            with self.subTest(term=term):
                measured,score=self.score(image,register=True)
                self.assertGreater(measured[term].max().item(),bound)
                self.assertLess(score,80)

    def test_empty_image_does_not_get_a_high_score(self):
        measured,score=self.score(np.zeros_like(self.target))
        self.assertEqual(score,0)
        self.assertTrue(torch.isfinite(measured['extra']).all())

    def test_affine_view_does_not_crop_away_added_border_ink(self):
        parameters=[10,10,0,.9,.9,0]
        metric=PreparedSourceMetric(self.target,self.strokes,parameters=parameters)
        image=affine_warp(self.target,parameters)
        added=image.copy();added[1:6,1:16]=1
        baseline=metric(torch.tensor(image))['extra'].item()
        result=metric(torch.tensor(added))
        self.assertGreater(result['extra'].item(),baseline+.1)
        self.assertGreater(result['extra_map'][0,0,2,3].item(),.9)

    def test_frozen_registration_keeps_image_gradients(self):
        metric=PreparedSourceMetric(self.target,self.strokes).double()
        image=torch.tensor(self.target*.8+.01,dtype=torch.float64,requires_grad=True)
        result=metric(image)
        gradient,=torch.autograd.grad(result['extra'].sum(),image)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[60,55].item(),0)
        direction=torch.zeros_like(image);direction[60,55]=1
        eps=1e-5
        finite=(metric(image.detach()+eps*direction)['extra']-metric(image.detach()-eps*direction)['extra'])/(2*eps)
        self.assertAlmostEqual(finite.item(),(gradient*direction).sum().item(),places=7)

    def test_bad_extractor_channels_do_not_establish_source_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            np.savez_compressed(root/'input.npz',image=self.target[None],reference_image=self.target[None],
                                reference_strokes=self.strokes,soft_masks=np.zeros_like(self.strokes))
            report={'results':[{'npz':str(root/'input.npz'),'strokes':[
                {'stroke':k+1,'missing':1.,'split':1.,'junction':1.} for k in range(3)]}]}
            result=score_structure_report(report,measurement_mode='source',save_dir=root/'scores',show_images=False)
            self.assertGreaterEqual(result['results'][0]['structural_score'],90)
            self.assertEqual(result['results'][0]['extractor_strokes'][0]['missing'],1)
            self.assertEqual(result['results'][0]['strokes'][0]['missing'],0)
            self.assertEqual(result['measurement_mode'],'source')

    def test_missing_full_reference_fails_instead_of_reusing_bad_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            report={'results':[{'strokes':[{'missing':0,'split':0,'junction':0}]}]}
            with self.assertRaisesRegex(ValueError,'full reference_image'):
                score_structure_report(report,measurement_mode='source',save_dir=Path(tmp)/'scores',show_images=False)


if __name__=='__main__':
    unittest.main()
