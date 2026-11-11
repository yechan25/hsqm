import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image,ImageDraw
from stroke_model.stroke_reference_fit import ReferenceFitConfig,fit_reference_stroke,fit_reference_glyph,render_reference


def raster(points,width=1,size=64):
    image=Image.new('L',(size,size));ImageDraw.Draw(image).line([tuple(p) for p in points],fill=255,width=width)
    return np.asarray(image,np.float32)/255


def iou(a,b):
    a=a>.5;b=b>.5
    return (a&b).sum()/max((a|b).sum(),1)


class ReferenceFitTests(unittest.TestCase):
    def test_straight_prototype_follows_position_without_copying_side_branch(self):
        reference=raster([(22,7),(22,56)])
        source=raster([(28,10),(28,58)],5)
        source=np.maximum(source,raster([(15,31),(28,31)],3))
        result=fit_reference_stroke(reference,source)
        self.assertEqual(result['metadata']['method'],'robust_straight_reference_fit')
        self.assertGreater(result['warped'][31,28],.9)
        self.assertLess(result['warped'][31,15],.1)
        self.assertLess(np.std(result['target_path'][:,0]),.2)
        self.assertGreater(result['metadata']['width_observed'],3)
        self.assertGreater(result['metadata']['max_displacement'],3)

    def test_curved_reference_deforms_smoothly_to_shifted_curved_extraction(self):
        t=np.linspace(0,1,90);points=np.stack((10+35*t,10+36*t*t),1)
        reference=raster(points)
        source=raster(points+np.array([5,4]),5)
        result=fit_reference_stroke(reference,source)
        self.assertEqual(result['metadata']['method'],'regularized_curved_reference_fit')
        before=render_reference(reference,result['reference_path'],result['reference_path'],result['width_scale'])
        self.assertGreater(iou(result['warped'],source),iou(before,source)+.2)
        self.assertTrue(np.isfinite(result['target_path']).all())
        self.assertLessEqual(result['metadata']['max_displacement'],16.001)

    def test_disconnected_decorative_mark_is_not_reconstructed(self):
        reference=raster([(22,7),(22,56)])
        source=raster([(27,8),(27,56)],5);source[54:58,48:52]=1
        result=fit_reference_stroke(reference,source)
        self.assertGreater(result['metadata']['components'],1)
        self.assertLess(result['warped'][54:58,48:52].sum(),.01)

    def test_loop_uses_shape_preserving_fallback_and_reports_it(self):
        reference=Image.new('L',(64,64));ImageDraw.Draw(reference).ellipse((15,15,45,45),outline=255,width=1)
        source=Image.new('L',(64,64));ImageDraw.Draw(source).ellipse((19,18,49,48),outline=255,width=3)
        result=fit_reference_stroke(np.asarray(reference,np.float32)/255,np.asarray(source,np.float32)/255)
        self.assertEqual(result['metadata']['method'],'affine_topology_preserving_fallback')
        self.assertLess(result['warped'][33,34],.1)
        self.assertTrue(result['metadata']['warnings'])

    def test_joint_contact_constraint_reduces_independent_fit_gap(self):
        references=np.stack((raster([(25,8),(25,53)]),raster([(25,29),(47,29)])))
        observed=np.stack((raster([(27,8),(27,53)],3),raster([(30,31),(49,31)],3)))
        fitted=fit_reference_glyph(references,observed)
        self.assertTrue(fitted['contacts'])
        self.assertTrue(all(c['gap_after']<=c['gap_before']+1e-6 for c in fitted['contacts']))
        json.dumps([f['metadata'] for f in fitted['fits']],allow_nan=False)

    def test_full_reference_reconnects_sparse_horizontal_hint_to_stem(self):
        stems=raster([(24,8),(24,53)])
        short=raster([(30,29),(38,29)])
        full=np.maximum(raster([(24,8),(24,53)],3),raster([(24,29),(39,29)],3))
        extracted=np.stack((raster([(25,8),(25,53)],5),raster([(33,29),(40,29)],3)))
        result=fit_reference_glyph(np.stack((stems,short)),extracted,glyph_reference=full)
        self.assertTrue(result['contacts'])
        self.assertGreater(result['candidate'][29,28],.5)
        self.assertEqual(result['fits'][1]['metadata']['method'],'robust_straight_reference_fit')

    def test_high_confidence_ink_is_not_gray_when_thickening_thin_reference(self):
        reference=raster([(20,8),(20,53)])
        observed=raster([(25,8),(25,53)],7)
        result=fit_reference_stroke(reference,observed)
        self.assertGreater(result['warped'][25,24],.95)
        self.assertGreater(result['warped'][25,26],.95)

    def test_empty_extraction_is_flagged_not_fabricated_as_success(self):
        reference=raster([(22,7),(22,56)])
        result=fit_reference_glyph(reference[None],np.zeros((1,64,64),np.float32))
        self.assertEqual(result['fits'][0]['metadata']['method'],'unreliable_reference_fallback')
        self.assertTrue(result['fits'][0]['metadata']['warnings'])

    def test_new_notebook_has_only_extraction_and_deformation(self):
        path=Path(__file__).resolve().parents[1]/'notebooks/stroke_reference_fit.ipynb'
        notebook=json.loads(path.read_text(encoding='utf-8'))
        cells=[c for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(len(cells),3)
        code='\n'.join(''.join(c['source']) for c in cells)
        self.assertIn("drive.mount('/content/drive'",code)
        self.assertIn('fit_reference_glyph(',code)
        self.assertNotIn('generate_guided_pair(',code)
        self.assertNotIn('StableDiffusionImg2ImgPipeline',code)
        self.assertIn("output.get('hints',strokes)",code)


if __name__=='__main__':unittest.main()
