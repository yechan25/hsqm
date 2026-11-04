import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import findfont
import numpy as np
import pandas as pd
from PIL import Image
import torch

from font_test.render import font_supports,render_glyph
from stroke_model.font_topology import reference_catalog,run_topology_fonts,run_scored_topology_fonts


class FontTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.font=findfont('DejaVu Sans')

    def test_render_is_deterministic_preserves_aspect_and_placement(self):
        ref=np.zeros((1,64,64),np.float32);ref[:,15:48,10:54]=1
        first=render_glyph('A',self.font,ref)
        np.testing.assert_array_equal(first,render_glyph('A',self.font,ref))
        self.assertEqual(first.shape,ref.shape)
        self.assertGreater(float(first.max()),.9);self.assertEqual(float(first.min()),0)
        yy,xx=np.where(first[0]>.5)
        self.assertTrue(14<=yy.min() and yy.max()<=48)
        self.assertTrue(9<=xx.min() and xx.max()<=54)
        # Width is NOT stretched to the target's wide box.
        self.assertLess(xx.max()-xx.min(),40)
        centered=render_glyph('A',self.font,ref,'centered')
        self.assertGreater(float(np.abs(first-centered).sum()),1)

    def test_missing_font_glyph_and_mismatched_text_fail(self):
        ref=np.ones((1,32,32),np.float32)
        self.assertTrue(font_supports(self.font,'A'))
        self.assertFalse(font_supports(self.font,'\U0010ffff'))
        for text in ('AB','','\U0010ffff'):
            with self.assertRaises(ValueError):render_glyph(text,self.font,ref)

    def test_catalog_never_guesses_numeric_id_as_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv=Path(tmp)/'paths.csv'
            pd.DataFrame([dict(input='missing',target_path='0.png',target_stroke_paths='s.png'),
                          dict(input='missing',target_path='G_아_2.png',target_stroke_paths='s.png')]).to_csv(csv,index=False)
            catalog=reference_catalog(csv)
            self.assertIsNone(catalog[0]['glyph']);self.assertEqual(catalog[1]['glyph'],'아')
            self.assertEqual(catalog[1]['csv_row'],2)

    def test_font_inference_keeps_reference_control_and_saves_raw_results(self):
        class Model(torch.nn.Module):
            output_mode='partition'
            def forward(self,image,reference,strokes):
                p=(strokes+1).softmax(1)
                return dict(probabilities=p,masks=image*p)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'best.pt';checkpoint.write_bytes(b'unchanged')
            ref=np.zeros((1,64,64),np.float32);ref[:,12:52,26:34]=1;ref[:,28:36,12:52]=1
            parts=np.zeros((2,64,64),np.float32);parts[0,12:52,30]=1;parts[1,32,12:52]=1
            for name,ink in [('reference.png',ref[0]),('s1.png',parts[0]),('s2.png',parts[1])]:
                Image.fromarray(np.uint8((1-ink)*255)).save(root/name)
            csv=root/'paths.csv'
            pd.DataFrame([dict(input='DOES_NOT_EXIST.png',target_path='reference.png',
                target_stroke_paths='s1.png;s2.png',input_stroke_paths='NEVER_READ.png')]).to_csv(csv,index=False)
            with patch('stroke_model.font_topology.load_topology',return_value=(Model(),64,3)):
                report=run_topology_fonts(checkpoint,csv,1,'A',[self.font,self.font],save_dir=root/'results',show=False)
            self.assertEqual(len(report['results']),2)
            self.assertTrue(report['results'][0]['control'])
            self.assertFalse(report['segmentation_accuracy_assessed'])
            self.assertTrue(Path(report['comparison']).is_file())
            for result in report['results']:
                with np.load(result['npz']) as arrays:
                    np.testing.assert_array_equal(arrays['reference'],ref)
                    np.testing.assert_array_equal(arrays['reference_strokes'],parts)
                    np.testing.assert_allclose(arrays['soft_masks'].sum(0),arrays['image'][0],atol=1e-6)
                    if result['control']:np.testing.assert_array_equal(arrays['image'],ref)
                    else:self.assertGreater(float(np.abs(arrays['image']-ref).sum()),10)
            self.assertEqual(checkpoint.read_bytes(),b'unchanged')
            self.assertEqual(json.loads((root/'results/font_report.json').read_text(encoding='utf-8'))['glyph'],'A')
            config=dict(translation=.01,rotation=.01,scale_min=.999,scale_max=1.001,shear=.001,
                        registration_iterations=1)
            with patch('stroke_model.font_topology.run_topology_fonts',return_value=report) as extract:
                scored=run_scored_topology_fonts('fixture',source_config=config,show=False)
                extract.assert_called_once();self.assertFalse(extract.call_args.kwargs['show'])
            scores=scored['font_scores']
            self.assertEqual([s['font'] for s in scores],[s['label'] for s in report['results']])
            self.assertGreaterEqual(scores[0]['structural_score'],99)
            self.assertLess(scores[1]['structural_score'],scores[0]['structural_score'])
            self.assertFalse(scored['topology_masks_used_for_score'])
            saved=pd.read_csv(Path(scored['save_dir'])/'font_scores.csv')
            self.assertEqual(len(saved),2)
            self.assertIn('extra_score_cap',saved.columns)
            self.assertTrue(Path(scored['debug_zip']).is_file())
            self.assertEqual(checkpoint.read_bytes(),b'unchanged')


if __name__=='__main__':unittest.main()
