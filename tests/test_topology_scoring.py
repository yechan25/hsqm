from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.ndimage import binary_dilation
import torch

from stroke_model.topology_scoring import (extraction_samples,score_topology_results,
    run_scored_topology_csv,run_scored_topology_canvas)


class TopologyScoringTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.strokes=np.zeros((2,64,64),np.float32)
        self.strokes[0,8:49,24]=1;self.strokes[1,48,10:51]=1
        self.reference=binary_dilation(self.strokes.max(0),iterations=2).astype(np.float32)[None]
        self.source_config=dict(translation=.1,rotation=.1,scale_min=.999,scale_max=1.001,
                               shear=.001,registration_iterations=1)

    def snapshot(self,path,image,owner):
        masks=np.zeros_like(self.strokes);masks[owner]=image[0]
        np.savez_compressed(path,image=image,reference=self.reference,reference_strokes=self.strokes,soft_masks=masks)

    def test_source_score_ignores_channel_swap_but_detects_actual_deleted_ink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[]
            missing=self.reference.copy();missing[:,45:54,32:57]=0
            originals={}
            for index,(image,owner) in enumerate(((self.reference,0),(self.reference,1),(missing,0))):
                path=root/f'{index}.npz';self.snapshot(path,image,owner)
                originals[path]=path.read_bytes();entries.append(dict(npz=str(path),input=f'case{index}.png',csv_row=index+1))
            (root/'best.pt').write_bytes(b'unchanged checkpoint')
            extraction=dict(checkpoint=str(root/'best.pt'),checkpoint_epoch=3,output_mode='partition',results=entries)
            result=score_topology_results(extraction,source_config=self.source_config,save_dir=root/'scores',show=False)
            scores=[row['structural_score'] for row in result['indicator']['results']]
            self.assertAlmostEqual(scores[0],scores[1],places=9)
            self.assertGreaterEqual(scores[0],99)
            self.assertLess(scores[2],scores[0]-10)
            self.assertFalse(result['topology_masks_used_for_score'])
            self.assertEqual(result['indicator']['measurement_mode'],'source')
            self.assertTrue((root/'scores/glyph_scores.csv').is_file())
            self.assertTrue((root/'scores/combined_0001.png').is_file())
            with zipfile.ZipFile(result['debug_zip']) as archive:
                self.assertIn('arrays/sample_0001.npz',archive.namelist())
                self.assertIn('measurements/sample_0001.npz',archive.namelist())
            for path,content in originals.items():self.assertEqual(path.read_bytes(),content)
            self.assertEqual((root/'best.pt').read_bytes(),b'unchanged checkpoint')

    def test_canvas_snapshot_adapter_supplies_ids_not_invented_zero_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'canvas.npz';self.snapshot(path,self.reference,0)
            samples=extraction_samples(dict(npz=str(path)))
            self.assertEqual(len(samples[0]['strokes']),2)
            self.assertNotIn('missing',samples[0]['strokes'][0])
            self.assertNotIn('junction',samples[0]['strokes'][0])
            np.savez_compressed(path,image=self.reference,soft_masks=self.strokes,reference_strokes=self.strokes)
            with self.assertRaisesRegex(ValueError,'full reference'):extraction_samples(dict(npz=str(path)))

    def test_csv_and_canvas_wrappers_reuse_extraction_once(self):
        extraction={'results':[{'npz':'fixture.npz'}]};scored={'save_dir':'fixture'}
        with patch('stroke_model.topology_scoring.run_topology_inference',return_value=extraction) as infer, \
             patch('stroke_model.topology_scoring.score_topology_results',return_value=scored) as score:
            self.assertEqual(run_scored_topology_csv('best.pt','paths.csv'),scored)
            infer.assert_called_once();score.assert_called_once()
            self.assertFalse(infer.call_args.kwargs['show'])
        with patch('stroke_model.topology_scoring.run_topology_canvas',return_value=extraction) as canvas, \
             patch('stroke_model.topology_scoring.score_topology_results',return_value=scored) as score:
            self.assertEqual(run_scored_topology_canvas('best.pt','paths.csv'),scored)
            canvas.assert_called_once();score.assert_called_once()


if __name__=='__main__':unittest.main()
