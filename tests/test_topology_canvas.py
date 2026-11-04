import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import torch

from stroke_model.canvas_topology import run_topology_canvas
from stroke_model.infer_topology import select_pattern_rows


class PartitionFixture(torch.nn.Module):
    output_mode = 'partition'
    def forward(self,image,reference,strokes):
        p = torch.ones_like(strokes)/strokes.shape[1]
        return dict(probabilities=p,masks=image*p)


class CanvasTests(unittest.TestCase):
    def test_modified_pattern_matches_all_filenames_not_parent_folders(self):
        frame = pd.DataFrame({'input':['normal.png','a/modified_one.png','b/Modified_two.PNG',
                                      'modified/normal.png','c/modified_(300).png']})
        self.assertEqual(select_pattern_rows(frame,'input','*modified*.png'),[2,3,5])

    def test_canvas_inference_keeps_width_and_does_not_read_input_or_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'best.pt').write_bytes(b'fixture')
            gray=np.full((32,32),255,np.uint8);gray[8:24,14:18]=0
            Image.fromarray(gray).save(root/'reference.png')
            pd.DataFrame([dict(input_path='MISSING_INPUT.png',target_path='reference.png',
                target_stroke_paths='reference.png;reference.png',input_stroke_paths='MISSING_GT.png')]).to_csv(root/'path.csv',index=False)
            stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
            answer=dict(row=0,png='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
            with patch('stroke_model.canvas_topology.load_topology',return_value=(PartitionFixture(),32,3)), \
                 patch('stroke_model.canvas_topology.canvas_prompt',return_value=answer) as prompt:
                report=run_topology_canvas(root/'best.pt',root/'path.csv',save_dir=root/'result',show=False)
            self.assertEqual(len(prompt.call_args.args[0]),1)
            self.assertEqual(report['output_mode'],'partition')
            self.assertFalse(report['uses_input_stroke_annotations'])
            with np.load(report['npz']) as arrays:
                np.testing.assert_allclose(arrays['soft_masks'].sum(0),arrays['image'][0])
                np.testing.assert_array_equal(arrays['hard_masks'].sum(0),arrays['image'][0]>.5)
            self.assertTrue((root/'result/canvas_overview.png').is_file())


if __name__ == '__main__':
    unittest.main()
