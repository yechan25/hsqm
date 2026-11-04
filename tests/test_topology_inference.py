import json
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
from torch import nn

from stroke_model.reference_topology import TopologyStrokeModel, AlignmentConfig
from stroke_model.infer_topology import ink_diagnostics, load_topology, run_topology_inference


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 2, 1)
    def forward(self, x):
        return [self.conv(x)]


class TinyTopo(TopologyStrokeModel):
    def __init__(self, **kwargs):
        super().__init__(backbone=TinyBackbone(), channels=[8], **kwargs)


class TopologyInferenceTests(unittest.TestCase):
    def test_thin_outputs_are_not_filled_and_coverage_exposes_missing_ink(self):
        image = np.zeros((1, 16, 16), np.float32)
        image[:, 2:14, 6:10] = 1
        masks = np.zeros((1, 16, 16), np.float32)
        masks[:, 2:14, 7] = 1
        result = ink_diagnostics(image, masks)
        self.assertEqual(result['covered_ink_fraction'], .25)
        self.assertEqual(result['uncovered_ink_fraction'], .75)
        self.assertEqual(result['overlapping_ink_fraction'], 0)

    def test_checkpoint_csv_prediction_export_without_annotations(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TinyTopo(image_size=64, width=8, context_width=4, pretrained=False)
            from dataclasses import asdict
            torch.save(dict(model=model.state_dict(), architecture=model.architecture,
                image_size=64, width=8, context_width=4, epoch=5,
                alignment_config=asdict(AlignmentConfig())), root/'best.pt')
            ink = np.zeros((64,64), np.uint8)
            ink[8:56,28:35] = 255
            for name in ('input', 'reference', 'stroke'):
                Image.fromarray(255-ink).save(root/f'{name}.png')
            Image.fromarray(np.full((64,64),255,np.uint8)).save(root/'empty.png')
            pd.DataFrame([
                dict(input_path='input.png', target_path='reference.png', target_stroke_paths=stroke,
                     input_stroke_paths='DO_NOT_READ_MISSING_ANNOTATION.png')
                for stroke in ('empty.png', 'stroke.png')]).to_csv(root/'path.csv', index=False)
            with patch('stroke_model.infer_topology.TopologyStrokeModel', TinyTopo):
                report = run_topology_inference(root/'best.pt', root/'path.csv', sample_rows=[1,2],
                    save_dir=root/'result', show=False)
            self.assertEqual(report['checkpoint_epoch'],5)
            self.assertFalse(report['uses_input_stroke_annotations'])
            self.assertEqual(len(report['skipped']),1)
            self.assertEqual(len(report['results']),1)
            with np.load(root/'result/row_00002.npz') as arrays:
                np.testing.assert_allclose(arrays['soft_masks'], arrays['image']*arrays['probabilities'])
                self.assertEqual(arrays['alignment_theta'].shape,(2,3))
            self.assertTrue((root/'result/row_00002_overview.png').is_file())
            self.assertTrue((root/'result/row_00002_strokes.png').is_file())
            self.assertEqual(json.loads((root/'result/inference.json').read_text())['postprocessing'],'none')

    def test_v2_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'old.pt'
            torch.save({'architecture':'reference_v2'},path)
            with self.assertRaisesRegex(ValueError,'Topo best.pt required'):
                load_topology(path,'cpu')


if __name__ == '__main__':
    unittest.main()
