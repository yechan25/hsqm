import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from diffusion_model.score_pilot import training_catalog, scored_ink, generate_pilot, score_pilot


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        strokes = np.zeros((2, 32, 32), np.float32)
        strokes[0, 5:26, 8:12] = 1
        strokes[1, 22:26, 8:25] = 1
        self.reference = strokes.max(0)[None]
        # Deliberately omit input labels/target masks: evaluator must not need them.
        np.savez(self.root/'sample.npz', reference=self.reference, strokes=strokes)
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps([
            dict(path='sample.npz', split='val', group_id='v', char_id='val-only'),
            dict(path='sample.npz', split='train', group_id='t', char_id='train-only')]), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_train_references_can_be_selected(self):
        entries = training_catalog(self.manifest)
        self.assertEqual([e['manifest_row'] for e in entries], [2])
        with self.assertRaisesRegex(ValueError, 'only manifest'):
            generate_pilot(self.manifest, [1], self.root/'out', pipeline=object())
        self.assertFalse((self.root/'out').exists())

    def test_fixed_polarity_and_transparency_no_auto_flip(self):
        white = scored_ink(Image.new('RGB', (8, 8), 'white'), 8)
        black = scored_ink(Image.new('RGB', (8, 8), 'black'), 8)
        transparent = scored_ink(Image.new('RGBA', (8, 8), (0, 0, 0, 0)), 8)
        self.assertEqual(float(white.sum()), 0)
        self.assertEqual(float(black.sum()), 64)
        self.assertEqual(float(transparent.sum()), 0)

    def test_generated_snapshots_reach_existing_scorer(self):
        class FakePipeline:
            scheduler = SimpleNamespace(config={'class': 'test double'})
            config = {}
            def __call__(self, **kwargs):
                # Entirely missing glyph, must not silently be removed/replaced.
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'white')],
                                       nsfw_content_detected=[False])

        class Extractor(torch.nn.Module):
            output_mode = 'partition'
            def forward(self, image, reference, strokes):
                probabilities = (strokes+1).softmax(1)
                return dict(probabilities=probabilities, masks=image*probabilities)

        before = (self.root/'sample.npz').read_bytes()
        report = generate_pilot(self.manifest, [2], self.root/'out', strengths=[.3], seeds=[7],
                                pipeline=FakePipeline())
        self.assertEqual(len(report['results']), 2)
        self.assertFalse(report['free_text_generation'])
        with np.load(report['results'][1]['npz']) as data:
            np.testing.assert_array_equal(data['reference'], self.reference)
            self.assertEqual(data['image'].sum(), 0)
        with patch('stroke_model.infer_topology.load_topology', return_value=(Extractor(), 32, 1)):
            result, table = score_pilot(self.root/'out/generation_report.json',
                                       self.root/'unused_checkpoint.pt', show=False)
        self.assertGreater(table.iloc[0]['structural_score'], 90)
        self.assertEqual(table.iloc[1]['structural_score'], 0)
        self.assertTrue(Path(result['debug_zip']).is_file())
        import zipfile
        with zipfile.ZipFile(result['debug_zip']) as archive:
            self.assertIn('generation_report.json', archive.namelist())
            self.assertIn('generated_originals/0001.png', archive.namelist())
        self.assertEqual((self.root/'sample.npz').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
