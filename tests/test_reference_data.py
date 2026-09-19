import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from stroke_model.train_reference import read_manifest, StrokeDataset, collate
from stroke_model.augment_reference import augment


class ReferenceDataTests(unittest.TestCase):
    def test_split_leak_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.json'
            path.write_text(json.dumps([
                {'path': 'a.npz', 'group_id': 'writer1', 'split': 'train'},
                {'path': 'b.npz', 'group_id': 'writer1', 'split': 'val'}]))
            with self.assertRaises(ValueError):
                read_manifest(path)

    def test_npz_and_variable_stroke_collation(self):
        with tempfile.TemporaryDirectory() as folder:
            samples = []
            for k in (1, 3):
                image = np.ones((1, 16, 16), np.float32)
                path = Path(folder) / f'{k}.npz'
                np.savez(path, image=image, reference=image,
                         strokes=np.ones((k, 16, 16), np.float32), labels=np.zeros((16, 16), np.int64))
                samples.append(StrokeDataset([{'path': str(path)}], size=16)[0])
            batch = collate(samples)
            self.assertEqual(batch[2].shape, (2, 3, 16, 16))
            self.assertEqual(batch[2][0, 1:].count_nonzero(), 0)

    def test_augmentation_preserves_ids_and_unknown(self):
        image = torch.zeros(1, 32, 32)
        image[:, 6:26, 6:26] = 1
        labels = torch.full((32, 32), -1, dtype=torch.long)
        labels[6:16, 6:26] = 0
        labels[16:26, 6:26] = 1
        x, y = augment(image, labels, torch.Generator().manual_seed(1))
        self.assertEqual(set(y.unique().tolist()), {-1, 0, 1})
        self.assertTrue(torch.isfinite(x).all())
        self.assertGreater(x[y[None] >= 0].min().item(), 0)


if __name__ == '__main__':
    unittest.main()
