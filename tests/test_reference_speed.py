import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
import torch
from test_reference_model import TinyBackbone
from stroke_model.reference_model import ReferenceStrokeModel, resize, warp, partition_loss
from stroke_model.local_reference_data import stage_manifest


def legacy_score(self, features, reference, strokes, xy, flow, hints, image):
    scores = []
    for j in range(strokes.shape[1]):
        r = self.reference(torch.cat((reference, strokes[:, j:j+1], xy), 1))
        r = warp(resize(r, image.shape[-2:]), flow)
        scores.append(self.head(torch.cat((features, r, hints[:, j:j+1], image, xy), 1)))
    return torch.cat(scores, 1)


class SpeedTests(unittest.TestCase):
    def test_batched_head_matches_original_outputs_and_gradients(self):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        fast = ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16], stroke_chunk_size=2)
        slow = copy.deepcopy(fast)
        slow.score_strokes = types.MethodType(legacy_score, slow)
        reference = torch.rand(2, 1, 16, 16)
        strokes = torch.rand(2, 3, 16, 16)
        x = torch.rand(2, 1, 16, 16, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = fast(x, reference, strokes), slow(y, reference, strokes)
        torch.testing.assert_close(a['logits'], b['logits'], atol=2e-6, rtol=2e-5)
        a['masks'][:, 0, :8].sum().backward()
        b['masks'][:, 0, :8].sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-4)
        for (name, p), (_, q) in zip(fast.named_parameters(), slow.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-4, rtol=2e-3, msg=name)

    def test_cpu_bfloat16_backward_is_finite(self):
        torch.set_num_threads(2)
        model = ReferenceStrokeModel(width=8, backbone=TinyBackbone(), channels=[8, 16])
        x = torch.rand(1, 1, 16, 16)
        s = torch.rand(1, 3, 16, 16)
        s[:, 2] = 0
        labels = torch.randint(0, 2, (1, 16, 16))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(x, x, s)
            loss = partition_loss(out, labels)['loss']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_staging_uses_local_paths_and_reuses_completed_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.npz'
            source.write_bytes(b'test immutable data')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps([{'path': str(source), 'split': 'train', 'group_id': 'a'}]))
            local = stage_manifest(manifest, root / 'local', workers=1)
            row = json.loads(local.read_text())[0]
            self.assertEqual(Path(row['path']).read_bytes(), source.read_bytes())
            source.unlink()
            self.assertEqual(stage_manifest(manifest, root / 'local'), local)


if __name__ == '__main__':
    unittest.main()
