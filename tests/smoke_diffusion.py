"""Offline CPU end-to-end train/resume equivalence and inference, no model download."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT/'outputs/diffusion_smoke'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    rows = []
    for i in range(4):
        ink = np.zeros((1, 16, 16), dtype=np.float32)
        ink[:, 3:13, 3+i:5+i] = 1
        ink[:, 3:5, 3:13] = 1
        path = root/f'{i}.npz'
        np.savez_compressed(path, image=ink, reference=ink)
        rows.append({'path': str(path), 'group_id': 'source'+str(i),
                     'char_id': '가', 'split': 'train' if i < 2 else 'val'})
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    common = [sys.executable, '-m', 'diffusion_model.train', '--manifest', str(manifest),
              '--size', '16', '--small', '--cpu', '--batch-size', '1', '--accumulation', '1',
              '--checkpoint-every', '1', '--validate-every', '1', '--sample-every', '1', '--sample-steps', '2']
    def run(args):
        subprocess.run(args, cwd=ROOT, check=True, env={**os.environ, 'PYTHONUTF8': '1'})
    resumed, uninterrupted = root/'resumed', root/'uninterrupted'
    run(common + ['--output', str(resumed), '--steps', '1'])
    run(common + ['--output', str(resumed), '--steps', '2', '--resume', str(resumed/'last.pt')])
    run(common + ['--output', str(uninterrupted), '--steps', '2'])
    a = torch.load(resumed/'last.pt', map_location='cpu', weights_only=True)
    b = torch.load(uninterrupted/'last.pt', map_location='cpu', weights_only=True)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
        torch.testing.assert_close(a['ema'][key], b['ema'][key], rtol=0, atol=0)
    assert a['step'] == b['step'] == 2
    assert [json.loads(s)['step'] for s in (resumed/'metrics.jsonl').read_text().splitlines()] == [1, 2]
    from PIL import Image
    with np.load(root/'2.npz') as item:
        Image.fromarray(((1-item['reference'][0])*255).astype('uint8')).save(root/'ref.png')
    run([sys.executable, '-m', 'diffusion_model.generate', '--checkpoint', str(resumed/'best_weights.pt'),
         '--reference', str(root/'ref.png'), '--output', str(root/'generated'), '--steps', '2', '--count', '1'])
    assert (root/'generated/seed_1234.png').is_file()
    print('PASS: actual CPU DDPM train / bitwise resume / EMA / preview / inference:', root)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
