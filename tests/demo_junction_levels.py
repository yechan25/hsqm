"""Reproduce weak-branch regression and save the simplified diagnostic figure."""
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

from stroke_model.score_stroke_structure import score_saved_strokes
from test_stroke_structure import line_cases


def main():
    torch.set_num_threads(1)
    reference, cases = line_cases()
    weak_arm = cases['normal'].clone()
    weak_arm[0, 15:30, 30:35] = .35
    masks = torch.cat([cases['normal'], cases['branch']*.6, weak_arm, cases['noise']])
    destination = ROOT / 'outputs' / 'junction_levels_demo' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'normal_weakT_weakArm_noise.npz', soft_masks=masks.numpy(),
                        reference_strokes=reference.repeat(4, 1, 1).numpy())
    score_saved_strokes(destination / 'normal_weakT_weakArm_noise.npz',
                        save_dir=destination / 'scored', show=False)
    print('Demo:', destination)


if __name__ == '__main__':
    main()
