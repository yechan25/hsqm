"""Generate a reproducible synthetic comparison, with no model or Drive access.

Run from the repository: python tests/demo_stroke_structure.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

from stroke_model.stroke_structure import StructureConfig, StrokeStructureLoss
from test_stroke_structure import line_cases


def main():
    torch.set_num_threads(1)
    reference, cases = line_cases()
    config = StructureConfig(width_weight=0)
    criterion = StrokeStructureLoss(reference, config)
    destination = ROOT / 'outputs' / 'stroke_structure_demo'
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    names = list(cases)
    fig, axes = plt.subplots(3, len(names), figsize=(2.2 * len(names), 9),
                             squeeze=False, layout='constrained')
    for col, (name, mask) in enumerate(cases.items()):
        result = criterion(mask, return_maps=True)
        values = {key: float(value[0, 0]) for key, value in result['per_stroke'].items()}
        rows.append({'case': name, **values})
        axes[0, col].imshow(mask[0].numpy(), cmap='gray_r', vmin=0, vmax=1)
        axes[0, col].set_title(name)
        axes[1, col].imshow(result['maps'][0]['split'][0, 0].numpy(), cmap='magma', vmin=0, vmax=1)
        axes[1, col].set_title(f"M {values['missing']:.3f} / D {values['split']:.3f}", fontsize=10)
        axes[2, col].imshow(result['maps'][0]['junction'][0, 0].numpy(), cmap='magma', vmin=0, vmax=1)
        axes[2, col].set_title(f"J {values['junction']:.3f} / W {values['width']:.3f}", fontsize=10)
        for row in range(3):
            axes[row, col].axis('off')
    fig.suptitle('Synthetic checks: input / unreachable-ink diagnostic / junction response\n'
                 'Extractor diagnostics, NOT source quality. M=reference deficit, D=split, J=junction, W=width.')
    fig.savefig(destination / 'comparison.png', dpi=150)
    plt.close(fig)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / 'comparison.csv', index=False)
    print(frame.round(6).to_string(index=False))
    print('Preview:', destination / 'comparison.png')


if __name__ == '__main__':
    main()
