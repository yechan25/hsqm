"""Generate target-only geometric variants of reviewed training samples.

Reference glyph/stroke IDs stay fixed. Never augment val/test or cross splits.
python -m stroke_model.augment_reference --manifest data_v2/manifest.json --variants 20
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_reference import read_manifest, StrokeDataset


def augment(image, labels, generator):
    def uniform(low, high):
        return low + (high - low) * torch.rand((), generator=generator).item()
    angle = math.radians(uniform(-10, 10))
    scale = uniform(.92, 1.08)
    theta = torch.tensor([[[scale * math.cos(angle), -scale * math.sin(angle), uniform(-.06, .06)],
                           [scale * math.sin(angle), scale * math.cos(angle), uniform(-.06, .06)]]])
    grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
    # Smooth displacement in normalized coordinates: at S=128, about 1px RMS.
    elastic = torch.randn(1, 2, 4, 4, generator=generator) * .015
    elastic = F.interpolate(elastic, size=image.shape[-2:], mode='bicubic', align_corners=False)
    grid = grid + elastic.permute(0, 2, 3, 1)
    warped = F.grid_sample(image[None], grid, align_corners=False)[0]
    owners = F.grid_sample((labels + 1)[None, None].float(), grid, mode='nearest', align_corners=False)[0, 0].long() - 1
    # Preserve unknown/background; never fabricate an owner for an ambiguous pixel.
    warped = warped * uniform(.8, 1.)
    return warped, owners


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--variants', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, default=128)
    args = p.parse_args()
    rows = read_manifest(args.manifest)
    # Fresh output prevents silently overwriting a previous experiment.
    dest = Path(args.manifest).resolve().parent / f'augmented_seed{args.seed}'
    dest.mkdir(exist_ok=False)
    generator = torch.Generator().manual_seed(args.seed)
    seeds = [r for r in rows if r['split'] == 'train' and r.get('source') != 'augmented']
    dataset = StrokeDataset(seeds, args.image_size)
    extra = []
    for i, row in enumerate(seeds):
        image, reference, strokes, labels = dataset[i]
        for j in range(args.variants):
            x, y = augment(image, labels, generator)
            # Reject crops that completely remove a supervised stroke.
            original_ids = labels[labels >= 0].unique()
            if any(not (y == k).any() for k in original_ids):
                continue
            path = dest / f'{i:05d}_{j:03d}.npz'
            np.savez_compressed(path, image=x.numpy(), reference=reference.numpy(),
                                strokes=strokes.numpy(), labels=y.numpy())
            extra.append({**row, 'path': str(path), 'source': 'augmented', 'parent': row['path']})
    out = dest / 'manifest.json'
    out.write_text(json.dumps(rows + extra, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(extra)} training variants; original val/test unchanged. Manifest: {out}')


if __name__ == '__main__':
    main()
