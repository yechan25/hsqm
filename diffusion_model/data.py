"""Read V2 NPZ manifests or image-pair manifests without changing their splits."""
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F


def read_manifest(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty list')
    groups, seen = {}, set()
    for original in rows:
        row = dict(original)
        if row.get('split') not in ('train', 'val', 'test'):
            raise ValueError('Each row needs split=train/val/test')
        group = str(row.get('group_id', '')).strip()
        if not group:
            raise ValueError('Each row needs a writer/font/source group_id')
        if groups.setdefault(group, row['split']) != row['split']:
            raise ValueError('Group leakage: ' + group)
        key = 'path' if 'path' in row else 'image'
        if key not in row:
            raise ValueError('Row needs NPZ path or image/reference paths')
        for field in (key, 'reference') if key == 'image' else (key,):
            row[field] = str((path.parent / row[field]).resolve())
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        identity = row[key]
        if identity in seen:
            raise ValueError('Duplicate target file: ' + identity)
        seen.add(identity)
        row['text'] = unicodedata.normalize('NFC', str(row.get('text', row.get('char_id', ''))))
        yield row


def load_image(path):
    """Explicit convention: image files are dark ink on a white background."""
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        gray = Image.alpha_composite(white, rgba).convert('L')
        return torch.from_numpy(1 - np.asarray(gray, dtype=np.float32).copy() / 255)[None]


def resize(x, size):
    return F.interpolate(x[None], (size, size), mode='bilinear', align_corners=False)[0]


class GlyphDataset(torch.utils.data.Dataset):
    def __init__(self, rows, size=128):
        self.rows, self.size = list(rows), size
        if not self.rows:
            raise ValueError('Empty dataset split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        strokes = None
        if 'path' in row:
            with np.load(row['path'], allow_pickle=False) as sample:
                image = torch.from_numpy(sample['image'].copy()).float()
                reference = torch.from_numpy(sample['reference'].copy()).float()
                if 'strokes' in sample:
                    strokes = torch.from_numpy(sample['strokes'].copy()).float()
        else:
            image, reference = load_image(row['image']), load_image(row['reference'])
        for x in (image, reference):
            if x.ndim != 3 or x.shape[0] != 1 or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError('Expected finite 1xHxW ink tensors in [0,1]: ' + str(row))
        if image.sum() < 1 or reference.sum() < 1:
            raise ValueError('Empty target/reference ink: ' + str(row))
        if strokes is not None:
            if strokes.ndim != 3 or not len(strokes) or not torch.isfinite(strokes).all() or strokes.min() < 0 or strokes.max() > 1:
                raise ValueError('Invalid reference strokes')
            if strokes.shape[-2:] != reference.shape[-2:] or not (strokes.flatten(1).sum(1) > 0).any():
                raise ValueError('Reference strokes must match reference size and contain ink')
            strokes = resize(strokes, self.size)
        return {'image': resize(image, self.size), 'reference': resize(reference, self.size),
                'strokes': strokes, 'text': row['text'], 'index': index}


def collate(samples):
    masks = [s['strokes'] for s in samples]
    if any(m is not None for m in masks) and not all(m is not None for m in masks):
        raise ValueError('Do not mix stroke-annotated and unannotated rows in one dataset')
    strokes = None
    if all(m is not None for m in masks):
        k = max(len(m) for m in masks)
        strokes = torch.stack([F.pad(m, (0, 0, 0, 0, 0, k-len(m))) for m in masks])
    return {'image': torch.stack([s['image'] for s in samples]),
            'reference': torch.stack([s['reference'] for s in samples]),
            'strokes': strokes, 'text': [s['text'] for s in samples]}


def fingerprint(manifest):
    """Hash the manifest and actual training/validation files for safe resume."""
    h = hashlib.sha256(Path(manifest).read_bytes())
    paths = set()
    for row in read_manifest(manifest):
        if row['split'] == 'test':
            continue
        paths.update([row['path']] if 'path' in row else [row['image'], row['reference']])
    for path in sorted(paths):
        with open(path, 'rb') as source:
            while chunk := source.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest()


class BatchStream:
    """Deterministic epoch shuffles with checkpointable position, no worker RNG."""
    def __init__(self, count, batch_size, seed, epoch=0, position=0):
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.epoch, self.position = epoch, position
        if count <= 0 or batch_size <= 0 or not 0 <= position <= count:
            raise ValueError('Invalid stream state')

    def next_indices(self):
        if self.position == self.count:
            self.epoch += 1
            self.position = 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.count, generator=generator).tolist()
        stop = min(self.position + self.batch_size, self.count)
        indices = order[self.position:stop]
        self.position = stop
        return indices

    def state_dict(self):
        return {'epoch': self.epoch, 'position': self.position}
