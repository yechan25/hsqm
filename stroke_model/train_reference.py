"""V2 trainer with explicit, group-disjoint train/val/test manifest.

python -m stroke_model.train_reference --manifest data_v2/manifest.json --epochs 30
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .reference_model import ReferenceStrokeModel, partition_loss


def read_manifest(path):
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    groups = {}
    files = set()
    for row in rows:
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        previous = groups.setdefault(row['group_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Group leakage: ' + row['group_id'])
        full = (path.parent / row['path']).resolve()
        if full in files:
            raise ValueError('Duplicate sample: ' + str(full))
        files.add(full)
        row['path'] = str(full)
    return rows


class StrokeDataset(Dataset):
    def __init__(self, rows, size=128):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.rows[index]['path'], allow_pickle=False) as sample:
            image, reference, strokes = [torch.from_numpy(sample[key].copy()).float()
                                          for key in ('image', 'reference', 'strokes')]
            raw_labels = sample['labels'].copy()
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError('labels must be an integer array; no soft/overlapping labels')
        labels = torch.from_numpy(raw_labels).long()
        if image.shape != (1, self.size, self.size) or reference.shape != image.shape:
            raise ValueError('image/reference must be 1xSxS; normalize before export')
        if strokes.ndim != 3 or strokes.shape[-2:] != image.shape[-2:] or strokes.shape[0] == 0:
            raise ValueError('strokes must be KxSxS')
        for tensor in (image, reference, strokes):
            if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                raise ValueError('Images/masks must be finite in [0,1], ink=1')
        if labels.shape != image.shape[-2:] or labels.min() < -1 or labels.max() >= len(strokes):
            raise ValueError('Invalid labels')
        if not (labels >= 0).any():
            raise ValueError('Sample has no supervised foreground pixels')
        active = strokes.flatten(1).sum(1) > 0
        if not active.any() or not active[labels[labels >= 0]].all():
            raise ValueError('Annotations reference an absent stroke')
        return image, reference, strokes, labels


def collate(samples):
    max_k = max(s[2].shape[0] for s in samples)
    return (torch.stack([s[0] for s in samples]), torch.stack([s[1] for s in samples]),
            torch.stack([torch.nn.functional.pad(s[2], (0, 0, 0, 0, 0, max_k - len(s[2]))) for s in samples]),
            torch.stack([s[3] for s in samples]))


@torch.no_grad()
def evaluate(model, loader, device):
    """Macro Dice over annotated nonempty strokes, not padded channels."""
    scores = []
    model.eval()
    for batch in loader:
        image, ref, strokes, labels = [x.to(device) for x in batch]
        prediction = model(image, ref, strokes)['probabilities'].argmax(1)
        for b in range(len(image)):
            known = labels[b] >= 0
            for k in labels[b][known].unique():
                target, pred = labels[b] == k, (prediction[b] == k) & known
                scores.append((2 * (target & pred).sum().float() / (target.sum() + pred.sum())).item())
    if not scores:
        raise ValueError('Validation set has no labelled strokes')
    return sum(scores) / len(scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='outputs/reference_v2')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = read_manifest(args.manifest)
    loaders = {}
    for split in ('train', 'val'):
        dataset = StrokeDataset([r for r in rows if r['split'] == split], args.image_size)
        if not len(dataset):
            raise ValueError('Missing split: ' + split)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=split == 'train', collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceStrokeModel(image_size=args.image_size).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=.01)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    best, history = -1., []
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        for batch in loaders['train']:
            image, ref, strokes, labels = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = partition_loss(model(image, ref, strokes), labels)['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(image)
        score = evaluate(model, loaders['val'], device)
        row = {'epoch': epoch + 1, 'train_loss': total / len(loaders['train'].dataset), 'val_macro_dice': score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'architecture': 'reference_v2',
                        'image_size': args.image_size, 'width': 64, 'epoch': epoch + 1,
                        'val_macro_dice': score, 'seed': args.seed}, destination / 'best.pt')
        (destination / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
