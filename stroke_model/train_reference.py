"""V2 trainer with explicit, group-disjoint train/val/test manifest.

python -m stroke_model.train_reference --manifest data_v2/manifest.json --epochs 30
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
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
        self.validated = set()

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
        if index in self.validated:
            return image, reference, strokes, labels
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
        self.validated.add(index)
        return image, reference, strokes, labels


def collate(samples):
    max_k = max(s[2].shape[0] for s in samples)
    return (torch.stack([s[0] for s in samples]), torch.stack([s[1] for s in samples]),
            torch.stack([torch.nn.functional.pad(s[2], (0, 0, 0, 0, 0, max_k - len(s[2]))) for s in samples]),
            torch.stack([s[3] for s in samples]))


@torch.no_grad()
def evaluate(model, loader, device, amp=False, amp_dtype=torch.float16):
    """Macro Dice over annotated nonempty strokes, not padded channels."""
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    model.eval()
    for batch in loader:
        image, ref, strokes, labels = [x.to(device, non_blocking=True) for x in batch]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            prediction = model(image, ref, strokes)['probabilities'].argmax(1)
        known = labels[:, None] >= 0
        target = torch.nn.functional.one_hot(labels.clamp_min(0), strokes.shape[1]).permute(0, 3, 1, 2).bool() & known
        pred = torch.nn.functional.one_hot(prediction, strokes.shape[1]).permute(0, 3, 1, 2).bool() & known
        mass = target.sum((-2, -1))
        dice = 2 * (target & pred).sum((-2, -1)).float() / (mass + pred.sum((-2, -1))).clamp_min(1)
        total += dice[mass > 0].sum()
        count += (mass > 0).sum()
    if count.item() == 0:
        raise ValueError('Validation set has no labelled strokes')
    return (total / count).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='outputs/reference_v2')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--local-cache', default=None)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.local_cache:
        from .local_reference_data import stage_manifest
        args.manifest = str(stage_manifest(args.manifest, args.local_cache))
    amp = device.type == 'cuda' and not args.no_amp
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f'Device: {torch.cuda.get_device_name() if device.type == "cuda" else "CPU"}; batch={args.batch_size}; AMP={amp}; dtype={amp_dtype}', flush=True)
    rows = read_manifest(args.manifest)
    loaders = {}
    for split in ('train', 'val'):
        dataset = StrokeDataset([r for r in rows if r['split'] == split], args.image_size)
        if not len(dataset):
            raise ValueError('Missing split: ' + split)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=split == 'train', collate_fn=collate,
                                   num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                                   persistent_workers=args.num_workers > 0)
        print(f'{split}: {len(dataset)} samples, {len(loaders[split])} batches', flush=True)
    print('Loading Swin pretrained weights...', flush=True)
    model = ReferenceStrokeModel(image_size=args.image_size).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and amp_dtype == torch.float16)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    best, history = -1., []
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        started = time.perf_counter()
        previous = started
        data_seconds = 0.
        progress = tqdm(loaders['train'], desc=f'Epoch {epoch + 1}/{args.epochs}', mininterval=5)
        for batch in progress:
            data_seconds += time.perf_counter() - previous
            image, ref, strokes, labels = [x.to(device, non_blocking=True) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                loss = partition_loss(model(image, ref, strokes), labels)['loss']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            value = loss.item()
            total += value * len(image)
            progress.set_postfix(loss=f'{value:.4f}', refresh=False)
            previous = time.perf_counter()
        train_seconds = time.perf_counter() - started
        print('Validation...', flush=True)
        score = evaluate(model, loaders['val'], device, amp, amp_dtype)
        row = {'epoch': epoch + 1, 'train_loss': total / len(loaders['train'].dataset), 'val_macro_dice': score,
               'train_seconds': train_seconds, 'data_wait_seconds': data_seconds,
               'samples_per_second': len(loaders['train'].dataset) / train_seconds,
               'epoch_seconds': time.perf_counter() - started}
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
