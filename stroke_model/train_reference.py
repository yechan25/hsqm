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
    def __init__(self, rows, size=128, include_topology=False):
        self.rows, self.size = rows, size
        self.include_topology = include_topology
        self.validated = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.rows[index]['path'], allow_pickle=False) as sample:
            image, reference, strokes = [torch.from_numpy(sample[key].copy()).float()
                                          for key in ('image', 'reference', 'strokes')]
            raw_labels = sample['labels'].copy()
            if self.include_topology:
                if 'target_masks' not in sample:
                    raise ValueError('Junction supervision requires actual input target_masks')
                target = torch.from_numpy(sample['target_masks'].copy()).float()
                valid = (torch.from_numpy(sample['valid_pixels'].copy()).float()
                         if 'valid_pixels' in sample else torch.ones_like(image))
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError('labels must be an integer array; no soft/overlapping labels')
        labels = torch.from_numpy(raw_labels).long()
        result = (image, reference, strokes, labels)
        if self.include_topology:
            result = (*result, target, valid)
        if index in self.validated:
            return result
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
        return result


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
        image, ref, strokes, labels = [x.to(device, non_blocking=True) for x in batch[:4]]
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
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--junction-weight', type=float, default=0., help='0: original V2; positive: same V2 plus junction loss')
    p.add_argument('--junction-every', type=int, default=4)
    p.add_argument('--junction-max-strokes', type=int, default=4)
    p.add_argument('--ownership-weight', type=float, default=0., help='Optional cached weak-region CE; 0 disables')
    p.add_argument('--affinity-weight', type=float, default=0., help='Optional same-owner neighbour consistency')
    p.add_argument('--epoch-warning-seconds', type=float, default=None)
    p.add_argument('--init-checkpoint', default=None, help='Compatible V2 weights; starts a new optimizer/run')
    args = p.parse_args()
    from dataclasses import asdict
    from .reference_v2_topology import (ARCHITECTURE, JunctionConfig, TopologyDataset,
                                      collate_topology, v2_topology_loss)
    junction_config = JunctionConfig(weight=args.junction_weight, every=args.junction_every,
                                     max_strokes=args.junction_max_strokes)
    topology_enabled = args.junction_weight > 0
    refinement_enabled = args.ownership_weight > 0 or args.affinity_weight > 0
    if any(not np.isfinite(v) or v < 0 for v in (args.ownership_weight,args.affinity_weight)):
        raise ValueError('Refinement weights must be finite and nonnegative')
    if args.epoch_warning_seconds is not None and args.epoch_warning_seconds <= 0:
        raise ValueError('epoch warning budget must be positive')
    if args.lr <= 0 or args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('Invalid training settings')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.local_cache:
        from .local_reference_data import stage_manifest
        args.manifest = str(stage_manifest(args.manifest, args.local_cache))
    if refinement_enabled:
        from .ownership_refinement import (OwnershipConfig, prepare_ownership_manifest,
                                          OwnershipDataset, collate_ownership, ownership_losses)
        ownership_config=OwnershipConfig()
        args.manifest=str(prepare_ownership_manifest(args.manifest,ownership_config))
    amp = device.type == 'cuda' and not args.no_amp
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f'Device: {torch.cuda.get_device_name() if device.type == "cuda" else "CPU"}; batch={args.batch_size}; AMP={amp}; dtype={amp_dtype}', flush=True)
    rows = read_manifest(args.manifest)
    loaders = {}
    for split in ('train', 'val'):
        auxiliary = topology_enabled and split == 'train'
        refinement = refinement_enabled and split == 'train'
        dataset_type = OwnershipDataset if refinement else TopologyDataset if auxiliary else StrokeDataset
        dataset = dataset_type([r for r in rows if r['split'] == split], args.image_size)
        if not len(dataset):
            raise ValueError('Missing split: ' + split)
        loaders[split] = DataLoader(dataset, batch_size=args.batch_size,
                                   shuffle=split == 'train', collate_fn=collate_ownership if refinement else collate_topology if auxiliary else collate,
                                   num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                                   persistent_workers=args.num_workers > 0)
        print(f'{split}: {len(dataset)} samples, {len(loaders[split])} batches', flush=True)
    print('Loading Swin pretrained weights...', flush=True)
    model = ReferenceStrokeModel(image_size=args.image_size, pretrained=not bool(args.init_checkpoint)).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') not in ('reference_v2', ARCHITECTURE) or checkpoint.get('image_size') != args.image_size:
            raise ValueError('초기 가중치는 image_size가 같은 reference_v2 체크포인트여야 합니다')
        model.load_state_dict(checkpoint['model'], strict=True)
        print('Initialized V2 weights:', args.init_checkpoint, flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp and amp_dtype == torch.float16)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if refinement_enabled:
        import shutil
        shutil.copyfile(Path(args.manifest).parent/'preparation_report.json',
                        destination/'ownership_preparation_report.json')
    architecture = ARCHITECTURE if topology_enabled or refinement_enabled else 'reference_v2'
    (destination / 'run_config.json').write_text(json.dumps({**vars(args), 'architecture': architecture,
        'junction_config': asdict(junction_config),
        'ownership_config': asdict(ownership_config) if refinement_enabled else None}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Architecture:', architecture, '/ junction:', asdict(junction_config), flush=True)
    best, history = -1., []
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.
        started = time.perf_counter()
        previous = started
        data_seconds = 0.
        junction_seconds = 0.
        junction_total = 0.
        junction_applications = 0
        ownership_total=0.;affinity_total=0.;batch_count=0
        progress = tqdm(loaders['train'], desc=f'Epoch {epoch + 1}/{args.epochs}', mininterval=5)
        for batch in progress:
            data_seconds += time.perf_counter() - previous
            image, ref, strokes, labels = [x.to(device, non_blocking=True) for x in batch[:4]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(image, ref, strokes)
                if topology_enabled:
                    applying = global_step % junction_config.every == 0
                    target, valid = ([x.to(device,non_blocking=True) for x in batch[4:6]]
                                     if applying else (None,None))
                    auxiliary_started = time.perf_counter()
                    losses = v2_topology_loss(output, labels, target, valid, junction_config, global_step)
                    junction_seconds += time.perf_counter()-auxiliary_started
                    if applying:
                        junction_total += float(losses['junction'].detach())
                        junction_applications += 1
                    loss = losses['loss']
                else:
                    loss = partition_loss(output, labels)['loss']
                if refinement_enabled:
                    owners,reliability=[x.to(device,non_blocking=True) for x in batch[6:8]]
                    refined=ownership_losses(output,owners,reliability,
                        compute_ownership=args.ownership_weight>0,compute_affinity=args.affinity_weight>0)
                    loss=loss+args.ownership_weight*refined['ownership']+args.affinity_weight*refined['affinity']
                    ownership_total=ownership_total+refined['ownership'].detach()
                    affinity_total=affinity_total+refined['affinity'].detach()
                batch_count+=1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            value = loss.item()
            total += value * len(image)
            progress.set_postfix(loss=f'{value:.4f}',lr=args.lr, refresh=False)
            previous = time.perf_counter()
        train_seconds = time.perf_counter() - started
        print('Validation...', flush=True)
        score = evaluate(model, loaders['val'], device, amp, amp_dtype)
        row = {'epoch': epoch + 1, 'train_loss': total / len(loaders['train'].dataset), 'val_macro_dice': score,
               'train_seconds': train_seconds, 'data_wait_seconds': data_seconds,
               'samples_per_second': len(loaders['train'].dataset) / train_seconds,
               'learning_rate': args.lr, 'junction_mean_when_applied': junction_total/max(junction_applications,1),
               'junction_applications': junction_applications, 'junction_cpu_dispatch_seconds': junction_seconds,
               'ownership_loss':float(ownership_total)/max(batch_count,1),
               'affinity_loss':float(affinity_total)/max(batch_count,1),
               'epoch_seconds': time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if args.epoch_warning_seconds and row['epoch_seconds']>args.epoch_warning_seconds:
            print(f'Epoch time {row["epoch_seconds"]:.1f}s exceeds comparison budget '
                  f'{args.epoch_warning_seconds:.1f}s; compare with refinement weights=0 on the same runtime.',flush=True)
        if score > best:
            best = score
            torch.save({'model': model.state_dict(), 'architecture': architecture,
                        'image_size': args.image_size, 'width': 64, 'epoch': epoch + 1,
                        'val_macro_dice': score, 'seed': args.seed,
                        'junction_config': asdict(junction_config),
                        'ownership_config': asdict(ownership_config) if refinement_enabled else None,
                        'ownership_weight':args.ownership_weight,'affinity_weight':args.affinity_weight}, destination / 'best.pt')
        (destination / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
