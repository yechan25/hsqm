"""Read-only train/val diagnosis on original prepared samples; no fitting."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from .convert_reference_csv import existing_path
from .reference_model import ReferenceStrokeModel
from .train_reference import read_manifest, StrokeDataset


def sample_metrics(image, labels, output):
    p = output['probabilities'][0].detach().float().cpu()
    prediction = p.argmax(0)
    foreground = image[0] > .5
    known = labels >= 0
    unknown = foreground & ~known
    wrong = known & (prediction != labels)
    dice = []
    for k in labels[known].unique():
        target, pred = labels == k, (prediction == k) & known
        dice.append(float(2 * (target & pred).sum() / (target.sum() + pred.sum())))
    return {'annotated_macro_dice': float(np.mean(dice)), 'stroke_dice': dice,
            'foreground_pixels': int(foreground.sum()), 'known_pixels': int((foreground & known).sum()),
            'unknown_pixels': int(unknown.sum()),
            'annotation_coverage': float((foreground & known).sum() / foreground.sum().clamp_min(1)),
            'known_error_pixels': int(wrong.sum()),
            'confident_known_error_pixels': int((wrong & (p.max(0).values >= .9)).sum())}


def select_examples(records, count=3):
    """Deliberately diagnostic selection, not a random estimate of prevalence."""
    if not records or count <= 0:
        return []
    selected = []
    orders = [sorted(records, key=lambda r: r['annotated_macro_dice']),
              sorted(records, key=lambda r: r['annotation_coverage']),
              sorted(records, key=lambda r: r['annotated_macro_dice'], reverse=True)]
    for ordered in orders:
        for record in ordered:
            if record['index'] not in [r['index'] for r in selected]:
                selected.append(record)
                break
        if len(selected) >= count:
            return selected
    return selected


def plot_diagnosis(image, reference, strokes, labels, output, title, prefix):
    import matplotlib.pyplot as plt
    p = output['probabilities'][0].detach().float().cpu().numpy()
    masks = output['masks'][0].detach().float().cpu().numpy()
    hints = output['hints'][0].detach().float().cpu().numpy()
    ids = np.flatnonzero(output['active'][0].cpu().numpy())
    foreground, known = image[0].numpy() > .5, labels.numpy() >= 0
    lab, pred = labels.numpy(), p.argmax(0)
    palette = plt.get_cmap('tab20')(np.arange(len(strokes)) % 20)[:, :3]
    target_rgb, pred_rgb, errors = [np.ones((*lab.shape, 3)) for _ in range(3)]
    target_rgb[known] = palette[lab[known]]
    target_rgb[foreground & ~known] = .65
    pred_rgb[foreground] = palette[pred[foreground]]
    errors[known] = [.8, .95, .8]
    errors[known & (lab != pred)] = [1, .1, .1]
    errors[foreground & ~known] = .65
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, data, name in zip(axes, [image[0].numpy(), reference[0].numpy(), target_rgb, pred_rgb, errors],
                              ['Input', 'Reference', 'Labels', 'Exclusive', 'Known errors=red; unknown=gray']):
        ax.imshow(data, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(prefix) + '_overview.png', dpi=140)
    plt.show()
    plt.close(fig)
    for offset in range(0, len(ids), 5):
        selected = ids[offset:offset+5]
        fig, axes = plt.subplots(5, len(selected), figsize=(2.6*len(selected), 10), squeeze=False)
        for column, k in enumerate(selected):
            truth = np.ones((*lab.shape, 3))
            truth[foreground & ~known] = .65
            truth[lab == k] = [0, 0, 0]
            panels = [strokes[k].numpy(), hints[k], truth, masks[k], ((pred == k) & foreground).astype(float)]
            names = ['Reference', 'Warped hint', 'Target; gray=unknown', 'Soft mask', 'Exclusive']
            for row, (data, name) in enumerate(zip(panels, names)):
                axes[row, column].imshow(data, cmap='gray_r' if row in (0, 2, 4) else 'viridis', vmin=0, vmax=1)
                axes[row, column].set_title(f'{name} {k+1}')
                axes[row, column].axis('off')
        fig.suptitle(title + ' | heatmaps: 0=purple, 1=yellow')
        fig.tight_layout()
        fig.savefig(str(prefix) + f'_strokes_{offset//5+1}.png', dpi=140)
        plt.show()
        plt.close(fig)


def run_diagnosis(checkpoint, manifest=None, examples_per_split=3):
    checkpoint = existing_path(checkpoint)
    manifest = existing_path(manifest or checkpoint.parent.parent / 'manifest.json')
    rows = [r for r in read_manifest(manifest) if r['split'] in ('train', 'val') and r.get('source') != 'augmented']
    if {r['split'] for r in rows} != {'train', 'val'}:
        raise ValueError('원본 train과 val이 모두 있는 manifest가 필요합니다')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False).to(device).eval()
    model.load_state_dict(state['model'])
    dataset = StrokeDataset(rows, state['image_size'])
    destination = checkpoint.parent / ('diagnosis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir()
    records, hashes = [], {'train': set(), 'val': set()}
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc='Diagnose original train/val', mininterval=2):
            image, ref, strokes, labels = dataset[i]
            out = model(image[None].to(device), ref[None].to(device), strokes[None].to(device))
            records.append({'index': i, 'split': rows[i]['split'], 'char_id': rows[i].get('char_id', ''),
                            'source_image': rows[i].get('source_image', ''), 'path': rows[i]['path'],
                            **sample_metrics(image, labels, out)})
            hashes[rows[i]['split']].add(hashlib.sha256(image.numpy().tobytes()).hexdigest())
    summary = {'checkpoint': str(checkpoint), 'manifest': str(manifest), 'checkpoint_epoch': state['epoch'],
               'identical_image_hashes_across_splits': len(hashes['train'] & hashes['val']),
               'note': 'Scores exclude unknown pixels; label identity and writer independence require manual review.'}
    for split in ('train', 'val'):
        subset = [r for r in records if r['split'] == split]
        all_dice = [d for r in subset for d in r['stroke_dice']]
        summary[split] = {'samples': len(subset), 'annotated_macro_dice': float(np.mean(all_dice)),
                          'annotation_coverage': sum(r['known_pixels'] for r in subset) / max(sum(r['foreground_pixels'] for r in subset), 1),
                          'known_error_pixels': sum(r['known_error_pixels'] for r in subset),
                          'confident_known_error_pixels': sum(r['confident_known_error_pixels'] for r in subset)}
    (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(records).to_csv(destination / 'samples.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('진단 예시는 낮은 Dice / 낮은 정답 커버리지 / 높은 Dice 순으로 선택합니다. 무작위 표본이 아닙니다.')
    for split in ('train', 'val'):
        for record in select_examples([r for r in records if r['split'] == split], examples_per_split):
            i = record['index']
            image, ref, strokes, labels = dataset[i]
            with torch.no_grad():
                out = model(image[None].to(device), ref[None].to(device), strokes[None].to(device))
            title = f"{split} #{i}: annotated Dice={record['annotated_macro_dice']:.3f}, coverage={record['annotation_coverage']:.1%}"
            plot_diagnosis(image, ref, strokes, labels, out, title, destination / f'{split}_{i:04d}')
    print('진단 결과 저장:', destination)
    return summary
