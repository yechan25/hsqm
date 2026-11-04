"""Inference-only Topo checkpoint inspection; never fills or thickens predictions."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .convert_reference_csv import Resolver, existing_path
from .infer_reference import inference_columns, load_inference_row, select_input_files
from .reference_topology import TopologyStrokeModel, AlignmentConfig
from .reference_model import ReferenceStrokeModel
from .reference_v2_topology import ARCHITECTURE


def load_topology(checkpoint, device):
    state = torch.load(existing_path(checkpoint), map_location='cpu', weights_only=True)
    if state.get('architecture') == ARCHITECTURE:
        model = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False)
        model.load_state_dict(state['model'], strict=True)
        model.output_mode = 'partition'
        return model.to(device).eval(), state['image_size'], state.get('epoch')
    if state.get('architecture') != TopologyStrokeModel.architecture:
        raise ValueError('Topo best.pt required (reference_topology_aligned_v1); V2/BiFPN cannot be used')
    model = TopologyStrokeModel(image_size=state['image_size'], width=state['width'],
        context_width=state['context_width'], pretrained=False,
        alignment_config=AlignmentConfig(**state['alignment_config']))
    model.load_trained(state)
    return model.to(device).eval(), state['image_size'], state.get('epoch')


def export_hard_masks(image, masks, threshold=.5, output_mode='independent'):
    if output_mode == 'partition':
        return (np.arange(len(masks))[:,None,None] == masks.argmax(0)) & (image[0] > .5)
    return masks >= threshold


def ink_diagnostics(image, masks, threshold=.5, output_mode='independent'):
    foreground = image[0] > .5
    hard = export_hard_masks(image, masks, threshold, output_mode)
    union = hard.any(0)
    count = max(int(foreground.sum()), 1)
    return {'input_ink_pixels': int(foreground.sum()),
            'covered_ink_fraction': float((union & foreground).sum()/count),
            'uncovered_ink_fraction': float((~union & foreground).sum()/count),
            'overlapping_ink_fraction': float(((hard.sum(0)>1) & foreground).sum()/count)}


def preview_topology(image, reference, strokes, probabilities, masks, prefix, title, threshold, show,
                     output_mode='independent'):
    import matplotlib.pyplot as plt
    foreground = image[0] > .5
    hard = export_hard_masks(image, masks, threshold, output_mode)
    union = hard.any(0)
    soft_union = masks.sum(0) if output_mode == 'partition' else image[0]*(1-np.prod(1-probabilities, axis=0))
    missing = np.zeros((*foreground.shape, 3))
    missing[foreground] = .35
    missing[foreground & ~union] = (1, 0, 0)
    palette = plt.get_cmap('tab20')(np.arange(len(strokes)) % 20)[:, :3]
    colored = np.zeros((*foreground.shape, 3))
    colored[union] = palette[masks.argmax(0)[union]]
    colored[hard.sum(0)>1] = (1, 0, 1)
    figure, axes = plt.subplots(1, 6, figsize=(16, 3))
    panels = ((image[0], 'Input'), (reference[0], 'Reference'), (soft_union, 'Soft union'),
              (union, 'Exclusive union' if output_mode == 'partition' else f'Hard union >= {threshold}'), (colored, 'Strokes (overlap=magenta)'),
              (missing, 'Uncovered ink = red'))
    for ax, (content, label) in zip(axes, panels):
        ax.imshow(content, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(label); ax.axis('off')
    figure.suptitle(title); figure.tight_layout()
    figure.savefig(str(prefix)+'_overview.png', dpi=140)
    if show:
        plt.show()
    plt.close(figure)
    figure, axes = plt.subplots(3, len(strokes), figsize=(2*len(strokes), 6), squeeze=False)
    for k in range(len(strokes)):
        for row, content, label in ((0, strokes[k], 'Ref stroke'), (1, masks[k], 'Soft'),
                                     (2, hard[k], 'Exclusive' if output_mode == 'partition' else 'Hard')):
            axes[row,k].imshow(content, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[row,k].set_title(f'{label} {k+1}'); axes[row,k].axis('off')
    figure.tight_layout(); figure.savefig(str(prefix)+'_strokes.png', dpi=140)
    if show:
        plt.show()
    plt.close(figure)


def run_topology_inference(checkpoint, csv_path, dataset_root=None, input_files=None,
                           sample_rows=None, max_samples=8, seed=42, threshold=.5,
                           save_dir=None, show=True):
    if not 0 < threshold < 1:
        raise ValueError('threshold must be between 0 and 1')
    if input_files is not None and sample_rows is not None:
        raise ValueError('Specify input_files or sample_rows, not both')
    checkpoint, csv_path = existing_path(checkpoint), existing_path(csv_path)
    frame = pd.read_csv(csv_path)
    columns = inference_columns(frame)
    if input_files is not None:
        selected = select_input_files(frame, columns['image'], input_files)
    elif sample_rows is not None:
        selected = list(sample_rows)
    else:
        # Spread the default preview across reference glyphs, rather than their variants.
        shuffled = frame.sample(frac=1, random_state=seed)
        selected = [int(i)+1 for i in shuffled.drop_duplicates(columns['reference']).index]
        if max_samples is None:
            selected = list(range(1, len(frame)+1))
        else:
            if max_samples < 1:
                raise ValueError('max_samples must be positive or None')
            selected = selected[:max_samples]
    if not selected or any(not isinstance(n, int) or not 1 <= n <= len(frame) for n in selected):
        raise ValueError(f'Select CSV data rows in 1..{len(frame)}')
    resolve = Resolver(csv_path, existing_path(dataset_root) if dataset_root else csv_path.parent)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size, epoch = load_topology(checkpoint, device)
    output_mode = getattr(model, 'output_mode', 'independent')
    destination = Path(save_dir) if save_dir else checkpoint.parent/('topo_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True, exist_ok=False)
    report = {'checkpoint': str(checkpoint), 'checkpoint_epoch': epoch, 'csv': str(csv_path),
              'uses_input_stroke_annotations': False, 'threshold': threshold,
              'postprocessing': 'v2_argmax_export' if output_mode == 'partition' else 'none',
              'output_mode': output_mode, 'results': [], 'skipped': []}
    print(f'Topo checkpoint epoch={epoch}; {device}; samples={len(selected)}', flush=True)
    for number in tqdm(selected, desc='Topo inference'):
        try:
            image, reference, strokes, image_path, reference_path = load_inference_row(
                frame.iloc[number-1], columns, resolve, size)
        except ValueError as exc:
            if str(exc) != 'Empty reference stroke image':
                raise
            report['skipped'].append({'csv_row': number, 'reason': str(exc)})
            print(f'Skipped CSV row {number}: {exc}', flush=True)
            continue
        with torch.inference_mode():
            output = model(image[None].to(device), reference[None].to(device), strokes[None].to(device))
        probabilities, masks = [output[key][0].float().cpu().numpy() for key in ('probabilities', 'masks')]
        arrays = {'probabilities': probabilities, 'soft_masks': masks, 'hard_masks': masks >= threshold,
                  'image': image.numpy(), 'reference': reference.numpy(), 'reference_strokes': strokes.numpy()}
        arrays['hard_masks'] = export_hard_masks(image.numpy(), masks, threshold, output_mode)
        for key in ('flow', 'hints', 'alignment_theta', 'aligned_reference', 'aligned_strokes'):
            if key in output:
                arrays[key] = output[key][0].float().cpu().numpy()
        prefix = destination/f'row_{number:05d}'
        np.savez_compressed(str(prefix)+'.npz', **arrays)
        diagnostics = ink_diagnostics(image.numpy(), masks, threshold, output_mode)
        title = f'CSV {number}: {image_path.name} | ink coverage={diagnostics["covered_ink_fraction"]:.1%}'
        preview_topology(image.numpy(), reference.numpy(), strokes.numpy(), probabilities, masks,
                         prefix, title, threshold, show, output_mode)
        report['results'].append({'csv_row': number, 'input': str(image_path),
            'reference': str(reference_path), 'npz': str(prefix)+'.npz', **diagnostics})
        print(title, flush=True)
    (destination/'inference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(report['results']).to_csv(destination/'ink_coverage.csv', index=False, encoding='utf-8-sig')
    print('Saved:', destination, flush=True)
    return report
