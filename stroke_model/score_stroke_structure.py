"""CSV/NPZ evaluation and plots for the experimental structure losses."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stroke_model.stroke_structure import StructureConfig, StrokeStructureLoss


def _destination(save_dir):
    destination = Path(save_dir) if save_dir else Path('structure_results') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _measure(masks, geometry, config, prefix, reference=None, image=None,
             show=True, caption=''):
    evaluator = StrokeStructureLoss(geometry, config).to(masks.device)
    with torch.no_grad():
        result = evaluator(masks, return_maps=True)
    rows = []
    for k in range(len(geometry)):
        rows.append({'stroke': k + 1,
                     **{key: float(value[0, k]) for key, value in result['per_stroke'].items()},
                     'closed_reference': result['convergence'][k]['closed'],
                     'connectivity_converged': result['convergence'][k]['pairs'],
                     'attachment_converged': result['convergence'][k]['attachment']})
    frame = pd.DataFrame(rows)
    frame.to_csv(str(prefix) + '_errors.csv', index=False, encoding='utf-8-sig')
    payload = {'soft_masks': masks.detach().float().cpu().numpy(),
               'reference_geometry': torch.as_tensor(geometry).detach().cpu().numpy()}
    if reference is not None:
        payload['reference_strokes'] = torch.as_tensor(reference).detach().cpu().numpy()
    if image is not None:
        payload['image'] = torch.as_tensor(image).detach().cpu().numpy()
    for name in ('centerline', 'missing', 'split', 'junction', 'attached'):
        payload[name + '_maps'] = np.stack([m[name][0, 0].cpu().numpy() for m in result['maps']])
    np.savez_compressed(str(prefix) + '_structure.npz', **payload)
    figures = _plot(result['maps'], rows, prefix, caption, show)
    print(caption)
    print(frame.round(5).to_string(index=False))
    if not frame['connectivity_converged'].all() or not frame['attachment_converged'].all():
        print('Iteration cap reached: increase connectivity_steps/attachment_steps before interpreting these errors.')
    return {'strokes': rows, 'mean_errors': {key: float(value.mean()) for key, value in result['per_stroke'].items()},
            'worst_extraction_error': float(result['per_stroke']['extraction_error'].max()),
            'source_quality': result['source_quality'],
            'csv': str(prefix) + '_errors.csv', 'npz': str(prefix) + '_structure.npz', 'figures': figures}


def _plot(maps, rows, prefix, caption, show):
    import matplotlib.pyplot as plt
    paths = []
    for offset in range(0, len(maps), 4):
        page = maps[offset:offset + 4]
        fig, axes = plt.subplots(5, len(page), squeeze=False, figsize=(3.2*len(page), 12))
        for col, item in enumerate(page):
            k = offset + col
            for row, (name, title) in enumerate([
                    ('centerline', 'Reference centerline'), ('mask', 'Soft mask (absolute ink)'),
                    ('missing', 'Missing on reference path'),
                    ('split', 'Extractor connectivity diagnostic'), ('junction', 'Extractor junction defect')]):
                content = item[name][0, 0].cpu().numpy()
                ax = axes[row, col]
                if row >= 2:
                    ax.imshow(item['mask'][0, 0].cpu().numpy(), cmap='gray_r', vmin=0, vmax=1, alpha=.25)
                    artist = ax.imshow(np.ma.masked_less_equal(content, 1e-8), cmap='magma',
                                       vmin=0, vmax=1 if name != 'junction' else max(1., float(content.max())))
                    fig.colorbar(artist, ax=ax, fraction=.04, pad=.02)
                else:
                    ax.imshow(content, cmap='gray_r', vmin=0, vmax=1)
                ax.set_title(f'Stroke {k+1}: {title}', fontsize=9)
                ax.axis('off')
            values = rows[k]
            axes[4, col].text(.5, -.08,
                f"M={values['missing']:.4f}  D={values['split']:.4f}\n"
                f"J={values['junction']:.4f}  W={values['width']:.4f}",
                transform=axes[4, col].transAxes, ha='center', va='top', fontsize=9)
        fig.suptitle(caption + '\nExtractor defects + reference deficits. Source-image quality is NOT assessed.', fontsize=10)
        fig.tight_layout(rect=(0, .025, 1, .96))
        path = str(prefix) + f'_structure_{offset//4+1}.png'
        fig.savefig(path, dpi=140)
        paths.append(path)
        if show:
            plt.show()
        plt.close(fig)
    return paths


def score_saved_strokes(npz_paths, config=None, save_dir=None, show=True):
    """Use saved run_inference_csv NPZ files: soft_masks + reference_strokes.

    Also accepts the inference report dictionary returned by run_inference_csv.
    Uses unwarped reference geometry: input and reference must already be aligned.
    """
    config = config or StructureConfig()
    if isinstance(npz_paths, dict):
        npz_paths = [item['npz'] for item in npz_paths['results']]
    elif isinstance(npz_paths, (str, Path)):
        npz_paths = [npz_paths]
    npz_paths = list(npz_paths)
    if not npz_paths:
        raise ValueError('No NPZ files selected')
    destination = _destination(save_dir)
    results = []
    for i, path in enumerate(npz_paths):
        with np.load(path, allow_pickle=False) as data:
            masks = torch.tensor(data['soft_masks'], dtype=torch.float32)
            reference = torch.tensor(data['reference_strokes'], dtype=torch.float32)
            image = data['image'].copy() if 'image' in data else None
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim != 3:
            raise ValueError(f'Expected K,H,W masks in {path}')
        results.append({'input_npz': str(path), **_measure(masks, reference, config,
            destination / f'sample_{i+1:04d}', reference=reference, image=image,
            show=show, caption=Path(path).name)})
    report = {'mode': 'saved_npz', 'geometry': 'reference', 'config': asdict(config),
              'source_quality': {'assessed': False}, 'results': results}
    (destination / 'structure_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Structure results:', destination)
    return report


def score_structure_csv(checkpoint, csv_path, input_files=None, sample_rows=None,
                        dataset_root=None, config=None, geometry='reference',
                        save_dir=None, show=True):
    """Predict selected PNGs then measure errors, ignoring target annotations.

    geometry='reference': fixed reference coordinates, requires alignment.
    geometry='hints': detached model-warped reference; can hide alignment errors.
    Evaluation only. For guidance call StrokeStructureLoss on non-detached masks.
    """
    from stroke_model.convert_reference_csv import Resolver, existing_path
    from stroke_model.infer_reference import inference_columns, load_inference_row, select_input_files
    from stroke_model.reference_model import ReferenceStrokeModel

    if geometry not in ('reference', 'hints'):
        raise ValueError('geometry must be reference or hints')
    if input_files is not None and sample_rows is not None:
        raise ValueError('Specify input_files or sample_rows, not both')
    config = config or StructureConfig()
    checkpoint, csv_path = existing_path(checkpoint), existing_path(csv_path)
    frame = pd.read_csv(csv_path)
    columns = inference_columns(frame)
    selected = (select_input_files(frame, columns['image'], input_files) if input_files is not None else
                list(range(1, len(frame)+1)) if sample_rows is None else list(sample_rows))
    if not selected or any(not isinstance(i, int) or not 1 <= i <= len(frame) for i in selected):
        raise ValueError(f'Select CSV rows in 1..{len(frame)}')
    resolve = Resolver(csv_path, existing_path(dataset_root) if dataset_root else csv_path.parent)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False)
    model.load_state_dict(state['model'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    size = state['image_size']
    del state
    destination = _destination(save_dir or checkpoint.parent / ('structure_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    print('Checkpoint:', checkpoint)
    print('CSV:', csv_path)
    print('Reference geometry:', geometry, '/ annotations are not used')
    results = []
    for number in selected:
        image, reference, strokes, image_path, _ = load_inference_row(frame.iloc[number-1], columns, resolve, size)
        with torch.no_grad():
            output = model(image[None].to(device), reference[None].to(device), strokes[None].to(device))
        # Reference preparation uses CPU only; scalar scoring runs on model device.
        prepared_geometry = strokes if geometry == 'reference' else output['hints'][0].detach().cpu()
        result = _measure(output['masks'][0], prepared_geometry, config,
            destination / f'row_{number:04d}', reference=strokes, image=image,
            show=show, caption=f'CSV row {number}: {image_path.name}')
        results.append({'csv_row': number, 'input': str(image_path), **result})
    report = {'mode': 'csv', 'checkpoint': str(checkpoint), 'csv': str(csv_path),
              'geometry': geometry, 'config': asdict(config), 'uses_input_stroke_annotations': False,
              'source_quality': {'assessed': False}, 'results': results}
    (destination / 'structure_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Structure results:', destination)
    return report
