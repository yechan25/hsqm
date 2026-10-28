"""Evaluate an existing V2 checkpoint on all rows of a held-out legacy CSV."""
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .convert_reference_csv import convert_csv, existing_path
from .reference_cache import complete_manifest
from .reference_model import ReferenceStrokeModel
from .train_reference import read_manifest, StrokeDataset, collate, evaluate
from .preview_reference import preview_reference


def run_test(checkpoint, csv_path, preview_count=5):
    checkpoint, csv_path = existing_path(checkpoint), existing_path(csv_path)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    size = state['image_size']
    key = hashlib.sha256(str(csv_path).encode() + csv_path.read_bytes() + str(size).encode()).hexdigest()[:16]
    cache = checkpoint.parent / 'test_data'
    cache.mkdir(exist_ok=True)
    manifest = None
    for folder in sorted(cache.glob(key + '*'), reverse=True):
        candidate = folder / 'manifest.json'
        if complete_manifest(candidate) and (folder / 'conversion_report.json').is_file():
            if all(r['split'] == 'test' for r in read_manifest(candidate)):
                manifest = candidate
                print('저장된 test 변환 재사용:', manifest)
                break
    if manifest is None:
        folder = cache / (key + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        manifest = convert_csv(csv_path, folder, size=size, split='test')
    del state
    return run_manifest(checkpoint, manifest, split='test', preview_count=preview_count, csv_path=csv_path)


def run_manifest(checkpoint, manifest, split='val', preview_count=5, csv_path=None):
    """Reuse the existing evaluation/soft-preview pipeline for an explicit split."""
    if split not in ('val', 'test'):
        raise ValueError('Evaluation split must be val or test')
    checkpoint, manifest = existing_path(checkpoint), existing_path(manifest)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    size = state['image_size']
    all_rows = read_manifest(manifest)
    indices = [i for i, row in enumerate(all_rows) if row['split'] == split]
    rows = [all_rows[i] for i in indices]
    if not rows:
        raise ValueError('Missing evaluation split: ' + split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceStrokeModel(image_size=size, width=state['width'], pretrained=False).to(device)
    model.load_state_dict(state['model'])
    loader = DataLoader(StrokeDataset(rows, size), batch_size=8, collate_fn=collate)
    print(f'{split} 평가 시작: {len(rows)}개, checkpoint epoch={state["epoch"]}, device={device}', flush=True)
    score = evaluate(model, loader, device)
    conversion = json.loads((manifest.parent / 'conversion_report.json').read_text(encoding='utf-8'))
    annotation = [conversion['rows'][i] for i in indices]
    foreground = sum(r['foreground_pixels'] for r in annotation)
    excluded = sum(r['overlap_pixels'] + r['missing_pixels'] for r in annotation)
    metrics = {f'{split}_macro_dice_annotated': score, f'{split}_samples': len(rows), 'split': split,
               'annotation_coverage': 1 - excluded / max(foreground, 1),
               'excluded_foreground_pixels': excluded, 'checkpoint': str(checkpoint),
               'checkpoint_epoch': state['epoch'], 'csv': str(csv_path) if csv_path else conversion.get('csv'), 'manifest': str(manifest)}
    destination = checkpoint.parent / (split + '_results_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir()
    (destination / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print('주의: 교차/미할당 정답 픽셀은 Dice에서 제외됩니다. 전체 획 품질은 그림도 확인하세요.')
    del model, state
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    preview_reference(manifest, checkpoint, count=preview_count, split=split, save_dir=destination)
    print('평가 결과 저장:', destination)
    return metrics


def run_zip_validation(checkpoint, dataset_root, preview_count=5):
    """Rebuild the training split from its saved CSV, archive identity and seed."""
    from .prepare_stroke_zips import ARCHIVES, prepare_stroke_zips
    checkpoint = existing_path(checkpoint)
    run = checkpoint.parent
    config = json.loads((run / 'run_config.json').read_text(encoding='utf-8'))
    saved = json.loads((run / 'complete.json').read_text(encoding='utf-8'))
    source = existing_path(dataset_root)
    old = {Path(path).name: (size, modified) for path, size, modified in saved['archives']}
    for name in ARCHIVES:
        path = existing_path(source / name)
        if (path.stat().st_size, path.stat().st_mtime_ns) != old.get(name):
            raise ValueError(f'학습 당시와 다른 ZIP입니다: {name}. 같은 데이터의 val 분할을 복원할 수 없습니다.')
    manifest = prepare_stroke_zips(source, csv_path=run / 'selected_paths.csv',
                                   size=config['image_size'], seed=config['seed'])
    return run_manifest(checkpoint, manifest, split='val', preview_count=preview_count,
                        csv_path=run / 'selected_paths.csv')
