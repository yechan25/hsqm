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
    rows = read_manifest(manifest)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceStrokeModel(image_size=size, width=state['width'], pretrained=False).to(device)
    model.load_state_dict(state['model'])
    loader = DataLoader(StrokeDataset(rows, size), batch_size=8, collate_fn=collate)
    print(f'Test 평가 시작: {len(rows)}개, checkpoint epoch={state["epoch"]}, device={device}', flush=True)
    score = evaluate(model, loader, device)
    conversion = json.loads((manifest.parent / 'conversion_report.json').read_text(encoding='utf-8'))
    annotation = conversion['rows']
    foreground = sum(r['foreground_pixels'] for r in annotation)
    excluded = sum(r['overlap_pixels'] + r['missing_pixels'] for r in annotation)
    metrics = {'test_macro_dice_annotated': score, 'test_samples': len(rows),
               'annotation_coverage': 1 - excluded / max(foreground, 1),
               'excluded_foreground_pixels': excluded, 'checkpoint': str(checkpoint),
               'checkpoint_epoch': state['epoch'], 'csv': str(csv_path), 'manifest': str(manifest)}
    destination = checkpoint.parent / ('test_results_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir()
    (destination / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print('주의: 교차/미할당 정답 픽셀은 Dice에서 제외됩니다. 전체 획 품질은 그림도 확인하세요.')
    del model, state
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    preview_reference(manifest, checkpoint, count=preview_count, split='test', save_dir=destination)
    print('Test 결과 저장:', destination)
    return metrics
