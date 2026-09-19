"""Reuse completed Drive datasets. Source images edited in place need rebuild=True."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from .convert_reference_csv import convert_csv, existing_path, normalized


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_manifest(path):
    try:
        rows = json.loads(path.read_text(encoding='utf-8'))
        return bool(rows) and all((path.parent / row['path']).is_file() for row in rows)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_cached(csv_path, cache_root, size=128, seed=42, rebuild=False):
    csv_path = existing_path(csv_path)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    signature = {'csv_sha256': digest(csv_path), 'csv': normalized(csv_path.resolve()),
                 'size': size, 'seed': seed, 'version': 1}
    for folder in sorted(cache_root.iterdir(), reverse=True) if not rebuild else []:
        manifest, metadata = folder / 'manifest.json', folder / 'cache.json'
        report_path = folder / 'conversion_report.json'
        if not folder.is_dir() or not manifest.is_file() or not report_path.is_file():
            continue
        try:
            if metadata.is_file():
                matches = json.loads(metadata.read_text(encoding='utf-8')) == signature
            else:
                # Adopt the user's already-running pre-cache conversion (fixed 128/42).
                report = json.loads(report_path.read_text(encoding='utf-8'))
                matches = (size == 128 and seed == 42
                           and normalized(report['csv']) == normalized(csv_path)
                           and len(report['rows']) == len(pd.read_csv(csv_path))
                           and csv_path.stat().st_mtime <= report_path.stat().st_mtime)
            if matches and complete_manifest(manifest):
                if not metadata.is_file():
                    print('이전에 완료된 변환을 등록합니다. CSV와 이미지가 변환 후 수정되지 않았다는 전제입니다.')
                    metadata.write_text(json.dumps(signature), encoding='utf-8')
                print('저장된 변환 데이터 재사용:', manifest)
                print('원본 이미지를 같은 경로에서 수정했다면 rebuild=True로 다시 변환하세요.')
                return manifest
        except (OSError, ValueError, KeyError, TypeError):
            continue
    folder = cache_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    manifest = convert_csv(csv_path, folder, size=size, seed=seed)
    (folder / 'cache.json').write_text(json.dumps(signature), encoding='utf-8')
    return manifest


def ensure_augmented(manifest, variants=20, seed=42, size=128):
    manifest = Path(manifest).resolve()
    if variants <= 0:
        return manifest
    signature = {'manifest_sha256': digest(manifest), 'variants': variants,
                 'seed': seed, 'size': size, 'version': 1}
    for metadata in sorted(manifest.parent.glob('augmented_*/cache.json'), reverse=True):
        try:
            target = metadata.parent / 'manifest.json'
            if json.loads(metadata.read_text(encoding='utf-8')) == signature and complete_manifest(target):
                print('저장된 증강 데이터 재사용:', target)
                return target
        except (OSError, ValueError):
            continue
    dest = manifest.parent / ('augmented_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    subprocess.run([sys.executable, '-m', 'stroke_model.augment_reference', '--manifest', str(manifest),
                    '--variants', str(variants), '--seed', str(seed), '--image-size', str(size),
                    '--output', str(dest)], cwd=Path(__file__).resolve().parents[1], check=True)
    target = dest / 'manifest.json'
    if not complete_manifest(target):
        raise RuntimeError('증강 결과가 완성되지 않았습니다')
    (dest / 'cache.json').write_text(json.dumps(signature), encoding='utf-8')
    return target
