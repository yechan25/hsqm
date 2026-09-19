"""Strict legacy CSV adapter. Ambiguous/missing labels remain unknown (-1)."""
import ast
import hashlib
import json
import random
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def normalized(value):
    return unicodedata.normalize('NFC', str(value).replace('\\', '/'))


def existing_path(value):
    """Resolve NFC/NFD differences without assuming the shared drive spelling."""
    path = Path(value)
    if path.exists():
        return path
    current = Path(path.anchor) if path.is_absolute() else Path('.')
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        direct = current / part
        if direct.exists():
            current = direct
            continue
        if not current.is_dir():
            raise FileNotFoundError(value)
        matches = [p for p in current.iterdir() if normalized(p.name) == normalized(part)]
        if len(matches) != 1:
            raise FileNotFoundError(f'경로를 찾거나 구분할 수 없습니다: {value}')
        current = matches[0]
    return current


def path_list(value):
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none'):
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    for sep in (';', '|', '\n', ','):
        if sep in text:
            return [p.strip().strip('\"\'') for p in text.split(sep) if p.strip()]
    return [text.strip('\"\'')]


class Resolver:
    def __init__(self, csv_path, dataset_root):
        self.csv_path, self.root = csv_path, dataset_root
        self.index = None

    def __call__(self, value):
        raw = normalized(str(value).strip().strip('\"\''))
        for candidate in (Path(raw), self.csv_path.parent / raw, self.root / raw):
            try:
                found = existing_path(candidate)
                if found.is_file():
                    return found
            except FileNotFoundError:
                pass
        if self.index is None:
            self.index = {}
            for p in self.root.rglob('*'):
                if 'input_stroke_ga' in p.relative_to(self.root).parts:
                    continue  # Legacy config excludes this duplicate, unused folder.
                if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'):
                    self.index.setdefault(normalized(p.name), []).append(p)
        candidates = self.index.get(raw.split('/')[-1], [])
        if not candidates:
            raise FileNotFoundError(f'CSV 이미지가 없습니다: {value} (검색 범위: {self.root})')
        def suffix_score(p):
            score = 0
            for left, right in zip(reversed(normalized(p).split('/')), reversed(raw.split('/'))):
                if left != right:
                    break
                score += 1
            return score
        best = max(map(suffix_score, candidates))
        matches = [p for p in candidates if suffix_score(p) == best]
        if len(matches) != 1:
            raise ValueError(f'같은 파일명이 여러 곳에 있습니다. CSV 경로를 명확하게 해주세요: {value}: {matches}')
        return matches[0]


def load_ink(path, size):
    with Image.open(path) as source:
        # Composite transparent PNGs onto white before converting to grayscale.
        image = ImageOps.exif_transpose(source).convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert('L')
        x = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    if x.mean() > .5:
        x = 1 - x
    return x[None].copy()


def convert_csv(csv_path, output, size=128, seed=42, dataset_root=None):
    csv_path = existing_path(csv_path)
    root = existing_path(dataset_root) if dataset_root else csv_path.parent
    resolve = Resolver(csv_path, root)
    df = pd.read_csv(csv_path)
    names = {str(c).lower(): c for c in df.columns}
    def column(*aliases, optional=False):
        for name in aliases:
            if name.lower() in names:
                return names[name.lower()]
        if optional:
            return None
        raise ValueError(f'필수 컬럼 {aliases} 없음. 현재 CSV 컬럼: {list(df.columns)}')
    cols = {
        'image': column('I_prep', 'input_image_path', 'input_path', 'input_image', 'input', 'source_path'),
        'reference': column('I_g', 'reference_image', 'reference_path', 'target_image', 'target_path', 'glyph_path'),
        'strokes': column('I_G_strokes', 'reference_strokes', 'reference_stroke_paths', 'template_strokes', 'target_strokes', 'target_stroke_paths'),
        'targets': column('input_stroke_paths', 'input_strokes', 'input_strokes_paths', 'prep_stroke_paths', 'gt_stroke_paths', 'gt_strokes', 'answer_stroke_paths', 'stroke_paths'),
    }
    group_col = column('writer_id', 'writer', 'subject_id', 'group_id', optional=True)
    char_col = column('char_id', 'char', 'character', 'text', optional=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'samples').mkdir()
    rows, reports = [], []
    seen_images = {}
    for index, row in df.iterrows():
        try:
            image_path = resolve(row[cols['image']])
            image = load_ink(image_path, size)
            reference = load_ink(resolve(row[cols['reference']]), size)
            ref_paths, target_paths = path_list(row[cols['strokes']]), path_list(row[cols['targets']])
            if not ref_paths or len(ref_paths) != len(target_paths):
                raise ValueError(f'기준/정답 획 수 불일치: {len(ref_paths)} / {len(target_paths)}')
            strokes = np.concatenate([load_ink(resolve(p), size) for p in ref_paths])
            targets = np.concatenate([load_ink(resolve(p), size) for p in target_paths]) > .5
            if not (strokes.reshape(len(strokes), -1).max(1) > .5).all():
                raise ValueError('비어 있는 기준 획 이미지가 있습니다')
            foreground = image[0] > .5
            counts = targets.sum(0)
            known = foreground & (counts == 1)
            labels = np.full((size, size), -1, dtype=np.int64)
            labels[known] = targets.argmax(0)[known]
            if any(not (labels == k).any() for k in range(len(strokes))):
                raise ValueError('정답 획 중 단독 소유 픽셀이 없는 획이 있습니다. 획 마스크/극성을 확인해주세요')
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if group_col:
                if pd.isna(row[group_col]):
                    raise ValueError('필자/그룹 ID가 비어 있습니다')
                group = 'writer:' + str(row[group_col])
            else:
                group = 'image:' + digest
            if digest in seen_images and seen_images[digest] != group:
                raise ValueError('동일 입력 이미지에 서로 다른 필자/그룹 ID가 지정되었습니다')
            seen_images[digest] = group
            path = output / 'samples' / f'{index:05d}.npz'
            np.savez_compressed(path, image=image, reference=reference, strokes=strokes, labels=labels)
            coverage = float(known.sum() / max(foreground.sum(), 1))
            rows.append({'path': str(path.resolve()), 'group_id': group, 'source': 'legacy_csv_unreviewed',
                         'char_id': str(row[char_col]) if char_col else '', 'source_image': str(image_path)})
            reports.append({'row': int(index), 'foreground_pixels': int(foreground.sum()),
                            'annotation_coverage': coverage, 'overlap_pixels': int((foreground & (counts > 1)).sum()),
                            'missing_pixels': int((foreground & (counts == 0)).sum())})
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(f'CSV 데이터 {index + 1}번째 행: {exc}') from exc
    groups = sorted({r['group_id'] for r in rows})
    if len(groups) < 2:
        raise ValueError('검증 분리를 위해 최소 2개 필자/원본 그룹이 필요합니다')
    random.Random(seed).shuffle(groups)
    validation = set(groups[:max(1, round(len(groups) * .2))])
    for row in rows:
        row['split'] = 'val' if row['group_id'] in validation else 'train'
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'csv': str(csv_path), 'columns': cols, 'group_column': group_col,
              'split_note': '필자 그룹 분할' if group_col else '원본 이미지 분할: 필자 일반화 평가는 아님',
              'label_policy': '단독 소유 전경만 감독; 교차/미할당은 -1. CSV 획 순서가 대응한다고 가정.',
              'rows': reports}
    (output / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"변환 완료: train={sum(r['split']=='train' for r in rows)}, val={sum(r['split']=='val' for r in rows)}")
    print(report['split_note'])
    print(f"평균 정답 커버리지: {np.mean([r['annotation_coverage'] for r in reports]):.1%}; 겹침/누락은 평가에서 제외됩니다.")
    print('획 순서의 의미적 대응은 자동 검증되지 않습니다. conversion_report.json과 미리보기를 확인하세요.')
    return manifest
