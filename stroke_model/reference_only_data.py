"""Restore training-reference membership without converting training samples."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .convert_reference_csv import Resolver, load_ink, normalized, path_list
from .prepare_stroke_zips import stage_stroke_archives, discover_archives

REFERENCE_ROLES = ('target.zip', 'target_stroke.zip')


def prepare_reference_catalog(source, csv_path, conversion, size, seed, local_root):
    """Reproduce the original group split from the saved accepted CSV row IDs.

    Only reference PNGs are decoded to recover content-hash reference groups.
    Ordered stroke PNGs are opened later, when a glyph is selected. No NPZ here.
    """
    source, csv_path = Path(source), Path(csv_path)
    group_col = conversion.get('group_column')
    if conversion.get('supervision') != 'actual_input_multihot' and not group_col:
        raise ValueError('이전 image-hash 분할은 기준만으로 복원할 수 없습니다. '
                         'REFERENCE_ONLY=False로 기존 전체 변환을 사용하세요.')
    if 'rows' not in conversion:
        raise ValueError('저장된 conversion_report.json에 원래 학습 행 목록이 필요합니다.')
    cache, raw, _ = stage_stroke_archives(source, local_root, roles=REFERENCE_ROLES)
    fingerprint = hashlib.sha256(csv_path.read_bytes()+json.dumps(
        dict(version=1,size=size,seed=seed,conversion=conversion),sort_keys=True).encode()).hexdigest()[:20]
    destination = cache/f'reference_only_{fingerprint}'
    manifest = destination/'manifest.json'
    if manifest.is_file():
        rows=json.loads(manifest.read_text(encoding='utf-8'))
        if rows and all(Path(r['reference']).is_file() and all(Path(p).is_file() for p in r['reference_strokes']) for r in rows):
            print('기준 글자 목록 캐시 재사용 (학습 NPZ 변환 없음):',manifest,flush=True)
            return manifest
    frame=pd.read_csv(csv_path)
    columns=conversion['columns']
    accepted=[int(row['row']) for row in conversion['rows']]
    if not accepted or len(set(accepted))!=len(accepted) or min(accepted)<0 or max(accepted)>=len(frame):
        raise ValueError('저장된 학습 행과 CSV가 일치하지 않습니다.')
    resolve=Resolver(csv_path,raw,local_only=True)
    # Index ONLY target archives, even if input archives were extracted earlier.
    resolve.index={}
    for archive in discover_archives(source,roles=REFERENCE_ROLES):
        archive_id=hashlib.sha256(archive.relative_to(source).as_posix().encode()).hexdigest()[:16]
        for path in (raw/archive_id).rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'):
                resolve.index.setdefault(normalized(path.name),[]).append(path)
    char_col=next((c for c in frame.columns if str(c).lower() in ('char_id','char','character','text')),None)
    hashes, resolved, records, groups = {}, {}, [], set()
    seen=set()
    print('학습 입력/정답 획은 읽지 않고 기준 글자만 준비합니다.',flush=True)
    for index in tqdm(accepted,desc='Index reference glyphs (no NPZ)',unit='row',mininterval=2):
        row=frame.iloc[index]
        raw_reference=str(row[columns['reference']])
        if raw_reference not in resolved:
            resolved[raw_reference]=resolve(raw_reference)
        reference=resolved[raw_reference]
        if str(reference) not in hashes:
            hashes[str(reference)]=hashlib.sha256(load_ink(reference,size).tobytes()).hexdigest()
        if group_col:
            if pd.isna(row[group_col]):raise ValueError('학습 그룹 ID가 비어 있습니다.')
            group='writer:'+str(row[group_col])
        else:
            group='reference:'+hashes[str(reference)]
        groups.add(group)
        stroke_names=path_list(row[columns['strokes']])
        if not stroke_names:raise ValueError('기준 획 목록이 비어 있습니다.')
        key=(str(reference),tuple(stroke_names),group)
        if key in seen:continue
        seen.add(key)
        paths=[]
        for name in stroke_names:
            if name not in resolved:resolved[name]=resolve(name)
            paths.append(str(resolved[name].resolve()))
        signature=hashlib.sha256((hashes[str(reference)]+json.dumps(paths)).encode()).hexdigest()
        records.append(dict(reference=str(reference.resolve()),reference_strokes=paths,
            image_size=size,group_id=group,char_id=str(row[char_col]) if char_col else '',
            source_csv_row=index+1,reference_sha256=signature,format='reference_paths_v1'))
    groups=sorted(groups)
    if len(groups)<2:raise ValueError('원래 학습 분할을 복원할 그룹이 부족합니다.')
    random.Random(seed).shuffle(groups)
    validation=set(groups[:max(1,round(len(groups)*.2))])
    for record in records:record['split']='val' if record['group_id'] in validation else 'train'
    destination.mkdir(exist_ok=True)
    partial=manifest.with_suffix('.partial')
    partial.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    partial.replace(manifest)
    print(f'기준 {len(records)}개 준비 완료. 학습 표본 {len(accepted)}개의 NPZ 변환을 생략했습니다.',flush=True)
    return manifest
