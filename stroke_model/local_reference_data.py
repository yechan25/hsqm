"""Copy immutable prepared samples off Drive once per Colab runtime."""
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm.auto import tqdm
from .train_reference import read_manifest


def stage_manifest(manifest, cache_root='/content/reference_v2_local', workers=4):
    manifest = Path(manifest).resolve()
    key = hashlib.sha256(str(manifest).encode() + manifest.read_bytes()).hexdigest()[:20]
    dest = Path(cache_root) / key
    target = dest / 'manifest.json'
    if target.is_file():
        rows = read_manifest(target)
        if all(Path(row['path']).is_file() for row in rows):
            print('Colab 로컬 데이터 재사용:', target, flush=True)
            return target
    rows = [r for r in read_manifest(manifest) if r['split'] in ('train', 'val')]
    if not rows:
        raise ValueError('No training/validation samples')
    dest.mkdir(parents=True, exist_ok=True)
    def copy(pair):
        i, row = pair
        source = Path(row['path'])
        local = dest / f'{i:07d}.npz'
        if not local.is_file() or local.stat().st_size != source.stat().st_size:
            partial = local.with_suffix('.partial')
            shutil.copyfile(source, partial)
            partial.replace(local)
        return {**row, 'path': str(local.resolve()), 'drive_source': str(source)}
    print(f'Drive → Colab 로컬 복사: {len(rows)}개. 런타임당 한 번만 필요합니다.', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        local_rows = list(tqdm(pool.map(copy, enumerate(rows)), total=len(rows), desc='Copy to local', unit='file', mininterval=2))
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(local_rows, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)
    return target
