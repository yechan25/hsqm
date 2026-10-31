"""Stage immutable data off Drive; compare bytes, preserve explicit splits."""
import hashlib
import json
import shutil
from pathlib import Path
from .data import read_manifest


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        while chunk := source.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def stage_manifest(source, output='/content/hsqm_diffusion_data'):
    source, root = Path(source).resolve(), Path(output)
    folder = root/hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    copies = {}
    for row in read_manifest(source):
        if row['split'] == 'test':
            continue
        row = dict(row)
        for field in ('path',) if 'path' in row else ('image', 'reference'):
            path = Path(row[field])
            if path not in copies:
                target = folder/(hashlib.sha256(str(path).encode()).hexdigest()[:24]+path.suffix)
                source_hash = digest(path)
                if not target.is_file() or digest(target) != source_hash:
                    temporary = target.with_suffix('.partial')
                    shutil.copyfile(path, temporary)
                    temporary.replace(target)
                copies[path] = str(target.resolve())
            row[field] = copies[path]
        rows.append(row)
    manifest = folder/'manifest.json'
    temporary = manifest.with_suffix('.partial')
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(manifest)
    print(f'Local dataset: {len(rows)} train/val rows, {manifest}', flush=True)
    return manifest
