"""Stage the four stroke archives locally; use CSV for semantic correspondence."""
import hashlib
import json
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from tqdm.auto import tqdm

from .convert_reference_csv import convert_csv, existing_path

ARCHIVES = ('input_stroke.zip', 'input.zip', 'target_stroke.zip', 'target.zip')


def discover_archives(source):
    """Each numbered TAR is a complete archive, not a byte-split TAR volume."""
    paths = []
    for name in ARCHIVES:
        role = Path(name).stem
        try:
            folder = existing_path(source / role)
        except FileNotFoundError:
            folder = source / role
        parts = []
        if folder.is_dir():
            pattern = re.compile(re.escape(role) + r'_\((\d+)\)\.tar(?:\.gz)?$', re.IGNORECASE)
            for path in folder.iterdir():
                match = pattern.fullmatch(path.name)
                if path.is_file() and match:
                    parts.append((int(match[1]), path))
        if parts:
            parts.sort(key=lambda item: item[0])
            numbers = [number for number, _ in parts]
            if numbers != list(range(numbers[-1] + 1)):
                raise ValueError(f'{role}: TAR 번호가 중복되거나 누락되었습니다: {numbers}')
            paths.extend(path for _, path in parts)
        else:
            try:
                paths.append(existing_path(source / name))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f'{source / role} 안의 {role}_(0).tar, {role}_(1).tar, ... '
                    f'또는 {source / name}을 찾을 수 없습니다.') from exc
    return paths


def extract_checked(archive, destination):
    destination = Path(destination).resolve()
    if not zipfile.is_zipfile(archive):
        with tarfile.open(archive, 'r:*') as tar:
            members = tar.getmembers()
            for member in members:
                path = PurePosixPath(member.name.replace('\\', '/'))
                if (path.is_absolute() or '..' in path.parts or
                        any(':' in part for part in path.parts) or
                        not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unsafe TAR member: {member.name}')
            needed = sum(m.size for m in members if m.isfile())
            if needed > shutil.disk_usage(destination.parent).free:
                raise OSError(f'로컬 디스크 부족: 압축 해제에 {needed / 2**30:.1f} GiB 필요')
            for member in tqdm(members, desc=f'Extract {Path(archive).name}', unit='file', mininterval=2):
                target = destination.joinpath(*PurePosixPath(member.name.replace('\\', '/')).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
        return
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for member in members:
            path = PurePosixPath(member.filename.replace('\\', '/'))
            if (path.is_absolute() or '..' in path.parts or
                    any(':' in part for part in path.parts) or
                    stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        needed = sum(m.file_size for m in members)
        if needed > shutil.disk_usage(destination.parent).free:
            raise OSError(f'로컬 디스크 부족: 압축 해제에 {needed / 2**30:.1f} GiB 필요')
        for member in tqdm(members, desc=f'Extract {Path(archive).name}', unit='file', mininterval=2):
            z.extract(member, destination)


def find_csv(source, raw, csv_path=None):
    if csv_path:
        return existing_path(csv_path)
    # Only the selected dataset directory and extracted archives, never a test/old dataset.
    direct = [p for p in source.iterdir() if p.is_file() and p.name.casefold() in ('path.csv', 'paths.csv')]
    candidates = direct or [p for p in raw.rglob('*') if p.is_file() and p.name.casefold() in ('path.csv', 'paths.csv')]
    if not candidates:
        raise FileNotFoundError(
            f'path.csv / paths.csv가 아직 없습니다. {source}에 업로드한 뒤 이 셀만 다시 실행하세요. '
            '완료된 압축 파일 복사/해제는 재사용합니다. 다른 CSV 이름이면 CSV_PATH를 지정하세요.')
    if len(candidates) != 1:
        raise ValueError(f'CSV 후보가 여러 개입니다. CSV_PATH를 지정하세요: {candidates}')
    return candidates[0]


def prepare_stroke_zips(source, local_root='/content/hsqm_stroke_zip', csv_path=None, size=128, seed=42,
                        independent=False):
    source = existing_path(source)
    paths = discover_archives(source)
    print('사용할 압축 파일:', len(paths), '개', flush=True)
    signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
    cache = Path(local_root) / key
    raw = cache / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for path in paths:
        destination = raw / path.stem
        if (destination / '.complete.json').is_file():
            print('로컬 압축 데이터 재사용:', path.name, flush=True)
            continue
        # Failed copies/extractions never create the completion marker.
        local_zip = cache / path.name
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        if shutil.disk_usage(cache).free < before[0]:
            raise OSError('압축 파일 복사를 위한 로컬 디스크 공간이 부족합니다')
        print('Drive → 로컬 압축 파일 복사:', path.name, flush=True)
        partial = local_zip.with_suffix('.partial')
        with path.open('rb') as src, partial.open('wb') as dst:
            with tqdm(total=before[0], desc=path.name, unit='B', unit_scale=True, mininterval=2) as progress:
                while chunk := src.read(8 * 1024 * 1024):
                    dst.write(chunk)
                    progress.update(len(chunk))
        after = (path.stat().st_size, path.stat().st_mtime_ns)
        if before != after or partial.stat().st_size != before[0]:
            raise RuntimeError(f'{path.name}이 복사 중 변경됐습니다. 업로드 완료 후 재실행하세요.')
        partial.replace(local_zip)
        temporary = Path(tempfile.mkdtemp(prefix=path.stem + '_', dir=cache))
        extract_checked(local_zip, temporary)
        (temporary / '.complete.json').write_text(json.dumps(before), encoding='utf-8')
        temporary.rename(destination)
        local_zip.unlink()  # Keep extracted images, reclaim the local archive space.
    selected = find_csv(source, raw, csv_path)
    content = selected.read_bytes()
    conversion_key = hashlib.sha256(content + f'converter2:{size}:{seed}:{independent}'.encode()).hexdigest()[:16]
    prepared = cache / ('prepared_' + conversion_key)
    manifest = prepared / 'manifest.json'
    if (prepared / '.complete.json').is_file() and manifest.is_file():
        print('로컬 변환 데이터 재사용:', manifest, flush=True)
        return manifest
    print('CSV 선택:', selected, flush=True)
    local_csv = cache / 'selected_paths.csv'
    local_csv.write_bytes(content)
    temporary = Path(tempfile.mkdtemp(prefix='conversion_', dir=cache)) / 'prepared'
    manifest = convert_csv(local_csv, temporary, size=size, seed=seed, dataset_root=raw, local_only=True,
                           independent=independent)
    # Keep converter absolute NPZ paths stable: use the completed temp directory directly.
    prepared.mkdir(exist_ok=True)
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    (prepared / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copyfile(temporary / 'conversion_report.json', prepared / 'conversion_report.json')
    (prepared / 'selected_paths.csv').write_bytes(content)
    (prepared / '.complete.json').write_text(json.dumps({'csv_source': str(selected), 'archives': signature}), encoding='utf-8')
    return prepared / 'manifest.json'
