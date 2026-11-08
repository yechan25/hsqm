"""Recover the transient stroke-training data for the diffusion scoring pilot."""
import hashlib
import json
from pathlib import Path

from stroke_model.convert_reference_csv import existing_path
from stroke_model.prepare_stroke_zips import prepare_stroke_zips


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _usable_manifest(path):
    if not path.is_file():
        return False
    rows = _read_json(path)
    return (isinstance(rows, list) and bool(rows) and any(r.get('split') == 'train' for r in rows)
            and all((path.parent / row['path']).is_file() for row in rows))


def prepare_pilot_data(checkpoint, data_root=None, *, csv_path=None,
                       local_root='/content/hsqm_stroke_zip', manifest=None):
    """Reuse live local data, otherwise rerun the SAME archive/CSV preparation.

    Reads the CSV snapshot, seed, resolution and supervision saved beside best.pt.
    Does not train a model or modify checkpoint files. No whole-Drive search.
    """
    checkpoint = existing_path(checkpoint)
    run = checkpoint.parent
    config = _read_json(run/'run_config.json')
    recorded = config.get('manifest')
    candidates = [Path(manifest)] if manifest is not None else ([Path(recorded)] if recorded else [])
    for candidate in candidates:
        if _usable_manifest(candidate):
            print('현재 세션의 학습 데이터 재사용:', candidate, flush=True)
            return dict(manifest=str(candidate.resolve()), mode='reuse_live_training_data',
                        checkpoint=str(checkpoint.resolve()))
    if manifest is not None:
        print('지정한 학습 목록 또는 연결된 NPZ가 없어 원본 압축 자료로 다시 준비합니다.', flush=True)
    conversion = _read_json(run/'conversion_report.json')
    if 'image_size' not in config or 'seed' not in config:
        raise ValueError('체크포인트 옆 run_config.json에 image_size/seed가 필요합니다. '
                         '기존 학습 결과 폴더의 best.pt를 선택하세요.')
    supervision = conversion.get('supervision')
    if supervision not in ('actual_input_multihot', 'exclusive_annotated'):
        raise ValueError('체크포인트 옆 conversion_report.json에서 기존 데이터 변환 방식을 확인할 수 없습니다.')
    # A saved CSV is the precise input to the original preparation, unlike a new
    # CSV guessed from another dataset or a numeric character label.
    snapshot = run/'selected_paths.csv'
    if csv_path is not None:
        selected_csv = existing_path(csv_path)
        if snapshot.is_file() and selected_csv.read_bytes() != snapshot.read_bytes():
            raise ValueError('CSV_PATH가 체크포인트의 학습 CSV와 다릅니다. 같은 CSV를 사용하세요.')
    elif snapshot.is_file():
        selected_csv = snapshot
    else:
        raise FileNotFoundError('기존 학습 결과 폴더에 selected_paths.csv가 없습니다. '
                                'CSV_PATH에 해당 학습에 쓴 원본 CSV를 지정하세요.')

    # Saved preparation metadata contains original archive paths. Infer their
    # common dataset root only when the caller did not specify one.
    if data_root is None:
        import os
        complete = _read_json(run/'complete.json')
        archive_paths = [str(item[0]) for item in complete.get('archives', [])]
        if not archive_paths:
            raise ValueError('DATA_ROOT에 기존 학습 ZIP/TAR가 있는 폴더를 지정하세요.')
        data_root = Path(os.path.commonpath(archive_paths))
        if len(archive_paths) == 1:
            data_root = data_root.parent
    source = existing_path(data_root)
    print('기존 임시 manifest 없음 → 학습 ZIP/TAR·CSV로 자동 재생성', flush=True)
    print('학습 압축 자료:', source, flush=True)
    print('원래 학습 CSV:', selected_csv, flush=True)
    print('기존 설정: size=', config['image_size'], 'seed=', config['seed'],
          'supervision=', supervision, flush=True)
    rebuilt = prepare_stroke_zips(source, local_root=local_root, csv_path=selected_csv,
                                  size=int(config['image_size']), seed=int(config['seed']),
                                  independent=supervision == 'actual_input_multihot')
    rebuilt_conversion = _read_json(rebuilt.parent/'conversion_report.json')
    # Catch changed/missing source samples before claiming the restored split is
    # the original training population. Image identity itself was not fingerprinted
    # by old training runs and cannot be retroactively guaranteed.
    for key in ('supervision', 'group_column', 'skipped_count'):
        if key in conversion and conversion[key] != rebuilt_conversion.get(key):
            raise ValueError(f'재생성 데이터의 {key}가 원래 학습 기록과 다릅니다.')
    if 'rows' in conversion and len(conversion['rows']) != len(rebuilt_conversion.get('rows', [])):
        raise ValueError('재생성한 표본 수가 원래 학습 기록과 다릅니다.')
    if not _usable_manifest(rebuilt):
        raise ValueError('재생성한 학습 목록/NPZ를 확인할 수 없습니다.')
    result = dict(manifest=str(rebuilt.resolve()), mode='rebuilt_from_training_archives',
                  checkpoint=str(checkpoint.resolve()), data_root=str(source.resolve()),
                  csv_snapshot=str(selected_csv.resolve()),
                  csv_sha256=hashlib.sha256(selected_csv.read_bytes()).hexdigest(),
                  image_size=int(config['image_size']), seed=int(config['seed']),
                  supervision=supervision, original_image_content_fingerprint_available=False)
    print('학습 목록 준비 완료:', rebuilt, flush=True)
    return result
