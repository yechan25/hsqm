from pathlib import Path
import json
def export_indicator_debug(structure_report, indicator_report, destination=None):
    """Package only evaluated snapshots and reports, never a checkpoint/dataset."""
    import copy
    import zipfile
    source_path = None
    if not isinstance(structure_report, dict):
        source_path = Path(structure_report)
        structure_report = json.loads(source_path.read_text(encoding='utf-8-sig'))
    report = copy.deepcopy(structure_report)
    scores = indicator_report['results']
    if len(report['results']) != len(scores):
        raise ValueError('Structure and indicator reports must describe the same evaluated samples')
    path = Path(destination or Path(indicator_report['save_dir'])/'indicator_debug.zip')
    path.parent.mkdir(parents=True, exist_ok=True)
    summary, missing = [], []
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for number, (sample, score) in enumerate(zip(report['results'], scores), 1):
            found = None
            for key in ('npz', 'input_npz'):
                if not sample.get(key):
                    continue
                candidates = [Path(sample[key])]
                if source_path and not candidates[0].is_absolute():
                    candidates.append(source_path.parent/candidates[0])
                found = next((p for p in candidates if p.is_file()), None)
                if found:
                    break
            if found:
                name = f'arrays/sample_{number:04d}.npz'
                archive.write(found, name)
                sample['npz'] = name
                sample.pop('input_npz', None)
            else:
                missing.append(number)
            measurement=score.get('measurement_npz')
            if measurement and Path(measurement).is_file():
                archive.write(measurement,f'measurements/sample_{number:04d}.npz')
            caps = {key: score[key+'_score_cap'] for key in ('missing', 'extra', 'junction', 'split')}
            limiting = min(caps, key=caps.get)
            if score['structural_score'] < min(caps.values())-1e-5:
                limiting = 'combined'
            summary.append({'sample': number, 'input': score.get('input'),
                'score': score['structural_score'], 'limiting_term': limiting,
                'score_caps': caps, 'strokes': score['strokes']})
        archive.writestr('structure_report.json', json.dumps(report, ensure_ascii=False, indent=2))
        archive.writestr('indicator_report.json', json.dumps(indicator_report, ensure_ascii=False, indent=2))
        archive.writestr('diagnosis.json', json.dumps({'samples': summary, 'missing_arrays': missing,
            'purpose': 'Diagnose false penalties; not a validated quality assessment.'}, ensure_ascii=False, indent=2))
    print('Diagnostic ZIP:', path)
    if missing:
        print('Snapshots unavailable for samples:', missing)
    return str(path)

# 기존 평가 결과를 묶기만 합니다. 모델 추론/점수 계산을 반복하지 않습니다.
_debug_source = globals().get("indicator_source")
if _debug_source is None:
    _debug_source = globals().get("structure_results")
if _debug_source is None or "indicator_results" not in globals():
    raise ValueError("사진과 점수를 평가한 기존 런타임의 마지막 새 셀에서 실행하세요.")
debug_zip = export_indicator_debug(_debug_source, indicator_results)
from google.colab import files
files.download(debug_zip)
