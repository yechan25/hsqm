"""Build the controlled stage-2 comparison from the stage-1 notebook."""
import ast
import hashlib
import json
from pathlib import Path
from build_stage1 import build as build_stage1

ROOT = Path(__file__).resolve().parents[2]


def build():
    nb = json.loads(build_stage1().read_text(encoding='utf-8'))
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'RUNTIME_HELPERS=' in source:
            nodes = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)]
            helpers = next(ast.literal_eval(n.value) for n in nodes if any(
                isinstance(t, ast.Name) and t.id == 'RUNTIME_HELPERS' for t in n.targets))
            helpers['stroke_model/boundary_guarded_repair.py'] = (ROOT/'stroke_model/boundary_guarded_repair.py').read_text(encoding='utf-8')
            hashes = {k:hashlib.sha256(v.encode()).hexdigest() for k,v in helpers.items()}
            source = '\n'.join('RUNTIME_HELPERS='+repr(helpers) if line.startswith('RUNTIME_HELPERS=') else
                'SOURCE_HASHES='+repr(hashes) if line.startswith('SOURCE_HASHES=') else line
                for line in source.split('\n'))
            source = source.replace('/content/hsqm_stage1_audit_runtime', '/content/hsqm_stage2_audit_runtime')
        if cell['cell_type'] == 'code' and 'ARCHIVE = ' in source:
            source = source.replace("RUN_REFERENCE_WIDTH_PROBE = False", "RUN_REFERENCE_WIDTH_PROBE = False\nBOUNDARY_CONFIG = dict(uncertainty_pixels=1., responsibility_floor=.5)")
            source = source.replace("'repair_stage1_'", "'repair_stage2_'")
            source = source.replace("'/content/drive/MyDrive/HSQM_stage1_results'", "'/content/drive/MyDrive/HSQM_stage2_results'")
            source = source.replace("reference_width_probe=RUN_REFERENCE_WIDTH_PROBE)))",
                "reference_width_probe=RUN_REFERENCE_WIDTH_PROBE, boundary_config=BOUNDARY_CONFIG)))")
        if cell['cell_type'] == 'code' and 'METRICS, FAILURES' in source:
            source = source.replace('width_probe=RUN_REFERENCE_WIDTH_PROBE, progress=progress)',
                'width_probe=False, progress=progress, boundary_guard=True, boundary_config=BOUNDARY_CONFIG)')
        if cell['cell_type'] == 'markdown' and source.startswith('# 1단계:'):
            source = '''# 2단계: 삭제 경계 보호 수정 비교

**1단계 원인 진단 완료 → 경계 오탐 규칙 수정 → 같은 조건으로 효과 확인.**
새 L4 GPU 세션에서 코드 셀 1→5를 실행하세요. 학습은 하지 않습니다.

입력은 방금 실행한 `repair_stage1_*.zip` 또는 기존 cyclic/selectable ZIP입니다.
stage1 ZIP에서는 이미 보정한 PNG가 아니라 원래 input.png와 기준 획을 가져옵니다.
동일 best.pt가 필요하며 SHA256을 확인합니다.

비교는 원본 / VAE / 일반 재역확산 / 기존 cyclic / 수정 boundary_guarded입니다.
세 재역확산 경로는 동일 initial latent·잡음·시간 단계를 공유합니다.
강도 .25, 60 nominal steps(실제15), seed2026, CFG5, gradient step/budget, 추출기와 피팅은 유지합니다.

변경은 삭제 지도에 한정됩니다:
- 피팅 경계에 모델 좌표 1픽셀의 불확실 여유를 추가합니다. 획을 실제로 두껍게 그리는 처리가 아닙니다.
- seed를 1픽셀 확장한 뒤에도 각 픽셀이 삭제 근거를 충족하는지 재검사합니다.
- 확대 보간으로 허용 영역 밖에 가중치가 새지 않도록 같은 eligibility 지도를 적용합니다.

이 값은 첫 비교용이며 최적값이 아닙니다. 일반 재역확산 자체의 변형은 여전히 가능합니다.
기준 이미지 추종/획 추출 일반화까지 해결한 모델로 해석하지 않습니다.
'''
        if cell['cell_type'] == 'markdown' and source.startswith('## 2번 셀'):
            source = '''## 2번 셀 설정
- `ARCHIVE=''`: 업로드 창에서 **repair_stage1_20261118_054447_077301.zip**을 선택하면 됩니다.
- `CHECKPOINT`: 1단계와 같은 best.pt 경로를 지정합니다.
- `CASE_IDS=None`: 생성 입력 전부와 해당 정상 대조군을 비교합니다.
- `BOUNDARY_CONFIG`와 강도/횟수/시드는 첫 실행에서 그대로 둡니다.
- 이번에는 최초 생성부터 다시 하지 않습니다. stage1 ZIP의 원래 생성물과 정상 대조군을 재사용합니다.
'''
        if cell['cell_type'] == 'markdown' and source.startswith('## 결과 검토'):
            source = '''## 결과 검토
- comparison.png: cyclic_guided와 boundary_guarded를 직접 비교합니다.
- 정상 대조군의 개/곁: 획 끝과 접합부가 덜 깎이는지 확인합니다.
- 생성 가/꺄: 추가 돌출 제거가 유지되는지 확인합니다.
- boundary_guarded/erase_*.png와 cyclic_guided/erase_*.png: 같은 단계의 삭제 지도를 대조합니다.
- boundary_guarded_map_history.json: 삭제 근거·확장 후 거절 픽셀 수를 기록합니다.
- automatic_metrics.csv의 all_source_*는 기존 healthy 지도에서 빠졌던 정상 획 끝도 포함한 보조 지표입니다.
  실제 오류도 섞일 수 있으므로 사람의 오류 감소 판정과 구분합니다.
- human_review.csv에 각 결과의 error_reduced/thinner/new_error/style_changed를 yes/no/uncertain으로 기록할 수 있습니다.

5번 셀 결과 ZIP을 보내주면 수정 효과와 남은 실패를 판단할 수 있습니다.
자동 점수만으로 성공을 선언하지 않습니다.
'''
        if cell['cell_type'] == 'code':
            ast.parse(source)
        cell['source'] = source.strip().splitlines(keepends=True)
    path = ROOT/'notebooks/repair_audit/stage2.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
