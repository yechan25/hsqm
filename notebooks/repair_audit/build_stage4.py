"""Build an upload-only same-input observed-width ablation."""
import ast
import hashlib
import json
from pathlib import Path
from build_stage3 import build as build_stage3, embedded

ROOT = Path(__file__).resolve().parents[2]


def build():
    build_stage3()  # Keep existing standalone notebooks synchronized with shared runtime.
    nb = json.loads((ROOT/'notebooks/repair_audit/stage2.ipynb').read_text(encoding='utf-8'))
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'RUNTIME_HELPERS=' in source:
            helpers = embedded(source)
            for name in ['stroke_model/observed_width_repair.py', 'repair_audit/stage3.py']:
                helpers[name] = (ROOT/name).read_text(encoding='utf-8')
            hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k,v in helpers.items()}
            source = '\n'.join('RUNTIME_HELPERS='+repr(helpers) if line.startswith('RUNTIME_HELPERS=') else
                'SOURCE_HASHES='+repr(hashes) if line.startswith('SOURCE_HASHES=') else line
                for line in source.splitlines())
            source = source.replace('hsqm_stage2_audit_runtime','hsqm_stage4_audit_runtime')
            source += '\nfrom repair_audit.stage3 import shape_comparison, write_stage3_tables\n'
        if cell['cell_type'] == 'code' and 'ARCHIVE = ' in source:
            source = source.replace('CASES, EXPERIMENT = read_archive(ARCHIVE, OUTPUT, CASE_IDS, include_controls=True)',
                "CASES, EXPERIMENT = read_archive(ARCHIVE, OUTPUT, CASE_IDS, include_controls=True)\nif any(len(c.get('original_rows', [])) != 4 or len(c.get('char_ids', [])) != 4 for c in CASES):\n    raise ValueError('글자 row 정보가 있는 3단계 결과 ZIP을 선택하세요.')")
            source = source.replace('BOUNDARY_CONFIG = dict(uncertainty_pixels=1., responsibility_floor=.5)',
                'BOUNDARY_CONFIG = dict(uncertainty_pixels=1., responsibility_floor=.5)\nWIDTH_CONFIG = dict(minimum_samples=12, minimum_path_coverage=.7, maximum_increase=4., maximum_half_width=8.)')
            source = source.replace('repair_stage2_', 'repair_stage4_').replace('HSQM_stage2_results','HSQM_stage4_results')
            source = source.replace('boundary_config=BOUNDARY_CONFIG)))',
                                    'boundary_config=BOUNDARY_CONFIG, width_config=WIDTH_CONFIG)))')
        if cell['cell_type'] == 'code' and 'METRICS, FAILURES' in source:
            source = source.replace('boundary_config=BOUNDARY_CONFIG)',
                'boundary_config=BOUNDARY_CONFIG, observed_width=True, width_config=WIDTH_CONFIG)')
            source = source.replace("display(Image.open(Path(case['directory'])/'audit/comparison.png'))",
                "shape_comparison(case, arms=['input','vae_roundtrip','redenoise_only','cyclic_guided','boundary_guarded','observed_width'])\n        display(Image.open(Path(case['directory'])/'audit/shape_comparison.png'))")
        if cell['cell_type'] == 'code':
            source = source.replace('write_tables(OUTPUT, CASES, METRICS, FAILURES)',
                                    'write_stage3_tables(OUTPUT, CASES, METRICS, FAILURES)')
            ast.parse(source)
        elif source.startswith('# 2단계:'):
            source = '''# 4단계: 폐곡선의 관측 두께를 반영한 삭제 기준 비교

새 L4 GPU 세션에서 코드 셀 **1→5** 실행. 모델 학습·새 이미지 생성은 하지 않습니다.
입력은 **repair_stage3_20261118_064305_733955.zip**과 기존 best.pt입니다.
ZIP 안의 이미 보정된 결과가 아니라 원래 input.png와 기준 획을 읽습니다.
12개 기준 샘플의 정상 입력·생성 입력 모두 같은 보정 설정으로 다시 비교합니다.

비교: 입력 / VAE / 일반 재확산 / 기존 cyclic / 경계 보호 / observed_width.
네 재확산 경로는 같은 초기 latent·잡음·시간 단계를 공유합니다.
강도 .25, nominal60 steps(실제15), seed2026, CFG5와 나머지 보정 설정은 유지합니다.

**변경은 폐곡선 fallback 획의 삭제용 반경뿐입니다.**
- 원래 입력의 추출 마스크에서, 예상 경로 근처의 관측 중심선 두께 중앙값을 구합니다.
- 충분한 경로가 보일 때만 기존 반경을 늘립니다. 증가량/최대값에 상한을 둡니다.
- 직선·열린 곡선, 복원 경로·투영, 피팅, 모델은 바꾸지 않습니다.
- 글자를 기준 이미지로 덮거나 반듯하게 그리는 후처리를 하지 않습니다.

정상 ㅇ를 덜 지우는지와 실제 추가 획도 보호해버리는지 함께 확인해야 합니다.
열린 곡선·접합부 손상과 잔상까지 해결한 수정은 아닙니다. GPU 품질은 이번 실행으로 검증합니다.
'''
        elif source.startswith('## 2번 셀'):
            source = '''## 2번 셀: 경로와 입력
- ARCHIVE를 비우면 업로드 창이 열립니다. **3단계 결과 ZIP**을 선택하세요.
- CHECKPOINT는 이전과 같은 best.pt이며 SHA256을 확인합니다.
- CASE_IDS=None: 생성 3시트와 각 정상 대조군을 모두 비교합니다.
- BOUNDARY_CONFIG/WIDTH_CONFIG/강도·시드는 첫 비교에서 그대로 두세요.
- 기준 원본 ZIP이나 글자 선택을 다시 준비할 필요는 없습니다.
'''
        elif source.startswith('## 결과 검토'):
            source = '''## 결과 검토
- shape_comparison.png: 글자별 6개 결과 비교. boundary_guarded와 observed_width가 핵심 비교입니다.
- 정상 row148의 ㅇ, row202의 ㅇ와 row64: 굵기·윤곽 보존이 좋아지는지 확인합니다.
- row131 등 적용 대상이 아닌 획의 변형도 남는지 확인합니다.
- 생성 입력: 돌출·잡티 제거가 유지되는지 반드시 함께 봅니다.
- observed_width_calibration.json: 획별 기존/새 반경, 샘플 수, 적용/거절 사유.
- observed_width_initial_maps.npz와 boundary_initial_maps.npz: 원본 초기 삭제 지도 비교.
- 각 arm/erase_*.png, restore_*.png와 *_trace.json: 삭제 침범·복원 투영 기록.
- human_review.csv: 오류 감소·얇아짐·직선/곡선·끝점·접합부·잔상을 각각 판정합니다.
- 자동 폭 비율은 실제 굵기 변화율이나 정답률이 아닙니다.

5번 셀 결과 ZIP을 보내주세요. 초기 지도 개선과 최종 GPU 품질은 구분해서 판단합니다.
'''
        cell['source'] = source.strip().splitlines(keepends=True)
    path=ROOT/'notebooks/repair_audit/stage4.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
