"""Build a standalone M/D/J scoring notebook and a pasteable scoring cell."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = '''# ===== 점수 설정: 누락·분리·정션 세 값만 사용 =====
# auto: 현재 structure_results가 있으면 재사용, 없으면 REPORT_PATH 읽기
# values: 아래 VALUES의 숫자만으로 바로 계산
MODE = "auto"
REPORT_PATH = "/content/drive/Shareddrives/2026 자율연구/HSQM/stroke_zip_runs/20261028_082616_066020/structure_20261101_092625_060835/structure_report.json"
SAVE_DIR = None  # None = 원래 결과 폴더 아래 indicator_시간. 직접 지정 시 새 폴더

SCORE_CONFIG = dict(
    missing_weight=1.0,   # 누락 중요도. 0이면 이 항 제외
    split_weight=1.0,     # 분리 중요도
    junction_weight=1.0,  # 정션 중요도
    split_scale=0.2,      # 분리값이 이 크기일 때 이 항의 유지율은 exp(-1)
    junction_scale=0.01,  # 정션값이 이 크기일 때 이 항의 유지율은 exp(-1)
    worst_stroke_weight=0.5, # 글자 점수: 획 평균과 최악 획 점수의 혼합. 0=평균, 1=최악
    eps=1e-6,
)

# MODE="values"일 때 사용. 이미지나 모델 없이 이 숫자만 계산합니다.
VALUES = [
    dict(stroke=1, missing=0.0, split=0.0, junction=0.0),
    dict(stroke=2, missing=0.809, split=0.0, junction=0.0),
]
'''

RUN = '''from _structural_indicator_runtime import IndicatorConfig, score_structure_report
from pathlib import Path

if MODE == "values":
    indicator_source = {"results": [{"input": "manual_values", "strokes": VALUES}]}
elif MODE == "auto":
    indicator_source = globals().get("structure_results")
    if indicator_source is None:
        if not REPORT_PATH:
            raise ValueError("구조 진단을 먼저 실행하거나 REPORT_PATH에 structure_report.json 경로를 넣으세요.")
        if str(REPORT_PATH).startswith("/content/drive/"):
            from google.colab import drive
            drive.mount("/content/drive")
        indicator_source = Path(REPORT_PATH)
else:
    raise ValueError('MODE는 "auto" 또는 "values"여야 합니다.')

indicator_results = score_structure_report(
    indicator_source, config=IndicatorConfig(**SCORE_CONFIG), save_dir=SAVE_DIR,
)
'''


def build():
    source = (ROOT/'stroke_model/structural_indicator.py').read_text(encoding='utf-8-sig')
    loader = '''# 점수 모듈 내장: 추가 모델·가중치 다운로드 없이 실행
import sys, types
_score_module = types.ModuleType("_structural_indicator_runtime")
_score_module.__file__ = "<embedded:structural_indicator>"
sys.modules[_score_module.__name__] = _score_module
exec(compile(''' + repr(source) + ''', _score_module.__file__, "exec"), _score_module.__dict__)
'''
    def cell(kind, text):
        result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    notebook = {'cells': [cell('markdown', '''# 누락·분리·정션 구조 지표

설문·추가 학습 없이 이미 계산한 missing/split/junction 세 값만 0~100 지표로 바꿉니다.
높을수록 세 오류가 적습니다. 사람 선호나 글자 품질에 교정한 점수는 아닙니다.

1. 첫 셀에서 입력과 가중치/변환 크기를 설정합니다.
2. 모듈 로드 셀과 계산 셀을 실행합니다.
3. 글자 structural_score와 획별 stroke_score를 확인합니다.

기존 구조 진단 노트북에 추가 셀을 붙이면 structure_results를 바로 재사용합니다.
이 독립 파일을 새로 열었다면 REPORT_PATH를 결과 JSON 경로로 지정하세요.
MODE="values"는 숫자만 입력하는 테스트입니다. 기존 추출 모델을 재실행하지 않습니다.

획 점수 = 100 × (1-M)^wM × exp(-wD·D/sD-wJ·J/sJ)

글자 점수 = (1-최악획비중) × 획 평균 + 최악획비중 × 최저 획 점수

기본값에서 누락 1인 획은 0점, 세 오류가 0인 획은 100점입니다.
정션 측정의 불확실성은 점수에도 전달됩니다. 노란 후보 수·파임·설문값은 공식에 넣지 않습니다.
'''), cell('code', SETTINGS), cell('code', loader), cell('code', RUN)],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'colab': {'name': 'structural_indicator.ipynb'}}, 'nbformat': 4, 'nbformat_minor': 5}
    path = ROOT/'notebooks/structural_indicator.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    one = ROOT/'notebooks/structural_indicator_cell.py'
    one.write_text(SETTINGS+'\n'+loader+'\n'+RUN, encoding='utf-8')
    for item in notebook['cells']:
        if item['cell_type'] == 'code':
            compile(''.join(item['source']), str(path), 'exec')
    compile(one.read_text(encoding='utf-8'), str(one), 'exec')
    print('Built:', path.name, one.name)


if __name__ == '__main__':
    build()
