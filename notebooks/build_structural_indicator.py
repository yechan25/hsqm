"""Build a standalone M/D/J/extra scoring notebook and pasteable cell."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = '''# ===== 점수 설정: 누락·분리·정션 + 추가 잉크 =====
# auto: 현재 structure_results가 있으면 재사용, 없으면 REPORT_PATH 읽기
# values: 아래 VALUES의 숫자만으로 바로 계산
MODE = "auto"
REPORT_PATH = "/content/drive/Shareddrives/2026 자율연구/HSQM/stroke_zip_runs/20261028_082616_066020/structure_20261101_092625_060835/structure_report.json"
SAVE_DIR = None  # None = 원래 결과 폴더 아래 indicator_시간. 직접 지정 시 새 폴더
SHOW_IMAGES = True  # 사진과 점수를 카드로 함께 표시
IMAGES_PER_ROW = 3
IMAGES_PER_PAGE = 12  # 저장용 PNG 한 장에 들어갈 사진 수
SHOW_STROKE_TABLE = False  # 기본은 카드의 '획별 점수 보기'를 눌러 확인

SCORE_CONFIG = dict(
    missing_weight=1.0,   # 누락 중요도. 0이면 이 항 제외
    split_weight=1.0,     # 분리 중요도
    junction_weight=1.0,  # 정션 중요도
    split_scale=0.2,      # 기본 모드: 이 크기에서 분리 최대 감점의 절반
    junction_scale=0.01,  # 기본 모드: 이 크기에서 정션 최대 감점의 절반
    worst_stroke_weight=0.5, # 글자 점수: 획 평균과 최악 획 점수의 혼합. 0=평균, 1=최악
    priority_mode=True,   # 삭제/추가 강한 감점, 정션/분리 제한 감점. False=이전 공식
    split_max_penalty=0.30, # 분리 단독 최대 감점 30점
    junction_max_penalty=0.65, # 정션 단독 최대 감점 65점
    extra_weight=1.0,     # 추가 잉크 감점. 0은 명시적으로 추가 검사를 끄는 설정
    extra_scale=0.025,    # 허용량 초과 2.5%p일 때 추가 점수 상한 100/e
    extra_allowance=0.03, # 정렬 후 남은 작은 잉크 차이 3%까지 감점 유예
    extra_radius=8.0,     # 목표 기준 중심선 주변의 허용 반경 (128 해상도 픽셀)
    extra_transition=0.5, # 허용 영역 경계의 부드러운 전이 폭
    extra_alignment=6,    # 기준 위치를 ±6px 안에서만 이동 허용
    extra_alignment_step=2, # 이동 후보 간격
    extra_alignment_temperature=0.01, # 부드러운 위치 선택 온도
    extra_alignment_prior=0.001, # 큰 이동을 억제하는 비용
    extra_alignment_sigma=2.0, # 위치 일치도를 계산하는 중심선 주변 폭
    eps=1e-6,
)

# MODE="values"일 때 사용. 이미지나 모델 없이 이 숫자만 계산합니다.
VALUES = [
    dict(stroke=1, missing=0.0, split=0.0, junction=0.0),
    dict(stroke=2, missing=0.809, split=0.0, junction=0.0),
]
EXTRA_VALUE = 0.0  # MODE="values": 글자 전체의 추가 잉크 비율을 직접 지정
'''

RUN = '''from _structural_indicator_runtime import IndicatorConfig, score_structure_report
from pathlib import Path

if MODE == "values":
    indicator_source = {"results": [{"input": "manual_values", "strokes": VALUES, "extra": EXTRA_VALUE}]}
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
    show_images=SHOW_IMAGES, images_per_row=IMAGES_PER_ROW,
    images_per_page=IMAGES_PER_PAGE, show_stroke_table=SHOW_STROKE_TABLE,
)
'''


def build():
    source = (ROOT/'stroke_model/structural_indicator.py').read_text(encoding='utf-8-sig')
    extra_source = (ROOT/'stroke_model/extra_ink.py').read_text(encoding='utf-8-sig')
    source = source.replace('from .extra_ink import ExtraInkMetric', 'from _extra_ink_runtime import ExtraInkMetric')
    loader = '''# 점수 모듈 내장: 추가 모델·가중치 다운로드 없이 실행
import sys, types
_extra_module = types.ModuleType("_extra_ink_runtime")
_extra_module.__file__ = "<embedded:extra_ink>"
sys.modules[_extra_module.__name__] = _extra_module
exec(compile(''' + repr(extra_source) + ''', _extra_module.__file__, "exec"), _extra_module.__dict__)
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
    notebook = {'cells': [cell('markdown', '''# 사진 : 누락·분리·정션·추가 잉크 구조 점수

설문·추가 학습 없이 누락/분리/정션과 **기준으로 설명되지 않는 추가 잉크**를 감점합니다.
높을수록 측정 오류가 적습니다. 사람 선호나 글자 품질에 교정한 점수는 아닙니다.

1. 첫 셀에서 입력과 가중치/변환 크기를 설정합니다.
2. 모듈 로드 셀과 계산 셀을 실행합니다.
3. 사진 아래의 **전체 점수**를 비교합니다. 카드의 **획별 점수 보기**를 누르면 세 오류와 획별 점수가 펼쳐집니다.

저장된 NPZ의 실제 평가 입력 배열을 우선 표시하고, 없으면 원본 PNG 경로에서 사진을 읽습니다.
기준 획 이미지나 진단 몽타주를 원본 대신 표시하지 않습니다. 경로가 없으면 이미지 없음으로 알립니다.
사진과 점수를 함께 담은 HTML/PNG도 결과 폴더에 저장합니다.

기존 구조 진단 노트북에 추가 셀을 붙이면 structure_results를 바로 재사용합니다.
이 독립 파일을 새로 열었다면 REPORT_PATH를 결과 JSON 경로로 지정하세요.
MODE="values"는 숫자만 입력하는 테스트입니다. 기존 추출 모델을 재실행하지 않습니다.

기본 모드는 **획 추가·삭제를 강하게, 정션을 중간, 분리를 약하게** 감점하는 평가 규칙입니다.
분리 유지율 = 1 - 분리최대감점 × (wD·D/sD)/(1+wD·D/sD)
정션 유지율 = 1 - 정션최대감점 × (wJ·J/sJ)/(1+wJ·J/sJ)
획 점수 = 100 × (1-M)^wM × 분리 유지율 × 정션 유지율
평균/최악 획 혼합점수와 각 오류별 점수 상한 중 최솟값을 사용합니다.
삭제 상한 = 최악 획의 누락 유지율 × 100. 추가 상한 = 100 × exp(-wE·max(E-허용량,0)/추가scale).
단독 분리/정션은 기본값에서 최대 30/65점 감점이고, 복합 오류는 더 낮아질 수 있습니다.
카드의 '오류별 점수 상한 보기'에서 점수가 낮아진 원인을 확인하세요.

기본값에서 누락 1인 획은 0점, 세 오류가 0인 획은 100점입니다.
각 카드의 '추가 잉크 위치 보기'에서 빨간 영역을 확인하세요. 추가는 획 배정과 무관하게 입력 전체에서 측정합니다.
기준 위치는 제한된 범위에서 부드럽게 정렬합니다. 큰 변형이나 정렬 실패는 추가로 오검출할 수 있습니다.
반경과 허용량은 정상 자료에서 교정해야 합니다. 이름별로 점수나 오류 순서를 강제하지 않습니다.
오래된 JSON도 NPZ의 image/reference_geometry가 있으면 추가를 재측정합니다. 없으면 0으로 가정하지 않고 오류를 냅니다.
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
