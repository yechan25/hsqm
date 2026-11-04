"""Build Topo extraction + original-source score notebook."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[]
    def add(kind,source):
        if kind=='code':compile(source,'<colab>','exec')
        cell=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# Topo 획 추출 + 원본 구조 점수
기존 점수 공식은 그대로 사용합니다. 점수는 **원본 입력 잉크**의 누락·분리·정션·추가 잉크에서
계산하고, 같은 입력의 Topo soft/exclusive 획을 함께 표시합니다. 추출 파임/혼입을 글씨 감점에 넣지 않습니다.
추출 결과가 달라져도 원본/기준/점수 설정이 같으면 원본 점수는 같습니다.
이 점수는 기존 규칙 기반 구조 지표이며 사람 평가에 보정된 학습 모델 점수는 아닙니다.

1. 준비/Drive 연결
2. 모델·CSV·점수 설정
3. modified 전체 추출 + 원본 점수 (INPUT_PATTERN=None이면 일반 입력 선택)
4. 캔버스 추출 + 원본 점수 (첫 두 코드 셀 다음 바로 실행 가능)

재학습하거나 best.pt를 수정하지 않습니다. 점수 정렬/측정은 별도 연산이므로 순수 추론보다 시간이 걸릴 수 있습니다.
결과는 checkpoint 옆 새 topo_scores_시간 폴더에 PNG/글자·획별 CSV/JSON/진단 ZIP으로 저장합니다.
''')
    add('code','''from pathlib import Path
import json, subprocess, sys, tempfile
root=Path(tempfile.mkdtemp(prefix='hsqm_topo_score_',dir='/content'))
subprocess.run(['git','clone','--depth','1','--branch','main','https://github.com/yechan25/hsqm.git',str(root)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'requirements.txt')],check=True)
sys.path.insert(0,str(root))
for key in list(sys.modules):
    if key=='stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
print('코드 버전:',subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip())
from google.colab import drive
drive.mount('/content/drive')
''')
    add('code','''import torch
from stroke_model.convert_reference_csv import existing_path
from stroke_model.reference_v2_topology import ARCHITECTURE
from stroke_model.topology_scoring import run_scored_topology_csv, run_scored_topology_canvas

HSQM=existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
CHECKPOINT=None  # None: v2_topology_runs 최신 best.pt. 특정 After 모델은 경로 지정
DATA_ROOT=HSQM/'dataset/test_dataset'
CSV_PATH=None
DATASET_ROOT=None
INPUT_FILES=None
SAMPLE_ROWS=None
INPUT_PATTERN='*modified*.png'  # None이면 일반 입력 포함
SAMPLE_COUNT=8  # 패턴 모드는 제한 없이 전체. 패턴이 None일 때만 적용
SCORE_CONFIG={}  # 기존 IndicatorConfig 기본값. 필요한 가중치만 명시적으로 덮어쓰기
SOURCE_CONFIG={}  # 기존 원본 잉크 정렬/측정 기본값. 예: {'registration_iterations':45}
EXPORT_DEBUG=True

if CHECKPOINT is None:
    candidates=sorted((HSQM/'v2_topology_runs').glob('*/best.pt'),key=lambda p:p.parent.name)
    if not candidates:raise FileNotFoundError('학습한 Topo best.pt를 CHECKPOINT에 지정하세요.')
    CHECKPOINT=candidates[-1]
state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
if state.get('architecture')!=ARCHITECTURE:
    raise ValueError('V2 기반 Topo best.pt를 지정하세요.')
print('사용 모델:',CHECKPOINT,'/ epoch:',state.get('epoch'))
print('학습 감독:',{k:state.get(k) for k in ('ownership_weight','affinity_weight','seed_focal_weight')})
del state
if CSV_PATH is None:
    data_root=existing_path(DATA_ROOT)
    candidates=[p for p in data_root.iterdir() if p.is_file() and p.name.casefold() in ('path.csv','paths.csv','path (1).csv')]
    if len(candidates)!=1:raise ValueError(f'CSV_PATH를 지정하세요. 후보: {candidates}')
    CSV_PATH=candidates[0]
print('테스트 CSV:',CSV_PATH)
print('점수 대상: 원본 잉크 / Topo 추출 마스크는 함께 표시만 합니다.')

def debug_download_button(report):
    path=report.get('debug_zip')
    if not path:return
    try:
        import ipywidgets as widgets
        from IPython.display import display
        from google.colab import files
        button=widgets.Button(description='진단 ZIP 다운로드')
        button.on_click(lambda _:files.download(path))
        display(button)
    except ImportError:
        print('진단 ZIP:',path)
''')
    add('code','''csv_score_report=run_scored_topology_csv(
    CHECKPOINT,CSV_PATH,dataset_root=DATASET_ROOT or DATA_ROOT,
    input_files=INPUT_FILES,sample_rows=SAMPLE_ROWS,input_pattern=INPUT_PATTERN,max_samples=SAMPLE_COUNT,
    score_config=SCORE_CONFIG,source_config=SOURCE_CONFIG,export_debug=EXPORT_DEBUG)
debug_download_button(csv_score_report)
''')
    add('code','''# 캔버스: 첫 두 코드 셀 실행 후 바로 이 셀로 넘어와도 됩니다.
canvas_score_report=run_scored_topology_canvas(
    CHECKPOINT,CSV_PATH,dataset_root=DATASET_ROOT or DATA_ROOT,
    canvas_size=384,pen_width=18,score_config=SCORE_CONFIG,source_config=SOURCE_CONFIG,
    export_debug=EXPORT_DEBUG)
debug_download_button(canvas_score_report)
''')
    notebook=dict(cells=cells,metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5)
    (ROOT/'notebooks/topology_with_scores.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
