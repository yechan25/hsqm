"""Build a diagnostic notebook; no training or checkpoint mutation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    cells=[]
    def add(kind,text):
        if kind=='code':compile(text,'<colab>','exec')
        cell=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
        if kind=='code':cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# Topo 튜닝 전후 진단 — 같은 입력으로 비교
재학습하지 않고 두 best.pt를 읽습니다. 원본 파일은 바꾸지 않고 새 comparison 폴더에 저장합니다.
튜닝 후는 최신 v2_topology_runs, 튜닝 전은 그 실행의 initial_checkpoint_source.txt에서 찾습니다.
자동 확인이 안 되면 BEFORE_CHECKPOINT에 튜닝 전의 정확한 경로를 지정하세요.

기본은 modified 전체입니다. 전에 저장한 캔버스를 비교하려면 INPUT_IMAGE에 그 input.png 경로를
넣고 REFERENCE_ROW에 해당 기준 글자의 CSV 행 번호를 지정하세요. 다시 그리지 않습니다.
4·5번의 soft/exclusive를 전후 동일 스케일로 비교합니다. 변경된 픽셀 수나 성분 수는 정확도 점수가 아닙니다.
exclusive의 흰색은 높은 확신을 뜻하지 않습니다. winner margin은 1위와 2위 확률 차이입니다.

보조 주석 진단 셀은 실제로 저장된 학습용 주석만 확인합니다. 학습 런타임의 manifest/NPZ/캐시가
남아 있어야 하며, 없으면 그 사실을 표시합니다. 새 캔버스에 학습 정답이 있다고 가정하지 않습니다.
맨 마지막 캔버스 셀에서는 새로 한 번 그린 그림을 두 모델에 동시에 적용합니다.
준비/설정 코드 셀만 실행한 뒤 캔버스 셀로 바로 넘어갈 수 있습니다.
''')
    add('code','''from pathlib import Path
import json, subprocess, sys, tempfile
root=Path(tempfile.mkdtemp(prefix='hsqm_compare_',dir='/content'))
subprocess.run(['git','clone','--depth','1','--branch','main','https://github.com/yechan25/hsqm.git',str(root)],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(root/'requirements.txt')],check=True)
sys.path.insert(0,str(root))
for key in list(sys.modules):
    if key=='stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
from google.colab import drive
drive.mount('/content/drive')
''')
    add('code','''from stroke_model.convert_reference_csv import existing_path
from stroke_model.compare_topology import find_before_checkpoint, compare_checkpoints, inspect_training_supervision
HSQM=existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
AFTER_CHECKPOINT=None  # None: 최신 v2_topology_runs의 best.pt
BEFORE_CHECKPOINT=None # None: 위 모델을 초기화했던 원본 경로. 자동 확인 실패 시 직접 지정
DATA_ROOT=HSQM/'dataset/test_dataset'
CSV_PATH=None
DATASET_ROOT=None
SAMPLE_ROWS=None       # 예: [1,100]. 지정하면 패턴보다 우선
INPUT_PATTERN='*modified*.png'  # None이면 CSV 전체
INPUT_IMAGE=None       # 예: '.../canvas_실행시간/input.png'. 저장한 같은 그림을 두 모델에 적용
REFERENCE_ROW=1        # INPUT_IMAGE 사용 시 기준으로 삼을 CSV 행, 헤더 제외 1부터
FOCUS_STROKES=[4,5]    # None이면 모든 획
TRAIN_MANIFEST=None   # 주석 진단용. 학습 코랩 run_config에 기록된 로컬 manifest 자동 확인
TRAIN_ROWS=None       # None: 보조 주석이 저장된 train 그룹을 최대 3개 선택. 또는 [123,...]
TRAIN_INPUT_PATTERN=None  # 예: '*초*'. 학습 원본 파일명으로 목록/자동 선택 필터링

if AFTER_CHECKPOINT is None:
    candidates=sorted((HSQM/'v2_topology_runs').glob('*/best.pt'),key=lambda p:p.parent.name)
    if not candidates:raise FileNotFoundError('튜닝 후 best.pt를 지정하세요.')
    AFTER_CHECKPOINT=candidates[-1]
if BEFORE_CHECKPOINT is None:
    BEFORE_CHECKPOINT=find_before_checkpoint(AFTER_CHECKPOINT)
if CSV_PATH is None:
    data_root=existing_path(DATA_ROOT)
    candidates=[p for p in data_root.iterdir() if p.is_file() and p.name.casefold() in ('path.csv','paths.csv','path (1).csv')]
    if len(candidates)!=1:raise ValueError(f'CSV_PATH를 지정하세요. 후보: {candidates}')
    CSV_PATH=candidates[0]
if TRAIN_MANIFEST is None:
    config_path=Path(AFTER_CHECKPOINT).parent/'run_config.json'
    if config_path.is_file():
        candidate=Path(json.loads(config_path.read_text())['manifest'])
        if candidate.is_file():TRAIN_MANIFEST=candidate
print('Before:',BEFORE_CHECKPOINT)
print('After :',AFTER_CHECKPOINT)
print('학습 주석 manifest:',TRAIN_MANIFEST)
''')
    add('code','''comparison_report=compare_checkpoints(
    BEFORE_CHECKPOINT,AFTER_CHECKPOINT,CSV_PATH,dataset_root=DATASET_ROOT or DATA_ROOT,
    sample_rows=SAMPLE_ROWS,input_pattern=INPUT_PATTERN,input_image=INPUT_IMAGE,
    reference_row=REFERENCE_ROW,focus_strokes=FOCUS_STROKES)
print('비교한 이미지:',len(comparison_report['results']))
print('픽셀 이동 표의 before_4 → after_5는 4번에서 5번으로 바뀐 픽셀입니다. 개선/악화 판정은 아닙니다.')
''')
    add('code','''# 선택 사항: 실제 학습 보조 주석을 확인. 테스트/캔버스의 가짜 정답은 만들지 않습니다.
if TRAIN_MANIFEST is None:
    print('학습 런타임의 manifest/NPZ/보조 주석 캐시가 현재 런타임에 없습니다.')
    print('이 셀은 학습했던 런타임에서 실행하거나 TRAIN_MANIFEST를 지정해야 합니다.')
else:
    import pandas as pd, fnmatch
    from IPython.display import display
    training_rows=json.loads(Path(TRAIN_MANIFEST).read_text())
    candidates=[(i+1,r) for i,r in enumerate(training_rows) if r['split']=='train'
        and (TRAIN_INPUT_PATTERN is None or fnmatch.fnmatchcase(
            str(r.get('source_image',r['path'])).replace('\\\\','/').rsplit('/',1)[-1].casefold(),TRAIN_INPUT_PATTERN.casefold()))]
    display(pd.DataFrame([{'manifest_row':n,'source':r.get('source_image',r['path'])} for n,r in candidates]).head(30))
    audit_rows=TRAIN_ROWS
    if audit_rows is None:
        audit_rows=[];seen=set()
        for n,r in candidates:
            if r.get('ownership_path') and r['group_id'] not in seen:
                audit_rows.append(n);seen.add(r['group_id'])
            if len(audit_rows)>=3:break
    supervision_report=inspect_training_supervision(
        TRAIN_MANIFEST,sample_rows=audit_rows,
        save_dir=Path(AFTER_CHECKPOINT).parent/'training_supervision_audit')
''')
    add('code','''# 캔버스: 준비/설정(첫 2개 코드 셀) 후 이 셀로 바로 넘어와도 됩니다.
from stroke_model.compare_topology import compare_canvas
canvas_comparison_report=compare_canvas(
    BEFORE_CHECKPOINT,AFTER_CHECKPOINT,CSV_PATH,
    dataset_root=DATASET_ROOT or DATA_ROOT,
    focus_strokes=FOCUS_STROKES,canvas_size=384,pen_width=18)
print('저장한 원본 캔버스:',canvas_comparison_report['canvas_input'])
''')
    notebook=dict(cells=cells,metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        nbformat=4,nbformat_minor=5)
    (ROOT/'notebooks/compare_topology.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
