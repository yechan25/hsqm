"""Build the inference-only Topo test notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def add(kind, source):
        if kind == 'code':
            compile(source, '<colab>', 'exec')
        cell = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# Topo 모델 테스트 — 재학습 없음
학습된 Topo `best.pt`를 그대로 불러와 CSV의 입력 글자를 획별로 추출합니다.
기존 V2/BiFPN 체크포인트는 사용할 수 없습니다. GPU 런타임을 권장합니다.
입력 글자·기준 글자·기준 획만 사용하며 `input_stroke_paths`는 읽지 않습니다.
기본은 `HSQM/dataset/test_dataset`. 이미지가 CSV 폴더 밖에 있으면 DATASET_ROOT를 지정하세요.

첫 그림: 원본 / 기준 / soft 합집합 / hard 합집합 / 획별 색상 / 미추출 원본 잉크(빨강).
다음 그림: 획마다 기준 획 / 실제 soft 예측 / hard 예측.
출력을 굵히거나 원본 잉크를 강제로 배정하지 않습니다. 확률·마스크 표시 범위는 항상 0~1입니다.
`ink coverage`는 원본 잉크를 얼마나 덮는지 나타내는 진단값이며 획 분리 정확도나 품질 점수가 아닙니다.
학습에 사용한 이미지로 테스트하면 독립 test 평가가 아닙니다.
''')
    add('code', '''from pathlib import Path
import subprocess, sys, tempfile
root = Path(tempfile.mkdtemp(prefix='hsqm_topo_test_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/yechan25/hsqm.git', str(root)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root/'requirements.txt')], check=True)
sys.path.insert(0, str(root))
for key in list(sys.modules):
    if key == 'stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
print('코드 버전:', subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
from google.colab import drive
drive.mount('/content/drive')
''')
    add('code', '''from stroke_model.convert_reference_csv import existing_path
from stroke_model.infer_topology import run_topology_inference
HSQM = existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
CHECKPOINT = None  # None: topology_runs의 최신 실행 best.pt. 특정 모델은 전체 경로 입력.
DATA_ROOT = HSQM/'dataset/test_dataset'
CSV_PATH = None  # None: 위 폴더의 path (1).csv / path.csv / paths.csv 중 하나를 선택
DATASET_ROOT = None  # 이미지가 DATA_ROOT 밖이면 이미지들을 포함하는 상위 폴더 지정
INPUT_FILES = None  # 예: ['I_아_2.png', 'I_야_2.png']. CSV에 있는 입력 파일명 지정
SAMPLE_ROWS = None  # 또는 예: [1, 100, 300]. 헤더 제외 1부터 시작. INPUT_FILES와 동시 지정 불가
PREVIEW_COUNT = 8  # 둘 다 None이면 기준 글자별로 골고루 선택. None이면 CSV 전체 처리
THRESHOLD = 0.5

if CHECKPOINT is None:
    candidates = sorted((HSQM/'topology_runs').glob('*/best.pt'), key=lambda p: p.parent.name)
    if not candidates:
        raise FileNotFoundError('Topo best.pt가 없습니다. CHECKPOINT에 학습 결과 경로를 지정하세요.')
    CHECKPOINT = candidates[-1]
if CSV_PATH is None:
    data_root = existing_path(DATA_ROOT)
    candidates = [p for p in data_root.iterdir() if p.is_file() and p.name.casefold() in ('path (1).csv', 'path.csv', 'paths.csv')]
    if len(candidates) != 1:
        raise ValueError(f'CSV 후보: {candidates}. CSV_PATH에 사용할 파일을 지정하세요.')
    CSV_PATH = candidates[0]
print('불러올 Topo 모델:', CHECKPOINT)
print('테스트 CSV:', CSV_PATH)
''')
    add('code', '''report = run_topology_inference(
    CHECKPOINT, CSV_PATH, dataset_root=DATASET_ROOT or DATA_ROOT,
    input_files=INPUT_FILES, sample_rows=SAMPLE_ROWS,
    max_samples=PREVIEW_COUNT, threshold=THRESHOLD)
print('사용한 체크포인트 epoch:', report['checkpoint_epoch'])
print('완료:', len(report['results']), '/ 빈 기준 획 제외:', len(report['skipped']))
''')
    notebook = dict(cells=cells, metadata=dict(accelerator='GPU',
        colab=dict(name='test_topology.ipynb'),
        kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/test_topology.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
