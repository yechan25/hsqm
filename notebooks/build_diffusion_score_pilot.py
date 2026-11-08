"""Build a Colab notebook loading the published GitHub runtime."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def cell(kind, source):
        item = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
        if kind == 'code':
            ast.parse(source.strip())
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''# 확산 생성 이미지 → 기존 Topo 추출·원본 점수

새 DDPM을 학습하지 않고 사전학습 Stable Diffusion 1.5 + DDIM으로 시험합니다.
기존 학습 ZIP/TAR·CSV에서 학습 목록을 자동으로 다시 만들고, **train 행의 기준 글자만** 선택합니다.
그 글자에서 출발하는 img2img 실험으로, 텍스트만으로 같은 글자를 생성하는 성능 시험은 아닙니다.
strength 0.25/0.45/0.65, seed 100/101: 글자당 생성물 6장과 기준 대조군 1장.
모든 생성물을 보존합니다. 다른 글자가 됐거나 빈 이미지가 나와도 제외하지 않습니다.

**Colab에서 GPU 런타임을 선택한 다음 위에서 아래로 실행하세요.**
모델과 학습 자료는 기존 Drive 것을 사용합니다. 실행 코드는 GitHub main에서 받습니다.
실제 GPU 생성과 사용자 체크포인트 평가 결과는 이 노트북 실행 후에 얻습니다.
로컬 연결 검사는 실제 확산 생성 품질 검증이 아닙니다.

1. 준비·Drive 연결 → 2. 기존 체크포인트·학습 압축 자료에서 데이터 자동 준비
3. 학습 기준 글자 선택 → 4. 생성 → 5. 기존 추출·점수·진단 ZIP

점수는 폰트 실험과 같은 원본 잉크 기반 점수입니다. Topo 마스크는 함께 표시하지만 수치 점수에 사용하지 않습니다.
이 점수는 사람 평가에 보정된 정확도가 아니므로 원본·획 그림과 함께 판단하세요.
참고: [Diffusers img2img](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img),
[SD 1.5 모델](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).''')
    cell('code', '''
from pathlib import Path
import subprocess, sys, tempfile, json
root = Path(tempfile.mkdtemp(prefix='hsqm_diffusion_score_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
    'https://github.com/yechan25/hsqm.git', str(root)], check=True)
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'diffusers==0.35.1', 'transformers==4.49.0', 'accelerate', 'safetensors',
    'timm', 'pandas', 'numpy', 'tqdm', 'matplotlib', 'pillow', 'scipy',
    'scikit-image', 'pyyaml', 'fonttools', 'ipywidgets'], check=True)
sys.path.insert(0, str(root))
for key in list(sys.modules):
    if key.split('.')[0] in ('stroke_model', 'font_test', 'diffusion_model'):
        del sys.modules[key]
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Colab에서 GPU 런타임으로 바꿔주세요.')
print('GPU:', torch.cuda.get_device_name(0))
print('실행 GitHub 커밋:', commit)
from google.colab import drive, output
drive.mount('/content/drive')
output.enable_custom_widget_manager()
''')
    cell('code', '''
from stroke_model.convert_reference_csv import existing_path
from diffusion_model.score_pilot import training_catalog, generate_pilot, score_pilot
from diffusion_model.pilot_data import prepare_pilot_data
from IPython.display import display
import ipywidgets as widgets
HSQM = existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
CHECKPOINT = None  # 폰트 실험에서 사용한 best.pt의 전체 경로를 넣으면 그 모델 사용
DATA_ROOT = HSQM/'dataset/automated_train_dataset'  # 기존 학습 ZIP/TAR가 있는 폴더
CSV_PATH = None  # None: 체크포인트 옆에 저장된 학습 CSV를 자동 사용
TRAIN_MANIFEST = None  # 기존 세션의 유효한 목록이 있을 때만 선택적으로 지정. 없어도 자동 재생성
SCORE_CONFIG = {}  # 폰트 시험에서 값을 바꿨다면 동일한 설정 입력
SOURCE_CONFIG = {}
STRENGTHS = [.25, .45, .65]
SEEDS = [100, 101]
STEPS = 30
MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
if CHECKPOINT is None:
    candidates = sorted((HSQM/'v2_topology_runs').glob('*/best.pt'))
    if not candidates:
        raise FileNotFoundError('CHECKPOINT에 기존 Topo best.pt를 지정하세요.')
    CHECKPOINT = candidates[-1]
CHECKPOINT = existing_path(CHECKPOINT)
print('사용할 획 모델:', CHECKPOINT, flush=True)
preparation = prepare_pilot_data(CHECKPOINT, DATA_ROOT, csv_path=CSV_PATH, manifest=TRAIN_MANIFEST)
TRAIN_MANIFEST = Path(preparation['manifest'])
print('데이터 준비 완료. 3번 셀에서 기준 글자를 선택하세요.', flush=True)
''')
    cell('code', '''
import io
import numpy as np
from PIL import Image
catalog = training_catalog(TRAIN_MANIFEST)
by_row = {item['manifest_row']: item for item in catalog}
reference_select = widgets.Dropdown(
    options=[(f"train row {item['manifest_row']} | {item['char_id'] or '(문자명 없음: 그림 확인)'}", item['manifest_row']) for item in catalog],
    description='기준 글자:', layout=widgets.Layout(width='98%'))
preview = widgets.Image(format='png', width=200, height=200)
def update_preview(change=None):
    with np.load(by_row[reference_select.value]['path'], allow_pickle=False) as data:
        ink = data['reference'][0]
    buffer = io.BytesIO()
    Image.fromarray(np.uint8((1-ink)*255)).save(buffer, format='PNG')
    preview.value = buffer.getvalue()
reference_select.observe(update_preview, names='value')
update_preview()
display(widgets.VBox([reference_select, preview]))
print('train에 들어 있는 기준만 표시합니다. 먼저 한 글자로 6장 생성해 확인합니다.')
''')
    cell('code', '''
from datetime import datetime
from huggingface_hub import model_info
# 실제 다운로드하는 모델 버전을 커밋 SHA로 고정해 보고서에 기록합니다.
MODEL_REVISION = model_info(MODEL_ID).sha
OUTPUT_DIR = HSQM/'diffusion_score_pilots'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
generation = generate_pilot(TRAIN_MANIFEST, [int(reference_select.value)], OUTPUT_DIR,
    strengths=STRENGTHS, seeds=SEEDS, steps=STEPS, model_id=MODEL_ID, revision=MODEL_REVISION)
generation['data_preparation'] = preparation
(OUTPUT_DIR/'generation_report.json').write_text(json.dumps(generation, ensure_ascii=False, indent=2), encoding='utf-8')
(OUTPUT_DIR/'git_commit.txt').write_text(commit, encoding='utf-8')
print('생성 원본과 점수 입력:', OUTPUT_DIR)
# 함수 밖에 생성 파이프라인을 보관하지 않습니다. 다음 셀에서 GPU 메모리를 회수합니다.
''')
    cell('code', '''
result, scores = score_pilot(OUTPUT_DIR/'generation_report.json', CHECKPOINT,
    score_config=SCORE_CONFIG, source_config=SOURCE_CONFIG, show=True)
display(scores)
print('원본과 추출 그림을 보고 감점이 실제 획 오류와 맞는지 확인하세요.')
print('점수 계산 성공만으로 평가 정확도를 입증하지 않습니다.')
from google.colab import files
download = widgets.Button(description='진단 ZIP 다운로드')
debug_path = result['debug_zip']
download.on_click(lambda _, path=debug_path: files.download(path))
display(download)
''')
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        accelerator='GPU', colab=dict(name='확산_생성_글자_점수_시험.ipynb')),
        nbformat=4, nbformat_minor=5)
    path = ROOT/'notebooks/test_diffusion_scores.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path)
    return path


if __name__ == '__main__':
    build()
