"""Build the GitHub-backed Colab entry point and an optional source-only backup ZIP."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def md(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})
    def code(source):
        compile(source, '<colab>', 'exec')
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': source.splitlines(keepends=True)})
    md('''# A100: 한글 글자 조건부 확산 모델 학습
이 노트북은 **128×128 한 글자용 pixel DDPM을 새로 학습**합니다. AnyText/SD 장면 생성 모델의 미세조정이 아닙니다.
먼저 baseline을 실행한 뒤 같은 초기 checkpoint에서 OCR / 획 보조 학습을 비교합니다.
획 손실은 정상쌍 재구성 감독이며, 아직 검증된 획 오류 점수가 아닙니다. 역확산 중 guidance는 이 버전에 없습니다.

1. Colab 런타임 → GPU → A100 선택.
2. 첫 코드 셀에서 GitHub main의 최신 코드를 자동으로 받습니다.
3. Drive 데이터 경로 확인 후 나머지 셀을 순서대로 실행.
ZIP 업로드는 필요 없습니다. 실행한 Git 커밋이 출력됩니다.
''')
    code('''from pathlib import Path
import sys, subprocess, tempfile
ROOT = Path(tempfile.mkdtemp(prefix='hsqm_diffusion_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/yechan25/hsqm.git', str(ROOT)], check=True)
if not (ROOT/'diffusion_model/requirements.txt').is_file():
    raise FileNotFoundError('저장소에서 확산 학습 코드를 찾지 못했습니다.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(ROOT/'diffusion_model/requirements.txt')], check=True)
sys.path.insert(0, str(ROOT))
for name in list(sys.modules):
    if name in ('diffusion_model', 'stroke_model') or name.startswith(('diffusion_model.', 'stroke_model.')):
        del sys.modules[name]
commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Colab 런타임을 GPU로 바꿔주세요.')
print(torch.cuda.get_device_name(), round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), 'GiB')
print('Code:', ROOT, 'GitHub commit:', commit, 'torch:', torch.__version__)
''')
    md('''## 데이터와 실험 설정
기본은 기존 `paths.csv`를 변환합니다. 데이터가 이미 NPZ manifest라면 `DATA_MODE="manifest"`로 바꿉니다.
폰트 생성은 `DATA_MODE="fonts"`; 서로 다른 한글 폰트 family 최소 2개가 필요합니다. 폰트 모드에서는 획 손실을 사용할 수 없습니다.
동일 원본의 증강·같은 필자가 train/val에 섞이지 않게 원본 CSV의 writer_id/group_id를 확인하세요.
처음에는 `STAGE="baseline"`. `MAX_STEPS=100`은 실행 확인용이고, 생성 품질 판단용이 아닙니다.
''')
    code('''from google.colab import drive
drive.mount('/content/drive')
PROJECT = Path('/content/drive/Shareddrives/2026 자율연구/HSQM')
DATA_MODE = 'legacy_csv'  # legacy_csv / manifest / fonts
CSV_PATH = PROJECT/'dataset/train_dataset/paths.csv'
MANIFEST_PATH = ''  # manifest 모드에서는 기존 manifest.json의 전체 경로 입력
FONT_DIR = PROJECT/'fonts'
CHARACTERS = '가나다라마바사아자차카타파하한글서울연구'
STAGE = 'baseline'  # baseline / ocr / stroke / ocr_stroke
MAX_STEPS = 10000 if STAGE == 'baseline' else 2000
BATCH_SIZE = 8
ACCUMULATION = 2
RUN_NAME = 'glyph_' + STAGE + '_v1'
STROKE_CHECKPOINT = ''  # 획 보조학습 시 학습 완료된 reference_v2 best.pt 전체 경로
BASELINE_CHECKPOINT = PROJECT/'diffusion_runs/glyph_baseline_v1/last.pt'
RESUME = True  # 같은 run의 last.pt가 있으면 optimizer/RNG 포함 재개
if STAGE not in ('baseline', 'ocr', 'stroke', 'ocr_stroke'):
    raise ValueError('지원하지 않는 STAGE')
if DATA_MODE == 'legacy_csv':
    from stroke_model.reference_cache import prepare_cached
    source_manifest = prepare_cached(CSV_PATH, PROJECT/'reference_v2_data')
elif DATA_MODE == 'manifest':
    if not MANIFEST_PATH:
        raise ValueError('MANIFEST_PATH에 기존 manifest.json 경로를 입력하세요.')
    source_manifest = Path(MANIFEST_PATH)
elif DATA_MODE == 'fonts':
    from diffusion_model.prepare_fonts import prepare
    prepared = PROJECT/'diffusion_font_data_v1'
    source_manifest = prepared/'manifest.json'
    if not source_manifest.exists():
        source_manifest = prepare(FONT_DIR, CHARACTERS, prepared, variants=16)
    if 'stroke' in STAGE:
        raise ValueError('폰트 렌더링 데이터에는 개별 획 마스크가 없습니다.')
else:
    raise ValueError('지원하지 않는 DATA_MODE')
from diffusion_model.stage import stage_manifest
manifest = stage_manifest(source_manifest)
print('Source:', source_manifest)
print('Local:', manifest)
''')
    md('''## 학습 전 데이터 확인
검증 이미지와 기준 이미지가 어떤 글자인지 확인합니다. 밝은 배경에 검정 글씨로 표시합니다.
획 보조학습에서는 V2 모델의 출력 단위가 실제 획인지 자모인지 확인하세요. 자모 마스크라면 결과도 자모 단위 감독입니다.
''')
    code('''from diffusion_model.data import GlyphDataset, read_manifest, collate
from diffusion_model.train import save_preview
from IPython.display import display
from PIL import Image
rows = list(read_manifest(manifest))
print({split: sum(r['split']==split for r in rows) for split in ('train', 'val', 'test')})
validation = GlyphDataset([r for r in rows if r['split']=='val'])
preview = collate([validation[i] for i in range(min(4, len(validation)))])
preview_path = ROOT/'data_preview.png'
save_preview(preview_path, preview['reference'], preview['image'], preview['image'])
print('왼쪽=기준 / 가운데·오른쪽=정상 학습 대상, 글자:', preview['text'])
display(Image.open(preview_path))
''')
    md('''## 학습 실행 / 세션 재개
기본 유효 batch는 8×2=16, A100에서는 BF16을 사용합니다. OOM이면 batch를 4 또는 2로 낮추고 새 run으로 시작하세요.
검증 noise MSE는 250 update마다, 미리보기는 500 update마다 저장합니다. last.pt는 500 update·개선·종료 시 저장됩니다.
OCR/획 보조학습은 같은 baseline last.pt에서 새 optimizer로 시작합니다. 세 종류의 보조학습을 각각 비교할 수 있습니다.
Ctrl+C로 중단하면 가장 최근에 완성된 체크포인트에서 다시 시작합니다. 중단된 update는 저장하지 않습니다.
''')
    code('''import importlib.util
if 'ocr' in STAGE and importlib.util.find_spec('easyocr') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'easyocr==1.7.2'], check=True)
output = PROJECT/'diffusion_runs'/RUN_NAME
cmd = [sys.executable, '-u', '-m', 'diffusion_model.train', '--manifest', str(manifest),
       '--output', str(output), '--steps', str(MAX_STEPS), '--batch-size', str(BATCH_SIZE),
       '--accumulation', str(ACCUMULATION)]
if STAGE != 'baseline':
    if 'ocr' in STAGE:
        cmd += ['--ocr-weight', '0.01']
    if 'stroke' in STAGE:
        if not STROKE_CHECKPOINT or not Path(STROKE_CHECKPOINT).is_file():
            raise FileNotFoundError('STROKE_CHECKPOINT에 학습 완료된 V2 checkpoint를 지정하세요.')
        cmd += ['--stroke-weight', '0.1', '--stroke-checkpoint', STROKE_CHECKPOINT]
    cmd += ['--aux-start', '0']
if RESUME and (output/'last.pt').is_file():
    cmd += ['--resume', str(output/'last.pt')]
elif STAGE != 'baseline':
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError('먼저 baseline 학습을 완료하거나 BASELINE_CHECKPOINT를 지정하세요.')
    cmd += ['--init-from', str(BASELINE_CHECKPOINT)]
print('Training:', ' '.join(cmd), flush=True)
with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, bufsize=1) as proc:
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
        result = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
if result:
    raise RuntimeError(f'학습 실패: 종료코드 {result}. 위 로그를 확인하세요.')
print('Saved to:', output)
''')
    md('''## 결과 확인
왼쪽=기준 조건 / 가운데=생성 / 오른쪽=실제 정상 이미지.
`best_weights.pt`는 validation noise MSE로 선택합니다. 획 오류가 가장 적은 모델이라는 뜻은 아닙니다.
동일한 시드의 OCR-only / stroke-only / combined 결과를 비교하고 별도 사람 평가를 수행하세요.
''')
    code('''import json
images = sorted(output.glob('preview_*.png'))
if images:
    display(Image.open(images[-1]))
lines = (output/'metrics.jsonl').read_text().splitlines()
for line in lines[-3:]:
    print(json.loads(line))
print('Inference checkpoint:', output/'best_weights.pt')
''')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'language': 'python',
                'display_name': 'Python 3'}, 'accelerator': 'GPU', 'colab': {'name': 'train_diffusion_a100.ipynb'}},
                'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'diffusion-{i}'
    notebook_path = ROOT/'notebooks/train_diffusion_a100.ipynb'
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    files = sorted((ROOT/'diffusion_model').glob('*.py')) + [ROOT/'diffusion_model/requirements.txt']
    files += sorted((ROOT/'stroke_model').glob('*.py'))
    files += [notebook_path, ROOT/'docs/diffusion_model/README.md', ROOT/'tests/test_diffusion.py',
              ROOT/'tests/smoke_diffusion.py']
    bundle = ROOT/'outputs/diffusion_bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle/notebook_path.name).write_bytes(notebook_path.read_bytes())
    zip_path = bundle/'hsqm_diffusion_a100.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('bundle_sha256.json', json.dumps({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                                          for p in files}, indent=2))
    print(notebook_path)
    print(zip_path)


if __name__ == '__main__':
    main()
