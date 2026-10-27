"""Build the GitHub-backed stroke ZIP Colab notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))

    def code(text):
        compile(text, '<colab>', 'exec')
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True)))

    md('''# ZIP 데이터로 기존 기준 획 모델 V2 학습
GPU 런타임에서 위부터 실행하세요. 첫 셀에서 GitHub main의 최신 코드를 가져오고 실행 커밋을 출력합니다.
Swin 사전학습 가중치와 패키지는 다운로드합니다.

기본 데이터 폴더: 공유 Drive의 `2026 자율연구/HSQM/dataset/automated_train_dataset`.
`input_stroke.zip`, `input.zip`, `target_stroke.zip`, `target.zip`을 로컬로 가져옵니다.
CSV는 같은 폴더 또는 ZIP 안의 path.csv/paths.csv를 찾습니다. CSV 획 목록은 동일한 의미 순서여야 합니다.
CSV가 아직 없으면 업로드 후 2번 셀만 재실행하세요. 파일명만 보고 정답 대응을 추측하지 않습니다.

기본은 기존 V2 구조 + Swin 사전학습 가중치로 새 학습입니다. 기존 V2 best.pt에서 시작하려면 INIT_CHECKPOINT를 지정하세요.
이는 가중치 초기화이며 중단된 optimizer/epoch를 복원하는 resume는 아닙니다.
변환 데이터는 /content에만 두고 체크포인트/기록은 Drive에 저장합니다.
런타임이 삭제되면 데이터 복사·변환이 다시 필요합니다. 입력 이미지 그룹으로 80/20 분리하며,
CSV에 writer_id/group_id가 없다면 필자 단위 일반화를 보장하지 않습니다.
''')
    code('''from pathlib import Path
import json, subprocess, sys, tempfile
root = Path(tempfile.mkdtemp(prefix="hsqm_zip_code_", dir="/content"))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/yechan25/hsqm.git', str(root)], check=True)
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")], check=True)
sys.path.insert(0, str(root))
for key in list(sys.modules):
    if key == "stroke_model" or key.startswith("stroke_model."):
        del sys.modules[key]
import torch
if not torch.cuda.is_available():
    raise RuntimeError("런타임 → 런타임 유형 변경에서 GPU를 선택한 뒤 다시 실행하세요.")
print("GPU:", torch.cuda.get_device_name(), "GitHub 커밋:", commit)
''')
    code('''from google.colab import drive
drive.mount('/content/drive')
from stroke_model.prepare_stroke_zips import prepare_stroke_zips
DATA_ROOT = '/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/automated_train_dataset'
CSV_PATH = None  # path.csv/paths.csv 자동 검색. 후보가 여러 개이거나 다른 이름이면 직접 지정.
IMAGE_SIZE = 128
SEED = 42
manifest = prepare_stroke_zips(DATA_ROOT, csv_path=CSV_PATH, size=IMAGE_SIZE, seed=SEED)
report = json.loads((manifest.parent / 'conversion_report.json').read_text(encoding='utf-8'))
print('CSV 컬럼 대응:', report['columns'])
print('검증 분리:', report['split_note'])
from stroke_model.preview_reference import preview_reference
preview_reference(manifest, count=5)
print('기준 획과 정답 획의 순서, 입력/기준 역할이 맞는지 위 그림을 확인하세요.')
''')
    code('''# 미리보기 확인 후 실행하세요. ZIP 데이터는 이미 합성됐으므로 추가 증강은 기본 사용하지 않습니다.
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
EPOCHS = 30
BATCH_SIZE = 8  # CUDA 메모리 부족 시 4 또는 2
INIT_CHECKPOINT = None  # 예: '/content/drive/.../best.pt'; 같은 image_size의 reference_v2만 허용
output = Path(DATA_ROOT).parents[1] / 'stroke_zip_runs' / datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
output.mkdir(parents=True, exist_ok=False)
for name in ('conversion_report.json', 'selected_paths.csv', '.complete.json'):
    shutil.copyfile(manifest.parent / name, output / name.lstrip('.'))
shutil.copyfile(root / 'stroke_model/reference_model.py', output / 'reference_model.py')
(output / 'git_commit.txt').write_text(commit)
cmd = [sys.executable, '-u', '-m', 'stroke_model.train_reference',
       '--manifest', str(manifest), '--output', str(output), '--epochs', str(EPOCHS),
       '--batch-size', str(BATCH_SIZE), '--num-workers', '2',
       '--image-size', str(IMAGE_SIZE), '--seed', str(SEED)]
if INIT_CHECKPOINT:
    # Copy once so model loading also avoids repeated Drive access.
    local_checkpoint = Path('/content/hsqm_init_checkpoint.pt')
    shutil.copyfile(INIT_CHECKPOINT, local_checkpoint)
    cmd += ['--init-checkpoint', str(local_checkpoint)]
(output / 'initial_checkpoint_source.txt').write_text(str(INIT_CHECKPOINT))
print('학습 데이터: Colab 로컬 / 결과:', output, flush=True)
with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
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
    raise RuntimeError(f'학습 실패: 종료 코드 {result}. 위 로그를 확인하세요.')
print('가장 좋은 검증 모델:', output / 'best.pt')
''')
    code('''preview_reference(manifest, checkpoint=output / 'best.pt', count=5)
history = json.loads((output / 'history.json').read_text())
print('최고 검증 Dice (주석된 전경 픽셀):', max(row['val_macro_dice'] for row in history))
print('겹침/미할당 픽셀은 제외되므로 conversion_report의 annotation_coverage와 함께 해석하세요.')
''')
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    colab=dict(name='train_stroke_zips.ipynb'), accelerator='GPU'), nbformat=4, nbformat_minor=5)
    destination = ROOT / 'notebooks/train_stroke_zips.ipynb'
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
