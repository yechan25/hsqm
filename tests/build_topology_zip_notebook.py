"""Build the runnable Colab topology extractor training notebook."""
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

    add('markdown', '''# Topo 모델 학습 — 기존 V2 + 정션 보조 손실
GPU 런타임에서 위부터 실행하세요. 입력 글자의 실제 획 정답으로 학습합니다.
기존 V2의 동결 Swin-T/FPN + 국소 flow + 공유 추출 헤드 + softmax를 그대로 사용합니다.
전역 affine/ConvLSTM/독립 sigmoid는 사용하지 않습니다. 모든 soft 획의 합은 원본 잉크와 같습니다.
정션 손실은 예측과 실제 정답의 구조 차이를 비교합니다. 실제 정답에 있는 가지는 제거 대상이 아닙니다.

데이터: 공유 Drive의 `2026 자율연구/HSQM/dataset/automated_train_dataset` 안
아래와 같은 폴더별 TAR 묶음과 CSV를 사용합니다. 각 TAR은 독립 압축 파일입니다.
```
automated_train_dataset/
  path (1).csv
  input_stroke/  # 폴더와 하위 폴더의 압축 파일 전체
  input/
  target_stroke/
  target/
```
파일명·번호·개수를 제한하지 않고 하위 폴더까지 TAR/ZIP을 전부 찾습니다.
TAR.GZ/TGZ/TAR.BZ2/TAR.XZ도 지원합니다. 기존 네 ZIP 형식도 지원합니다.
`path (1).csv`가 있으면 우선 사용하고, 없으면 `path.csv`/`paths.csv`를 검색합니다.
입력=input, 기준=target, 정답=input_stroke, 기준 획=target_stroke입니다.
얇은 정답은 기존 V2처럼 주석된 픽셀만 CE/Dice로 감독합니다. 그 밖의 원본 잉크를 배경으로 벌주지 않습니다.
실제 정답 획 마스크는 정션 보조 손실에만 사용합니다. CSV는 완전한 실제 획 경로를 제공한다고 가정합니다.
기준 획이 비어 있는 샘플은 전체 행을 제외하고 계속 진행하며 `skipped_samples.csv`에 기록합니다.
기준/정답 획 순서와 극성은 2번 셀에서 확인하세요.

기본은 ImageNet Swin 사전학습에서 새 학습입니다. 기존 V2 `best.pt`를 지정하면
전체 V2를 초기화합니다(optimizer/epoch 복원 아님). 정션 성능 개선과 GPU 속도는 아직 검증 전입니다.
''')
    add('code', '''from pathlib import Path
import json, subprocess, sys, tempfile
root = Path(tempfile.mkdtemp(prefix='hsqm_topology_', dir='/content'))
subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'main',
                'https://github.com/yechan25/hsqm.git', str(root)], check=True)
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(root/'requirements.txt')], check=True)
sys.path.insert(0, str(root))
for key in list(sys.modules):
    if key == 'stroke_model' or key.startswith('stroke_model.'):
        del sys.modules[key]
import torch
if not torch.cuda.is_available():
    raise RuntimeError('런타임 유형을 GPU로 변경한 뒤 다시 실행하세요.')
print('GPU:', torch.cuda.get_device_name(), '\\n코드 버전:', commit)
''')
    add('code', '''from google.colab import drive
drive.mount('/content/drive')
from stroke_model.prepare_stroke_zips import prepare_stroke_zips
from stroke_model.convert_reference_csv import existing_path
from stroke_model.train_reference import read_manifest
from stroke_model.train_independent import IndependentDataset
import matplotlib.pyplot as plt
DATA_ROOT = '/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/automated_train_dataset'
CSV_PATH = None  # 직접 지정 가능. 기본은 path (1).csv 우선, 그다음 path.csv/paths.csv.
IMAGE_SIZE = 128
SEED = 42
data_root = existing_path(DATA_ROOT)
if CSV_PATH is None:
    try:
        CSV_PATH = existing_path(data_root/'path (1).csv')
    except FileNotFoundError:
        pass
manifest = prepare_stroke_zips(data_root, csv_path=CSV_PATH, size=IMAGE_SIZE, seed=SEED,
                               independent=True)
report = json.loads((manifest.parent/'conversion_report.json').read_text(encoding='utf-8'))
print('선택 CSV:', report['csv'], '\\n컬럼:', report['columns'])
print(report['label_policy'], '\\n검증 분리:', report['split_note'])
print('빈 기준 획으로 제외:', report['skipped_count'], '개 / 목록:', manifest.parent/'skipped_samples.csv')
rows = read_manifest(manifest)
print('train:', sum(r['split']=='train' for r in rows), 'val:', sum(r['split']=='val' for r in rows))
dataset = IndependentDataset(rows, IMAGE_SIZE)
for index in range(min(3, len(dataset))):
    image, reference, strokes, target, valid = dataset[index]
    k = len(strokes)
    fig, axes = plt.subplots(2, k+2, figsize=(2*(k+2), 4), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    axes[0,0].imshow(image[0], cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('Input')
    axes[1,0].imshow(reference[0], cmap='gray', vmin=0, vmax=1); axes[1,0].set_title('Reference')
    for s in range(k):
        axes[0,s+1].imshow(target[s], cmap='gray', vmin=0, vmax=1); axes[0,s+1].set_title(f'Actual GT {s+1}')
        axes[1,s+1].imshow(strokes[s], cmap='gray', vmin=0, vmax=1); axes[1,s+1].set_title(f'Ref stroke {s+1}')
    axes[0,-1].imshow(target.sum(0), vmin=0, vmax=max(2,k)); axes[0,-1].set_title('GT overlap count')
    plt.tight_layout(); plt.show()
''')
    add('code', '''# 2번 셀의 획 순서와 실제 정답을 확인한 뒤 실행하세요.
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
EPOCHS = 30
BATCH_SIZE = 8  # 기존 V2와 동일. 메모리 부족 시 4 또는 2
LEARNING_RATE = 2e-4
INIT_CHECKPOINT = None  # 선택: 같은 해상도/폭의 기존 reference_v2 best.pt
JUNCTION_WEIGHT = 0.05  # 0이면 기존 V2와 같은 모델/기본 손실만 사용
JUNCTION_EVERY = 4  # 4배치마다 정션 항 계산
JUNCTION_MAX_STROKES = 4  # 적용 배치마다 전체 획 중 최대 4개를 순환 선택
output = data_root.parents[1]/'v2_topology_runs'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
output.mkdir(parents=True, exist_ok=False)
for name in ('conversion_report.json', 'selected_paths.csv', 'skipped_samples.csv', '.complete.json'):
    shutil.copyfile(manifest.parent/name, output/name.lstrip('.'))
(output/'git_commit.txt').write_text(commit)
cmd = [sys.executable, '-u', '-m', 'stroke_model.train_reference',
       '--manifest', str(manifest), '--output', str(output), '--epochs', str(EPOCHS),
       '--batch-size', str(BATCH_SIZE), '--lr', str(LEARNING_RATE), '--image-size', str(IMAGE_SIZE),
       '--seed', str(SEED), '--num-workers', '2',
       '--junction-weight', str(JUNCTION_WEIGHT), '--junction-every', str(JUNCTION_EVERY),
       '--junction-max-strokes', str(JUNCTION_MAX_STROKES)]
if INIT_CHECKPOINT:
    local_checkpoint = root/'v2_initializer.pt'
    shutil.copyfile(INIT_CHECKPOINT, local_checkpoint)
    cmd += ['--init-checkpoint', str(local_checkpoint)]
(output/'initial_checkpoint_source.txt').write_text(str(INIT_CHECKPOINT))
print('모델: 기존 V2 + 정션 보조 손실 / 결과:', output, flush=True)
from IPython.display import display
live = display({'text/plain': '학습 프로세스 시작 중...'}, raw=True, display_id=True)
print('진행률: epoch, 배치 수, 경과/예상 남은 시간, loss, lr (약 5초마다 갱신)')
print('전체 실행 로그:', output/'training.log')
with (output/'training.log').open('w', encoding='utf-8') as log:
    with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', bufsize=1) as proc:
        try:
            for line in proc.stdout:
                log.write(line); log.flush()
                if line.strip():
                    live.update({'text/plain': line.rstrip()}, raw=True)
                if line.lstrip().startswith('{'):
                    print(line.rstrip(), flush=True)  # epoch별 결과도 출력에 보존
            returncode = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            raise
if returncode:
    raise subprocess.CalledProcessError(returncode, cmd)
print('학습 완료:', output/'best.pt')
''')
    add('code', '''# 학습에서 선택한 best.pt의 검증 예측. 독립 test 평가가 아닙니다.
from stroke_model.reference_model import ReferenceStrokeModel, exclusive_masks
from stroke_model.train_reference import StrokeDataset, collate, evaluate
from torch.utils.data import DataLoader
checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
model = ReferenceStrokeModel(image_size=checkpoint['image_size'], width=checkpoint['width'], pretrained=False)
model.load_state_dict(checkpoint['model'], strict=True); model = model.cuda().eval()
validation = StrokeDataset([r for r in rows if r['split']=='val'], checkpoint['image_size'])
metrics = {'val_macro_dice_annotated': evaluate(model, DataLoader(validation, batch_size=1, collate_fn=collate), 'cuda')}
(output/'validation_metrics.json').write_text(json.dumps(metrics, indent=2))
print(metrics)
for index in range(min(5, len(validation))):
    image, reference, strokes, labels = validation[index]
    with torch.no_grad():
        prediction = model(image[None].cuda(), reference[None].cuda(), strokes[None].cuda())
    masks = prediction['masks'][0].cpu()
    hard = exclusive_masks(prediction, (image[None].cuda() > .5))[0].cpu()
    k = len(strokes)
    fig, axes = plt.subplots(3, k+1, figsize=(2*(k+1), 6), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    axes[0,0].imshow(image[0], cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('Input')
    axes[1,0].imshow(reference[0], cmap='gray', vmin=0, vmax=1); axes[1,0].set_title('Reference')
    for s in range(k):
        for row_index, tensor, title in ((0,labels==s,'Annotation'), (1,masks[s],'Soft'), (2,hard[s],'Exclusive')):
            axes[row_index,s+1].imshow(tensor, cmap='gray', vmin=0, vmax=1)
            axes[row_index,s+1].set_title(f'{title} {s+1}')
    plt.tight_layout(); fig.savefig(output/f'val_{index:03d}.png', dpi=120); plt.show()
''')
    notebook = dict(cells=cells, metadata=dict(colab=dict(name='train_topology_zips.ipynb'),
        accelerator='GPU', kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        nbformat=4, nbformat_minor=5)
    (ROOT/'notebooks/train_topology_zips.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
