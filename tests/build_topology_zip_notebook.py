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

    add('markdown', '''# Topo 획 추출 모델 학습
GPU 런타임에서 위부터 실행하세요. 입력 글자의 실제 획 정답으로 학습합니다.
전역 affine 정렬 + 동결 Swin-T/FPN + 국소 flow + 양방향 ConvLSTM + 획별 sigmoid.
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
교차점의 겹친 획 정답과 빈 획 슬롯을 보존합니다. CSV는 모든 획의 정답을 제공한다고 가정합니다.
기준/정답 획 순서와 극성은 2번 셀에서 확인하세요.

기본은 ImageNet Swin 사전학습에서 새 학습입니다. 기존 V2 `best.pt`를 지정하면
특징 추출부를 초기화합니다(optimizer/epoch 복원 아님). Topology 성능 개선은 아직 검증 전입니다.
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
from dataclasses import asdict
from stroke_model.reference_topology import TopologyLossConfig, AlignmentConfig
import shutil
EPOCHS = 30
BATCH_SIZE = 4  # 메모리 부족 시 2 또는 1
LEARNING_RATE = 2e-4
INIT_CHECKPOINT = None  # 선택: 같은 해상도/폭의 기존 reference_v2 best.pt
LOSS_CONFIG = asdict(TopologyLossConfig())
ALIGNMENT_CONFIG = asdict(AlignmentConfig())
output = data_root.parents[1]/'topology_runs'/datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
output.mkdir(parents=True, exist_ok=False)
for name in ('conversion_report.json', 'selected_paths.csv', '.complete.json'):
    shutil.copyfile(manifest.parent/name, output/name.lstrip('.'))
(output/'git_commit.txt').write_text(commit)
loss_path = root/'topology_loss.json'; loss_path.write_text(json.dumps(LOSS_CONFIG))
align_path = root/'topology_alignment.json'; align_path.write_text(json.dumps(ALIGNMENT_CONFIG))
cmd = [sys.executable, '-u', '-m', 'stroke_model.train_independent', '--model', 'topology',
       '--manifest', str(manifest), '--output', str(output), '--epochs', str(EPOCHS),
       '--batch-size', str(BATCH_SIZE), '--lr', str(LEARNING_RATE), '--image-size', str(IMAGE_SIZE),
       '--seed', str(SEED), '--loss-config', str(loss_path), '--alignment-config', str(align_path),
       '--no-shuffle-strokes']  # 양방향 획 문맥은 CSV의 의미 순서를 유지
if INIT_CHECKPOINT:
    local_checkpoint = root/'v2_initializer.pt'
    shutil.copyfile(INIT_CHECKPOINT, local_checkpoint)
    cmd += ['--init-checkpoint', str(local_checkpoint)]
(output/'initial_checkpoint_source.txt').write_text(str(INIT_CHECKPOINT))
print('모델: reference_topology_aligned_v1 / 결과:', output, flush=True)
subprocess.run(cmd, cwd=root, check=True)
print('학습 완료:', output/'best.pt')
''')
    add('code', '''# 학습에서 선택한 best.pt의 검증 예측. 독립 test 평가가 아닙니다.
from stroke_model.reference_topology import TopologyStrokeModel, AlignmentConfig
from stroke_model.train_independent import collate_independent, evaluate_independent
from torch.utils.data import DataLoader
checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
model = TopologyStrokeModel(image_size=checkpoint['image_size'], width=checkpoint['width'],
    context_width=checkpoint['context_width'], pretrained=False,
    alignment_config=AlignmentConfig(**checkpoint['alignment_config']))
model.load_trained(checkpoint); model = model.cuda().eval()
validation = IndependentDataset([r for r in rows if r['split']=='val'], checkpoint['image_size'])
metrics = evaluate_independent(model, DataLoader(validation, batch_size=1, collate_fn=collate_independent), 'cuda')
(output/'validation_metrics.json').write_text(json.dumps(metrics, indent=2))
print(metrics)
for index in range(min(5, len(validation))):
    image, reference, strokes, target, valid = validation[index]
    with torch.no_grad():
        prediction = model(image[None].cuda(), reference[None].cuda(), strokes[None].cuda())
    masks = prediction['masks'][0].cpu()
    k = len(strokes)
    fig, axes = plt.subplots(3, k+1, figsize=(2*(k+1), 6), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    axes[0,0].imshow(image[0], cmap='gray', vmin=0, vmax=1); axes[0,0].set_title('Input')
    axes[1,0].imshow(reference[0], cmap='gray', vmin=0, vmax=1); axes[1,0].set_title('Reference')
    for s in range(k):
        for row_index, tensor, title in ((0,target[s],'GT'), (1,masks[s],'Soft'), (2,masks[s]>=.5,'Hard')):
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
