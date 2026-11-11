"""Build standalone L4 per-stroke adaptive constraints and live extraction demo."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def add(kind, source):
        if kind == 'code': ast.parse(source)
        cell = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
        if kind == 'code': cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    add('markdown', '''# 획별 tolerance + 주기적 획 재추출 → 국소 역확산 (L4)

1 → 2 → 3 → 4 실행. 이번 버전은 실제 획 추출기를 매 3단계마다 다시 실행합니다.
필요 자료: 원래 `sweep_20261111_062740_406072.zip`과 그 실험의 획 추출 체크포인트.
이전 ink_constraint 결과 ZIP에는 획별 기준과 모델이 없어 그것만으로는 실행할 수 없습니다.
2번에서 Drive를 연결하여 찾거나, ZIP·체크포인트를 직접 올릴 수 있습니다.
모델은 sweep ZIP에 기록된 SHA256과 정확히 일치해야 합니다. 새 학습은 없습니다.

기준 획 ID/초기 정합 경로/폭은 고정, 현재 이미지의 softmax 소속은 주기적으로 재추출합니다.
허용 여유는 정상 몸통의 굵기에 따라 획별로 다르게 정합니다.
정상 경로 보호 + 오류 영역 확장 + 이전 삭제 영역 유지. candidate RGB 일치 손실은 없습니다.
기본은 삭제만, 추가는 ENABLE_RESTORE=True. 마스크·폭 표를 확인하고 4번을 실행하세요.
재추출은 clean 예측에 no_grad로 적용합니다. 확산 및 추출기는 학습하지 않습니다.
''')
    helpers = {p: (ROOT/p).read_text(encoding='utf-8') for p in (
        'diffusion_model/fitted_candidate_diffusion.py', 'stroke_model/adaptive_ink_constraints.py',
        'stroke_model/reference_model.py', 'stroke_model/stroke_reference_fit.py')}
    add('code', '''import sys,subprocess,json,hashlib,io,zipfile
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','diffusers==0.37.0',
    'transformers==4.57.6','accelerate','safetensors','timm','numpy','pillow',
    'matplotlib','scipy','scikit-image'],check=True)
'''+'RUNTIME_HELPERS='+repr(helpers)+'''
ROOT=Path('/content/hsqm_adaptive_runtime');ROOT.mkdir(exist_ok=True)
for package in ('diffusion_model','stroke_model'):
    (ROOT/package).mkdir(exist_ok=True);(ROOT/package/'__init__.py').touch()
for name,code in RUNTIME_HELPERS.items():
    (ROOT/name).write_text(code,encoding='utf-8')
sys.path.insert(0,str(ROOT))
# Cell 1 can be rerun after a different notebook in the same runtime.
for name in list(sys.modules):
    if name.split('.')[0] in ('diffusion_model','stroke_model'):sys.modules.pop(name)
import torch,numpy as np,matplotlib.pyplot as plt
from PIL import Image
from google.colab import files
from diffusion_model.fitted_candidate_diffusion import load_case,image_tensor,run_pair
from stroke_model.adaptive_ink_constraints import load_stroke_bundle,checkpoint_hash,AdaptiveInkConstraints
from stroke_model.reference_model import ReferenceStrokeModel
if not torch.cuda.is_available():raise RuntimeError('런타임에서 L4 GPU를 선택하세요.')
print('GPU:',torch.cuda.get_device_name(0))
''')
    add('code', '''USE_DRIVE=True
ZIP_PATH=''  # 기존 sweep ZIP. 이전 ink_constraint/fitted_reverse 결과 ZIP 아님.
CHECKPOINT=''  # 알고 있다면 직접 경로 지정. 비워두면 Drive best.pt 검색.
CASE_ID='case_003_strength_0p65_seed_100'
drive_root=Path('/content/drive/MyDrive')
if USE_DRIVE and not drive_root.is_dir():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as exc:
        print('Drive 연결 불가. 아래 직접 업로드로 계속할 수 있습니다:',type(exc).__name__)
if not ZIP_PATH:
    local=Path('/content/sweep_20261111_062740_406072.zip')
    matches=([local] if local.is_file() else [])
    if not matches and USE_DRIVE and drive_root.is_dir():
        print('Drive에서 원래 sweep ZIP을 찾습니다...',flush=True)
        matches=list(drive_root.rglob('sweep_20261111_062740_406072.zip'))
    if matches:ZIP_PATH=str(matches[0]);print('ZIP:',ZIP_PATH)
if not ZIP_PATH:
    print('원래 sweep ZIP을 업로드하세요.')
    uploaded=files.upload();names=[n for n in uploaded if n.lower().endswith('.zip')]
    if len(names)!=1:raise ValueError('sweep ZIP 하나를 선택하세요.')
    ZIP_PATH=names[0]
try:records,expected_sha=load_stroke_bundle(ZIP_PATH,CASE_ID)
except KeyError as exc:raise ValueError('원래 sweep ZIP의 source_conditions/warped_slot 자료가 필요합니다.') from exc
source,candidate,metadata=load_case(ZIP_PATH,CASE_ID)
metadata.update(checkpoint_sha256=expected_sha,reference_bundle_case=CASE_ID)
print('필요한 모델 SHA256:',expected_sha)
if not CHECKPOINT:
    candidates=list(Path('/content').glob('*.pt'))
    if USE_DRIVE and drive_root.is_dir():
        print('Drive best.pt를 검사합니다. 이름이 같아도 해시가 다르면 건너뜁니다.',flush=True)
        candidates.extend(drive_root.rglob('best.pt'))
    for path in candidates:
        try:actual=checkpoint_hash(path)
        except OSError:continue
        if actual==expected_sha:
            CHECKPOINT=str(path);break
        print('다른 체크포인트, 건너뜀:',path,flush=True)
if not CHECKPOINT:
    print('일치하는 획 추출 체크포인트(.pt)를 업로드하세요.')
    uploaded=files.upload();names=[n for n in uploaded if n.lower().endswith('.pt')]
    if len(names)!=1:raise ValueError('체크포인트 하나를 선택하세요.')
    CHECKPOINT=names[0]
if checkpoint_hash(CHECKPOINT)!=expected_sha:
    raise ValueError('체크포인트 해시 불일치: 원래 sweep 실험에 사용한 모델이 아닙니다.')
state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
if state.get('architecture')!='reference_v2_topology_loss_v1':
    raise ValueError('이 노트북은 기존 reference_v2_topology_loss_v1 체크포인트를 사용합니다.')
extractor=ReferenceStrokeModel(image_size=state['image_size'],width=state['width'],pretrained=False)
extractor.load_state_dict(state['model'],strict=True)
extractor=extractor.to('cuda').eval().requires_grad_(False)
if any(r['strokes'].shape[-2:]!=(state['image_size'],state['image_size']) for r in records):
    raise ValueError('기준 획과 모델 해상도가 다릅니다.')
del state
print('검증된 모델:',CHECKPOINT)
''')
    add('code', '''ENABLE_RESTORE=False
REFRESH_EVERY=3
MARGIN_RATIO=.20  # 정상 몸통 전체 굵기의 20%를 허용 여유로 사용
source_rgb=image_tensor(source,'cuda')
objective=AdaptiveInkConstraints(source_rgb,extractor,records,refresh_every=REFRESH_EVERY,
    enable_restore=ENABLE_RESTORE,margin_ratio=MARGIN_RATIO)
print('획별 허용 여유 (모델 좌표와 원본 이미지 좌표):')
for row in objective.bands:print(row)
erase=objective.erase[0,0].cpu().numpy();restore=objective.restore[0,0].cpu().numpy()
rgb=np.asarray(source,dtype=np.float32)/255
alpha=.7*np.maximum(erase,restore)[...,None]
color=np.stack([erase,np.zeros_like(erase),restore],-1)
color/=np.maximum(color.max(-1,keepdims=True),1e-8)
overlay=rgb*(1-alpha)+color*alpha
fig,axes=plt.subplots(1,4,figsize=(20,5))
for ax,im,title in zip(axes,[source,overlay,erase,restore],
        ['Original input','Red=erase / blue=restore','Initial erase pressure','Restore centerlines']):
    ax.imshow(im,cmap='magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
plt.show();plt.close(fig)
print('正常 / 정상 획을 넓게 덮는다면 MARGIN_RATIO를 높여 재생성하세요.')
'''.replace('正常 / ', ''))
    add('code', '''from datetime import datetime
from huggingface_hub import model_info
import shutil
STEPS=60
STRENGTH=.25  # 실제 추가 역확산 15단계. 이전 버전과 동일.
SEED=2026
INNER_STEPS=4
STEP_RATIO=.012
MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION=model_info(MODEL_ID).sha
OUTPUT=Path('/content')/('adaptive_ink_reverse_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
# Cell 4 reruns always reset ownership/mask history to the same original input.
objective=AdaptiveInkConstraints(source_rgb,extractor,records,refresh_every=REFRESH_EVERY,
    enable_restore=ENABLE_RESTORE,margin_ratio=MARGIN_RATIO)
def progress(method,step,total,row):
    if step==1 or step%3==0 or step==total or row.get('constraint_refresh'):
        print(f"{method} {step}/{total} | refresh={bool(row.get('constraint_refresh'))} | "
              f"accepted={row['accepted_inner_steps']} | {row.get('loss_before')} → {row.get('loss_after')}",flush=True)
report=run_pair(source,candidate,metadata,OUTPUT,steps=STEPS,strength=STRENGTH,seed=SEED,
    inner_steps=INNER_STEPS,step_ratio=STEP_RATIO,model_id=MODEL_ID,revision=REVISION,
    progress=progress,guidance_mode='adaptive_ink',objective_override=objective)
panels=[('Original input',source),('Re-denoise only',Image.open(OUTPUT/'redenoise_only.png')),
        ('Adaptive stroke-guided reverse',Image.open(OUTPUT/'adaptive_guided.png'))]
fig,axes=plt.subplots(1,3,figsize=(18,6))
for ax,(title,im) in zip(axes,panels):ax.imshow(im);ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(OUTPUT/'comparison.png',dpi=140);plt.show();plt.close(fig)
print('공통 초기 마스크에서 평가한 진단값 (정확도 아님):',json.dumps(report['metrics'],indent=2))
print('재추출 이력:',[(r['step'],r['erase_pixels']) for r in report['constraint_history']])
mask_paths=sorted(OUTPUT.glob('erase_step_*.png'))
if mask_paths:
    fig,axes=plt.subplots(1,len(mask_paths),figsize=(4*len(mask_paths),4),squeeze=False)
    for ax,path in zip(axes.flat,mask_paths):
        ax.imshow(source);ax.imshow(Image.open(path),cmap='Reds',vmin=0,vmax=255,alpha=.5)
        ax.set_title(path.stem);ax.axis('off')
    fig.tight_layout();fig.savefig(OUTPUT/'refresh_masks.png',dpi=120);plt.show();plt.close(fig)
paths=sorted(OUTPUT.glob('adaptive_guided_step_*.png'))
fig,axes=plt.subplots(1,len(paths),figsize=(4*len(paths),4),squeeze=False)
for ax,path in zip(axes.flat,paths):
    ax.imshow(Image.open(path));ax.set_title(path.stem);ax.axis('off')
fig.tight_layout();fig.savefig(OUTPUT/'adaptive_trajectory.png',dpi=120);plt.show();plt.close(fig)
(OUTPUT/'runtime_helpers.json').write_text(json.dumps(RUNTIME_HELPERS,ensure_ascii=False),encoding='utf-8')
# Preserve the small stroke bundle for subsequent experiments, not the checkpoint.
with zipfile.ZipFile(ZIP_PATH) as z:
    from pathlib import PurePosixPath
    for name in z.namelist():
        member=PurePosixPath(name)
        if member.is_absolute() or '..' in member.parts or '\\\\' in name:continue
        if name.startswith('source_conditions/') or name.startswith(CASE_ID+'/warped_slot_'):
            path=OUTPUT/'stroke_bundle'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
archive=shutil.make_archive(str(OUTPUT),'zip',OUTPUT)
print('실제 GPU 결과와 정상 획 변형을 함께 확인하세요:',archive)
files.download(archive)
''')
    nb = dict(cells=cells, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
              language_info=dict(name='python'), accelerator='GPU', colab=dict(provenance=[])), nbformat=4, nbformat_minor=5)
    path = Path(__file__).with_name('adaptive_ink_diffusion.ipynb')
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':print(build())
