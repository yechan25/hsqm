"""Three-cell saved-error experiment; embeds local helpers, so no Git commit is required."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def add(kind,source):
        if kind=='code':ast.parse(source)
        item=dict(cell_type=kind,metadata={},source=source.strip().splitlines(keepends=True))
        if kind=='code':item.update(outputs=[],execution_count=None)
        cells.append(item)
    add('markdown','''# 저장된 오류 이미지 → 획 추출 가이드 → 실제 재역확산

L4에서 실행하는 3셀 실험입니다. 새 text-to-image 생성 및 Qwen 로딩은 없습니다.
1 준비·기존 ZIP 선택 → 2 기존 획 모델·가이드 확인 → 3 동일 잡음으로 재역확산 비교.
`guidance_debug (4).zip`을 권장합니다. 최초 baseline을 그대로 사용하며 이전 guided 결과는 입력에 쓰지 않습니다.
기준 획이 주어진 정렬된 조건의 보정 실험입니다. 기준 자료는 첫 생성 입력이 아니지만, 보정 가이드에 사용됩니다.
글자 외부 잉크는 단순 기준 윤곽 비교만으로도 찾을 수 있습니다. 따라서 가이드의 효과와 획 모델 자체의 기여를 구분합니다.
첫 실행은 1번 슬롯(가)만 보정합니다. 결과가 좋아진다고 보장하지 않으며 정상 획 손상도 함께 확인하세요.
실제 GPU 결과는 코랩 실행 후 얻습니다. 로컬 수정 코드를 포함하므로 별도 Git 커밋이 필요 없습니다.
''')
    helpers={p:(ROOT/p).read_text(encoding='utf-8') for p in (
        'diffusion_model/saved_repair_demo.py','stroke_model/saved_extra_guide.py')}
    setup='''from pathlib import Path
from datetime import datetime
import subprocess,sys,tempfile,json,hashlib,gc
root=Path(tempfile.mkdtemp(prefix='saved_repair_',dir='/content'))
subprocess.run(['git','clone','--depth','1','https://github.com/yechan25/hsqm.git',str(root)],check=True)
commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
subprocess.run([sys.executable,'-m','pip','install','--prefer-binary',
    'diffusers==0.37.0','transformers==4.57.6','huggingface-hub==0.36.0',
    'accelerate','safetensors','timm','pandas','numpy','matplotlib','pillow',
    'scipy','scikit-image','pyyaml','fonttools','ipywidgets'],check=True)
'''
    setup+='runtime_helpers='+repr(helpers)+'\n'
    setup+='''for name,code in runtime_helpers.items():
    (root/name).write_text(code,encoding='utf-8')
sys.path.insert(0,str(root))
for name in list(sys.modules):
    if name.split('.')[0] in ('stroke_model','diffusion_model'):del sys.modules[name]
import torch,numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
from google.colab import drive,files
drive.mount('/content/drive')
if not torch.cuda.is_available():raise RuntimeError('L4 GPU 런타임을 선택하세요.')
from stroke_model.convert_reference_csv import existing_path
from diffusion_model.saved_repair_demo import prepare_saved_demo
HSQM=existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
ARCHIVE=None  # Drive ZIP 경로를 지정하거나, None이면 guidance_debug (4).zip 업로드
CHECKPOINT=None  # None: ZIP에 기록된 당시 획 추출기. 다른 모델을 자동 선택하지 않음
if ARCHIVE is None:
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('오류 이미지가 든 ZIP 하나를 선택하세요.')
    ARCHIVE=Path(next(iter(uploaded)))
RUN_DIR=HSQM/'saved_repair_demo'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
demo=prepare_saved_demo(ARCHIVE,RUN_DIR/'input')
if CHECKPOINT is None:CHECKPOINT=demo['checkpoint']
CHECKPOINT=existing_path(CHECKPOINT)
actual_sha=hashlib.sha256(Path(CHECKPOINT).read_bytes()).hexdigest()
if demo['checkpoint_sha256'] and actual_sha!=demo['checkpoint_sha256']:
    raise ValueError('ZIP 당시 사용한 획 모델과 SHA256이 다릅니다. CHECKPOINT를 확인하세요.')
print('GPU:',torch.cuda.get_device_name(0),'| 실제 입력:',demo['source'])
print('입력 SHA256:',demo['provenance']['source_sha256'])
print('이전 생성 방식:',demo['provenance']['source_initialization'])
display(Image.open(demo['source']))
'''
    add('code',setup)
    add('code','''from diffusion_model.score_pilot import training_catalog
from diffusion_model.multiglyph_pilot import compose_sheet
from stroke_model.infer_topology import load_topology
from stroke_model.saved_extra_guide import extracted_extra_map,fixed_region_metrics
from stroke_model.detected_extra_guidance import DetectedExtraGuidance

FOCUS_SLOTS=(0,)  # 0=좌상단 가. 모든 칸이면 (0,1,2,3)
TOLERANCE=3       # 기준 획의 여유: 모델 해상도에서 3픽셀. 기존 (4).zip의 정상 획 경계 보호
THRESHOLD=.25
KEEP_WEIGHT=20.
extractor,size,epoch=load_topology(CHECKPOINT,torch.device('cuda'))
extractor.requires_grad_(False)
_,slots=compose_sheet(training_catalog(demo['manifest']),demo['size'])
for slot,box in zip(slots,demo['boxes']):slot['bbox']=box
if any(slot['reference'].shape[-1]!=size for slot in slots):raise ValueError('모델/기준 해상도 불일치')
with Image.open(demo['source']) as image:array=np.asarray(image.convert('RGB'),np.float32)/255
source_rgb=torch.from_numpy(array.copy()).permute(2,0,1)[None].cuda()
probability,geometry,extraction=extracted_extra_map(extractor,source_rgb,slots,
    focus_slots=FOCUS_SLOTS,tolerance=TOLERANCE)
fixed_objective=DetectedExtraGuidance(source_rgb,probability,slots,THRESHOLD,KEEP_WEIGHT)
np.savez_compressed(RUN_DIR/'guides.npz',probability=probability.cpu().numpy(),
    geometry_only=geometry.cpu().numpy(),mask=fixed_objective.mask.cpu().numpy())
guide_metadata=dict(kind='extractor_assignment_vs_aligned_reference_strokes',
    uses_reference_conditions=True,focus_slots=FOCUS_SLOTS,tolerance=TOLERANCE,
    threshold=THRESHOLD,keep_weight=KEEP_WEIGHT,checkpoint_sha256=actual_sha,
    geometry_mask_identical=bool(torch.equal(probability>=THRESHOLD,geometry>=THRESHOLD)))
model_mask=probability>=THRESHOLD;geometry_mask=geometry>=THRESHOLD
guide_metadata['geometry_mask_iou']=float((model_mask&geometry_mask).sum()/(model_mask|geometry_mask).sum().clamp_min(1))
(RUN_DIR/'guide_metadata.json').write_text(json.dumps(guide_metadata,indent=2))
fig,axes=plt.subplots(1,4,figsize=(16,4))
axes[0].imshow(array);axes[0].set_title('Saved erroneous input')
axes[1].imshow(probability[0,0].cpu(),cmap='magma',vmin=0,vmax=1);axes[1].set_title('Extractor + reference guide')
axes[2].imshow(array);axes[2].imshow(fixed_objective.mask[0,0].cpu(),cmap='Reds',alpha=.45,vmin=0,vmax=1);axes[2].set_title('Actual edit region')
axes[3].imshow(geometry[0,0].cpu(),cmap='magma',vmin=0,vmax=1);axes[3].set_title('Geometry-only control map')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'guide_preview.png');plt.show()
for index in FOCUS_SLOTS:
    masks=extraction[index]['masks'];strokes=extraction[index]['reference_strokes']
    fig,axes=plt.subplots(2,len(masks),figsize=(3*len(masks),6),squeeze=False)
    for k in range(len(masks)):
        axes[0,k].imshow(strokes[k],cmap='gray');axes[0,k].set_title(f'Condition stroke {k+1}')
        axes[1,k].imshow(masks[k],cmap='magma');axes[1,k].set_title(f'Extracted stroke {k+1}')
    for ax in axes.flat:ax.axis('off')
    plt.show()
    np.savez_compressed(RUN_DIR/f'extracted_slot_{index+1}.npz',**{k:v for k,v in extraction[index].items() if k not in ('slot','bbox')})
if guide_metadata['geometry_mask_iou']>=.98:
    print('이 예제의 삭제 위치는 기하 비교만으로도 동일합니다. 모델 고유의 이득으로 해석하지 마세요.')
print('오류 위치를 가이드가 잡는지 확인한 뒤 3번을 실행하세요. 정상 획이 빨갛다면 진행하지 마세요.')
''')
    add('code','''from diffusion_model.run_topo_guidance import generate_guided_pair
from huggingface_hub import model_info
import pandas as pd
import zipfile

MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION=model_info(MODEL_ID).sha
STEPS=100
STRENGTH=.35  # 실제 재역확산 35단계
SEED=101
RUN_GEOMETRY_CONTROL=True
GUIDANCE=dict(local_edit=False,clean_step_ratio=.04,start_fraction=.1,end_fraction=1.,
    every=1,backtracks=6,snapshot_every=2,state_update='clean_residual',max_new_local_ink=None)
if hashlib.sha256(Path(demo['source']).read_bytes()).hexdigest()!=demo['provenance']['source_sha256']:
    raise RuntimeError('입력이 변경됐습니다. 1번부터 실행하세요.')
extractor.cpu();gc.collect();torch.cuda.empty_cache()
def run_pair(objective,directory):
    report=generate_guided_pair(demo['manifest'],demo['rows'],CHECKPOINT,directory,
        model_id=MODEL_ID,revision=REVISION,steps=STEPS,canvas_size=demo['size'],
        strength=STRENGTH,seed=SEED,init_mode='external',input_sheet=demo['source'],
        guidance=GUIDANCE,guidance_signal='localized_extra',objective_override=objective,
        slot_bboxes=demo['boxes'],prompt_text='Black Korean lettering on a plain white background. Preserve the existing glyphs and layout.',
        compare_structure=False,compare_uniform=False,appearance_weight=0,extra_weight=0,
        source_spill_weight=0,direct_reference=False)
    report.update(primary_signal=guide_metadata['kind'],guide_uses_reference_conditions=True,
        geometry_mask_iou=guide_metadata['geometry_mask_iou'],
        editing_mode='fixed_reference_conditioned_extra_map')
    (directory/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
generation=run_pair(fixed_objective,RUN_DIR/'extractor_guided')
pages={p['method']:p['input'] for p in generation['pages']}
panels=[('Input',demo['source']),('Re-denoise only',pages['baseline']),('Extractor/reference-guided',pages['guided'])]
if RUN_GEOMETRY_CONTROL and guide_metadata['geometry_mask_iou']<.98 and (geometry>=THRESHOLD).any():
    geometry_objective=DetectedExtraGuidance(source_rgb,geometry,slots,THRESHOLD,KEEP_WEIGHT)
    geometry_generation=run_pair(geometry_objective,RUN_DIR/'geometry_control')
    if generation['initial_latent_sha256']!=geometry_generation['initial_latent_sha256']:
        raise RuntimeError('기하 대조군의 초기 잠재값이 다릅니다.')
    gp={p['method']:p['input'] for p in geometry_generation['pages']}
    if hashlib.sha256(Path(pages['baseline']).read_bytes()).hexdigest()!=hashlib.sha256(Path(gp['baseline']).read_bytes()).hexdigest():
        raise RuntimeError('대조군 baseline이 재현되지 않았습니다.')
    panels.append(('Geometry-only guided',gp['guided']))
fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),5))
metrics=[]
for ax,(label,path) in zip(axes,panels):
    with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'),np.float32)/255
    ax.imshow(rgb);ax.set_title(label);ax.axis('off')
    tensor=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()
    metrics.append(dict(method=label,**fixed_region_metrics(tensor,fixed_objective)))
fig.tight_layout();fig.savefig(RUN_DIR/'comparison.png',dpi=150);plt.show()
fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),5))
x0,y0,x1,y1=slots[FOCUS_SLOTS[0]]['bbox']
for ax,(label,path) in zip(axes,panels):
    with Image.open(path) as image:ax.imshow(image.crop((x0,y0,x1,y1)))
    ax.set_title(label);ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'focused_comparison.png',dpi=160);plt.show()
frame=pd.DataFrame(metrics);display(frame)
frame.to_csv(RUN_DIR/'region_diagnostics.csv',index=False)
print('위 수치는 고정 편집 영역의 진단값이며 독립적인 품질 점수가 아닙니다.')
print('채택된 가이드 단계:',generation['guided_accepted_steps'])
print('같은 초기 잠재값:',generation['initial_latent_sha256'])
snapshots=sorted((RUN_DIR/'extractor_guided').glob('guided_step_*.png'))
snapshots=[p for p in snapshots if p.stem.endswith(('after','estimate'))]
if snapshots:
    frames=[]
    for path in snapshots:
        with Image.open(path) as image:frames.append(image.convert('RGB').copy())
    frames[0].save(RUN_DIR/'reverse_process.gif',save_all=True,append_images=frames[1:],duration=350,loop=0)
    from IPython.display import Image as DisplayImage
    display(DisplayImage(filename=str(RUN_DIR/'reverse_process.gif')))
    print('GIF는 각 단계의 clean 예측입니다. 최종 결과는 comparison을 확인하세요.')
(RUN_DIR/'runtime_helpers.json').write_text(json.dumps(runtime_helpers,ensure_ascii=False))
(RUN_DIR/'git_commit.txt').write_text(commit)
archive_path=RUN_DIR.with_suffix('.zip')
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in RUN_DIR.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(RUN_DIR))
files.download(str(archive_path))
''')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    path=Path(__file__).with_name('saved_repair_demo.ipynb')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


def build_no_drive():
    """Same experiment with explicit file uploads and session-local outputs."""
    original=build()
    notebook=json.loads(original.read_text(encoding='utf-8'))
    header=''.join(notebook['cells'][0]['source'])
    header+='\nDrive 마운트를 사용하지 않습니다. ZIP과 당시 획 추출기 best.pt를 각각 업로드하세요.\n결과는 /content에 저장되며 세션 종료 시 삭제되므로 마지막 ZIP을 다운로드하세요.\n'
    notebook['cells'][0]['source']=header.splitlines(keepends=True)
    source=''.join(notebook['cells'][1]['source'])
    source=source.replace('from google.colab import drive,files','from google.colab import files')
    source=source.replace("drive.mount('/content/drive')\n",'')
    source=source.replace("HSQM=existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')",
        "HSQM=Path('/content/saved_repair_workspace');HSQM.mkdir(exist_ok=True)")
    source=source.replace('ARCHIVE=None  # Drive ZIP 경로를 지정하거나, None이면 guidance_debug (4).zip 업로드',
        'ARCHIVE=None  # None이면 guidance_debug (4).zip 업로드; /content 파일 경로도 가능')
    source=source.replace('CHECKPOINT=None  # None: ZIP에 기록된 당시 획 추출기. 다른 모델을 자동 선택하지 않음',
        'CHECKPOINT=None  # None이면 ZIP 당시 사용한 획 추출기 best.pt 업로드')
    source=source.replace("if CHECKPOINT is None:CHECKPOINT=demo['checkpoint']",'''if CHECKPOINT is None:
    print('이번에는 ZIP 당시 사용한 획 추출기 best.pt 파일을 업로드하세요.')
    uploaded_checkpoint=files.upload()
    if len(uploaded_checkpoint)!=1:raise ValueError('획 추출기 checkpoint 하나를 선택하세요.')
    CHECKPOINT=Path(next(iter(uploaded_checkpoint)))
    del uploaded_checkpoint''')
    ast.parse(source)
    notebook['cells'][1]['source']=source.splitlines(keepends=True)
    destination=original.with_name('saved_repair_demo_no_drive.ipynb')
    destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return destination


if __name__=='__main__':
    print(build())
    print(build_no_drive())
