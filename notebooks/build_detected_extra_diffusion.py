"""Inference-only reverse diffusion using an already trained extra-ink detector."""
import ast,json
from pathlib import Path
from build_structure_generalization import build as base

def build():
    nb=json.loads(base().read_text(encoding='utf-8'))
    def put(i,text):
        if nb['cells'][i]['cell_type']=='code':ast.parse(text)
        nb['cells'][i]['source']=text.strip().splitlines(keepends=True)
    put(0,"""# 학습된 오류 탐지기 → 실제 재역확산 (학습 없음)

새 오류 탐지기 best.pt와 그때 사용한 기존 획 추출기 best.pt를 불러옵니다.
1 준비 → 2 저장 모델·생성 이미지 → 3 문자 선택 → 4 오류 지도 확인 → 5 실제 재역확산 → 6 비교·중간 영상·ZIP.
표준 글자와의 픽셀 차이는 보정 신호가 아닙니다. 표준 획은 기존 추출기의 조건으로만 사용합니다.
삭제 위치는 학습된 탐지기에서 얻고, 삭제 목표 배경은 현재 생성물 주변의 밝은 단색 배경에서 추정합니다.
실사진/어두운 배경 복원은 아직 지원하지 않습니다. 이 경로는 삭제만 하며 누락 획을 추가하지 않습니다.
아래 비교는 원본 생성물 / 재역확산만 / 오류 지도 가이드 재역확산이며 최종 RGB를 직접 칠하지 않습니다.
""")
    settings=''.join(nb['cells'][2]['source']).split('if CHECKPOINT is None:')[0]
    # Keep the established source data prep; the checkpoint selected below must
    # match the hash stored in the trained detector.
    settings=settings.replace('CHECKPOINT = None', 'CHECKPOINT = None')
    put(2,settings+"""
from stroke_model.detected_extra_guidance import load_error_head,predict_extra_sheet,DetectedExtraGuidance
ERROR_CHECKPOINT=None  # 학습한 새 오류 탐지기 best.pt. None이면 최신 학습 결과를 선택
FIRST_IMAGE=None  # 기존 생성물 2x2 이미지. None이면 저장된 두 단계 실험의 최근 stage1 결과
THRESHOLD=.5  # 미보정 진단 임계값. 3번에서 실제 지도를 확인하세요.
KEEP_WEIGHT=20.
REPAIR_STEPS=100
REPAIR_STRENGTH=.2
REPAIR_SEED=101
GUIDANCE=dict(local_edit=False,clean_step_ratio=.01,start_fraction=.1,end_fraction=1.,
    every=1,backtracks=6,snapshot_every=2,state_update='clean_residual',
    max_new_local_ink=None,max_state_change_to_base_step_ratio=.5)
if ERROR_CHECKPOINT is None:
    candidates=sorted((HSQM/'error_localizer_runs').glob('*/best.pt'))
    if not candidates:raise FileNotFoundError('ERROR_CHECKPOINT에 이미 학습한 새 오류 탐지기 best.pt 경로를 넣으세요. 이 노트북에서는 학습하지 않습니다.')
    ERROR_CHECKPOINT=candidates[-1]
# Read the detector metadata to use its EXACT frozen extractor, not an unrelated latest one.
info=torch.load(ERROR_CHECKPOINT,map_location='cpu',weights_only=True)
recorded=info.get('metadata',{}).get('extractor_checkpoint')
if CHECKPOINT is None and recorded:CHECKPOINT=existing_path(recorded)
if CHECKPOINT is None:raise ValueError('CHECKPOINT에 학습 당시 사용한 획 추출기를 지정하세요.')
preparation=prepare_pilot_data(CHECKPOINT,DATA_ROOT,csv_path=CSV_PATH,manifest=TRAIN_MANIFEST,reference_only=True)
TRAIN_MANIFEST=Path(preparation['manifest'])
from stroke_model.infer_topology import load_topology
extractor,size,_=load_topology(CHECKPOINT,torch.device('cuda'))
head,detector_info=load_error_head(ERROR_CHECKPOINT,CHECKPOINT,'cuda')
if FIRST_IMAGE is None:
    candidates=sorted((HSQM/'two_stage_repair').glob('*/stage1/page_001_baseline.png'))
    if not candidates:raise FileNotFoundError('FIRST_IMAGE에 실제로 생성된 네 글자 이미지 경로를 넣으세요.')
    FIRST_IMAGE=candidates[-1]
FIRST_IMAGE=existing_path(FIRST_IMAGE)
print('오류 탐지기:',ERROR_CHECKPOINT)
print('정확히 대응하는 획 추출기:',CHECKPOINT)
print('재역확산할 생성물:',FIRST_IMAGE)
print('threshold calibrated:',detector_info['threshold_calibrated'])
""")
    preview=''.join(nb['cells'][3]['source'])
    start=preview.index('font_paths=')
    preview=preview[:start]+"""
from PIL import Image
import numpy as np
from diffusion_model.multiglyph_pilot import compose_sheet
from diffusion_model.score_pilot import training_catalog
characters_box=glyph_box
print('같은 4문자를 선택하고, 문자 칸에 생성물의 목표 문자를 읽기 순서대로 입력하세요.')
display(characters_box)
with Image.open(FIRST_IMAGE) as im:display(im.convert('RGB'))
"""
    # The original preview includes glyph_box creation before font_paths.
    put(3,preview)
    put(4,"""
from datetime import datetime
from huggingface_hub import model_info
from diffusion_model.run_topo_guidance import generate_guided_pair
chars=characters_box.value.split()
if len(chars)==1 and len(chars[0])==4:chars=list(chars[0])
if len(chars)!=4:raise ValueError('예: 가 걔 곁 괴 — 4문자를 입력하세요.')
selected=list(reference_select.value)
if len(selected)!=4:raise ValueError('표준 획 조건을 정확히 4개 선택하세요.')
by_row={r['manifest_row']:r for r in training_catalog(TRAIN_MANIFEST)}
_,slots=compose_sheet([by_row[r] for r in selected],CANVAS_SIZE)
with Image.open(FIRST_IMAGE) as im:
    if im.size!=(CANVAS_SIZE,CANVAS_SIZE):raise ValueError('생성 이미지 크기와 CANVAS_SIZE를 맞추세요.')
    array=np.asarray(im.convert('RGB'),np.float32)/255
import hashlib
SOURCE_SHA=hashlib.sha256(Path(FIRST_IMAGE).read_bytes()).hexdigest()
source_rgb=torch.from_numpy(array.copy()).permute(2,0,1)[None].cuda()
head.cuda();extractor.cuda()
probability=predict_extra_sheet(head,extractor,source_rgb,slots,chars)
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,5))
axes[0].imshow(array);axes[0].set_title('Generated input')
axes[1].imshow(probability[0,0].cpu(),cmap='magma',vmin=0,vmax=1);axes[1].set_title('Trained detector probability')
axes[2].imshow(array);axes[2].imshow((probability[0,0].cpu()>=THRESHOLD),cmap='Reds',alpha=.4,vmin=0,vmax=1);axes[2].set_title('Fixed edit mask')
for ax in axes:ax.axis('off')
plt.show()
fixed_objective=DetectedExtraGuidance(source_rgb,probability,slots,THRESHOLD,KEEP_WEIGHT)
print('이 오류 지도가 재역확산에 실제 사용됩니다. 잘못된 위치이면 threshold/선택을 확인한 뒤 실행하세요.')
""")
    # Separate map inspection from expensive inference; no extra training cell.
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[]))
    put(5,"""
# Actual reverse diffusion: SAME initial re-noised latent for baseline and guided.
if hashlib.sha256(Path(FIRST_IMAGE).read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('입력 이미지가 바뀌었습니다. 4번 셀부터 다시 실행하세요.')
extractor.cpu();head.cpu();torch.cuda.empty_cache()
OUTPUT_DIR=HSQM/'detected_extra_diffusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
generation=generate_guided_pair(TRAIN_MANIFEST,selected,CHECKPOINT,OUTPUT_DIR,
    model_id=MODEL_ID,revision=model_info(MODEL_ID).sha,steps=REPAIR_STEPS,canvas_size=CANVAS_SIZE,
    strength=REPAIR_STRENGTH,seed=REPAIR_SEED,init_mode='external',input_sheet=FIRST_IMAGE,
    guidance=GUIDANCE,guidance_signal='localized_extra',objective_override=fixed_objective,
    compare_structure=False,compare_uniform=False,appearance_weight=0,extra_weight=0,
    source_spill_weight=0,direct_reference=False)
np.savez_compressed(OUTPUT_DIR/'detector_plan.npz',probability=probability.cpu().numpy(),mask=fixed_objective.mask.cpu().numpy(),target=fixed_objective.target.cpu().numpy())
(OUTPUT_DIR/'detector_info.json').write_text(json.dumps(dict(detector_info,checkpoint=str(ERROR_CHECKPOINT),threshold=THRESHOLD,keep_weight=KEEP_WEIGHT,source_sha256=SOURCE_SHA),indent=2))
(OUTPUT_DIR/'git_commit.txt').write_text(commit)
print('채택된 가이드 단계:',generation['guided_accepted_steps'])
""")
    put(6,"""
pages={p['method']:p for p in generation['pages']}
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,(title,path) in zip(axes,[('First generation',FIRST_IMAGE),('Re-denoise only',pages['baseline']['input']),('Detector-guided reverse diffusion',pages['guided']['input'])]):
    with Image.open(path) as image:ax.imshow(image.convert('RGB'))
    ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(OUTPUT_DIR/'comparison.png',dpi=140);plt.show()
snapshots=sorted(OUTPUT_DIR.glob('guided_step_*_after.png'))
if snapshots:
    fig,axes=plt.subplots(1,min(6,len(snapshots)),figsize=(18,3),squeeze=False)
    for ax,path in zip(axes[0],snapshots[-6:]):
        with Image.open(path) as image:ax.imshow(image.convert('RGB'))
        ax.set_title(path.stem);ax.axis('off')
    plt.show()
import zipfile
from google.colab import files
archive=OUTPUT_DIR.parent/(OUTPUT_DIR.name+'_detected_diffusion.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in OUTPUT_DIR.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(OUTPUT_DIR))
files.download(str(archive))
""")
    path=Path(__file__).with_name('detected_extra_diffusion.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path
if __name__=='__main__':print(build())
