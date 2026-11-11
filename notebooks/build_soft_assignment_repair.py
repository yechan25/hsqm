"""Build the self-contained, Drive-free softmax-guided repair experiment."""
import ast
import json
from pathlib import Path
from build_saved_repair_demo import build_no_drive

ROOT=Path(__file__).resolve().parents[1]


def build():
    notebook=json.loads(build_no_drive().read_text(encoding='utf-8'))
    def put(index,source):
        if notebook['cells'][index]['cell_type']=='code':ast.parse(source)
        notebook['cells'][index]['source']=source.strip().splitlines(keepends=True)
    put(0,'''# Softmax 역확산 보정 v2 — 네 칸 / 무가이드 대비 보호 한도 / Drive 없음

1 ZIP·기존 획 모델 업로드 → 2 획별 softmax/연속 오류 압력 확인 → 3 실제 재역확산 및 세 방법 비교.
권장 입력: guidance_debug (4).zip과 당시 사용한 획 추출기 best.pt.
기존 오류 이미지를 그대로 재잡음합니다. 새 이미지 생성·Qwen·정답 이미지의 VAE 초기화는 없습니다.
기준 획의 위치와 폭 조건을 사용하는 실험입니다. 참조 없는 검출이나 누락 획 복원 모델이 아닙니다.
기본은 네 칸 모두의 추가 잉크 삭제입니다. 같은 단계의 무가이드 결과와 비교해 정상 획·배경 보호 한도를 검사합니다.
결과는 /content 임시 저장소에 있으므로 마지막 ZIP을 다운로드하세요. GPU 품질은 실제 실행 후 확인해야 합니다.
''')
    source=''.join(notebook['cells'][1]['source'])
    position=source.index('runtime_helpers=');finish=source.index('for name,code in runtime_helpers.items():',position)
    helpers={p:(ROOT/p).read_text(encoding='utf-8') for p in (
        'diffusion_model/saved_repair_demo.py','stroke_model/saved_extra_guide.py',
        'diffusion_model/guided_ddim.py','stroke_model/soft_assignment_guidance.py')}
    source=source[:position]+'runtime_helpers='+repr(helpers)+'\n'+source[finish:]
    source=source.replace("HSQM/'saved_repair_demo'","HSQM/'soft_assignment_repair'")
    source=source.replace('ARCHIVE=None',"ARCHIVE=Path('/content/guidance_debug (4).zip') if Path('/content/guidance_debug (4).zip').is_file() else None",1)
    source=source.replace('CHECKPOINT=None',"CHECKPOINT=Path('/content/best.pt') if Path('/content/best.pt').is_file() else None",1)
    put(1,source)
    put(2,'''from diffusion_model.score_pilot import training_catalog
from diffusion_model.multiglyph_pilot import compose_sheet
from stroke_model.infer_topology import load_topology
from stroke_model.soft_assignment_guidance import SoftAssignmentGuidance

RUN_VERSION='soft-assignment-v2-matched-baseline-all-slots'
FOCUS_SLOTS=(0,1,2,3)  # 네 칸 모두. 한 칸만 시험하려면 예: (0,)
ALLOWANCE=3.0     # 모델 좌표의 최소 허용 반폭. 실제 소스 두께로 상향 보정 후 고정
FALLOFF=3.0       # 거리 페널티가 완만하게 증가하는 폭
extractor,size,epoch=load_topology(CHECKPOINT,torch.device('cuda'))
extractor.requires_grad_(False)
_,slots=compose_sheet(training_catalog(demo['manifest']),demo['size'])
for slot,box in zip(slots,demo['boxes']):slot['bbox']=box
if any(slot['reference'].shape[-1]!=size for slot in slots):raise ValueError('모델/기준 해상도 불일치')
with Image.open(demo['source']) as image:array=np.asarray(image.convert('RGB'),np.float32)/255
source_rgb=torch.from_numpy(array.copy()).permute(2,0,1)[None].cuda()
options=dict(focus_slots=FOCUS_SLOTS,allowance=ALLOWANCE,falloff=FALLOFF,
    preserve_weight=.1,healthy_rms_limit=.025,healthy_local_limit=.12,healthy_mass_drop=.04,
    baseline_rms_slack=.002,baseline_local_slack=.02,baseline_mass_slack=.01)
soft_objective=SoftAssignmentGuidance(extractor,slots,source_rgb,mode='softmax',**options)
geometry_objective=SoftAssignmentGuidance(extractor,slots,source_rgb,mode='geometry',**options)
np.savez_compressed(RUN_DIR/'initial_guides.npz',softmax_weights=soft_objective.initial_weights.cpu().numpy(),
    geometry_weights=geometry_objective.initial_weights.cpu().numpy(),healthy_regions=soft_objective.healthy.cpu().numpy())
guide_info=dict(version=RUN_VERSION,kind='dynamic_detached_softmax_stroke_distance',uses_reference_conditions=True,
    options=options,calibrated_radii=soft_objective.bands,checkpoint_sha256=actual_sha,
    geometry_mean_absolute_map_difference=float((soft_objective.initial_weights-geometry_objective.initial_weights).abs().mean()))
(RUN_DIR/'guide_info.json').write_text(json.dumps(guide_info,indent=2))
fig,axes=plt.subplots(1,4,figsize=(17,4))
axes[0].imshow(array);axes[0].set_title('Saved erroneous input')
axes[1].imshow(soft_objective.initial_weights[0,0].cpu(),cmap='magma',vmin=0,vmax=1);axes[1].set_title('Softmax-weighted distance')
axes[2].imshow(geometry_objective.initial_weights[0,0].cpu(),cmap='magma',vmin=0,vmax=1);axes[2].set_title('Nearest-stroke distance only')
axes[3].imshow(array);axes[3].imshow(soft_objective.initial_weights[0,0].cpu(),cmap='Reds',alpha=.6,vmin=0,vmax=1);axes[3].set_title('Continuous edit pressure')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'guide_preview.png',dpi=140);plt.show()
for index in FOCUS_SLOTS:
    slot=slots[index];ref=getattr(soft_objective,f'reference_{index}');strokes=getattr(soft_objective,f'strokes_{index}')
    ink=soft_objective.crop(soft_objective.source_ink,slot,ref.shape[-2:])
    with torch.no_grad():prediction=extractor(ink,ref,strokes)
    p=prediction['probabilities'][0].cpu();m=prediction['masks'][0].cpu()
    fig,axes=plt.subplots(3,len(p),figsize=(3*len(p),8),squeeze=False)
    for k in range(len(p)):
        axes[0,k].imshow(strokes[0,k].cpu(),cmap='gray');axes[0,k].set_title(f'Condition {k+1}')
        axes[1,k].imshow(p[k],cmap='viridis',vmin=0,vmax=1);axes[1,k].set_title(f'Softmax p{k+1}')
        axes[2,k].imshow(m[k],cmap='magma',vmin=0,vmax=1);axes[2,k].set_title(f'Extracted ink {k+1}')
    for ax in axes.flat:ax.axis('off')
    plt.show()
    np.savez_compressed(RUN_DIR/f'extracted_slot_{index+1}.npz',probabilities=p.numpy(),masks=m.numpy())
print('보정은 위 연속값 전체를 사용합니다. 정상 획 전체에 압력이 강하면 ALLOWANCE/조건 정렬을 확인하세요.')
print('허용 반폭:',soft_objective.bands)
print('실제 보정 슬롯:',FOCUS_SLOTS)
''')
    put(3,'''from diffusion_model.run_topo_guidance import generate_guided_pair
from huggingface_hub import model_info
import pandas as pd
import zipfile

STEPS=100
STRENGTH=.35
SEED=101
RUN_GEOMETRY_CONTROL=True
MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION=model_info(MODEL_ID).sha
GUIDANCE=dict(local_edit=False,clean_step_ratio=.01,start_fraction=.1,end_fraction=1.,
    every=1,backtracks=5,snapshot_every=2,state_update='clean_residual',max_new_local_ink=None,
    inner_iterations=4,final_inner_iterations=24,max_clean_change_ratio=.08)
if hashlib.sha256(Path(demo['source']).read_bytes()).hexdigest()!=demo['provenance']['source_sha256']:
    raise RuntimeError('입력이 바뀌었습니다. 1번 셀부터 다시 실행하세요.')
def run_pair(objective,name):
    # Reset from the same source before each method, including guard masks.
    objective.refresh(source_rgb)
    directory=RUN_DIR/name
    report=generate_guided_pair(demo['manifest'],demo['rows'],CHECKPOINT,directory,
        model_id=MODEL_ID,revision=REVISION,steps=STEPS,canvas_size=demo['size'],
        strength=STRENGTH,seed=SEED,init_mode='external',input_sheet=demo['source'],
        guidance=GUIDANCE,guidance_signal='localized_extra',objective_override=objective,
        slot_bboxes=demo['boxes'],prompt_text='Black Korean lettering on a plain white background. Preserve the existing glyphs and layout.',
        compare_structure=False,compare_uniform=False,appearance_weight=0,extra_weight=0,
        source_spill_weight=0,direct_reference=False)
    report.update(version=RUN_VERSION,primary_signal='dynamic_'+objective.mode+'_stroke_distance',
        editing_mode='iterative_clean_latent_with_separate_edit_and_healthy_guards',
        primary_score='continuous edit loss; not calibrated image quality',
        gradient_path='clean latent -> VAE -> RGB; detached extractor ownership refreshed per diffusion step',
        guide_uses_reference_conditions=True,ownership_backward=False,
        focus_slots=list(objective.focus_slots),normal_guard_reference='matched_unguided_trajectory',
        baseline_guard_steps=len(objective.baseline_guards))
    (directory/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    np.savez_compressed(directory/'last_guidance.npz',weights=objective.weights.cpu().numpy(),protect=objective.protect.cpu().numpy())
    return report
generation=run_pair(soft_objective,'softmax_guided')
pages={p['method']:p['input'] for p in generation['pages']}
panels=[('Input',demo['source']),('Re-denoise only',pages['baseline']),('Softmax-guided',pages['guided'])]
if RUN_GEOMETRY_CONTROL:
    geometry_generation=run_pair(geometry_objective,'geometry_control')
    if generation['initial_latent_sha256']!=geometry_generation['initial_latent_sha256']:
        raise RuntimeError('대조군의 초기 잠재값이 다릅니다.')
    gp={p['method']:p['input'] for p in geometry_generation['pages']}
    if Path(pages['baseline']).read_bytes()!=Path(gp['baseline']).read_bytes():raise RuntimeError('Baseline 재현 실패')
    panels.append(('Distance-only guided',gp['guided']))
fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),5));metrics=[]
# Common, fixed source diagnostic maps for ALL methods. Do not compare each
# method on its own changing guide. These are still optimization-related metrics.
soft_objective.refresh(source_rgb)
geometry_objective.refresh(source_rgb)
evaluation_region=(geometry_objective.initial_weights>.03).float()
for ax,(label,path) in zip(axes,panels):
    with Image.open(path) as image:rgb=np.asarray(image.convert('RGB'),np.float32)/255
    ax.imshow(rgb);ax.set_title(label);ax.axis('off')
    tensor=torch.from_numpy(rgb.copy()).permute(2,0,1)[None].cuda()
    metric=soft_objective.diagnostics(tensor)
    metric['initial_geometry_edit_loss']=geometry_objective.diagnostics(tensor)['initial_map_edit_loss']
    metric['fixed_error_region_mean_ink']=float((soft_objective.ink(tensor)*evaluation_region).sum()/evaluation_region.sum().clamp_min(1))
    metrics.append(dict(method=label,**metric))
fig.tight_layout();fig.savefig(RUN_DIR/'comparison.png',dpi=150);plt.show()
fig,axes=plt.subplots(len(FOCUS_SLOTS),len(panels),figsize=(4*len(panels),4*len(FOCUS_SLOTS)),squeeze=False)
for row,index in enumerate(FOCUS_SLOTS):
    box=slots[index]['bbox']
    for ax,(label,path) in zip(axes[row],panels):
        with Image.open(path) as image:ax.imshow(image.crop(tuple(box)))
        ax.set_title(f'Slot {index+1}: {label}');ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'focused_comparison.png',dpi=160);plt.show()
frame=pd.DataFrame(metrics);display(frame);frame.to_csv(RUN_DIR/'region_diagnostics.csv',index=False)
trace=[json.loads(line) for line in (RUN_DIR/'softmax_guided/trace_guided.jsonl').read_text().splitlines()]
print('채택한 보정 단계:',generation['guided_accepted_steps'])
print('채택한 내부 수정:',sum(r.get('accepted_inner_steps',0) for r in trace))
print('삭제 손실/정상 획 한도는 trace_guided.jsonl에서 각 시도별로 확인할 수 있습니다.')
display(pd.DataFrame([{k:r.get(k) for k in ('step','accepted_inner_steps','edit_loss_before',
    'edit_loss_after','protected_rms','protected_local_rms','healthy_mass_drop')} for r in trace if r['active']]))
snapshots=sorted((RUN_DIR/'softmax_guided').glob('guided_step_*.png'))
snapshots=[p for p in snapshots if p.stem.endswith(('after','estimate'))]
if snapshots:
    frames=[]
    for path in snapshots:
        with Image.open(path) as image:frames.append(image.convert('RGB').copy())
    frames[0].save(RUN_DIR/'reverse_process.gif',save_all=True,append_images=frames[1:],duration=350,loop=0)
    from IPython.display import Image as DisplayImage
    display(DisplayImage(filename=str(RUN_DIR/'reverse_process.gif')))
print('GIF는 중간 clean 예측입니다. 최종 출력과 정상 획 손상도 함께 확인하세요.')
(RUN_DIR/'runtime_helpers.json').write_text(json.dumps(runtime_helpers,ensure_ascii=False))
(RUN_DIR/'git_commit.txt').write_text(commit)
archive_path=RUN_DIR.with_suffix('.zip')
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in RUN_DIR.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(RUN_DIR))
files.download(str(archive_path))
''')
    path=Path(__file__).with_name('soft_assignment_repair_no_drive.ipynb')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


def build_drive():
    """Primary Colab: mount Drive, recover original experiment and exact model."""
    notebook=json.loads(build().read_text(encoding='utf-8'))
    notebook['cells'][0]['source']='''# Softmax 역확산 보정 v2 — Drive 마운트 / 네 칸

L4에서 1→2→3번 실행합니다. 1번에서 학습 자료가 있는 계정으로 Google Drive 연결을 승인하세요.
기존 Drive의 ZIP 또는 원래 생성 실험 폴더와 당시 획 추출기 best.pt를 사용합니다.
기본은 가·걔·곁·괴 네 칸 모두입니다. 기준 획 조건을 사용하는 추가 잉크 삭제 실험입니다.
새 첫 생성과 Qwen 로딩은 없습니다. 기존 오류 이미지를 추가 역확산합니다.
계산과 중간 저장은 /content에서 수행하고 최종 ZIP은 Drive에 복사합니다.
Google 인증/네트워크 오류 자체를 해결했다고 보장하지 않으며 마운트 실패 시 원래 오류와 함께 중단합니다.
'''.strip().splitlines(keepends=True)
    source=''.join(notebook['cells'][1]['source'])
    mount='''from google.colab import drive
import os
FORCE_REMOUNT=False  # 연결이 깨진 기존 mount를 다시 연결할 때만 True
if FORCE_REMOUNT or not os.path.ismount('/content/drive'):
    print('Google Drive 연결을 승인하세요. 기존 학습 자료가 있는 계정을 선택하세요.',flush=True)
    try:
        drive.mount('/content/drive',force_remount=FORCE_REMOUNT,timeout_ms=180000)
    except Exception as exc:
        raise RuntimeError('Drive 인증/마운트에 실패했습니다. 위 Google 오류를 확인한 뒤 1번 셀을 다시 실행하세요.') from exc
print('Drive mount 확인 완료',flush=True)
'''
    source=source.replace('root=Path(tempfile.mkdtemp(',mount+'root=Path(tempfile.mkdtemp(',1)
    start=source.index("HSQM=Path('/content/saved_repair_workspace')")
    source=source[:start]+'''import shutil
from diffusion_model.saved_repair_demo import pack_saved_generation
HSQM_PATH='/content/drive/Shareddrives/2026 자율연구/HSQM'
HSQM=existing_path(HSQM_PATH)
ARCHIVE=None  # Drive ZIP 경로를 명시할 수 있음. None이면 아래 위치/원래 생성 폴더 확인
CHECKPOINT=None  # None이면 당시 기록된 checkpoint 경로. 최신 모델 임의 선택 없음
SOURCE_REPORT=HSQM/'topo_guided_diffusion/20261108_103932_929716/generation_report.json'
if ARCHIVE is None:
    candidates=[HSQM/'guidance_debug (4).zip',HSQM/'debug/guidance_debug (4).zip',
        Path('/content/drive/MyDrive/guidance_debug (4).zip'),
        Path('/content/drive/MyDrive/Downloads/guidance_debug (4).zip')]
    ARCHIVE=next((p for p in candidates if p.is_file()),None)
if ARCHIVE is None:
    try:report_path=existing_path(SOURCE_REPORT)
    except FileNotFoundError:report_path=None
    if report_path is not None:
        print('Drive의 기존 생성 실험에서 입력을 복원합니다:',report_path,flush=True)
        try:ARCHIVE=pack_saved_generation(report_path,root/'source_experiment.zip')
        except FileNotFoundError as exc:print('기존 생성 폴더의 파일이 부족합니다:',exc)
if ARCHIVE is None:
    print('기존 ZIP/생성 폴더를 찾지 못했습니다. guidance_debug (4).zip만 업로드하세요.')
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('ZIP 하나를 선택하세요.')
    ARCHIVE=Path(next(iter(uploaded)))
archive_origin=str(ARCHIVE)
local_archive=root/'saved_input.zip'
shutil.copyfile(existing_path(ARCHIVE),local_archive)
RUN_DIR=Path('/content/soft_assignment_runs')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
demo=prepare_saved_demo(local_archive,RUN_DIR/'input')
if CHECKPOINT is None:
    try:CHECKPOINT=existing_path(demo['checkpoint'])
    except FileNotFoundError:
        CHECKPOINT=HSQM/'v2_topology_runs/20261104_214705_208224/best.pt'
checkpoint_origin=existing_path(CHECKPOINT)
CHECKPOINT=root/'extractor_best.pt'
print('Drive 획 모델을 실행용 임시 저장소로 복사합니다:',checkpoint_origin,flush=True)
shutil.copyfile(checkpoint_origin,CHECKPOINT)
actual_sha=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
if demo['checkpoint_sha256'] and actual_sha!=demo['checkpoint_sha256']:
    raise ValueError('ZIP 당시 획 모델과 SHA256이 다릅니다. CHECKPOINT를 확인하세요.')
(RUN_DIR/'drive_sources.json').write_text(json.dumps(dict(archive=archive_origin,checkpoint=str(checkpoint_origin)),ensure_ascii=False,indent=2))
print('GPU:',torch.cuda.get_device_name(0),'| 실제 입력:',demo['source'])
print('획 모델:',checkpoint_origin,'| SHA256:',actual_sha)
display(Image.open(demo['source']))
'''
    ast.parse(source)
    notebook['cells'][1]['source']=source.strip().splitlines(keepends=True)
    comparison=''.join(notebook['cells'][3]['source'])
    comparison=comparison.replace("files.download(str(archive_path))",'''drive_output=HSQM/'soft_assignment_repair_results'
drive_output.mkdir(parents=True,exist_ok=True)
drive_archive=drive_output/archive_path.name
shutil.copy2(archive_path,drive_archive)
print('Drive 결과 ZIP:',drive_archive)
files.download(str(archive_path))''')
    ast.parse(comparison)
    notebook['cells'][3]['source']=comparison.strip().splitlines(keepends=True)
    path=Path(__file__).with_name('soft_assignment_repair.ipynb')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
    print(build_drive())
