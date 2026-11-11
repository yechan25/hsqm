"""Freeze the pre-guard adaptive runtime and add selectable four-glyph sweeps."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PIN='d74e26e19d1e13360fdecde5d9d80d889344d990'
PINNED=('diffusion_model/fitted_candidate_diffusion.py','stroke_model/adaptive_ink_constraints.py',
        'stroke_model/reference_model.py','stroke_model/stroke_reference_fit.py')
SUPPORT=('diffusion_model/selectable_adaptive_sweep.py','diffusion_model/score_pilot.py',
         'diffusion_model/multiglyph_pilot.py','diffusion_model/pilot_data.py',
         'stroke_model/convert_reference_csv.py','stroke_model/prepare_stroke_zips.py',
         'stroke_model/reference_only_data.py')


def build():
    helpers={p:subprocess.check_output(['git','show',f'{PIN}:{p}'],cwd=ROOT).decode('utf-8') for p in PINNED}
    hashes={p:hashlib.sha256(code.encode()).hexdigest() for p,code in helpers.items()}
    helpers.update({p:(ROOT/p).read_text(encoding='utf-8') for p in SUPPORT})
    cells=[]
    def add(kind,source):
        if kind=='code':ast.parse(source)
        cell=dict(cell_type=kind,metadata={},source=source.strip().splitlines(keepends=True))
        if kind=='code':cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','''# 이전 adaptive 복귀 — 네 글자 선택 → 여러 생성 오류 → 실제 재역확산

**보정 코드는 안정화판 직전 d74e26e로 고정했습니다.** 새로운 거절/최종 선택/원본 대체 기능은 없습니다.
L4에서 1→2→3→4→5 실행. 3번에서 실제 글자 그림을 보고 네 칸을 선택합니다.
한 조합만 선택해도 강도 .45/.65/.85 × seed 100/101의 6가지 생성물과 정상 대조군을 처리합니다.
여러 조합을 추가할 수도 있습니다. 모든 생성은 같은 기준 이미지에서 독립적으로 시작합니다.
이 실험은 선택한 기준 글자 4개를 img2img로 변형하는 통제 실험이며 프롬프트만으로 생성하는 시험이 아닙니다.

필요 자료: 기존 best.pt와 그 학습 폴더의 run_config.json/selected_paths.csv/conversion_report.json,
기준 글자·기준 획 원본 압축 자료(보통 target.zip/target_stroke.zip). Drive에서 기존 경로를 사용합니다.
전체 글자 목록은 기존 학습 목록에서 복구합니다. 네 글자만 든 디버그 ZIP으로 대체하지 않습니다.
체크포인트 SHA256을 확인합니다. 학습은 하지 않습니다.
5번 결과는 입력 / 재역확산만 / 이전 adaptive 보정입니다. 심한 실패도 자동으로 숨기지 않습니다.
''')
    add('code','''import sys,subprocess,json,hashlib,gc,zipfile,shutil
from pathlib import Path
from datetime import datetime
subprocess.run([sys.executable,'-m','pip','install','diffusers==0.37.0','transformers==4.57.6',
    'accelerate','safetensors','timm','numpy','pandas','tqdm','pillow','matplotlib',
    'scipy','scikit-image','ipywidgets'],check=True)
'''+f'ADAPTIVE_COMMIT={PIN!r}\nPINNED_HASHES={hashes!r}\nRUNTIME_HELPERS={helpers!r}\n'+'''
ROOT=Path('/content/hsqm_previous_adaptive');ROOT.mkdir(exist_ok=True)
for package in ('diffusion_model','stroke_model'):
    (ROOT/package).mkdir(exist_ok=True);(ROOT/package/'__init__.py').touch()
for name,code in RUNTIME_HELPERS.items():(ROOT/name).write_text(code,encoding='utf-8')
for name,expected in PINNED_HASHES.items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError('고정 코드 해시 불일치: '+name)
sys.path.insert(0,str(ROOT))
for name in list(sys.modules):
    if name.split('.')[0] in ('diffusion_model','stroke_model'):sys.modules.pop(name)
import torch,numpy as np,matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display,clear_output
from google.colab import drive,files
from stroke_model.convert_reference_csv import existing_path
from stroke_model.adaptive_ink_constraints import checkpoint_hash
from stroke_model.reference_model import ReferenceStrokeModel
from diffusion_model.pilot_data import prepare_pilot_data
from diffusion_model.score_pilot import training_catalog,load_catalog_reference
from diffusion_model.selectable_adaptive_sweep import select_entries,prepare_groups,generate_cases,repair_cases
if not torch.cuda.is_available():raise RuntimeError('런타임에서 L4 GPU를 선택하세요.')
print('GPU:',torch.cuda.get_device_name(0),'| 보정 버전:',ADAPTIVE_COMMIT)
''')
    add('code','''# 기존 코랩 세션의 manifest가 남아 있으면 MANIFEST에 직접 지정해도 됩니다.
CHECKPOINT=''
MANIFEST=''
DATA_ROOT=''  # 비우면 학습 폴더의 complete.json에서 기존 압축 자료 위치 복구
CSV_PATH=''   # 비우면 체크포인트 옆 selected_paths.csv 사용
USE_DRIVE=True
EXPECTED_SHA='92b8d5dfac59a4e44d632464836877e1cc242b2a4bc72cd6e58d5fcef84cbbf7'
if USE_DRIVE and not Path('/content/drive/MyDrive').is_dir():drive.mount('/content/drive')
if not CHECKPOINT:
    recorded='/content/drive/Shareddrives/2026 자율연구/HSQM/v2_topology_runs/20261104_214705_208224/best.pt'
    try:
        candidate=existing_path(recorded)
        if checkpoint_hash(candidate)==EXPECTED_SHA:CHECKPOINT=str(candidate)
    except FileNotFoundError:pass
if not CHECKPOINT:
    for root in (Path('/content/drive/Shareddrives'),Path('/content/drive/MyDrive')):
        if not root.is_dir():continue
        print('기존 학습 체크포인트 검색:',root,flush=True)
        for path in root.rglob('best.pt'):
            try:matched=checkpoint_hash(path)==EXPECTED_SHA
            except OSError:continue
            if matched:CHECKPOINT=str(path);break
        if CHECKPOINT:break
if not CHECKPOINT:raise FileNotFoundError('CHECKPOINT에 기존 학습 폴더의 best.pt 경로를 지정하세요.')
if checkpoint_hash(CHECKPOINT)!=EXPECTED_SHA:raise ValueError('이전 adaptive에서 쓴 체크포인트와 다릅니다.')
preparation=prepare_pilot_data(CHECKPOINT,data_root=DATA_ROOT or None,csv_path=CSV_PATH or None,
    manifest=MANIFEST or None,reference_only=True)
MANIFEST=preparation['manifest'];CATALOG=training_catalog(MANIFEST)
if len(CATALOG)<4:raise ValueError('선택 가능한 서로 다른 기준 글자가 4개 미만입니다.')
state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
MODEL_SIZE=state['image_size']
if state.get('architecture')!='reference_v2_topology_loss_v1':raise ValueError('기존 V2 topology 체크포인트가 필요합니다.')
extractor=ReferenceStrokeModel(image_size=state['image_size'],width=state['width'],pretrained=False)
extractor.load_state_dict(state['model'],strict=True);extractor=extractor.eval().requires_grad_(False)
del state
print('글자 목록:',len(CATALOG),'개 | 모델:',CHECKPOINT)
print('다음 셀에서 그림을 보고 네 글자를 선택하세요.')
''')
    add('code','''import ipywidgets as widgets
OPTIONS=[(f"row {r['manifest_row']} | {r.get('char_id','') or '그림으로 확인'}",r['manifest_row']) for r in CATALOG]
by_row={r['manifest_row']:r for r in CATALOG}
selectors=[widgets.Dropdown(options=OPTIONS,value=CATALOG[i]['manifest_row'],description=f'칸 {i+1}',
           layout=widgets.Layout(width='95%')) for i in range(4)]
page=widgets.IntSlider(value=0,min=0,max=(len(CATALOG)-1)//16,description='목록 페이지',continuous_update=False)
gallery=widgets.Output();preview=widgets.Output();group_output=widgets.Output()
SELECTED_GROUPS=[]
def draw_gallery(change=None):
    with gallery:
        clear_output(wait=True)
        shown=CATALOG[page.value*16:(page.value+1)*16]
        fig,axes=plt.subplots(4,4,figsize=(10,10))
        for ax in axes.flat:ax.axis('off')
        for ax,entry in zip(axes.flat,shown):
            reference,_=load_catalog_reference(entry)
            ax.imshow(1-reference[0],cmap='gray',vmin=0,vmax=1)
            ax.set_title(f"row {entry['manifest_row']} | {entry.get('char_id','')}")
        fig.tight_layout();plt.show();plt.close(fig)
def draw_selected(change=None):
    with preview:
        clear_output(wait=True)
        rows=[s.value for s in selectors]
        print('현재 네 칸 순서:',rows)
        fig,axes=plt.subplots(2,2,figsize=(6,6))
        for ax,row in zip(axes.flat,rows):
            reference,_=load_catalog_reference(by_row[row])
            ax.imshow(1-reference[0],cmap='gray',vmin=0,vmax=1);ax.set_title(f'row {row}');ax.axis('off')
        plt.show();plt.close(fig)
def add_group(button):
    with group_output:
        clear_output(wait=True)
        rows=tuple(s.value for s in selectors)
        try:select_entries(MANIFEST,rows)
        except ValueError as exc:print(exc);return
        if rows not in SELECTED_GROUPS:SELECTED_GROUPS.append(rows)
        print('등록된 조합:',SELECTED_GROUPS)
def clear_groups(button):
    SELECTED_GROUPS.clear()
    with group_output:clear_output();print('등록 해제. 4번을 실행하면 현재 네 칸을 사용합니다.')
add=widgets.Button(description='현재 4자 조합 추가',button_style='primary');add.on_click(add_group)
reset=widgets.Button(description='조합 목록 초기화');reset.on_click(clear_groups)
page.observe(draw_gallery,names='value')
for selector in selectors:selector.observe(draw_selected,names='value')
display(page,gallery,widgets.VBox(selectors),preview,widgets.HBox([add,reset]),group_output)
draw_gallery();draw_selected()
print('한 조합만 시험하려면 네 칸을 고른 뒤 바로 4번 실행. 여러 조합은 추가 버튼으로 등록하세요.')
''')
    add('code','''from diffusers import StableDiffusionImg2ImgPipeline,DDIMScheduler
STRENGTHS=(.45,.65,.85)
SEEDS=(100,101)
GENERATION_STEPS=100
INCLUDE_CLEAN_CONTROL=True
MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION='451f4fe16113bff5a5d2269ed5ad43b0592e9a14'  # 이전 실험과 같은 확산 모델 revision
OUTPUT=Path('/content')/('selectable_adaptive_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
selections=list(SELECTED_GROUPS) or [tuple(s.value for s in selectors)]
GROUPS=prepare_groups(MANIFEST,selections,OUTPUT)
for group in GROUPS:
    for entry in training_catalog(group['manifest']):
        ref,strokes=load_catalog_reference(entry)
        if ref.shape[-2:]!=(MODEL_SIZE,MODEL_SIZE):raise ValueError('선택한 기준 글자와 모델 해상도가 다릅니다.')
print('실제 생성 조합:',selections,'| 강도:',STRENGTHS,'| seed:',SEEDS)
extractor.cpu();gc.collect();torch.cuda.empty_cache()
generation_pipe=StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID,revision=REVISION,
    torch_dtype=torch.float16,use_safetensors=True)
generation_pipe.scheduler=DDIMScheduler.from_config(generation_pipe.scheduler.config)
generation_pipe.enable_attention_slicing();generation_pipe.to('cuda')
try:
    CASES=generate_cases(GROUPS,strengths=STRENGTHS,seeds=SEEDS,steps=GENERATION_STEPS,
        revision=REVISION,pipeline=generation_pipe,include_control=INCLUDE_CLEAN_CONTROL)
finally:
    del generation_pipe;gc.collect();torch.cuda.empty_cache()
for group in GROUPS:
    failure=Path(group['root'])/'generation_failure.json'
    if failure.is_file():print('생성 중 실패 기록:',failure.read_text())
(OUTPUT/'cases.json').write_text(json.dumps(CASES,ensure_ascii=False,indent=2),encoding='utf-8')
(OUTPUT/'experiment.json').write_text(json.dumps(dict(adaptive_commit=ADAPTIVE_COMMIT,
    pinned_source_hashes=PINNED_HASHES,checkpoint_sha256=EXPECTED_SHA,selections=selections,
    generation_strengths=STRENGTHS,generation_seeds=SEEDS,generation_steps=GENERATION_STEPS,
    model_id=MODEL_ID,revision=REVISION,free_text_generation=False,
    reference_initialized_generation=True,data_preparation=preparation),ensure_ascii=False,indent=2),encoding='utf-8')
if not CASES:raise RuntimeError('생성된 사례가 없습니다. generation_failure.json을 확인하세요.')
fig,axes=plt.subplots((len(CASES)+2)//3,3,figsize=(15,5*((len(CASES)+2)//3)),squeeze=False)
for ax in axes.flat:ax.axis('off')
for ax,case in zip(axes.flat,CASES):
    ax.imshow(Image.open(case['input']));ax.set_title(case['case_id']+'\\n'+case['label'])
fig.tight_layout();fig.savefig(OUTPUT/'generation_gallery.png',dpi=100);plt.show();plt.close(fig)
print('생성 사례:',len(CASES),'개. 다음 셀은 기본적으로 모두 보정합니다.')
''')
    add('code','''# None이면 모든 사례. 특정 경우만 시험하려면 위에 표시된 case_id 목록을 지정하세요.
REPAIR_CASE_IDS=None
REPAIR_STEPS=60
REPAIR_STRENGTH=.25
REPAIR_SEED=2026
selected_cases=CASES if REPAIR_CASE_IDS is None else [c for c in CASES if c['case_id'] in REPAIR_CASE_IDS]
if REPAIR_CASE_IDS is not None and set(REPAIR_CASE_IDS)!={c['case_id'] for c in selected_cases}:
    raise ValueError('REPAIR_CASE_IDS에 없는 case_id가 있습니다.')
if not selected_cases:raise ValueError('보정할 사례가 없습니다.')
attempt='attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
selected_cases=[dict(c,output=str(Path(c['output'])/attempt)) for c in selected_cases]
extractor.to('cuda')
def progress(method,step,total,row):
    if step==1 or step%5==0 or step==total:print(method,step,'/',total,'accepted=',row['accepted_inner_steps'],flush=True)
def show_case(row):
    print(row['case_id'],row['status'],flush=True)
    if row['status']!='completed':print(row.get('error'));return
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(key,title) in zip(axes,[('input','Generated input'),('baseline','Re-denoise only'),('guided','Previous adaptive')]):
        ax.imshow(Image.open(row[key]));ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(Path(row['output'])/'comparison.png',dpi=120);plt.show();plt.close(fig)
SUMMARIES=repair_cases(selected_cases,extractor,checkpoint_sha256=EXPECTED_SHA,revision=REVISION,
    seed=REPAIR_SEED,steps=REPAIR_STEPS,strength=REPAIR_STRENGTH,on_case=show_case,progress=progress)
fig,axes=plt.subplots(len(SUMMARIES),3,figsize=(15,5*len(SUMMARIES)),squeeze=False)
for axes_row,row in zip(axes,SUMMARIES):
    for ax,(key,title) in zip(axes_row,[('input','Generated input'),('baseline','Re-denoise only'),('guided','Previous adaptive')]):
        ax.axis('off');ax.set_title(row['case_id']+'\\n'+title)
        if row.get(key):ax.imshow(Image.open(row[key]))
        else:ax.text(.1,.5,'FAILED: '+row.get('error',''),wrap=True)
fig.tight_layout();fig.savefig(OUTPUT/'all_comparisons.png',dpi=100);plt.show();plt.close(fig)
(OUTPUT/'runtime_helpers.json').write_text(json.dumps(RUNTIME_HELPERS,ensure_ascii=False),encoding='utf-8')
archive=shutil.make_archive(str(OUTPUT),'zip',OUTPUT)
print('모든 성공/실패 결과:',archive)
files.download(archive)
''')
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),
        language_info=dict(name='python'),accelerator='GPU',colab=dict(provenance=[])),nbformat=4,nbformat_minor=5)
    path=Path(__file__).with_name('selectable_adaptive_sweep.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
