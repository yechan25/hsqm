"""Separate generate -> inspect -> re-noise -> repair notebook."""
import ast,json
from pathlib import Path
from build_structure_generalization import build as build_generalization

def build():
    notebook=json.loads(build_generalization().read_text(encoding='utf-8'))
    def put(i,text):
        if notebook['cells'][i]['cell_type']=='code':ast.parse(text)
        notebook['cells'][i]['source']=text.strip().splitlines(keepends=True)
    put(0,"""# 먼저 생성하고, 필요하면 다시 역확산해서 획 보정

1~3번: 모델/자료 준비와 네 문자 선택. 최초 생성은 INIT_MODE=noise만 허용합니다.
4번: 획 유도 없이 최초 이미지를 생성하고 저장합니다. 이 결과를 먼저 확인하세요.
이미 저장한 생성물을 쓰려면 FIRST_IMAGE에 같은 크기의 2×2 이미지 경로를 넣습니다.
5번: 최초 생성물에 작은 잡음을 다시 넣고 동일 잠재값으로 두 경로를 비교합니다.
6번: 처음 생성 / 재역확산만 / 획 구조 유도 재역확산을 비교하고 진단 ZIP을 받습니다.

재보정은 깨끗한 정답 픽셀을 사용하지 않습니다. 표준 글자는 획 추출·구조 특성의 조건입니다.
첫 생성물과의 RGB 오차는 원래 모양 보존용이며, 구조 손실은 기존 정규화 획 특성을 사용합니다.
이 구조 지표의 추가 획 탐지 한계는 여전히 남아 있으므로 개선을 보장하지 않습니다.
최초 생성은 프롬프트와 잡음만 사용합니다. 폰트 입력 기본값은 폐기했습니다.
오류 판정의 검증된 임계값은 없어서 4번 결과를 보고 사용자가 5번 실행 여부를 정합니다.
""")
    put(2,''.join(notebook['cells'][2]['source']).split('GUIDANCE=dict')[0]+"""
FIRST_IMAGE=None  # Optional existing generated 2x2 image path
REPAIR_STEPS=100
REPAIR_STRENGTH=.2
REPAIR_SEED=101
PRESERVATION_WEIGHT=1.
REPAIR_GUIDANCE=dict(local_edit=False,clean_step_ratio=.003,start_fraction=.25,end_fraction=1.,
    every=1,backtracks=6,state_update='clean_residual',max_new_local_ink=None,
    max_state_change_to_base_step_ratio=.25)
""")
    source=''.join(notebook['cells'][4]['source'])
    source=source.replace('from diffusion_model.run_topo_guidance import generate_guided_pair',
                          'from diffusion_model.two_stage_repair import generate_first_stage')
    start=source.index("OUTPUT_DIR=HSQM/")
    end=source.index('input_metadata=None',start)
    source=source[:start]+"""RUN_DIR=HSQM/'two_stage_repair'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR.mkdir(parents=True,exist_ok=False)
MODEL_REVISION=model_info(MODEL_ID).sha
OUTPUT_DIR=RUN_DIR/'stage1'
"""+source[end:]
    start=source.index('generation=generate_guided_pair(')
    prefix=source[:start]
    # Existing-image mode must not render or generate another source.
    marker=prefix.index('input_metadata=None')
    prefix=prefix[:marker]+"""if FIRST_IMAGE is None:
"""+''.join('    '+line if line.strip() else line for line in prefix[marker:].splitlines(keepends=True))
    source=prefix+"""    first=generate_first_stage(TRAIN_MANIFEST,list(reference_select.value),CHECKPOINT,OUTPUT_DIR,
        model_id=MODEL_ID,revision=MODEL_REVISION,steps=STEPS,canvas_size=CANVAS_SIZE,
        seed=SEED,strength=STRENGTH,init_mode=INIT_MODE,input_sheet=run_input_sheet,
        prompt_text=prompt_box.value.strip())
    FIRST_RESULT=Path(first['generated_image'])
    if input_metadata:
        (OUTPUT_DIR/'input_font.json').write_text(json.dumps(input_metadata,ensure_ascii=False,indent=2),encoding='utf-8')
else:
    FIRST_RESULT=Path(FIRST_IMAGE)
    if not FIRST_RESULT.is_file():raise FileNotFoundError(FIRST_RESULT)
(RUN_DIR/'git_commit.txt').write_text(commit,encoding='utf-8')
print('최초 생성 결과 — 보정이 필요하면 5번 셀을 실행하세요:',FIRST_RESULT)
display(Image.open(FIRST_RESULT))
"""
    put(4,source)
    put(5,"""
from diffusion_model.two_stage_repair import repair_generated_image
OUTPUT_DIR=RUN_DIR/('stage2_'+datetime.now().strftime('%H%M%S_%f'))
generation=repair_generated_image(TRAIN_MANIFEST,list(reference_select.value),CHECKPOINT,
    FIRST_RESULT,OUTPUT_DIR,model_id=MODEL_ID,revision=MODEL_REVISION,
    canvas_size=CANVAS_SIZE,steps=REPAIR_STEPS,strength=REPAIR_STRENGTH,seed=REPAIR_SEED,
    preservation_weight=PRESERVATION_WEIGHT,guidance=REPAIR_GUIDANCE,
    prompt_text=prompt_box.value.strip())
print('두 재보정 경로의 동일 시작 잠재값:',generation['initial_latent_sha256'])
print('최초 생성물 보존 해시:',generation['first_generated_sha256'])
print('구조 손실 및 보존 오차:',generation['structure_audit'])
""")
    notebook['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[]))
    put(6,"""
import matplotlib.pyplot as plt
pages={p['method']:p for p in generation['pages']}
items=[('First generation',FIRST_RESULT),('Re-denoise only',pages['baseline']['input']),
       ('Stroke-guided re-denoise',pages['guided']['input'])]
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,(title,path) in zip(axes,items):
    with Image.open(path) as image:ax.imshow(image.convert('RGB'))
    ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'two_stage_comparison.png',dpi=140);plt.show()
import zipfile
from google.colab import files
archive=RUN_DIR.parent/(RUN_DIR.name+'_two_stage_debug.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in RUN_DIR.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(RUN_DIR))
    z.write(FIRST_RESULT,'first_generated_input.png')
files.download(str(archive))
""")
    path=Path(__file__).with_name('two_stage_repair.ipynb')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path
if __name__=='__main__':print(build())
