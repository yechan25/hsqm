"""Build separate template-geometry generalization experiment."""
import ast,json
from pathlib import Path
from build_topo_guided_diffusion import build_guided

def build():
    notebook=json.loads(build_guided().read_text(encoding='utf-8'))
    def put(i,text):
        if notebook['cells'][i]['cell_type']=='code':ast.parse(text)
        notebook['cells'][i]['source']=text.strip().splitlines(keepends=True)
    put(0,"""# 정답 픽셀 없는 획 구조 유도 — 일반화 시험

표준 글자의 획 추출 결과에서 정규화한 방향·비대칭·상대 위치·접촉 위치를 구합니다.
생성물에서 같은 획 추출 모델로 구한 특성의 smooth L1 손실을 최적화합니다.
표준 글자 RGB와의 픽셀 손실·삭제 지도·정답 윤곽 맞추기는 보정에 사용하지 않습니다.
이 지표는 기하 모멘트 기반 대리 지표이며 완전한 위상/문자 인식 지표가 아닙니다.

기본 INIT_MODE='noise': 첫 생성은 프롬프트와 잡음만 사용합니다. 이전 폰트 입력 기본값은 폐기했습니다.
3번 셀에서 표준 글자 네 개를 선택하고 같은 문자를 순서대로 입력한 뒤 폰트를 선택하세요.
이전 보정 실험과 같은 STEPS=100, strength=0.65, seed=100을 사용합니다(실제 역확산 65단계).
잡음 생성은 INIT_MODE='noise'로 명시한 실험에만 사용합니다. SD1.5의 네 글자 생성은 보장되지 않습니다.
INPUT_SHEET 경로를 지정하면 직접 준비한 2×2 이미지를 사용합니다.
외부 이미지의 정답 획/깨끗한 짝 이미지는 필요하지 않습니다. 문자 순서·배치는 사용자가 맞춰야 합니다.

기존 훈련 목록의 표준 글자를 조건으로 사용하므로 미학습 문자 일반화가 입증되는 실험은 아닙니다.
다른 폰트/잡음 seed의 성공률, 정상 획 손상, 문자 판독 결과로 평가해야 합니다.
SD1.5가 네 글자를 만들지 못한 결과도 제외하지 않습니다. 템플릿으로 형태가 끌리는 편향이 남을 수 있습니다.
실제 GPU 품질은 아직 검증 전입니다. 1→5번 셀을 실행하세요.
""")
    put(1,''.join(notebook['cells'][1]['source'])+"\nsubprocess.run(['apt-get','update','-qq'],check=True)\nsubprocess.run(['apt-get','install','-y','-qq','fonts-nanum'],check=True)\n")
    settings=''.join(notebook['cells'][2]['source']).split('from diffusion_model.run_topo_guidance import')[0]
    settings=settings.replace('STEPS = 100  # strength=.65에서 실제 역확산 65단계','STEPS = 100')
    settings+="""
# Separate experimental path; old pixel-control notebook is preserved.
SEED=100  # Repeat with 101, 102, ...; keep all results
INIT_MODE='noise'  # First generation must not use an input glyph image
STRENGTH=.65
AUTO_FONT_INPUT=False
INPUT_SHEET=None
GUIDANCE=dict(local_edit=False,clean_step_ratio=.01,start_fraction=.5,end_fraction=1.,
              every=1,backtracks=5,state_update='clean_residual',max_new_local_ink=None)
"""
    put(2,settings)
    put(3,''.join(notebook['cells'][3]['source'])+"""
import ipywidgets as widgets
glyph_box=widgets.Text(description='4문자:',placeholder='예: 가 걔 곁 괴 (선택 순서와 일치)')
font_paths=sorted(Path('/usr/share/fonts/truetype/nanum').glob('*.ttf'))
default_font=next((str(p) for p in font_paths if p.name=='NanumMyeongjo.ttf'),None)
font_box=widgets.Dropdown(options=[(p.name,str(p)) for p in font_paths],value=default_font,description='입력 폰트:')
prompt_box=widgets.Textarea(description='Prompt',value='Four large Korean Hangul characters in a 2 by 2 layout. Preserve each input character and its position. Black ink on plain white background. No additional text, tables, grid lines, captions or logos.',layout=widgets.Layout(width='95%',height='90px'))
if INIT_MODE=='external' and AUTO_FONT_INPUT:
    display(glyph_box,font_box,prompt_box)
else:
    display(glyph_box,prompt_box)
""")
    put(4,"""
from datetime import datetime
from huggingface_hub import model_info
from diffusion_model.run_topo_guidance import generate_guided_pair
if not prompt_box.value.strip():raise ValueError('선택한 4문자와 순서가 일치하는 프롬프트를 입력하세요.')
OUTPUT_DIR=HSQM/'structure_generalization'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
input_metadata=None
run_input_sheet=INPUT_SHEET
if INIT_MODE=='external' and AUTO_FONT_INPUT and INPUT_SHEET is None:
    from diffusion_model.crossfont_input import compose_font_input
    from diffusion_model.score_pilot import training_catalog
    catalog={r['manifest_row']:r for r in training_catalog(TRAIN_MANIFEST)}
    glyphs=glyph_box.value.split()
    if len(glyphs)==1 and len(glyphs[0])==4:glyphs=list(glyphs[0])
    if not font_box.value:raise ValueError('입력 폰트를 선택하세요.')
    image,input_metadata=compose_font_input([catalog[r] for r in reference_select.value],glyphs,font_box.value,CANVAS_SIZE)
    import tempfile
    input_path=Path(tempfile.mkdtemp(prefix='hsqm_crossfont_',dir='/content'))/'input.png'
    image.save(input_path)
    run_input_sheet=input_path
    print('독립 폰트 입력 (선택한 표준 문자/순서와 맞는지 확인):');display(image)
generation=generate_guided_pair(TRAIN_MANIFEST,list(reference_select.value),CHECKPOINT,OUTPUT_DIR,
    model_id=MODEL_ID,revision=model_info(MODEL_ID).sha,steps=STEPS,canvas_size=CANVAS_SIZE,
    guidance=GUIDANCE,seed=SEED,strength=STRENGTH,guidance_signal='template_structure',init_mode=INIT_MODE,
    input_sheet=run_input_sheet,prompt_text=prompt_box.value.strip(),
    appearance_weight=0,extra_weight=0,source_spill_weight=0,direct_reference=False,
    compare_structure=False,compare_uniform=False)
if input_metadata:
    (OUTPUT_DIR/'input_font.json').write_text(json.dumps(input_metadata,ensure_ascii=False,indent=2),encoding='utf-8')
(OUTPUT_DIR/'git_commit.txt').write_text(commit,encoding='utf-8')
import matplotlib.pyplot as plt
display_pages=list(generation['pages'])
if INIT_MODE=='external':display_pages.insert(1,dict(method='external_input',input=str(OUTPUT_DIR/'external_input.png')))
fig,axes=plt.subplots(1,len(display_pages),figsize=(5*len(display_pages),5))
for ax,page in zip(axes,display_pages):
    with Image.open(page['input']) as im:ax.imshow(im.convert('RGB'))
    ax.set_title('Template (conditioning only)' if page['method']=='reference' else page['method']);ax.axis('off')
plt.show()
print('Primary audit:',generation['structure_audit'])
print('Template encoded as input:',generation['reference_encoded_as_input'])
print('Pixel audit is auxiliary only; it is NOT the optimization metric.')
""")
    put(5,"""
import json,zipfile,pandas as pd
from google.colab import files
rows=[]
for method,data in generation['structure_audit'].items():
    for slot,loss in enumerate(data['per_glyph']):rows.append(dict(method=method,slot=slot+1,descriptor_loss=loss))
frame=pd.DataFrame(rows);display(frame)
frame.to_csv(OUTPUT_DIR/'descriptor_scores.csv',index=False)
archive=OUTPUT_DIR.parent/(OUTPUT_DIR.name+'_structure_debug.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUTPUT_DIR.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUTPUT_DIR))
files.download(str(archive))
""")
    path=Path(__file__).with_name('structure_generalization.ipynb')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path
if __name__=='__main__':print(build())
