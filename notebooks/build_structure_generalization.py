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

기본 INIT_MODE='noise': 표준 이미지를 VAE로 인코딩하지 않고 잡음에서 생성합니다.
3번 셀에서 선택한 네 표준 글자와 동일한 문자를 순서대로 포함한 영문 프롬프트를 입력하세요.
INIT_MODE='external'이면 같은 2×2 배치의 다른 폰트 이미지 경로를 INPUT_SHEET에 넣습니다.
외부 이미지의 정답 획/깨끗한 짝 이미지는 필요하지 않습니다. 문자 순서·배치는 사용자가 맞춰야 합니다.

기존 훈련 목록의 표준 글자를 조건으로 사용하므로 미학습 문자 일반화가 입증되는 실험은 아닙니다.
다른 폰트/잡음 seed의 성공률, 정상 획 손상, 문자 판독 결과로 평가해야 합니다.
SD1.5가 네 글자를 만들지 못한 결과도 제외하지 않습니다. 템플릿으로 형태가 끌리는 편향이 남을 수 있습니다.
실제 GPU 품질은 아직 검증 전입니다. 1→5번 셀을 실행하세요.
""")
    settings=''.join(notebook['cells'][2]['source']).split('from diffusion_model.run_topo_guidance import')[0]
    settings=settings.replace('STEPS = 100  # strength=.65에서 실제 역확산 65단계','STEPS = 50')
    settings+="""
# Separate experimental path; old pixel-control notebook is preserved.
SEED=100  # Repeat with 101, 102, ...; keep all results
INIT_MODE='noise'  # 'external' for a different font/input sheet
INPUT_SHEET=None
GUIDANCE=dict(local_edit=False,clean_step_ratio=.01,start_fraction=.5,end_fraction=1.,
              every=1,backtracks=5,state_update='clean_residual',max_new_local_ink=None)
"""
    put(2,settings)
    put(3,''.join(notebook['cells'][3]['source'])+"""
import ipywidgets as widgets
prompt_box=widgets.Textarea(description='Prompt',placeholder='Describe the SAME four selected characters in 2x2 reading order',layout=widgets.Layout(width='95%',height='90px'))
display(prompt_box)
""")
    put(4,"""
from datetime import datetime
from huggingface_hub import model_info
from diffusion_model.run_topo_guidance import generate_guided_pair
if not prompt_box.value.strip():raise ValueError('선택한 4문자와 순서가 일치하는 프롬프트를 입력하세요.')
OUTPUT_DIR=HSQM/'structure_generalization'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
generation=generate_guided_pair(TRAIN_MANIFEST,list(reference_select.value),CHECKPOINT,OUTPUT_DIR,
    model_id=MODEL_ID,revision=model_info(MODEL_ID).sha,steps=STEPS,canvas_size=CANVAS_SIZE,
    guidance=GUIDANCE,seed=SEED,guidance_signal='template_structure',init_mode=INIT_MODE,
    input_sheet=INPUT_SHEET,prompt_text=prompt_box.value.strip(),
    appearance_weight=0,extra_weight=0,source_spill_weight=0,direct_reference=False,
    compare_structure=False,compare_uniform=False)
(OUTPUT_DIR/'git_commit.txt').write_text(commit,encoding='utf-8')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,len(generation['pages']),figsize=(15,5))
for ax,page in zip(axes,generation['pages']):
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
