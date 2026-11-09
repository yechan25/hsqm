"""True prompt-first generation, then optional detected-map reverse diffusion."""
import ast,json
from pathlib import Path
from build_detected_extra_diffusion import build as base

def build():
    original=json.loads(base().read_text(encoding='utf-8'))
    nb=dict(original,cells=[])
    def add(kind,text):
        if kind=='code':ast.parse(text)
        c=dict(cell_type=kind,metadata={},source=text.strip().splitlines(keepends=True))
        if kind=='code':c.update(execution_count=None,outputs=[])
        nb['cells'].append(c)
    add('markdown',"""# 프롬프트 → 순수 text-to-image → 오류 지도 → 재역확산

첫 생성에는 이미지·폰트·표준 글자·획 모델을 전달하지 않습니다. 입력은 프롬프트와 seed뿐입니다.
1 준비 → 2 프롬프트로 실제 생성 → 3 저장 모델/문자 조건 선택 → 4 생성 사진의 문자 영역과 오류 지도 → 5 재역확산 → 6 비교.
실제 문자가 틀리거나 배치가 달라도 폰트 렌더링으로 바꾸거나 결과를 골라내지 않습니다.
문자 위치를 고정 2×2로 가정하지 않습니다. 생성 결과에서 각 글자의 bbox를 지정해야 합니다.
현재 오류 지도 보정은 밝은 단색 배경의 추가 잉크 삭제용입니다. 사진 배경이 복잡하면 명시적으로 중단하며, 폰트 이미지로 대체하지 않습니다.
""")
    setup=''.join(original['cells'][1]['source'])
    setup='\n'.join(line for line in setup.splitlines() if 'apt-get' not in line)
    add('code',setup)
    add('code',"""
from datetime import datetime
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from stroke_model.convert_reference_csv import existing_path
from diffusion_model.prompt_generation import generate_prompt_image
from huggingface_hub import model_info
HSQM=existing_path('/content/drive/Shareddrives/2026 자율연구/HSQM')
MODEL_ID='stable-diffusion-v1-5/stable-diffusion-v1-5'
MODEL_REVISION=model_info(MODEL_ID).sha
CANVAS_SIZE=576
PROMPT='A photograph of a white sign displaying the Korean text "가나다라" in black ink. Exactly these four Hangul characters in a single line. No other text.'
EXPECTED_TEXT='가나다라'  # User intention, not rendered into an input image
GENERATION_STEPS=50
GENERATION_SEED=100
RUN_DIR=HSQM/'prompt_then_repair'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
first=generate_prompt_image(RUN_DIR/'stage1',prompt=PROMPT,
    model_id=MODEL_ID,revision=MODEL_REVISION,steps=GENERATION_STEPS,seed=GENERATION_SEED,size=CANVAS_SIZE)
FIRST_IMAGE=Path(first['generated_image'])
fig,ax=plt.subplots(figsize=(7,7))
with Image.open(FIRST_IMAGE) as im:ax.imshow(im.convert('RGB'))
ax.set_title('Actual prompt-only generation; pixel coordinates for character ROIs');plt.show()
print('First-stage image/template/font inputs:',first['input_image_used'],first['template_used'],first['font_rendering_used'])
""")
    settings=''.join(original['cells'][2]['source'])
    settings='\n'.join(line for line in settings.splitlines() if not line.startswith(('FIRST_IMAGE=None','MODEL_ID =','CANVAS_SIZE =')))
    choose=''.join(original['cells'][3]['source'])
    add('code',settings+'\n'+choose+"""
characters_box.value=EXPECTED_TEXT
CHAR_BOXES=[]  # Fill [(x0,y0,x1,y1), ...] for the four intended characters, in reading order
print('사진에서 각 글자 bbox를 CHAR_BOXES에 지정하세요. 잘못 생성된 문자/위치도 숨기지 않습니다.')
""")
    maps=''.join(original['cells'][4]['source'])
    needle='_,slots=compose_sheet([by_row[r] for r in selected],CANVAS_SIZE)'
    maps=maps.replace(needle,needle+"""
if len(CHAR_BOXES)!=4:raise ValueError('CHAR_BOXES에 실제 생성 사진 속 네 글자의 bbox를 지정하세요. 자동 2x2 영역은 사용하지 않습니다.')
for slot,box in zip(slots,CHAR_BOXES):
    if len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('Integer bboxes required')
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=CANVAS_SIZE and 0<=y0<y1<=CANVAS_SIZE):raise ValueError('BBox outside image')
    slot['bbox']=list(box)
""")
    add('code',maps)
    repair=''.join(original['cells'][5]['source']).replace("OUTPUT_DIR=HSQM/'detected_extra_diffusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')", "OUTPUT_DIR=RUN_DIR/('repair_'+datetime.now().strftime('%H%M%S_%f'))")
    repair=repair.replace('objective_override=fixed_objective,','objective_override=fixed_objective,slot_bboxes=CHAR_BOXES,')
    repair=repair.replace('revision=model_info(MODEL_ID).sha','revision=MODEL_REVISION')
    add('code',repair)
    comparison=''.join(original['cells'][6]['source'])
    comparison=comparison.replace("OUTPUT_DIR.parent/(OUTPUT_DIR.name+'_detected_diffusion.zip')", "RUN_DIR.parent/(RUN_DIR.name+'_prompt_repair.zip')")
    comparison=comparison.replace("for path in OUTPUT_DIR.rglob('*'):","for path in RUN_DIR.rglob('*'):").replace("z.write(path,path.relative_to(OUTPUT_DIR))", "z.write(path,path.relative_to(RUN_DIR))")
    add('code',comparison)
    path=Path(__file__).with_name('prompt_then_repair.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path
if __name__=='__main__':print(build())
