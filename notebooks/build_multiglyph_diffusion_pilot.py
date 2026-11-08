"""Reuse the five-cell scoring flow, but jointly generate entire glyph sheets."""
import ast
import json
from pathlib import Path
from build_diffusion_score_pilot import build

ROOT = Path(__file__).resolve().parents[1]


def build_multiglyph():
    notebook = json.loads(build().read_text(encoding='utf-8'))

    def source(index, text):
        text = text.strip()
        if notebook['cells'][index]['cell_type'] == 'code':
            ast.parse(text)
        notebook['cells'][index]['source'] = text.splitlines(keepends=True)

    source(0, '''# 여러 글자를 한 번에 생성 → 칸별로 잘라 획 분석

**여러 글자를 따로 생성해서 합치는 방식이 아닙니다.** 학습 기준 글자들을 한 장에 배치하고,
그 전체 이미지를 Stable Diffusion img2img에 넣어 한 번에 역확산합니다.
생성한 전체 이미지를 저장한 뒤, 미리 고정한 칸 좌표로 잘라 기존 Topo와 원본 점수로 분석합니다.
OCR 성공 여부로 자르거나 결과를 제외하지 않습니다. 빈 칸·합쳐진 글자·다른 글자도 남깁니다.

기본 시험: 같은 기준 목록의 앞 1/4/9개, 전체 576×576, strength 0.65/0.85, seed 100.
공통으로 포함된 글자를 비교할 수 있습니다. 서로 다른 글자들을 전체 평균만으로 비교하지 마세요.
글자가 많아질수록 글자당 픽셀도 줄어듭니다. 개수 효과와 해상도 효과를 분리해 보려면
2번 셀에서 FIXED_CELL_PIXELS=192로 설정하세요(전체 크기가 달라져 계산량도 달라집니다).
글자 수가 많으면 오류가 더 생길지는 실제 결과로 확인해야 합니다.

1. 준비·Drive → 2. 기존 학습 ZIP/CSV 자동 준비 → 3. 학습 글자 여러 개 선택
4. 전체 이미지 공동 생성·원본 보기 → 5. 잘라낸 글자별 추출·점수·감점 지도

GPU 런타임에서 실행하세요. 5번 셀은 기존 평가를 실행하며 역확산 guidance는 적용하지 않습니다.
현재 원본 점수는 옅은 생성 배경에도 민감하므로 감점 지도와 원본을 함께 확인하세요.''')
    setup = ''.join(notebook['cells'][2]['source'])
    setup = setup.replace('from diffusion_model.pilot_data import prepare_pilot_data',
        'from diffusion_model.pilot_data import prepare_pilot_data\nfrom diffusion_model.multiglyph_pilot import generate_multiglyph_pilot')
    setup = setup.replace('STRENGTHS = [.25, .45, .65]\nSEEDS = [100, 101]',
        'GLYPH_COUNTS = [1, 4, 9]\nCANVAS_SIZE = 576\nFIXED_CELL_PIXELS = None  # None: 전체 크기 고정. 192: 글자당 크기 고정 비교\nSTRENGTHS = [.65, .85]\nSEEDS = [100]')
    source(2, setup)
    source(3, '''
import io, math
import numpy as np
from PIL import Image, ImageDraw
catalog = training_catalog(TRAIN_MANIFEST)
by_row = {item['manifest_row']: item for item in catalog}
needed = max(GLYPH_COUNTS)
if len(catalog) < needed:
    raise ValueError(f'사용 가능한 학습 기준은 {len(catalog)}개입니다. GLYPH_COUNTS를 줄이세요.')
reference_select = widgets.SelectMultiple(
    options=[(f"train row {item['manifest_row']} | {item['char_id'] or '(그림 확인)'}", item['manifest_row']) for item in catalog],
    value=tuple(item['manifest_row'] for item in catalog[:needed]), rows=12,
    description='여러 글자:', layout=widgets.Layout(width='98%'))
preview = widgets.Image(format='png', width=500, height=500)
def update_preview(change=None):
    selected = list(reference_select.value)[:needed]
    side = math.ceil(math.sqrt(needed))
    sheet = Image.new('RGB', (160*side, 180*side), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        with np.load(by_row[row]['path'], allow_pickle=False) as data:
            glyph = Image.fromarray(np.uint8((1-data['reference'][0])*255)).convert('RGB')
        x, y = (index%side)*160, (index//side)*180
        sheet.paste(glyph.resize((160,160)), (x,y))
        draw.text((x+4,y+160), f'slot {index+1} | row {row}', fill='black')
    buffer=io.BytesIO(); sheet.save(buffer,format='PNG'); preview.value=buffer.getvalue()
reference_select.observe(update_preview,names='value'); update_preview()
display(widgets.VBox([reference_select,preview]))
print(f'최소 {needed}개를 선택하세요. 각 개수 조건은 선택 목록의 앞부분을 사용합니다.')
print('문자명이 없는 경우 그림으로 확인하세요. 서로 다른 글자를 비교하려면 같은 글자의 폰트 변형을 중복 선택하지 마세요.')
''')
    source(4, '''
from datetime import datetime
from huggingface_hub import model_info
MODEL_REVISION = model_info(MODEL_ID).sha
OUTPUT_DIR = HSQM/'multiglyph_diffusion_pilots'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
generation = generate_multiglyph_pilot(TRAIN_MANIFEST, list(reference_select.value), OUTPUT_DIR,
    glyph_counts=GLYPH_COUNTS, canvas_size=CANVAS_SIZE, cell_pixels=FIXED_CELL_PIXELS,
    strengths=STRENGTHS, seeds=SEEDS, steps=STEPS, model_id=MODEL_ID, revision=MODEL_REVISION)
generation['data_preparation'] = preparation
(OUTPUT_DIR/'generation_report.json').write_text(json.dumps(generation,ensure_ascii=False,indent=2),encoding='utf-8')
(OUTPUT_DIR/'git_commit.txt').write_text(commit,encoding='utf-8')
import matplotlib.pyplot as plt
pages=generation['pages']
fig,axes=plt.subplots(math.ceil(len(pages)/3),3,figsize=(15,5*math.ceil(len(pages)/3)),squeeze=False)
for ax in axes.flat: ax.axis('off')
for ax,page in zip(axes.flat,pages):
    with Image.open(page['input']) as image: ax.imshow(image.convert('RGB'))
    ax.set_title(page['label'])
plt.tight_layout(); plt.show()
print('전체 이미지와 고정 좌표 글자 crop:',OUTPUT_DIR)
print('생성 오류가 늘었는지는 원본과 잘라낸 결과를 확인하세요. 5번 셀에서 글자별로 평가합니다.')
''')
    notebook['metadata']['colab']['name'] = '여러_글자_공동생성_점수시험.ipynb'
    path = ROOT/'notebooks/test_multiglyph_diffusion_scores.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)
    return path


if __name__ == '__main__':
    build_multiglyph()
