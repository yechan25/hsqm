"""Build the main Colab: choose glyphs, generate, extract, and repair. No stage ZIP."""
import ast
import hashlib
import json
from pathlib import Path
from build_target_guided_repair import build as build_archive

ROOT=Path(__file__).resolve().parents[2]


def build():
    base=json.loads(build_archive().read_text(encoding='utf-8'))
    old_codes=[''.join(c['source']) for c in base['cells'] if c['cell_type']=='code']
    setup=old_codes[0]
    tree=ast.parse(setup)
    helpers=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='RUNTIME_HELPERS')
    for name in ['diffusion_model/pilot_data.py','diffusion_model/score_pilot.py',
                 'stroke_model/convert_reference_csv.py','stroke_model/prepare_stroke_zips.py',
                 'stroke_model/reference_only_data.py','repair_audit/selectable_target_inputs.py']:
        helpers[name]=(ROOT/name).read_text(encoding='utf-8')
    hashes={name:hashlib.sha256(code.encode()).hexdigest() for name,code in helpers.items()}
    setup='\n'.join('RUNTIME_HELPERS='+repr(helpers) if line.startswith('RUNTIME_HELPERS=') else
                    'SOURCE_HASHES='+repr(hashes) if line.startswith('SOURCE_HASHES=') else line
                    for line in setup.splitlines())
    setup=setup.replace("'timm', 'numpy', 'scipy', 'scikit-image', 'pillow'",
                        "'timm', 'numpy', 'scipy', 'scikit-image', 'pillow', 'pandas', 'tqdm', 'ipywidgets'")
    setup=setup.replace('from repair_audit.stage1 import read_archive, load_conditions, file_hash, json_write',
                        'from repair_audit.stage1 import load_conditions, file_hash, json_write')
    setup=setup.replace('repair_generated_sheet, _cached_masks, extract_sheet_masks',
                        'repair_generated_sheet, extract_sheet_masks')
    setup+='''\nfrom google.colab import output
output.enable_custom_widget_manager()
import ipywidgets as widgets
import numpy as np
from diffusion_model.pilot_data import prepare_pilot_data
from diffusion_model.score_pilot import training_catalog, load_catalog_reference
from repair_audit.selectable_target_inputs import (prepare_selected_sheet, lock_prototype,
    load_prototype, load_reference_image)
'''
    cells=[]
    def add(kind,source):
        source=source.strip()+'\n'
        if kind=='code':ast.parse(source)
        cells.append(dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True),
            **(dict(outputs=[],execution_count=None) if kind=='code' else {})))
    add('markdown','''# 원하는 글자 선택 → 2×2 생성 → 획별 목표 가이드 보정

**이전 실험 결과 ZIP은 필요 없습니다.** 기존 Drive의 획 모델 best.pt와 기준 글자·기준 획 데이터를 사용합니다.
GPU(T4 가능)에서 코드 셀 **1→6** 실행하세요.

1. 설치 → 2. 기준 데이터 준비 → 3. 원하는 글자 4칸 선택 → 4. 모델 로딩 → 5. 생성·보정 → 6. ZIP 다운로드.
다른 글자를 시험할 때는 **3번의 선택을 바꾸고 5번을 다시 실행**하면 됩니다. 모델은 다시 로딩하지 않습니다.
매번 별도 실행 폴더를 만들어 기존 결과를 보존합니다.

기본: 선택한 2×2 시트 × 시드 100/101 = 생성 2장 + 정상 시트 1장, 총 3장 보정.
생성할 입력 폰트와 보정 기준 획은 따로 선택합니다. 기준은 글자마다 한 종류를 Drive에 잠그고 모든 입력 폰트에 공유합니다.
기준을 입력 폰트에 맞춰 자동 선택하지 않습니다. 원하는 글자가 여러 칸에 반복돼도 됩니다.
글자 이름이 없으면 썸네일과 row 번호로 고릅니다. 숫자 ID를 한글 문자로 추측하지 않습니다.
점수는 외부 평가 점수가 없을 때 미검증 기하 지표를 씁니다. 실제 GPU 품질은 미검증입니다.
''')
    add('code',setup)
    add('code','''# 2. Drive의 기존 체크포인트·기준 자료 준비 (결과 ZIP 불필요)
CHECKPOINT = '/content/drive/Shareddrives/2026 자율연구/HSQM/v2_topology_runs/20261104_214705_208224/best.pt'
DATA_ROOT = '/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/automated_train_dataset'
CSV_PATH = None  # 기본: best.pt 옆 selected_paths.csv
TRAIN_MANIFEST = None  # 현재 세션의 유효한 manifest가 있으면 지정 가능
PROTOTYPE_BANK = '/content/drive/MyDrive/HSQM_fixed_character_prototypes_v1'
GENERATION_STRENGTHS = [.65]  # [.45, .65]로 강도 비교 가능
GENERATION_SEEDS = [100, 101]
GENERATION_STEPS = 60
CANVAS_SIZE = 576
SCORES_JSON = ''  # 선택 사항: 생성 case_id -> 네 칸의 획별 점수(0~100)
DRIVE_RESULT_DIR = '/content/drive/MyDrive/HSQM_target_guided_results'
REPAIR_STEPS, REPAIR_STRENGTH, REPAIR_SEED = 60, .25, 2026
PLAN_CONFIG = dict(keep_score=90., full_edit_score=40., maximum_step_blend=.8)
MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
if not Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive')
if not Path(CHECKPOINT).is_file(): raise FileNotFoundError('CHECKPOINT 경로를 확인하세요.')
PREPARATION = prepare_pilot_data(CHECKPOINT,DATA_ROOT,csv_path=CSV_PATH,manifest=TRAIN_MANIFEST,
    local_root='/content/hsqm_selectable_reference_data',reference_only=True)
TRAIN_MANIFEST = Path(PREPARATION['manifest'])
CATALOG = training_catalog(TRAIN_MANIFEST)
if not CATALOG: raise ValueError('선택 가능한 기준 글자가 없습니다.')
print('선택 가능한 기준:',len(CATALOG),'개. 3번 셀에서 선택하세요.')
''')
    add('code','''# 3. 원하는 글자 선택. 칸 순서: 왼쪽 위 / 오른쪽 위 / 왼쪽 아래 / 오른쪽 아래
LABEL_TO_ENTRY = {f"row {e['manifest_row']} | {e.get('char_id') or '(이름 없음: 아래 그림 확인)'}":e for e in CATALOG}
LABELS = list(LABEL_TO_ENTRY)
bank_index = Path(PROTOTYPE_BANK)/'index.json'
LOCKED_KEYS = list(json.loads(bank_index.read_text(encoding='utf-8'))) if bank_index.is_file() else []
reference_selectors = [widgets.Combobox(options=LABELS,value=LABELS[min(i,len(LABELS)-1)],
    ensure_option=True,description=f'칸 {i+1}',layout=widgets.Layout(width='98%')) for i in range(4)]
character_keys = [widgets.Combobox(options=LOCKED_KEYS,value='',description='글자 이름',
    layout=widgets.Layout(width='98%')) for _ in range(4)]
prototype_selectors = [widgets.Combobox(options=LABELS,value='',ensure_option=True,description='고정 기준',
    layout=widgets.Layout(width='98%')) for _ in range(4)]
pin_buttons = [widgets.Button(description='이 글자의 기준 잠금') for _ in range(4)]
identity_checks = [widgets.Checkbox(value=False,description='입력과 기준이 같은 글자임을 확인') for _ in range(4)]
bank_status = widgets.Output()
selection_preview = widgets.Output()
catalog_preview = widgets.Output()
catalog_page = widgets.IntSlider(value=0,min=0,max=(len(CATALOG)-1)//24,description='목록 페이지')
def selected_rows():
    values=[w.value for w in reference_selectors]
    if any(v not in LABEL_TO_ENTRY for v in values):
        raise ValueError('각 칸에 목록에서 유효한 항목을 선택하세요.')
    return [LABEL_TO_ENTRY[v]['manifest_row'] for v in values]
def selected_keys():
    keys=[w.value.strip() for w in character_keys]
    for key,check in zip(keys,identity_checks):
        if not key or not check.value:raise ValueError('각 칸의 글자 이름과 동일 글자 확인이 필요합니다.')
        load_prototype(PROTOTYPE_BANK,key)
    return keys
def pin(index):
    with bank_status:
        bank_status.clear_output(wait=True)
        try:
            value=prototype_selectors[index].value
            if value not in LABEL_TO_ENTRY:raise ValueError('한 기준 폰트에서 기준 획을 먼저 선택하세요.')
            record=lock_prototype(CATALOG,LABEL_TO_ENTRY[value]['manifest_row'],character_keys[index].value,PROTOTYPE_BANK)
            print('고정됨:',record['character'],'row',record['manifest_row'],record['sha256'][:12])
            for key_widget in character_keys:
                key_widget.options=list(json.loads((Path(PROTOTYPE_BANK)/'index.json').read_text(encoding='utf-8')))
            update_selection()
        except (ValueError,FileNotFoundError) as exc:print(str(exc))
def update_selection(change=None):
    with selection_preview:
        selection_preview.clear_output(wait=True)
        try:
            entries=[LABEL_TO_ENTRY[w.value] for w in reference_selectors]
            conditions=[dict(reference=load_reference_image(e)) for e in entries]
            print('생성용 입력 폰트:')
            display(canonical_sheet(conditions,384))
            print('선택 row:',selected_rows())
            keys=[w.value.strip() for w in character_keys]
            try:
                fixed=[load_prototype(PROTOTYPE_BANK,key)[0] for key in keys]
                print('폰트에 무관하게 공유하는 고정 기준 획:')
                display(canonical_sheet(fixed,384))
            except (FileNotFoundError,ValueError,KeyError):print('글자별 기준을 한 번 잠근 뒤 사용할 수 있습니다.')
        except (KeyError,ValueError) as exc: print('목록 항목을 선택하세요:',str(exc))
def update_catalog(change=None):
    with catalog_preview:
        catalog_preview.clear_output(wait=True)
        entries=CATALOG[catalog_page.value*24:(catalog_page.value+1)*24]
        page=Image.new('RGB',(6*100,4*120),'white');draw=ImageDraw.Draw(page)
        for i,e in enumerate(entries):
            ref=load_reference_image(e)
            glyph=Image.fromarray(np.uint8(np.clip(1-ref[0],0,1)*255)).convert('RGB')
            x,y=(i%6)*100,(i//6)*120
            page.paste(glyph.resize((100,100)),(x,y));draw.text((x+3,y+102),f"row {e['manifest_row']}",fill='black')
        display(page)
def source_changed(index):
    identity_checks[index].value=False
    update_selection()
for i,w in enumerate(reference_selectors):w.observe(lambda change,i=i:source_changed(i),names='value')
for i,w in enumerate(character_keys):w.observe(lambda change,i=i:source_changed(i),names='value')
for i,b in enumerate(pin_buttons):b.on_click(lambda button,i=i:pin(i))
catalog_page.observe(update_catalog,names='value')
cards=[widgets.VBox([reference_selectors[i],character_keys[i],prototype_selectors[i],pin_buttons[i],identity_checks[i]]) for i in range(4)]
display(widgets.VBox([widgets.HTML('입력 row는 생성할 이미지입니다. 글자 이름을 입력해도 이미지가 바뀌지 않습니다.<br>그림에서 같은 글자인지 확인하고, 고정 기준에는 그 글자의 기준 획 row를 선택해 한 번 잠그세요. 이미 잠긴 글자는 이름만 지정합니다.'),
    *cards,bank_status,selection_preview,widgets.HTML('전체 글자 목록: 입력 폰트 및 최초 고정 기준 선택용'),
    catalog_page,catalog_preview]))
update_selection();update_catalog()
print('선택 후 처음에는 4번→5번. 이후 다른 글자는 선택만 바꾸고 5번 재실행.')
''')
    load=old_codes[2].replace('# 3.','# 4.')
    start=load.index('for case in CASES:')
    end=load.index('extractor.eval()',start)
    load=load[:start]+"EXTRACTOR_SIZE = state['image_size']\n"+load[end:]
    load=load.replace("json_write(OUTPUT/'scheduler_config.json',dict(pipe.scheduler.config))",
                      'SCHEDULER_CONFIG = dict(pipe.scheduler.config)')
    load=load.replace("json_write(OUTPUT/'versions.json',{name:importlib.metadata.version(name) for name in",
                      'VERSIONS = {name:importlib.metadata.version(name) for name in')
    load=load.replace("('torch','diffusers','transformers','timm','numpy','scipy','scikit-image')})",
                      "('torch','diffusers','transformers','timm','numpy','scipy','scikit-image')}")
    load=load.replace("print('로딩 완료. 4번 실행.')","print('로딩 완료. 5번 실행.')")
    add('code',load)
    run=old_codes[3]
    run=run.replace('# 4. 두 경로 비교. 같은 세션에서 재실행하면 완료 사례는 건너뜀',
                    '''# 5. 현재 선택으로 새 실행. 다른 글자를 선택한 뒤 이 셀만 다시 실행 가능
SELECTED_ROWS = selected_rows()
PROTOTYPE_KEYS = selected_keys()
OUTPUT = Path('/content')/('selected_target_repair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUTPUT.mkdir()
SEED_CASES = prepare_selected_sheet(CATALOG,SELECTED_ROWS,OUTPUT/'selected_reference',size=CANVAS_SIZE,
    prototype_keys=PROTOTYPE_KEYS,prototype_bank=PROTOTYPE_BANK)
for condition in load_conditions(SEED_CASES[0]['directory']):
    if condition['reference'].shape[-2:] != (EXTRACTOR_SIZE,EXTRACTOR_SIZE):
        raise ValueError('선택한 기준과 모델 해상도가 다릅니다.')
SCORES = json.loads(Path(SCORES_JSON).read_text(encoding='utf-8')) if SCORES_JSON else {}
INPUT_MODE = 'fresh'
json_write(OUTPUT/'runtime_manifest.json',dict(source_hashes=SOURCE_HASHES,
    selected_rows=SELECTED_ROWS,prototype_keys=PROTOTYPE_KEYS,prototype_bank=str(PROTOTYPE_BANK),
    prototype_records=SEED_CASES[0]['prototype_records'],
    data_preparation=PREPARATION,checkpoint_sha256=file_hash(CHECKPOINT),
    manifest_sha256=file_hash(TRAIN_MANIFEST),model_id=MODEL_ID,revision=REVISION,
    scores=SCORES,gpu=torch.cuda.get_device_name(0),
    settings=dict(generation_strengths=GENERATION_STRENGTHS,generation_seeds=GENERATION_SEEDS,
        generation_steps=GENERATION_STEPS,steps=REPAIR_STEPS,strength=REPAIR_STRENGTH,
        seed=REPAIR_SEED,plan=PLAN_CONFIG)))
json_write(OUTPUT/'scheduler_config.json',SCHEDULER_CONFIG)
json_write(OUTPUT/'versions.json',VERSIONS)
print('생성용 입력 폰트（고정 기준과 별도）')
display(Image.open(Path(SEED_CASES[0]['directory'])/'input.png').resize((384,384)))''')
    run=run.replace("        if INPUT_MODE == 'archive':\n            with zipfile.ZipFile(ARCHIVE) as z: masks = _cached_masks(z,case,source,conditions)\n        else: masks = None  # 새 생성물에 옛 입력의 추출 마스크를 재사용하지 않음",
                    '        masks = None  # 현재 생성물에서 새로 추출')
    run=run.replace('같은 세션에서 4번을 다시 실행하면 미완료 사례부터 재계산합니다.',
                    '현재 결과는 6번으로 다운로드하세요. 5번 재실행은 새로운 실험입니다.')
    add('code',run)
    add('markdown','''## 결과 확인과 다음 선택
- fresh_inputs/<case_id>/diffusion_comparison.png: 입력 / 목표 / 일반 재역확산 / 가이드 재역확산.
- 정상 대조군도 함께 보고 획 두께·원본 스타일·오류 수정·배경 조화를 확인하세요.
- 목표 자체가 기준 폰트처럼 보일 수 있습니다. 실제 확산 결과와 구분해서 판단하세요.
- 입력 폰트와 고정 기준이 같은 경우에는 교차 폰트 일반화로 해석하지 마세요. 기존 모델의 학습 폰트 분리도 별도로 확인해야 합니다.
- 현재 자료의 문자/폰트 메타데이터가 불완전하므로, 입력과 기준의 문자 대응 및 기준 폰트 통일은 직접 확인해야 합니다.
- 각 생성 시트 저장 및 보정 사례 완료/실패/수동 중단 시 Drive에 ZIP을 백업합니다.
- 5번을 다시 실행하면 현재 선택으로 **새 실험**을 시작합니다. 이전 실행은 별도 폴더·ZIP으로 남습니다.
- 6번은 가장 최근 실행을 다운로드합니다. 중간 latent나 세션 간 재개는 지원하지 않습니다.
''')
    add('code',old_codes[4].replace('# 5.','# 6.'))
    nb=dict(base,cells=cells)
    for i,cell in enumerate(cells):cell['id']=f'selectable-target-{i}'
    path=ROOT/'notebooks/repair_audit/target_guided_repair.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
