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
                 'stroke_model/reference_only_data.py','repair_audit/selectable_target_inputs.py',
                 'repair_audit/selection_cache.py']:
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
    setup=setup.replace("if not torch.cuda.is_available():\n    raise RuntimeError('런타임 유형을 GPU로 바꾼 뒤 다시 실행하세요.')\nprint('GPU:', torch.cuda.get_device_name(0))",
        "print('실행 장치:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (전처리·선택 가능, 모델 실행은 GPU 필요)')")
    setup+='''\nfrom google.colab import output
output.enable_custom_widget_manager()
import ipywidgets as widgets
import numpy as np
from diffusion_model.pilot_data import prepare_pilot_data
from diffusion_model.score_pilot import training_catalog, load_catalog_reference
from repair_audit.selectable_target_inputs import (prepare_selected_sheet, save_prototype,
    load_prototype, load_reference_image)
from repair_audit.selection_cache import prepare_selection_cache,selection_thumbnails,thumbnail,reference_identity
'''
    cells=[]
    def add(kind,source):
        source=source.strip()+'\n'
        if kind=='code':ast.parse(source)
        cells.append(dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True),
            **(dict(outputs=[],execution_count=None) if kind=='code' else {})))
    add('markdown','''# 원하는 글자 선택 → 2×2 생성 → 획별 목표 가이드 보정

**이전 실험 결과 ZIP은 필요 없습니다.** 기존 Drive의 획 모델 best.pt와 기준 글자·기준 획 데이터를 사용합니다.
GPU(T4 가능)에서 코드 셀 **1→7** 실행하세요.
전처리와 선택 확인만 할 때는 CPU 런타임에서 1→4까지만 실행해도 됩니다.

1. 설치 → 2. 데이터 목록 → 3. 선택용 전처리 → 4. 글자 선택 → 5. 모델 로딩 → 6. 생성·보정 → 7. ZIP 다운로드.
3번에서 로컬 자료·썸네일을 한 번 준비합니다. 4번의 검색/미리보기에서는 원본 파일을 읽지 않습니다.
다른 글자를 시험할 때는 **4번의 선택을 바꾸고 6번을 다시 실행**하면 됩니다. 모델은 다시 로딩하지 않습니다.
매번 별도 실행 폴더를 만들어 기존 결과를 보존합니다.

기본: 선택한 2×2 시트 × 시드 100/101 = 생성 2장 + 정상 시트 1장, 총 3장 보정.
생성할 입력 폰트와 보정 기준 획은 따로 선택합니다. 한 실험의 같은 글자는 기준 한 종류를 공유합니다.
기준은 실행 전에 변경할 수 있으며 실행 때 복사한 조건은 이전 결과와 함께 보존됩니다.
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
SOURCE_CATALOG = CATALOG
print('기준 목록:',len(CATALOG),'개. 3번 전처리를 실행하세요.')
''')
    add('code','''# 3. 선택용 전처리. 원본 읽기·로컬 자료 복사·썸네일 생성은 여기서 한 번 수행
def preparation_progress(done,total):
    if done==1 or done%10==0 or done==total: print('선택 자료 준비',done,'/',total,flush=True)
CATALOG = prepare_selection_cache(SOURCE_CATALOG,'/content/hsqm_picker_cache',
    source_fingerprint=file_hash(TRAIN_MANIFEST),progress=preparation_progress)
SELECTION_THUMBNAILS = selection_thumbnails(CATALOG)
bank_index = Path(PROTOTYPE_BANK)/'index.json'
PROTOTYPE_RECORDS = json.loads(bank_index.read_text(encoding='utf-8')) if bank_index.is_file() else {}
PROTOTYPE_PREVIEWS = {}
for key in PROTOTYPE_RECORDS:
    condition,record = load_prototype(PROTOTYPE_BANK,key)
    PROTOTYPE_PREVIEWS[key] = thumbnail(condition['reference'])
print('전처리 완료:',len(CATALOG),'개. 4번 선택 화면을 여세요.')
print('문자명 자동 연결:',sum(bool(e['resolved_character']) for e in CATALOG),'/',len(CATALOG))
''')
    add('code', (ROOT/'notebooks/repair_audit/selection_ui.py').read_text(encoding='utf-8'))
    load=old_codes[2].replace('# 3.','# 5.')
    load="if not torch.cuda.is_available():\n    raise RuntimeError('모델 실행에는 GPU 런타임이 필요합니다. 전처리·선택은 CPU에서도 가능합니다.')\n"+load
    start=load.index('for case in CASES:')
    end=load.index('extractor.eval()',start)
    load=load[:start]+"EXTRACTOR_SIZE = state['image_size']\n"+load[end:]
    load=load.replace("json_write(OUTPUT/'scheduler_config.json',dict(pipe.scheduler.config))",
                      'SCHEDULER_CONFIG = dict(pipe.scheduler.config)')
    load=load.replace("json_write(OUTPUT/'versions.json',{name:importlib.metadata.version(name) for name in",
                      'VERSIONS = {name:importlib.metadata.version(name) for name in')
    load=load.replace("('torch','diffusers','transformers','timm','numpy','scipy','scikit-image')})",
                      "('torch','diffusers','transformers','timm','numpy','scipy','scikit-image')}")
    load=load.replace("print('로딩 완료. 4번 실행.')","print('로딩 완료. 6번 실행.')")
    add('code',load)
    run=old_codes[3]
    run=run.replace('# 4. 두 경로 비교. 같은 세션에서 재실행하면 완료 사례는 건너뜀',
                    '''# 6. 현재 선택으로 새 실행. 다른 글자를 선택한 뒤 이 셀만 다시 실행 가능
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
                    '현재 결과는 7번으로 다운로드하세요. 6번 재실행은 새로운 실험입니다.')
    add('code',run)
    add('markdown','''## 결과 확인과 다음 선택
- fresh_inputs/<case_id>/diffusion_comparison.png: 입력 / 목표 / 일반 재역확산 / 가이드 재역확산.
- 정상 대조군도 함께 보고 획 두께·원본 스타일·오류 수정·배경 조화를 확인하세요.
- 목표 자체가 기준 폰트처럼 보일 수 있습니다. 실제 확산 결과와 구분해서 판단하세요.
- 입력 폰트와 고정 기준이 같은 경우에는 교차 폰트 일반화로 해석하지 마세요. 기존 모델의 학습 폰트 분리도 별도로 확인해야 합니다.
- 현재 자료의 문자/폰트 메타데이터가 불완전하므로, 입력과 기준의 문자 대응 및 기준 폰트 통일은 직접 확인해야 합니다.
- 각 생성 시트 저장 및 보정 사례 완료/실패/수동 중단 시 Drive에 ZIP을 백업합니다.
- 6번을 다시 실행하면 현재 선택으로 **새 실험**을 시작합니다. 이전 실행은 별도 폴더·ZIP으로 남습니다.
- 7번은 가장 최근 실행을 다운로드합니다. 중간 latent나 세션 간 재개는 지원하지 않습니다.
''')
    add('code',old_codes[4].replace('# 5.','# 7.'))
    nb=dict(base,cells=cells)
    for i,cell in enumerate(cells):cell['id']=f'selectable-target-{i}'
    path=ROOT/'notebooks/repair_audit/target_guided_repair.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
