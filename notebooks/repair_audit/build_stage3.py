"""Build new-glyph generation + frozen stage-2 repair, in one standalone Colab."""
import ast
import hashlib
import json
from pathlib import Path
from build_stage2 import build as build_stage2

ROOT = Path(__file__).resolve().parents[2]


def embedded(source):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == 'RUNTIME_HELPERS' for t in n.targets))


def build():
    base = json.loads(build_stage2().read_text(encoding='utf-8'))
    codes = [''.join(c['source']) for c in base['cells'] if c['cell_type'] == 'code']
    legacy = json.loads((ROOT/'notebooks/cyclic_stroke_repair.ipynb').read_text(encoding='utf-8'))
    historical = embedded(''.join(legacy['cells'][1]['source']))
    helpers = embedded(codes[0])
    # Generation support is taken from the same standalone historical notebook.
    names = ['diffusion_model/selectable_adaptive_sweep.py', 'diffusion_model/score_pilot.py',
             'diffusion_model/multiglyph_pilot.py', 'diffusion_model/pilot_data.py',
             'stroke_model/convert_reference_csv.py', 'stroke_model/prepare_stroke_zips.py',
             'stroke_model/reference_only_data.py', 'stroke_model/adaptive_ink_constraints.py']
    helpers.update({name: historical[name] for name in names})
    helpers['repair_audit/stage3.py'] = (ROOT/'repair_audit/stage3.py').read_text(encoding='utf-8')
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in helpers.items()}
    setup = '\n'.join('RUNTIME_HELPERS='+repr(helpers) if line.startswith('RUNTIME_HELPERS=') else
                      'SOURCE_HASHES='+repr(hashes) if line.startswith('SOURCE_HASHES=') else line
                      for line in codes[0].splitlines())
    setup = setup.replace('/content/hsqm_stage2_audit_runtime', '/content/hsqm_stage3_audit_runtime')
    setup = setup.replace("'timm', 'numpy'", "'pandas', 'tqdm', 'timm', 'numpy'")
    setup = setup.replace('진단 코드 준비 완료. 다음 셀에서 기존 결과 ZIP과 best.pt를 지정하세요.',
                          '실행 코드 준비 완료. 다음 셀에서 기존 학습 자료 경로를 확인하세요.')
    setup += '''
from repair_audit.stage3 import choose_rows, validate_rows, materialize_inputs, shape_comparison, write_stage3_tables
from diffusion_model.pilot_data import prepare_pilot_data
from diffusion_model.score_pilot import training_catalog, load_catalog_reference
from diffusion_model.selectable_adaptive_sweep import prepare_groups, generate_cases
import matplotlib.pyplot as plt
print('3단계 준비 완료. 기존 결과 ZIP 업로드는 필요 없습니다.')
'''
    cells = []
    def add(kind, source):
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# 3단계: 다른 글자에서 삭제 효과·획 형태 검증

새 L4 GPU 세션에서 **코드 셀 1→7** 실행. 학습하지 않습니다.
기존 4자를 제외한 **12자(4자씩 3묶음)**를 골라 오류 이미지를 새로 생성합니다.
각 묶음의 정상 폰트 입력도 반드시 함께 보정합니다. 생성에 오류가 없어도 제외하지 않습니다.

비교: 입력 / VAE / 보정 없는 재확산 / 기존 보정 / 2단계 경계 보호 보정.
보정 설정과 획 추출기는 그대로 고정하며, 반듯하게 그리는 새 규칙은 추가하지 않습니다.
기존 학습 목록의 다른 글자를 평가하므로, 미학습 문자·다른 폰트 일반화 실험은 아닙니다.

필요 자료: 기존 best.pt, 옆의 run_config.json·selected_paths.csv·conversion_report.json,
학습 때 사용한 기준 글자·기준 획 압축 자료(target.zip / target_stroke.zip 등).
2번에서 기존 경로를 복구합니다. 결과 ZIP만으로는 새 글자의 기준 획을 만들 수 없습니다.
''')
    add('code', setup)
    add('markdown', '''## 2번: 자료 경로 확인
기본 체크포인트 경로는 이전과 같습니다. 자료 자동 복구가 실패하면 `DATA_ROOT`,
`CSV_PATH` 또는 기존 `MANIFEST` 경로를 지정합니다. Drive 원본은 수정하지 않습니다.
''')
    add('code', '''# 2. 기존 자료 복구·체크포인트 확인
CHECKPOINT = '/content/drive/Shareddrives/2026 자율연구/HSQM/v2_topology_runs/20261104_214705_208224/best.pt'
MANIFEST = ''
DATA_ROOT = ''
CSV_PATH = ''
EXPECTED_SHA = '92b8d5dfac59a4e44d632464836877e1cc242b2a4bc72cd6e58d5fcef84cbbf7'
REPAIR_STEPS, REPAIR_STRENGTH, REPAIR_SEED = 60, .25, 2026
BOUNDARY_CONFIG = dict(uncertainty_pixels=1., responsibility_floor=.5)
GENERATION_STEPS, GENERATION_STRENGTH, GENERATION_SEED = 100, .65, 100
MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
DRIVE_RESULT_DIR = '/content/drive/MyDrive/HSQM_stage3_results'
if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
if file_hash(CHECKPOINT) != EXPECTED_SHA:
    raise ValueError('이전 단계와 다른 체크포인트입니다. 경로를 확인하세요.')
preparation = prepare_pilot_data(CHECKPOINT, data_root=DATA_ROOT or None,
    csv_path=CSV_PATH or None, manifest=MANIFEST or None, reference_only=True,
    local_root='/content/hsqm_stage3_reference_data')
MANIFEST = preparation['manifest']
CATALOG = training_catalog(MANIFEST)
print('사용 가능한 기준 글자:', len(CATALOG), '| 다음 셀에서 12자를 미리 확인하세요.')
''')
    add('markdown', '''## 3번: 새 글자 12자 미리 확인
기본값은 목록에 고르게 분포한 12자입니다. 구조별 균형을 보장하는 표집은 아닙니다.
그림을 보고 받침·곡선·여러 접합부가 포함되도록 `SELECTED_ROWS`를 수정해도 됩니다.
수정 시 화면에 표시된 **row 번호** 12개를 입력합니다. 기존 4자(row 1,2,3,5)는 제외합니다.
선택은 생성 전에 고정합니다. 결과가 안 좋은 글자를 나중에 빼지 않습니다.
''')
    add('code', '''# 3. 생성 전에 선택 확정. 직접 고르려면 None 대신 [row번호, ...] 입력
SELECTED_ROWS = None
NUMBER_OF_GLYPHS = 12
if SELECTED_ROWS is None:
    SELECTED_ROWS = choose_rows(CATALOG, NUMBER_OF_GLYPHS)
ROW_GROUPS = validate_rows(CATALOG, SELECTED_ROWS)
by_row = {r['manifest_row']: r for r in CATALOG}
fig, axes = plt.subplots(len(ROW_GROUPS), 4, figsize=(10, 3*len(ROW_GROUPS)), squeeze=False)
for ax, row in zip(axes.flat, SELECTED_ROWS):
    ref, _ = load_catalog_reference(by_row[row])
    ax.imshow(1-ref[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'row {row}'); ax.axis('off')
plt.tight_layout(); plt.show(); plt.close(fig)
print('생성할 묶음:', ROW_GROUPS)
print('글자가 충분히 다양하면 4번 실행. 바꾸려면 위 SELECTED_ROWS를 수정하고 이 셀 재실행.')
''')
    add('code', '''# 4. 새 오류 이미지 생성 + 정상 대조군 준비 (보정 설정은 변경하지 않음)
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
OUTPUT = Path('/content')/('repair_stage3_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
OUTPUT.mkdir(exist_ok=False)
ROW_GROUPS = validate_rows(CATALOG, SELECTED_ROWS)
GROUPS = prepare_groups(MANIFEST, ROW_GROUPS, OUTPUT/'generation_sources')
EXPERIMENT = dict(checkpoint_sha256=EXPECTED_SHA, selections=ROW_GROUPS,
    generation_strength=GENERATION_STRENGTH, generation_seed=GENERATION_SEED,
    generation_steps=GENERATION_STEPS, model_id=MODEL_ID, revision=REVISION,
    selection_scope='new glyphs in training reference catalog; not unseen-font generalization',
    data_preparation=preparation)
json_write(OUTPUT/'experiment.json', EXPERIMENT)
json_write(OUTPUT/'runtime_manifest.json', dict(source_hashes=SOURCE_HASHES,
    model_id=MODEL_ID, revision=REVISION, checkpoint_sha256=EXPECTED_SHA,
    torch_version=torch.__version__, settings=dict(steps=REPAIR_STEPS,
    strength=REPAIR_STRENGTH, seed=REPAIR_SEED, boundary_config=BOUNDARY_CONFIG)))
# If rerunning from generation after repair, release the previous GPU models.
for name in ('pipe', 'extractor'):
    globals().pop(name, None)
gc.collect(); torch.cuda.empty_cache()
generation_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, revision=REVISION,
    torch_dtype=torch.float16, use_safetensors=True)
generation_pipe.scheduler = DDIMScheduler.from_config(generation_pipe.scheduler.config)
generation_pipe.enable_attention_slicing(); generation_pipe.to('cuda')
try:
    GENERATED = generate_cases(GROUPS, strengths=(GENERATION_STRENGTH,), seeds=(GENERATION_SEED,),
        steps=GENERATION_STEPS, revision=REVISION, pipeline=generation_pipe, include_control=False)
finally:
    del generation_pipe; gc.collect(); torch.cuda.empty_cache()
CASES, GENERATION_FAILURES = materialize_inputs(GROUPS, OUTPUT, EXPECTED_SHA)
json_write(OUTPUT/'archive_metadata.json', dict(experiment=EXPERIMENT, selected_cases=CASES))
METRICS, FAILURES = [], list(GENERATION_FAILURES)
write_stage3_tables(OUTPUT, CASES, METRICS, FAILURES)
print('입력:', [(c['case_id'], c['kind']) for c in CASES])
print('생성 실패:', GENERATION_FAILURES)
for c in CASES:
    if c['kind'] == 'generated':
        print(c['case_id']); display(Image.open(Path(c['directory'])/'input.png').resize((384,384)))
print('오류가 적거나 심한 입력도 그대로 평가합니다. 다음 5번 실행.')
''')
    load = codes[2].replace('# 3.', '# 5.').replace('4번 셀', '6번 셀')
    add('code', load)
    run = codes[3].replace('# 4.', '# 6.').replace('METRICS, FAILURES = [], []',
        'METRICS, FAILURES = [], list(GENERATION_FAILURES)')
    run = run.replace('write_tables(OUTPUT, CASES, METRICS, FAILURES)',
                      'write_stage3_tables(OUTPUT, CASES, METRICS, FAILURES)')
    run = run.replace("display(Image.open(Path(case['directory'])/'audit/comparison.png'))",
        "shape_comparison(case)\n        display(Image.open(Path(case['directory'])/'audit/shape_comparison.png'))")
    run = run.replace('4번 셀 재실행', '6번 셀 재실행').replace('2번 셀부터', '4번 셀부터').replace('2번부터', '4번부터')
    add('code', run)
    add('markdown', '''## 판정 항목: 오류 제거와 획 형태를 따로 보기
- `shape_comparison.png`: 글자별 입력/VAE/일반/기존/수정 결과 확대 비교.
- 정상 입력에서도 곧은 획이 휘거나, 곡선이 울거나, 끝점·접합부가 어색해지는지 봅니다.
- 일반 재확산과 비교해서 우리 보정이 추가한 변형인지 구분합니다.
- `human_review.csv`: `input_has_error`, `error_reduced`, `thinner`, `new_error`,
  `straight_or_curve_worsened`, `endpoint_deformed`, `junction_deformed`, `residual_artifact`,
  `style_changed`를 yes/no/uncertain으로 기록할 수 있습니다.
- 정상 입력의 error_reduced는 n/a. 생성 입력에 실제 오류가 없으면 input_has_error=no,
  error_reduced=n/a로 적습니다. 이름을 모르는 글자는 row 번호만으로 추적 가능합니다.
- 자동 폭 지표는 위치 이동·임계값에도 영향을 받습니다. 반듯함/정오 점수로 해석하지 않습니다.

7번 결과 ZIP을 보내주세요. 검토표를 먼저 채우지 않아도 이미지로 함께 판단할 수 있습니다.
실패와 심한 변형도 ZIP에 남깁니다. 형태 보정 알고리즘은 이 결과를 보고 다음 단계에서 결정합니다.
''')
    save = codes[4].replace('# 5.', '# 7.').replace('write_tables(OUTPUT, CASES, METRICS, FAILURES)',
        'write_stage3_tables(OUTPUT, CASES, METRICS, FAILURES)')
    add('code', save)
    base['cells'] = cells
    path = ROOT/'notebooks/repair_audit/stage3.ipynb'
    path.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
