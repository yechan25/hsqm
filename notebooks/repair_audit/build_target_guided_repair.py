"""Build the standalone, ZIP-input target-guided Colab pilot."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = [
    'stroke_model/reference_model.py', 'stroke_model/stroke_structure.py',
    'stroke_model/source_structure.py', 'stroke_model/target_stroke_plan.py',
    'diffusion_model/stroke_projected_diffusion.py', 'repair_audit/stage1.py',
    'repair_audit/target_guided_repair.py',
    'repair_audit/fresh_target_inputs.py',
]


def build():
    helpers = {name: (ROOT/name).read_text(encoding='utf-8') for name in MODULES}
    hashes = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in helpers.items()}
    cells = []
    def add(kind, source):
        source = source.strip() + '\n'
        if kind == 'code': ast.parse(source)
        cells.append(dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True),
                          **(dict(outputs=[], execution_count=None) if kind == 'code' else {})))

    add('markdown', '''# 관측 두께·정답 스켈레톤으로 목표 이미지를 만들고 재역확산

GPU 런타임(T4 가능)에서 **코드 셀 1→5** 실행. 기본은 정답 2×2 시트를 새로 확산·역확산한 뒤 보정합니다.
2번 셀에서 **repair_stage4_20261123_132010_265667.zip**을 업로드하세요.
기존 Drive의 best.pt도 필요합니다. ZIP은 기준 획을 가져오는 데 사용합니다.
기본: 3개 글자 그룹 × seed 100/101 = 생성 6장(24칸) + 정상 대조군 3장, 총 **9장** 보정.
strength .65에서 오류가 반드시 발생하지는 않습니다. 생성 실패/정상 결과도 제외하지 않습니다.

획 추출 → 관측 두께 → 정답 스켈레톤 렌더링 → 글자 공통 affine → 획별 수정 강도 → 원본에서 재역확산.
비교는 입력 / 목표(확산 아님) / 일반 재역확산 / 목표 가이드 재역확산입니다.
두 확산 경로는 같은 초기 latent를 사용합니다. 최종 목표 이미지 덮어쓰기는 없습니다.

**점수 주의:** 외부 점수가 없으면 미검증 기하 지표를 사용합니다. 논문 평가기의 재현이나 정오 판정기가 아닙니다.
앞선 CPU 사례에서 24획 중 23획을 수정 대상으로 잡았으므로 정상 대조군의 과잉 수정을 함께 확인하세요.
60 nominal steps × strength .25 = 경로당 실제 15스텝, 시트당 두 경로입니다. GPU 품질은 이번에 검증합니다.
''')
    setup = '''# 1. 설치 및 고정 런타임 준비. 이 노트북만으로 실행 가능
import sys, subprocess, json, hashlib, gc, shutil, traceback, time, zipfile
from pathlib import Path
from datetime import datetime
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'diffusers==0.37.0', 'transformers==4.57.6', 'accelerate', 'safetensors',
    'timm', 'numpy', 'scipy', 'scikit-image', 'pillow'], check=True)
'''
    setup += 'RUNTIME_HELPERS='+repr(helpers)+'\nSOURCE_HASHES='+repr(hashes)+'\n'
    setup += '''RUNTIME = Path('/content/hsqm_target_guided_runtime')
for name, code in RUNTIME_HELPERS.items():
    if hashlib.sha256(code.encode()).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError('Embedded source hash mismatch: '+name)
    target = RUNTIME/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(code, encoding='utf-8')
    (target.parent/'__init__.py').touch()
# A rerun must not keep modules imported by an older notebook in this session.
import importlib
for module_name in list(sys.modules):
    if module_name.split('.')[0] in ('repair_audit', 'stroke_model', 'diffusion_model'):
        del sys.modules[module_name]
if str(RUNTIME) in sys.path: sys.path.remove(str(RUNTIME))
sys.path.insert(0, str(RUNTIME))
importlib.invalidate_caches()
import torch
from PIL import Image, ImageDraw
from IPython.display import display
from google.colab import drive, files
from repair_audit.stage1 import read_archive, load_conditions, file_hash, json_write
from repair_audit.target_guided_repair import repair_generated_sheet, _cached_masks, extract_sheet_masks
from stroke_model.target_stroke_plan import TargetPlanConfig
from repair_audit.fresh_target_inputs import generate_fresh_cases, canonical_sheet
if not torch.cuda.is_available():
    raise RuntimeError('런타임 유형을 GPU로 바꾼 뒤 다시 실행하세요.')
print('GPU:', torch.cuda.get_device_name(0))
'''
    add('code', setup)
    add('markdown', '''## 입력과 설정
- ARCHIVE를 비워 두면 ZIP 업로드 창이 열립니다. best.pt는 업로드하지 말고 Drive 경로를 확인하세요.
- INPUT_MODE='fresh'는 정답 시트에서 새 생성, 'archive'는 ZIP의 기존 생성물을 사용합니다.
- CASE_IDS=None은 ZIP의 3개 그룹 전체. 빠른 시험은 ['group_000_page_001']로 바꾸세요.
- GENERATION_STRENGTHS/SEEDS로 오류 강도와 반복 수를 조절합니다. 생성 설정과 보정 설정은 별개입니다.
- SCORES_JSON은 선택 사항: `{case_id: [[첫칸 획별 0~100점], [...], [...], [...]]}`.
- 기존 ZIP의 저장 추출을 우선 사용하고, 없는 정상 대조군 등은 같은 체크포인트로 추출합니다.
- Drive에 사례 완료/실패/수동 중단 시 ZIP을 백업합니다. 강제 연결 종료는 마지막 백업까지만 보존됩니다.
''')
    add('code', '''# 2. 입력 선택 및 Drive 연결
ARCHIVE = ''
CHECKPOINT = '/content/drive/Shareddrives/2026 자율연구/HSQM/v2_topology_runs/20261104_214705_208224/best.pt'
INPUT_MODE = 'fresh'  # 'archive': 이전 ZIP 생성물만 사용
CASE_IDS = None  # 빠른 시험: ['group_000_page_001']
GENERATION_STRENGTHS = [.65]  # 강도 비교 시 [.45, .65] (사례 수 증가)
GENERATION_SEEDS = [100, 101]
GENERATION_STEPS = 60
SCORES_JSON = ''  # 선택: Drive의 획별 점수 JSON 경로
DRIVE_RESULT_DIR = '/content/drive/MyDrive/HSQM_target_guided_results'
REPAIR_STEPS, REPAIR_STRENGTH, REPAIR_SEED = 60, .25, 2026
PLAN_CONFIG = dict(keep_score=90., full_edit_score=40., maximum_step_blend=.8)
MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
if not Path('/content/drive/MyDrive').is_dir(): drive.mount('/content/drive')
if not ARCHIVE:
    print('4단계 결과 ZIP 하나를 선택하세요.')
    uploaded = files.upload()
    if len(uploaded) != 1: raise ValueError('ZIP 하나만 선택하세요.')
    ARCHIVE = str(Path(next(iter(uploaded))).resolve())
    del uploaded
if not Path(CHECKPOINT).is_file():
    raise FileNotFoundError('CHECKPOINT를 기존 best.pt 경로로 수정한 뒤 2번을 다시 실행하세요.')
OUTPUT = Path('/content')/('target_guided_repair_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
CASES, EXPERIMENT = read_archive(ARCHIVE, OUTPUT, CASE_IDS, include_controls=True)
if INPUT_MODE not in ('fresh','archive'): raise ValueError('INPUT_MODE는 fresh 또는 archive')
SEED_CASES = list(CASES)
if file_hash(CHECKPOINT) != EXPERIMENT['checkpoint_sha256'].lower():
    raise ValueError('기존 ZIP과 best.pt SHA256이 다릅니다.')
SCORES = json.loads(Path(SCORES_JSON).read_text(encoding='utf-8')) if SCORES_JSON else {}
json_write(OUTPUT/'runtime_manifest.json', dict(source_hashes=SOURCE_HASHES,
    model_id=MODEL_ID, revision=REVISION, checkpoint_sha256=file_hash(CHECKPOINT),
    input_archive_sha256=file_hash(ARCHIVE), torch_version=torch.__version__,
    gpu=torch.cuda.get_device_name(0), scores=SCORES,
    settings=dict(steps=REPAIR_STEPS, strength=REPAIR_STRENGTH, seed=REPAIR_SEED, plan=PLAN_CONFIG,
        input_mode=INPUT_MODE,generation_strengths=GENERATION_STRENGTHS,
        generation_seeds=GENERATION_SEEDS,generation_steps=GENERATION_STEPS)))
PREVIEW_CASES = [c for c in CASES if c['kind']=='normal_control'] if INPUT_MODE=='fresh' else CASES
for case in PREVIEW_CASES:
    print(case['case_id'], case['kind'])
    with Image.open(Path(case['directory'])/'input.png') as old:
        preview = canonical_sheet(load_conditions(case['directory']),old.width) if INPUT_MODE=='fresh' else old.copy()
    display(preview.resize((288,288)))
if INPUT_MODE=='fresh':
    n = len(PREVIEW_CASES)
    print('예정: 새 생성',n*len(GENERATION_STRENGTHS)*len(GENERATION_SEEDS),'장 + 정상',n,'장')
print('저장 위치:', OUTPUT)
''')
    add('code', '''# 3. 기존 획 추출기와 SD1.5 로딩 (추가 학습 없음)
from stroke_model.reference_model import ReferenceStrokeModel
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
if state.get('architecture') != 'reference_v2_topology_loss_v1':
    raise ValueError('reference_v2_topology_loss_v1 체크포인트가 필요합니다.')
extractor = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False)
extractor.load_state_dict(state['model'], strict=True)
for case in CASES:
    if any(c['reference'].shape[-2:] != (state['image_size'],state['image_size'])
           for c in load_conditions(case['directory'])):
        raise ValueError('조건 이미지와 추출기 해상도가 다릅니다.')
extractor.eval().requires_grad_(False)  # 추출할 때만 GPU에 올림
del state; gc.collect(); torch.cuda.empty_cache()
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, revision=REVISION,
    torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, thresholding=False)
pipe.to('cuda'); pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
for module in (pipe.unet, pipe.vae, pipe.text_encoder): module.eval().requires_grad_(False)
pipe.vae.to(dtype=torch.float32)
json_write(OUTPUT/'scheduler_config.json',dict(pipe.scheduler.config))
import importlib.metadata
json_write(OUTPUT/'versions.json',{name:importlib.metadata.version(name) for name in
    ('torch','diffusers','transformers','timm','numpy','scipy','scikit-image')})
print('로딩 완료. 4번 실행.')
''')
    add('code', '''# 4. 두 경로 비교. 같은 세션에서 재실행하면 완료 사례는 건너뜀
def save_snapshot():
    archive = shutil.make_archive(str(OUTPUT)+'_checkpoint', 'zip', OUTPUT)
    if DRIVE_RESULT_DIR:
        destination = Path(DRIVE_RESULT_DIR); destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination/Path(archive).name)
    return archive

# 기존 오류 생성물이 아닌 정답 기준 시트에서 새 입력을 생성합니다.
if INPUT_MODE == 'fresh':
    generation_root = OUTPUT/'fresh_inputs'
    generation_report = generation_root/'generation_report.json'
    expected_count = sum(c['kind']=='normal_control' for c in SEED_CASES)*(1+len(GENERATION_STRENGTHS)*len(GENERATION_SEEDS))
    cached_generation = json.loads(generation_report.read_text(encoding='utf-8')) if generation_report.is_file() else None
    if cached_generation and len(cached_generation['cases']) == expected_count:
        if (cached_generation['strengths'] != GENERATION_STRENGTHS or cached_generation['seeds'] != GENERATION_SEEDS
                or cached_generation['steps'] != GENERATION_STEPS):
            raise ValueError('생성 설정이 바뀌었습니다. 2번부터 새로 실행하세요.')
        CASES = cached_generation['cases']
        print('완료된 새 생성 결과 재사용:',len(CASES))
    else:
        if generation_root.exists():
            generation_root.rename(OUTPUT/('interrupted_generation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        try:
            # 표준 img2img 입력은 fp16 UNet/text embedding dtype을 따릅니다.
            # 기존 커스텀 보정 샘플러는 이후 fp32 VAE로 돌립니다.
            pipe.vae.to(dtype=torch.float16)
            CASES = generate_fresh_cases(pipe,SEED_CASES,generation_root,
                strengths=GENERATION_STRENGTHS,seeds=GENERATION_SEEDS,steps=GENERATION_STEPS,
                on_saved=lambda case,cases: save_snapshot())
        finally:
            pipe.vae.to(dtype=torch.float32)
            save_snapshot()
    print('새 생성 + 정상 대조군:',len(CASES),'장. 이제 각 시트를 보정합니다.')

def show_comparison(case):
    root = Path(case['directory']); repair = root/'target_repair'
    panels = [('input', root/'input.png'), ('target: NOT diffusion', repair/'target_plan/aligned_target.png'),
              ('redenoise only', repair/'reverse/redenoise_only.png'),
              ('target guided', repair/'reverse/stroke_projected.png')]
    side = 288; image = Image.new('RGB',(side*4,side+30),'white'); draw = ImageDraw.Draw(image)
    for i,(label,path) in enumerate(panels):
        draw.text((i*side+5,8),label,fill='black')
        with Image.open(path) as panel: image.paste(panel.convert('RGB').resize((side,side)),(i*side,30))
    image.save(root/'diffusion_comparison.png'); display(image)

for case in CASES:
    root = Path(case['directory']); result = root/'target_repair'
    try:
        if (result/'pipeline.json').is_file():
            print('완료 사례 건너뜀:',case['case_id']); show_comparison(case); continue
        if result.exists():
            result.rename(root/('interrupted_attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        source = Image.open(root/'input.png').convert('RGB')
        conditions = load_conditions(root)
        if INPUT_MODE == 'archive':
            with zipfile.ZipFile(ARCHIVE) as z: masks = _cached_masks(z,case,source,conditions)
        else: masks = None  # 새 생성물에 옛 입력의 추출 마스크를 재사용하지 않음
        if masks is None:
            print(case['case_id'],'저장 추출 없음: 기존 모델로 추출',flush=True)
            try:
                extractor.to('cuda')
                masks = extract_sheet_masks(source,conditions,extractor)
            finally:
                extractor.to('cpu'); gc.collect(); torch.cuda.empty_cache()
        else: print(case['case_id'],'동일 입력의 저장 추출 재사용',flush=True)
        cache = root/'cached_extraction'; cache.mkdir(exist_ok=True)
        import numpy as np
        for i,mask in enumerate(masks,1): np.savez_compressed(cache/f'slot_{i}.npz',extracted_masks=mask)
        arm_clock = {}
        def progress(arm,step,total,row):
            now = time.monotonic()
            arm_clock.setdefault(arm,now)
            if step == 1 or step%3 == 0 or step == total:
                elapsed = now-arm_clock[arm]
                eta = elapsed/(step-1)*(total-step) if step>1 else None
                print(case['case_id'],arm,f'{step}/{total}',
                      '남은 초≈'+str(round(eta)) if eta is not None else '시간 측정 중',flush=True)
        guide, report = repair_generated_sheet(pipe,extractor,source,conditions,result,
            cached_masks=masks,scores=SCORES.get(case['case_id']),config=TargetPlanConfig(**PLAN_CONFIG),
            steps=REPAIR_STEPS,strength=REPAIR_STRENGTH,seed=REPAIR_SEED,
            prompt=case.get('prompt',''),negative_prompt=case.get('negative_prompt',''),progress=progress)
        selected = sum(s['edit_strength']>0 for s in guide.report['strokes'])
        print('수정 대상:',selected,'/',len(guide.report['strokes']), '획; 실제 스텝:',report['actual_steps'])
        show_comparison(case)
        (root/'failure.json').unlink(missing_ok=True)
    except KeyboardInterrupt:
        print('수동 중단: 부분 결과 백업. 같은 세션에서 4번을 다시 실행하면 미완료 사례부터 재계산합니다.')
        break
    except Exception as exc:
        json_write(root/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        print('실패:',case['case_id'],str(exc),flush=True)
    finally:
        snapshot = save_snapshot()
        print('백업:',snapshot,flush=True)
        gc.collect(); torch.cuda.empty_cache()
print('현재 완료',sum((Path(c['directory'])/'target_repair/pipeline.json').is_file() for c in CASES),'/',len(CASES))
''')
    add('markdown', '''## 결과 확인
- fresh 모드의 실제 입력과 보정 결과는 fresh_inputs/<case_id>/에 있습니다.
- generation_report.json에 생성 강도·시드·안전 필터 여부가 기록됩니다. 기존 ZIP 입력 폴더와 구분해서 보세요.
- diffusion_comparison.png에서 정상 대조군도 함께 확인하세요. 목표 이미지는 확산 결과가 아닙니다.
- 획이 얇아지는지, 정상 획이 변하는지, 오류 획이 목표 위치로 이동하는지, 배경이 자연스러운지 봅니다.
- target_plan/plan.json에 관측 두께·정합·점수·수정 강도가 있습니다.
- reverse/report.json에 동일 초기 latent 해시와 경로별 진단이 있습니다. 진단값 감소가 글자 정답률을 뜻하지는 않습니다.
- 이번 가이드는 매 스텝 VAE를 왕복합니다. T4에서도 실행 가능하도록 구성했지만 속도·메모리·품질은 실제 GPU 검증 전입니다.
- 재연결로 런타임이 사라지면 Drive 백업을 확보하세요. 이 노트북은 **세션 간 중간 latent 재개를 지원하지 않습니다**.
''')
    add('code', '''# 5. 결과 ZIP 다운로드 (완료/부분 결과 모두 가능)
archive = save_snapshot()
print('Drive:',DRIVE_RESULT_DIR)
print('다운로드 후 이 ZIP을 보내주세요:',archive)
files.download(archive)
''')
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(colab=dict(name='target_guided_repair.ipynb'),
                                  accelerator='GPU', kernelspec=dict(name='python3',display_name='Python 3')))
    for index, cell in enumerate(cells): cell['id'] = f'target-repair-{index}'
    path = ROOT/'notebooks/repair_audit/target_guided_repair_archive.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__ == '__main__': print(build())
