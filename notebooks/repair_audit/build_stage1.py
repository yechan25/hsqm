"""Build a standalone upload-to-Colab notebook; no unpublished GitHub files needed."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build():
    legacy = json.loads((ROOT / 'notebooks/cyclic_stroke_repair.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(legacy['cells'][1]['source'])
    assignments = [n for n in ast.parse(setup).body if isinstance(n, ast.Assign)]
    embedded = next(ast.literal_eval(n.value) for n in assignments if any(
        isinstance(t, ast.Name) and t.id == 'RUNTIME_HELPERS' for t in n.targets))
    names = ['stroke_model/reference_model.py', 'stroke_model/stroke_reference_fit.py',
             'diffusion_model/fitted_candidate_diffusion.py', 'stroke_model/restore_settle_erase.py']
    helpers = {name: embedded[name] for name in names}
    helpers['repair_audit/stage1.py'] = (ROOT / 'repair_audit/stage1.py').read_text(encoding='utf-8')
    hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in helpers.items()}
    cells = []
    def add(kind, source):
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# 1단계: 획 보정 실패 원인 진단

**새 Colab의 L4 GPU에서 코드 셀 1→5 실행. 모델 학습·새 이미지 생성은 하지 않습니다.**
기존 `cyclic_stroke_repair_*.zip` 또는 `selectable_adaptive_*.zip`과 그 실행의 획 추출기 `best.pt`가 필요합니다.
ZIP에는 `experiment.json`, `cases.json`, `group_*/source_conditions/condition_1~4.npz`, 원래 generation 이미지가 있어야 합니다.
이 노트북은 필요한 소스가 내장되어 있어 GitHub 업로드·별도 코드 ZIP이 필요 없습니다.

비교: 원본 / VAE 왕복 / 일반 재역확산 / 기존 복원·정리·삭제 보정.
정상 폰트 입력 대조군도 같은 보정을 실행하며 실패 사례도 기록합니다.
원본 기준 폰트는 기존 controller의 조건입니다. 기준 전체 RGB를 보정 정답으로 넣지 않습니다.

**주의:** 비교용 VAE는 모든 경로에서 float32 posterior mode를 공유합니다.
과거 노트북의 확률적 VAE 인코딩과 달라 과거 PNG의 완전 재현 실험은 아닙니다.
일반/가이드 재역확산은 이 실행 안에서 정확히 같은 초기 latent·잡음·시간 단계를 공유합니다.
자동 두께 값은 고정 중심점의 거리변환 지표이며 이동·임계값에 영향을 받습니다. 정오·손상 여부는 사람이 판정합니다.
''')
    add('code', '''# 1. 설치·내장 코드 준비 (새 GPU 세션 권장)
import sys, subprocess, json, hashlib, gc, shutil, traceback
from pathlib import Path
from datetime import datetime
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'diffusers==0.37.0', 'transformers==4.57.6', 'accelerate', 'safetensors',
    'timm', 'numpy', 'scipy', 'scikit-image', 'pillow', 'matplotlib'], check=True)
''' + f'RUNTIME_HELPERS={helpers!r}\nSOURCE_HASHES={hashes!r}\n' + '''
ROOT = Path('/content/hsqm_stage1_audit_runtime')
ROOT.mkdir(exist_ok=True)
for package in ('stroke_model', 'diffusion_model', 'repair_audit'):
    (ROOT/package).mkdir(exist_ok=True)
    (ROOT/package/'__init__.py').touch()
for name, source in RUNTIME_HELPERS.items():
    path = ROOT/name
    path.write_text(source, encoding='utf-8')
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASHES[name]:
        raise RuntimeError('Embedded source hash mismatch: '+name)
sys.path.insert(0, str(ROOT))
for name in list(sys.modules):
    if name.split('.')[0] in ('stroke_model', 'diffusion_model', 'repair_audit'):
        sys.modules.pop(name)
import torch
from PIL import Image
from IPython.display import display
from google.colab import drive, files
from repair_audit.stage1 import *
if not torch.cuda.is_available():
    raise RuntimeError('런타임 유형을 GPU(L4 권장)로 바꾼 뒤 새 세션에서 실행하세요.')
print('GPU:', torch.cuda.get_device_name(0))
print('진단 코드 준비 완료. 다음 셀에서 기존 결과 ZIP과 best.pt를 지정하세요.')
''')
    add('markdown', '''## 2번 셀에서 바꿀 값
- `ARCHIVE`: 기존 실험 ZIP의 Drive 경로. 비워두면 ZIP 하나를 업로드합니다.
- `CHECKPOINT`: 그 실험에 사용한 best.pt의 Drive 경로. 기존 알려진 경로를 초기값으로 넣었습니다.
- `CASE_IDS=None`: ZIP 안의 생성 사례 전부. 일부를 고르면 선택 목록이 기록되며 전체 성능으로 해석하지 않습니다.
- `RUN_REFERENCE_WIDTH_PROBE=False`: 첫 실행은 그대로 둡니다. True는 기준 획을 1모델픽셀 두껍게 한 민감도 스트레스 시험입니다. 다른 폰트 일반화 검증이 아닙니다.
- 정상 대조군은 항상 포함됩니다. 첫 시험은 기존 한 장(4글자) + 정상 대조군으로 시작해도 됩니다.
''')
    add('code', '''# 2. 입력·설정 (이 셀의 경로만 수정)
ARCHIVE = ''
CHECKPOINT = '/content/drive/Shareddrives/2026 자율연구/HSQM/v2_topology_runs/20261104_214705_208224/best.pt'
CASE_IDS = None
REPAIR_STEPS = 60
REPAIR_STRENGTH = .25
REPAIR_SEED = 2026
RUN_REFERENCE_WIDTH_PROBE = False
DRIVE_RESULT_DIR = '/content/drive/MyDrive/HSQM_stage1_results'
MODEL_ID = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
REVISION = '451f4fe16113bff5a5d2269ed5ad43b0592e9a14'

if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
if not ARCHIVE:
    print('기존 실험 결과 ZIP 하나를 업로드하세요. best.pt 업로드 창이 아닙니다.')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('ZIP 하나만 선택하세요.')
    ARCHIVE = str(Path(next(iter(uploaded))).resolve())
    del uploaded
if not Path(CHECKPOINT).is_file():
    raise FileNotFoundError('CHECKPOINT를 실제 best.pt 경로로 고친 뒤 2번 셀을 다시 실행하세요.')
OUTPUT = Path('/content')/('repair_stage1_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
CASES, EXPERIMENT = read_archive(ARCHIVE, OUTPUT, CASE_IDS, include_controls=True)
if file_hash(CHECKPOINT) != EXPERIMENT['checkpoint_sha256'].lower():
    raise ValueError('best.pt SHA256이 기존 실험과 다릅니다. 올바른 체크포인트를 지정하세요.')
json_write(OUTPUT/'runtime_manifest.json', dict(source_hashes=SOURCE_HASHES,
    model_id=MODEL_ID, revision=REVISION, archive=str(ARCHIVE), checkpoint_sha256=file_hash(CHECKPOINT),
    torch_version=torch.__version__, settings=dict(steps=REPAIR_STEPS, strength=REPAIR_STRENGTH,
    seed=REPAIR_SEED, reference_width_probe=RUN_REFERENCE_WIDTH_PROBE)))
print('진단 대상:', [(c['case_id'], c['kind']) for c in CASES])
for case in CASES:
    print(case['case_id']); display(Image.open(Path(case['directory'])/'input.png').resize((384,384)))
print('결과 폴더:', OUTPUT)
''')
    add('code', '''# 3. 기존 획 추출기와 SD 로딩 (학습 안 함)
from stroke_model.reference_model import ReferenceStrokeModel
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
if state.get('architecture') != 'reference_v2_topology_loss_v1':
    raise ValueError('이 진단은 기존 reference_v2_topology_loss_v1 체크포인트용입니다.')
extractor = ReferenceStrokeModel(image_size=state['image_size'], width=state['width'], pretrained=False)
extractor.load_state_dict(state['model'], strict=True)
for case in CASES:
    if any(c['reference'].shape[-2:] != (state['image_size'], state['image_size'])
           for c in load_conditions(case['directory'])):
        raise ValueError('조건 해상도와 체크포인트 해상도가 다릅니다.')
extractor = extractor.eval().requires_grad_(False).to('cuda')
del state; gc.collect(); torch.cuda.empty_cache()
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, revision=REVISION,
    torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False, thresholding=False)
pipe.to('cuda'); pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
for module in (pipe.unet, pipe.vae, pipe.text_encoder):
    module.eval().requires_grad_(False)
pipe.vae.to(dtype=torch.float32)
json_write(OUTPUT/'scheduler_config.json', dict(pipe.scheduler.config))
import importlib.metadata
json_write(OUTPUT/'versions.json', {name: importlib.metadata.version(name) for name in
    ('torch','diffusers','transformers','timm','numpy','scipy','scikit-image')})
print('로딩 완료. 4번 셀을 실행하세요.')
''')
    add('code', '''# 4. 진단 실행. 사례 실패 시에도 기존 결과를 숨기지 않고 기록합니다.
if any((Path(c['directory'])/'audit').exists() for c in CASES):
    raise FileExistsError('이미 실행한 결과를 보존합니다. 새 실행은 2번 셀부터 시작하세요.')
METRICS, FAILURES = [], []
def progress(case_id, arm, step, total, row):
    if step == 1 or step % 3 == 0 or step == total:
        print(case_id, arm, f'{step}/{total}', row['phase'],
              'accepted=', row['accepted_inner_steps'], flush=True)
for case in CASES:
    try:
        METRICS.extend(run_case(pipe, extractor, case, steps=REPAIR_STEPS,
            strength=REPAIR_STRENGTH, seed=REPAIR_SEED,
            width_probe=RUN_REFERENCE_WIDTH_PROBE, progress=progress))
        display(Image.open(Path(case['directory'])/'audit/comparison.png'))
    except Exception as exc:
        failure = dict(case_id=case['case_id'], type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
        FAILURES.append(failure)
        json_write(Path(case['directory'])/'failure.json', failure)
        print('FAILED:', case['case_id'], str(exc), flush=True)
    finally:
        write_tables(OUTPUT, CASES, METRICS, FAILURES)
        gc.collect(); torch.cuda.empty_cache()
print('실행 끝. 자동 판정이 아닌 시각 검토가 필요합니다. 실패 수:', len(FAILURES))
print('4번 셀 재실행은 기존 audit 폴더를 덮어쓰지 않습니다. 새 실험은 2번부터 실행하세요.')
''')
    add('markdown', '''## 결과 검토
1. `comparison.png`: reference / input / VAE / 재역확산 / cyclic 비교. 기준 유사도는 정답률이 아닙니다.
2. `extraction_slot_*.png`: 잘못된 돌출이 어느 획에 포함되는지 확인.
3. `diagnostic_regions.png`: 자동 healthy 영역에 실제 오류가 섞였는지 확인.
4. `cyclic_guided/erase_*.png`, `restore_*.png`, `preserve_*.png`: 정상 획에 삭제가 걸리는지 확인.
5. `human_review.csv`: target_character, input_error_type, extraction_ok, error_reduced, thinner,
   new_error, style_changed, notes를 작성. 판단 항목은 yes/no/uncertain, 정상 입력의 오류 감소는 n/a.
6. `automatic_metrics.csv`: width ratio <1은 고정 중심점에서 얇아짐/이동의 징후이며 손상 확정이 아닙니다.
   healthy_ink_drop은 정상 후보 영역의 잉크 감소량입니다. healthy 영역 자체는 검수해야 합니다.

일단 한 번 실행한 ZIP을 가지고 다음 수정 대상을 결정합니다. 낮은 loss/높은 coverage를 성공으로 간주하지 않습니다.
''')
    add('code', '''# 5. 모든 성공/실패 결과 저장·다운로드
write_tables(OUTPUT, CASES, METRICS, FAILURES)
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
if DRIVE_RESULT_DIR:
    destination = Path(DRIVE_RESULT_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    copied = destination/Path(archive).name
    if copied.exists():
        raise FileExistsError(copied)
    shutil.copy2(archive, copied)
    print('Drive 백업:', copied)
print('다운로드:', archive)
files.download(archive)
''')
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(name='python3', display_name='Python 3', language='python'),
        language_info=dict(name='python'), accelerator='GPU', colab=dict(provenance=[])))
    target = ROOT / 'notebooks/repair_audit/stage1.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
