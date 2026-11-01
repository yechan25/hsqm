"""Build self-contained Colab artifacts from the current local module sources.

Run from the repo: python notebooks/build_stroke_structure.py
The notebook embeds the scoring modules, so it does not depend on their having
been pushed to GitHub. The existing V2 model/runtime is cloned from main.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SETTINGS = '''# ===== 여기의 설정만 바꾸세요 =====
# "csv": PNG 파일명으로 모델 추론 + 구조 평가
# "saved_npz": 이미 저장된 추론 결과 평가 (모델 추론을 반복하지 않음)
MODE = "csv"
CHECKPOINT = "/content/drive/Shareddrives/2026 자율연구/HSQM/stroke_zip_runs/20261028_082616_066020/best.pt"
CSV_PATH = "/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/test_dataset/paths.csv"
INPUT_FILES = ["I_아_2.png", "I_야_2.png"]  # 보고 싶은 PNG 파일명. None = 전체 CSV
DATASET_ROOT = None  # CSV 폴더 밖에 이미지가 있으면 데이터의 상위 폴더 지정
NPZ_FILES = []  # saved_npz 모드: ["/content/drive/.../row_0001.npz"]
SAVE_DIR = None  # None = 시간별 새 폴더. 직접 지정할 때는 아직 없는 폴더
SHOW_PLOTS = True

# reference: 원래 기준 좌표. 입력과 기준이 정렬돼 있어야 함.
# hints: 모델이 변형한 기준 좌표. 정렬 오류를 숨길 수 있으므로 비교용으로 사용.
REFERENCE_GEOMETRY = "reference"

# 모든 거리 단위는 추론 해상도의 픽셀 (기본 모델은 128 x 128).
STRUCTURE_CONFIG = dict(
    reference_threshold=0.5,   # 기준 skeleton을 만들 때만 사용하는 이진화 기준
    coverage_radius=2,         # 누락 측정: 기준 경로 주변 위치 오차 허용 반경
    required_ink=0.7,          # 이 값 이상의 잉크 강도를 충분하다고 봄
    anchor_spacing=10,        # 분리 검사 지점 사이의 경로 길이
    anchor_radius=1,          # 시작/도착 검사 영역의 반경
    corridor_radius=3,        # 연결 경로가 이동할 수 있는 기준 주변 반경
    pair_spans=(1, 2),        # 이웃 지점 / 한 지점 건너 연결을 함께 검사
    max_pairs=32,             # 획 하나당 검사할 경로 쌍의 최대 개수
    connectivity_steps=None,  # None = 수렴까지. 작게 제한하면 가짜 끊김 가능
    skeleton_iterations=12,   # soft skeleton 반복 횟수. 두꺼운 획이면 늘릴 것
    junction_radii=(1, 2, 3), # 작은 접합부 분기까지 검사
    junction_margin=0.0,      # 기본값: 폰트/짧은 가지에 대한 정션 면제 없음
    junction_seed_radius=2,   # 기준 주변 본체에서만 연결된 가지를 탐색
    attachment_steps=None,    # None = 수렴까지. 제한 시 수렴 여부가 표에 표시됨
    expected_width=5.0,       # 정상 획의 기대 굵기. 기준 skeleton 굵기가 아님
    width_radius=4,           # 파임 검사 영역 반경
    width_tolerance=0.25,     # 정상 대비 굵기/잉크량 부족 허용 비율
    missing_weight=1.0,       # 별도 reference_deficit 계산에만 사용
    split_weight=1.0,         # 추출 구조 오류의 분리 가중치
    junction_weight=1.0,      # 추출 구조 오류의 정션 가중치
    width_weight=0.0,         # 별도 reference_deficit 계산. 굵기 교정 후 켜기
    eps=1e-6,
)

# 선택적 추출기 추가 학습 API용. 아래 기본 실행은 평가만 하며 학습하지 않습니다.
EXTRACTION_TRAINING_CONFIG = dict(
    structure_weight=0.05,    # 기존 CE/Dice에 추가할 구조 손실 가중치
    mass_guard_weight=0.2,    # 기존 획을 지워서 정션을 피하는 것을 억제
    consistency_weight=0.1,   # 기존 모델의 획 배정을 보존하는 보조 항
    mass_tolerance=0.15,      # 기존 모델 대비 허용하는 획별 잉크 감소 비율
    eps=1e-6,
)
'''

SETUP = '''#@title 2. 기존 V2 모델 코드 준비 (Colab에서 한 번 실행)
from pathlib import Path
import subprocess, sys, tempfile
_structure_repo = Path(tempfile.mkdtemp(prefix="structure_runtime_", dir="/content"))
subprocess.run(["git", "clone", "--depth", "1", "--branch", "main",
                "https://github.com/yechan25/hsqm.git", str(_structure_repo)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(_structure_repo / "requirements.txt")], check=True)
sys.path.insert(0, str(_structure_repo))
for _name in list(sys.modules):
    if _name == "stroke_model" or _name.startswith("stroke_model."):
        del sys.modules[_name]
print("V2 준비 완료. 평가 모듈은 이 노트북에 포함된 버전을 사용합니다.")
'''

RUN = '''#@title 4. 실행: PNG 선택 → 획별 오류 표와 위치 그림
from stroke_model.stroke_structure import StructureConfig
from stroke_model.score_stroke_structure import score_structure_csv, score_saved_strokes

structure_config = StructureConfig(**STRUCTURE_CONFIG)
if MODE == "csv":
    from google.colab import drive
    drive.mount("/content/drive")
    structure_results = score_structure_csv(
        CHECKPOINT, CSV_PATH, input_files=INPUT_FILES, dataset_root=DATASET_ROOT,
        config=structure_config, geometry=REFERENCE_GEOMETRY,
        save_dir=SAVE_DIR, show=SHOW_PLOTS,
    )
elif MODE == "saved_npz":
    selected_npz = NPZ_FILES or globals().get("inference_results")
    if not selected_npz:
        raise ValueError("NPZ_FILES에 기존 추론 NPZ 경로를 넣으세요.")
    # 기존 셀에서 Drive를 연결하지 않았다면 먼저 연결하세요.
    structure_results = score_saved_strokes(
        selected_npz, config=structure_config, save_dir=SAVE_DIR, show=SHOW_PLOTS,
    )
else:
    raise ValueError('MODE는 "csv" 또는 "saved_npz"여야 합니다.')
'''


def build():
    modules = {}
    for name in ('infer_reference', 'stroke_structure', 'score_stroke_structure', 'extraction_training'):
        modules['stroke_model.' + name] = (ROOT / 'stroke_model' / (name + '.py')).read_text(encoding='utf-8-sig')
    embedded = '''#@title 3. 평가 모듈 로드 (노트북에 코드 포함; GitHub 업로드 불필요)
import sys, types
_structure_sources = ''' + repr(modules) + '''
for _module_name, _source in _structure_sources.items():
    _module = types.ModuleType(_module_name)
    _module.__package__ = "stroke_model"
    _module.__file__ = "<embedded:" + _module_name + ">"
    sys.modules[_module_name] = _module
    exec(compile(_source, _module.__file__, "exec"), _module.__dict__)
print("구조 평가 모듈 로드 완료")
'''
    def cell(kind, text):
        result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    introduction = '''# 추출된 획의 구조 오류 진단

**1번 설정 → 2번 준비 → 3번 모듈 로드 → 4번 실행** 순서입니다.
설정은 첫 번째 코드 셀에 모았습니다. 첫 실행은 PNG 1~2개로 확인하세요.

- 기준 획은 누락된 획까지 전부 준비해야 합니다. 입력 획 정답 주석은 사용하지 않습니다.
- 원본 기준과 입력이 정렬돼 있어야 합니다. 그림 첫 줄의 기준 중심선을 확인하세요.
- 정션(J)은 폰트와 무관하게 개별 추출 획의 구조 오류로 봅니다. 짧은 가지도 면제하지 않습니다.
- extraction_error는 D/J 구조 이상 지표이고, reference_deficit은 M/W 기준 대비 부족량입니다.
- 원본 글자 품질은 판정하지 않습니다. 정상 원본에서도 추출기가 정션을 만들 수 있습니다.
- 정션은 분기 반응 근사값이고, 100점 환산이나 사람 선호 교정은 하지 않았습니다.
- 모델 추론은 GPU를 쓸 수 있습니다. 연결 전파는 반복 연산이므로 여러 PNG는 시간이 걸릴 수 있습니다.
- 정상 굵기를 모르는 경우 W는 참고만 하세요. 기본 width_weight=0입니다.
- 이 파일에는 새 평가 코드가 포함돼 있어, GitHub main에 해당 코드가 없어도 실행됩니다.
'''
    guidance = '''## 선택적 추출기 추가 학습 API

정션은 우선 추출기 개선용 신호입니다. 원본 글자 품질 보상으로 역확산에 직접 연결하지 않습니다.
다음은 학습 루프에 넣을 손실 구성 예시이며 위 평가 셀은 가중치를 갱신하지 않습니다.
배치의 기준 geometry는 입력 좌표에 고정 정렬되어 있어야 합니다. 샘플마다 다르면 criterion도 샘플별로 준비해야 합니다.

```python
from copy import deepcopy
from stroke_model.stroke_structure import StrokeStructureLoss
from stroke_model.extraction_training import ExtractionTrainingConfig, extraction_training_loss

# 추가 학습 전에 한 번만 생성합니다.
teacher = deepcopy(extractor).eval().requires_grad_(False)
criterion = StrokeStructureLoss(aligned_reference_strokes, structure_config).to(device)
train_config = ExtractionTrainingConfig(**EXTRACTION_TRAINING_CONFIG)

# 정답 labels가 있는 학습 배치에서 사용합니다.
extractor.train()
with torch.no_grad():
    teacher_output = teacher(image, reference, reference_strokes)
output = extractor(image, reference, reference_strokes)
structure = criterion(output["masks"])  # student 경로는 detach/no_grad 금지
# B,K bool 주석: 원본에서 정상적으로 완전한 획임을 확인한 채널만 True.
# 실제 누락/손상 또는 정상 여부가 불명확한 원본 채널은 False로 지정합니다.
losses = extraction_training_loss(output, labels, structure, teacher_output, train_config,
                                 topology_valid=known_complete_strokes)
optimizer.zero_grad(set_to_none=True)
losses["loss"].backward()
optimizer.step()
```

기존 정답 CE/Dice + 구조 손실 + 잉크 감소 방지 + 기존 출력 보존을 함께 사용합니다.
M/W는 이 학습 손실에 넣지 않습니다. 입력에서 실제로 없어진 획을 만들어내게 해서는 안 됩니다.
교차점의 주석 공백과 기존 모델 오류가 남으므로, 정답 Dice와 구조 오류를 함께 검증해야 합니다.
'''
    notebook = {'cells': [cell('markdown', introduction), cell('code', SETTINGS),
                cell('code', SETUP), cell('code', embedded), cell('code', RUN), cell('markdown', guidance)],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'colab': {'name': 'stroke_structure_v2.ipynb'}, 'accelerator': 'GPU'},
                'nbformat': 4, 'nbformat_minor': 5}
    output = ROOT / 'notebooks' / 'stroke_structure_v2.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    one_cell = ROOT / 'notebooks' / 'stroke_structure_v2_cell.py'
    one_cell.write_text(SETTINGS + '\n# 기존 V2 준비 셀 실행 후 이 파일 전체를 추가 셀에 붙여 넣으세요.\n\n' +
                        embedded + '\n' + RUN, encoding='utf-8')
    for item in notebook['cells']:
        if item['cell_type'] == 'code':
            compile(''.join(item['source']), str(output), 'exec')
    compile(one_cell.read_text(encoding='utf-8'), str(one_cell), 'exec')
    print('Built and compiled:', output.name, one_cell.name)


if __name__ == '__main__':
    build()
