# 추출된 획의 구조 오류 진단과 학습 손실

입력부터 진단·추가 학습까지의 순서는 [과정 설명서](STROKE_STRUCTURE_PROCESS.md)에 그림과 예시로 정리했다.

사용자가 제공한 저장 배열에서 확인한 과검출 원인과 수정 전후 수치는 [실제 배열 회귀 기록](JUNCTION_REGRESSION.md)에 정리했다. 빨간 원은 세 경로와 수준 간 지속성이 확인된 후보, 노란 원은 미확정 후보다. J_soft 단독 판정은 아직 신뢰하지 않는다.

정션은 폰트와 관계없이 개별 추출 획에서 없어야 할 분기로 취급한다. 짧은 가지를 폰트 특성으로 면제하거나 정상 폰트의 정션 반응을 빼지 않는다. 다만 **추출 결과의 구조 이상은 원본 글자의 오류라는 증거가 아니다.** 정상 원본에서도 추출기가 잘못 배정할 수 있으므로 원본 품질은 평가하지 않는다.

실제 체크포인트의 판별 성능과 추가 학습 효과는 검증 전이다. 이 구현은 가중치를 자동으로 갱신하지 않는다.

## 실행

- [독립 Colab 파일](../../notebooks/stroke_structure_v2.ipynb): 업로드 후 설정 → 준비 → 모듈 로드 → 실행.
- [기존 Colab에 붙일 추가 셀](../../notebooks/stroke_structure_v2_cell.py): 기존 V2 코드 준비 후 파일 전체를 새 셀에 붙인다.
- [진단 모듈](../../stroke_model/stroke_structure.py), [PNG/NPZ 실행 모듈](../../stroke_model/score_stroke_structure.py), [추출기 학습 손실](../../stroke_model/extraction_training.py).

코드는 노트북에 내장하므로 새 모듈의 GitHub 업로드는 실행 전제 조건이 아니다. 기존 V2 모델은 GitHub main에서 받는다. 재생성: `python notebooks/build_stroke_structure.py`.

첫 코드 셀에서 MODE, CHECKPOINT, CSV_PATH, INPUT_FILES와 STRUCTURE_CONFIG를 바꾼다. 입력 정답 획 주석은 평가에 필요하지 않다. 누락된 획까지 목표 글자의 전체 기준 목록에 포함한다.

MODE="saved_npz"에서는 NPZ_FILES에 기존 추론 NPZ를 넣거나 같은 런타임의 inference_results를 재사용한다. 필요한 키는 soft_masks와 reference_strokes다. Drive 파일을 읽기 전에는 Drive가 연결되어 있어야 한다.

## 설정

모든 설정은 첫 셀에 모았다. 거리는 추론 해상도의 픽셀 단위다.

| 설정 | 의미 |
|---|---|
| coverage_radius / required_ink | 기준 경로의 위치 허용 반경 / 충분한 잉크 강도 |
| anchor_spacing / anchor_radius | 연결 검사 지점 간격 / 검사 영역 반경 |
| corridor_radius / pair_spans / max_pairs | 경로 탐색 폭 / 검사 범위 / 최대 검사 쌍 |
| connectivity_steps / attachment_steps | 전파 상한. None이면 격자 크기 이내에서 수렴까지 |
| skeleton_iterations | soft skeleton 계산 깊이. 획 반경보다 충분히 크게 설정 |
| junction_radii | 기본 (1,2,3). 작은 접합부 분기도 검사 |
| junction_margin | 기본 0. 짧은 정션을 폰트 특성으로 면제하지 않음 |
| junction_thresholds / junction_temperature | 공통 절대 잉크 수준 / 학습용 occupancy 전이 폭. 기본 (0.15,0.30,0.45,0.60,0.75)와 0.025 |
| junction_seed_radius | 기준 주변 본체에서 연결 탐색을 시작하는 영역 |
| expected_width / width_radius / width_tolerance | 파임 참고값의 정상 굵기 / 검사 반경 / 허용 비율 |
| split_weight / junction_weight | extraction_error의 D/J 가중치 |
| missing_weight / width_weight | 별도 reference_deficit의 M/W 가중치 |

width_weight는 정상 굵기 교정 전에는 0이다. 예측 획을 최댓값으로 정규화하지 않는다. 미약한 잔상이 온전한 획으로 증폭될 수 있기 때문이다.

## 측정과 출력의 의미

- M: 기준 중심선 각 위치 주변의 잉크 강도가 required_ink보다 부족한 비율의 제곱 평균.
- D: 기준 경로의 두 검사 영역 사이에서 실제 잉크를 따라 연결되는 강도의 부족. 양쪽 잉크가 존재하는 경우에 가중한다. 고리는 지역 통로로 검사하여 반대편 우회를 제한한다.
- J: 여러 절대 수준에서 soft skeleton의 분기 제안과 원본 occupancy의 원형 고리에서 세 방향이 유지되는지를 함께 계산한다. 원형 탐색 반경은 미분 가능한 국소 거리 추정에 비례한다. 각도 평활로 픽셀 진동을 줄이고, 수준별 반응은 평균한 뒤 기준 길이로 정규화한다. 원래 잉크 강도를 곧바로 가닥 수로 세지 않는다.
- W: 정상 기대 굵기의 기준 영역과 국소 평균 잉크량의 부족을 비교한다. 정확한 법선 단면 대신 정사각형 영역을 사용하는 보조 근사값이다.

D의 연결 전파:

```text
F = mask * local_corridor
R = F * start_region
repeat until convergence:
    R = min(F, max_pool(R))
u = max(F * start_region)
v = max(F * end_region)
c = max(R * end_region)
pair_error = u*v * relu((min(u,v)-c) / max(min(u,v), eps))**2
```

외딴 작은 조각을 추가해도 정상 본체 경로의 연결성은 유지된다. mask 자체를 팽창/closing해서 틈을 메우지 않는다. 실제 결손으로 검사 지점이 사라진 경우 D가 작아도 M이 검출할 수 있다.

**혼합 total은 제거했다.** extraction_error는 가중 D+J이고 reference_deficit은 가중 M+W다. `loss`는 전자 평균, `reference_loss`는 후자 평균이다. 어느 값도 오류 원인이 원본인지 추출기인지 판정하지 않는다. 실제 입력에서 누락된 획의 빈 출력은 올바른 추출일 수 있다. 보고서에는 source_quality.assessed=False를 명시한다.

CSV 표, 오류 위치 PNG, 원본과 진단 맵 NPZ, 설정을 포함한 structure_report.json을 저장한다. 도달 불가 그림은 D의 정확한 픽셀별 분해가 아닌 진단 지도다. 구조 결과가 원본 글자의 점수로 자동 합산되지 않는다.

기본 화면은 원본 soft mask / 그래프 검사 / 학습용 soft 반응의 3줄이다. branch_regions_max는 접합부 밖으로 세 경로가 이어지고 기본 두 수준 이상에서 유지된 영역 수의 최댓값이다. unresolved_regions_max는 짧거나 불안정해 확정하지 못한 후보 수다. 빨간 원은 전자, 노란 원은 후자다. 원이 없다고 정상임이 증명되지는 않으며 노란 후보가 있으면 특히 유의한다. 이산 검사는 학습에 사용하지 않고 soft 근사와 일치가 보장되지 않는다. 전체 M/D/J 화면은 SHOW_DETAILS=True로 표시할 수 있으며 False여도 파일로 저장한다.

기존 저장 결과로 다시 검사하려면 MODE="saved_npz"와 NPZ_FILES를 설정한다. 모델 추론을 반복하지 않는다. 저장된 reference_geometry가 있으면 이를 보존해 비교 좌표가 달라지지 않도록 한다. 실제 실패 예제 재현에는 스크린샷보다 row_0043_structure.npz 같은 배열 파일이 필요하다.

## 추출기 추가 학습 API

기존 정답 CE/Dice에 D/J 구조 항을 추가한다. 정션을 줄이려고 획을 지우거나 다른 채널로 넘기지 못하도록, 추가 학습 시작 시점에 고정한 teacher의 잉크량과 출력을 보존하는 보조 항을 함께 쓴다. 이러한 항이 편법을 완전히 차단한다고 보장하지는 않는다.

EXTRACTION_TRAINING_CONFIG도 첫 셀에 있다. structure_weight, mass_guard_weight, consistency_weight, mass_tolerance로 조절한다. M/W 기준 부족량은 추출기 학습 항에서 제외한다. 원본에 없는 획을 추출기에 만들어내라고 요구해서는 안 된다.

구조 항은 원본에서 완전한 획임을 확인한 채널에만 적용한다. topology_valid는 별도 B,K bool 메타데이터다. 실제 누락/손상 또는 정상 여부가 불명확한 채널은 False로 두며, student 예측으로 이 주석을 정하지 않는다. 모두 False면 기존 supervised 손실만 남는다.

```python
from copy import deepcopy
from stroke_model.stroke_structure import StructureConfig, StrokeStructureLoss
from stroke_model.extraction_training import ExtractionTrainingConfig, extraction_training_loss

# 학습 전에 한 번. aligned_reference_strokes는 고정된 입력 좌표의 K,H,W.
criterion = StrokeStructureLoss(aligned_reference_strokes, StructureConfig(**STRUCTURE_CONFIG)).to(device)
teacher = deepcopy(extractor).eval().requires_grad_(False)
train_cfg = ExtractionTrainingConfig(**EXTRACTION_TRAINING_CONFIG)

# 정답 labels가 있는 학습 배치. 샘플별 기준이 다르면 criterion도 별도로 준비한다.
extractor.train()
with torch.no_grad():
    teacher_output = teacher(image, reference, reference_strokes)
output = extractor(image, reference, reference_strokes)
structure = criterion(output['masks'])
losses = extraction_training_loss(output, labels, structure, teacher_output, train_cfg,
                                  topology_valid=known_complete_strokes)
optimizer.zero_grad(set_to_none=True)
losses['loss'].backward()
optimizer.step()
```

평가·그림 함수는 no_grad로 실행하지만 학습 student 경로는 detach/no_grad/argmax/numpy 변환을 사용하지 않는다. min/max는 거의 모든 곳에서 부분 기울기를 전달하지만 병목과 동률에서 희소하거나 불안정할 수 있다. 전파 수렴 확인은 값만 detach하며 그래프는 유지한다. 학습 함수는 전파 미수렴을 거부한다.

이 API는 기존 trainer를 자동 실행하거나 수정하지 않는다. 아직 실제 추가 학습하지 않았다. D/J를 원본 품질 보상으로 역확산에 직접 넣지 않는다. 원본 오류를 평가할 점수 모델은 별도 검증해야 한다.

## 정렬과 한계

REFERENCE_GEOMETRY="reference"는 원래 기준 좌표이므로 입력과 기준이 정렬돼 있어야 한다. hints는 모델 변형 기준을 detach해 쓰지만 정렬이 결손을 피하며 오류를 숨길 수 있다. 기준은 열린 단일 경로나 고리여야 하며, 비거나 분기하거나 여러 조각이면 오류로 알린다.

soft skeleton 자체의 가짜 분기, 굵기·곡률 변화, 본체에 닿은 잡음은 추가 검증이 필요하다. 현재 J에는 폰트별 면제나 정상 반응 차감이 없다. teacher 보존 항은 기존 잘못된 배정을 붙잡을 수도 있으므로 정답 Dice와 구조 진단을 함께 확인한다.

## 검증·문헌

`python -m unittest discover -s tests -p "test_*structure*.py" -v`

27개 검사에서 합성 누락·분리·짧은 정션·외딴 잡음·고리 절단, 약한 가지·흐린 경계·픽셀 끝·정상 꺾임·짧은 잔상, 기울기, 파일 입출력, teacher 기울기 차단, 채널 삭제 방지, 손상 입력 제외를 확인한다. 실제 체크포인트 대신 mock을 쓰는 통합 검사가 포함되며 학습 효과를 증명하지는 않는다. `python tests/demo_junction_levels.py`로 3줄 그림을 재현하고, 실제 배열의 재실행은 회귀 기록의 명령을 사용한다.

- [clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation.html), [저자 구현](https://github.com/jocpae/clDice): soft skeleton 기반.
- [Clough et al.](https://arxiv.org/abs/1910.01877): 위상 비교 기준. Persistent homology 자체는 현재 모듈에 구현하지 않았다.
- [Morphological reconstruction 공식 문서와 문헌](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.reconstruction): 연결 전파 기반.
- [Menten et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Menten_A_Skeletonization_Algorithm_for_Gradient-Based_Optimization_ICCV_2023_paper.html): skeleton 근사의 한계.

지역 연결 검사·본체 연결 가중 정션·학습 보호 항의 조합은 이 연구용 구현안이다. 인용 논문들이 이 한글 구현을 검증한 것은 아니다.
