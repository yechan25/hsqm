# 오류 위치 탐지 모델 v1: 실제 코드 아키텍처

상태: 구조·손실 코드 구현, 학습 전. 체크포인트나 성능이 검증된 모델이 아니다. 기존 two_stage_repair 노트북은 이 새 모델을 사용하지 않는다. 기존 실험의 결과를 새 모델의 성능으로 보고하지 않는다.

상위 설계/선행 연구: [오류 위치 탐지·국소 재보정 설계](ERROR_LOCALIZATION_REPAIR_DESIGN.md).

## 구현 모듈

- stroke_model/error_localizer.py: ErrorLocalizerConfig, StrokeErrorLocalizer, FrozenExtractorErrorLocalizer, error_localizer_loss.
- stroke_model/local_repair_proposal.py: LocalRepairProposal, proposal_loss, FixedRepairPlan.
- tests/test_error_localizer.py: 의미 정확도가 아닌 구조와 학습 연결 검증.

## 입력 계약

모델에 한 문자 ROI씩 넣는다. 임의 사진의 문자 검출·문자열 정렬은 이 모듈의 범위 밖이다.

| 이름 | 형태/의미 |
|---|---|
| rgb | B×3×H×W, [0,1] 실제 생성 이미지, H/W>=16 |
| probabilities | B×K×H×W, 기존 추출기의 획 소속 확률. 정답 마스크가 아님 |
| codepoints | B, int64 Unicode 완성형 한글 음절. 초성19·중성21·종성28 토큰으로 분해 |
| active | B×K bool. 서로 다른 획 수를 한 배치에 넣기 위한 패딩 구분 |
| stroke_types | 선택 B×K int64. unknown/horizontal/vertical/diagonal/curve/bent/loop |
| relations | 선택 B×K×K int64. unknown/contact_expected/separate_expected/continuation_expected |

stroke_types와 relations를 생략하면 unknown이다. unknown은 '정상/분리'가 아니다. 관계 주석을 실제로 공급하지 않은 실험에서 관계 지식을 학습했다고 주장하지 않는다. 관계 메타데이터 자동 생성기도 아직 없다.

FrozenExtractorErrorLocalizer는 기존 V2를 eval+requires_grad=False로 고정하고, 표준 템플릿/획 마스크를 기존 추출기의 소속 판별 조건으로만 사용한다. 새 오류 탐지 헤드는 템플릿 RGB를 입력받지 않는다. 그러나 추출 확률에 템플릿 편향이 남을 수 있으므로 다른 기준 폰트를 쓰는 ablation이 필요하다.

## 계산 구조: 기본 width=32

1. 각 획에 동일한 작은 CNN을 적용한다. 입력은 현재 잉크·소속 확률·소속 잉크·xy·배정 엔트로피의 6채널이다.
2. 획별 면적 비율/중심/공분산의 6개 값과 선택적 획 유형·관계 임베딩을 만든다. 이전처럼 이 숫자를 기준과 맞추는 손실을 쓰는 것이 아니라 학습할 탐지기의 입력 특징으로만 사용한다.
3. 유효 획에 대해 평균/최대 pooling한다. 획 순서를 동시에 바꾸어도 결과가 같고 패딩 획은 집계에서 제외된다.
4. RGB3 + 잉크1 + 엔트로피1 + 평균/최대 획 특징64 = 69채널을 U-Net형 encoder로 보낸다.
5. 128×128 예시: encoder 32×128×128 → 64×64×64 → 128×32×32. 자모 임베딩 및 획 context로 bottleneck에 FiLM 조건을 적용한다. 알려진 관계는 상대 획의 기하/유형 토큰과 관계 임베딩을 한 차례 message passing하고, unknown 관계는 메시지를 보내지 않는다.
6. Skip decoder 64×64×64 → 32×128×128에서 출력 헤드를 계산한다.

| 출력 | 의미 |
|---|---|
| extra_logits / extra_probability | B×1×H×W 추가 잉크 위치. 아직 학습되지 않은 확률 |
| kind_logits | B×2×H×W, 부착 가지 / 분리 잡획. 오류 positive 영역에서만 종류를 감독 |
| has_error_logit | B×1, 주석으로 감독할 오류 존재 여부. uncertainty 또는 보정된 신뢰도가 아님 |
| features | B×32×H×W, 별도 수정안 모듈용 영상 특징 |

신뢰도 보정·ensemble·abstain 임계값은 아직 구현하지 않았다. 임의 초기값/임의 threshold로 생성물을 자동 삭제하는 실행 경로는 추가하지 않았다. 기본 width=32에서 새 오류 탐지기는 288,164개, 수정안 모듈은 19,875개 파라미터다(기존 추출기 제외). 실제 GPU 속도/활성 메모리는 미측정이며 속도 개선을 주장하지 않는다.

## 학습 손실

error_localizer_loss는 다음을 사용한다.

L_detector = focal_BCE(gamma=2) + soft_Dice + 0.2 * kind_CE + 0.1 * presence_BCE.

가중치는 실험 시작용이며 최적값이 아니다. extra_target/valid는 B×1×H×W의 dense 합성/검수 라벨이다. 기존 희소 중심선에서 extra_target을 임의로 만들면 안 된다. valid=0은 loss에서 제외되며, 일부 정상 영역만 주석된 이미지를 전체 정상이라는 presence 정답으로 바꾸지 않는다. 모든 영역이 unknown인 배치는 loss와 gradient가 0이다. 종류 라벨은 B×H×W: -1 unknown, 0 정상, 1 부착 추가 획, 2 분리 잡획이다. 정상/unknown 픽셀에서 kind_CE를 학습하지 않는다.

현재 고정 추출기의 확률은 detach되며 영상 encoder/오류 헤드만 학습된다. 추출 모델은 기존 가중치로 유지한다. cached probabilities를 사용할 수도 있지만, 데이터/추출기 checkpoint hash와 split을 함께 저장해야 한다.

## 수정안 모듈과 국소 계획

LocalRepairProposal은 rgb, localizer features, error_probability를 받아 RGB residual을 예측한다. 특징과 확률은 detach하여 복원 손실이 탐지기를 바꾸거나 오류를 숨기지 못하게 한다. 출력 residual의 마지막 층을 0으로 초기화하므로 최초 출력은 정확히 원래 RGB다. 탐지기와 복원기를 따로 학습하는 첫 단계에 맞춘 구조다.

proposal_loss = 손상 영역에서 같은 글씨체 clean RGB와의 L1 + identity_weight * 정상 영역에서 corrupted RGB와의 L1. valid=0은 제외한다. clean 이미지는 학습 감독에만 존재하며, LocalRepairProposal.forward 입력에는 없다. 합성 손상 데이터와 오류 탐지기 학습 실행기는 [학습 코랩](TRAIN_ERROR_LOCALIZER.md)에 추가했다. 복원 제안기 학습은 아직 후속 작업이다.

FixedRepairPlan은 현재 생성 이미지·예측 수정안·외부에서 검증한 edit mask를 clone+detach해 고정한다. 생성된 RGB 후보에 대해 편집 영역 smooth L1과 비편집 영역 보존 smooth L1을 각각 면적으로 정규화한다. decoder RGB를 clamp하지 않아 범위 밖 값에도 수정 gradient가 남는다. 이 클래스는 아직 기존 sampler에 맞춘 adapter/실행 경로에 연결하지 않았다.

composite 메서드는 명시적인 최종 RGB 합성용이다. mask=0 영역은 source를 그대로 유지한다. 이를 사용한 결과를 '확산 자체가 정상 영역을 보존했다'고 해석하지 않는다.

## 학습·실행을 위한 다음 순서

1. 문자/폰트 family/원본 sample 기준으로 split을 먼저 고정한다.
2. 정상 세리프를 포함한 clean negative와 실제 픽셀 변경이 알려진 부착 돌출/잡획 sample을 만든다. 실제 생성물의 소량 검수 라벨도 확보한다.
3. FrozenExtractorErrorLocalizer에서 추출기 동결을 유지하고 오류 탐지기만 학습한다. 주석이 없는 영역은 valid=0으로 둔다.
4. 정상 세리프 오탐과 부착 돌출 미탐을 held-out 폰트/문자별로 확인한다. 기준 템플릿 교체 시 결과 변화도 확인한다.
5. Gate A를 통과한 뒤 LocalRepairProposal을 별도로 학습한다. 탐지 입력은 GT/예측 두 조건을 비교한다.
6. Gate B를 통과한 뒤 FixedRepairPlan을 재역확산에 연결한다. 무보정/재역확산만/수정안만/수정안+확산을 비교한다.

합성 extra 데이터 생성과 탐지기 trainer/optimizer loop는 구현했다. checkpoint에는 threshold_calibrated=False를 기록한다. 실제 confidence calibration, OCR, 사진 ROI 검출, 누락·끊김 복원, 복원 제안기 학습, diffusion adapter는 아직 구현하지 않았다. 이번 구현은 이 기능들의 성능을 대신하지 않는다.
