# Jamo: 클래스에서 자모를 골라 합성·학습

[Colab에서 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/jamo.ipynb)

기존 ReferenceStrokeModel과 학습·평가 코드를 사용한다. 굵은 원본 자모를 skeleton으로 변환하지 않는다.
unique_images는 원본 자모 은행이며, 여기서 별도의 합성 학습 데이터를 생성한다.
실제 Drive 데이터 학습과 성능은 Colab에서 확인해야 한다.

## 원본 파일 해석

입력: 공유 Drive의 `2026 자율연구/jamo seperation/unique_images/폰트이름_unique_images`.

`N_M_글자.png`에서 N은 0~5 클래스, M은 0 초성/1 중성/2 종성이다.
글자 이름은 **M에 해당하는 자모 종류**를 확인하기 위한 대표 글자다.
동일 글자의 파일이 모두 있을 필요가 없다. 예를 들어 `0_0_가.png`의 ㄱ과
`0_1_아.png`의 ㅏ를 합쳐 가를 구성할 수 있다.
초성·중성·종성 종류는 Hangul 음절의 Unicode 분해로 판별한다. NFC/NFD 파일명을 지원한다.

첨부 Generate Jamo Image의 분류를 사용한다.

| 클래스 | 중성 Unicode index | 구성 |
|---|---|---|
| 0 / 3 | 0,1,2,3,4,5,6,7,20 | 초성+중성 / 초성+중성+종성 |
| 1 / 4 | 8,12,13,17,18 | 초성+중성 / 초성+중성+종성 |
| 2 / 5 | 9,10,11,14,15,16,19 | 초성+중성 / 초성+중성+종성 |

같은 폰트·클래스·M·자모 종류에 대표 파일이 여러 개면 정규화 파일명 순으로 하나를 선택하고
conversion_report.json의 duplicate_semantic_candidates에 기록한다.
클래스 안에서 공유하는 다른 대표 글자의 자모를 사용하며, 다른 클래스의 자모로 대체하지 않는다.
파일명 클래스와 Unicode 분해가 맞지 않으면 중단한다.

## 학습 데이터 생성

1. 모든 폰트를 약 80/20 train/val로 먼저 분할한다. 최소 2개 폰트가 필요하다.
   각 샘플에서 reference와 input은 같은 폰트다. 원본 자모를 reference로, 같은 자모의 변형·합성을 input/정답으로 쓴다. 단일 기준 폰트를 따로 선택하지 않는다.
2. 같은 폰트의 각 클래스에서 존재하는 초성·중성·종성 종류를 찾는다.
3. 공통 자모 종류의 조합을 seed에 따라 선택한다. 기본 폰트·클래스당 최대 100개 조합이다.
   빠른 확인은 SAMPLES_PER_CLASS=10으로 시작한다. 가능한 조합이 적으면 그 수만 생성한다.
4. 기준은 원본 마스크로 합치고, 타깃 자모마다 scale 0.8~1.0, rotation ±5도,
   shift ±20/256 캔버스 비율을 독립 적용한다. 마스크와 정답은 같은 변형 결과에서 만든다.
5. 첨부 코드의 모든 자모 AND 대신 두 자모 이상이 겹친 영역을 검사한다.
   겹친 픽셀/전체 캔버스 > MAX_OVERLAP(기본 0.001)이면 최대 20회 재시도한다.
   계속 실패하거나 자모별 단독 소유 픽셀이 없으면 그 조합을 제외하고 보고한다.
6. 합성 이미지는 자모 마스크의 max다. 겹치는 정답 픽셀은 -1로 제외한다.
   label 0/1/2는 초성/중성/종성 순서이며, NPZ에 원본 기준 마스크와 변형 타깃 label을 함께 저장한다.

모델은 합성 데이터로 학습한다. 실제 폰트 렌더링과 완전히 같은 글리프를 보장하거나
실제 손글씨 성능을 검증하는 데이터는 아니다. 합성과 분리는 원래의 자모 캔버스 위치를 사용한다.

## 속도와 저장

- 원본 PNG를 병렬로 한 번씩 읽고 메모리에 보관한다. SOURCE_WORKERS 기본값은 8이다.
- 합성 NPZ/manifest와 미리보기는 `/content/jamo_prepared` 로컬 디스크에서 처리한다.
- 생성된 데이터 전체를 ZIP 하나로 묶어 `jamo seperation/jamo_training/data`에 저장한다.
  Drive에 샘플 파일을 하나씩 쓰거나 학습 때 하나씩 읽지 않는다.
- 같은 런타임에서는 로컬 캐시를 재사용한다. 새 런타임에서는 ZIP을 한 번 복사해 로컬로 푼다.
- 빠른 재사용을 위해 원본 목록을 매번 재검사하지 않는다. 원본 추가/수정 시 REBUILD=True로 다시 생성한다.
- 합성 때 자모별 변형을 적용하므로 기본 VARIANTS=0이다. 추가 전체 글자 증강은 선택적으로 사용할 수 있다.
- 데이터 생성과 미리보기 셀을 분리했다. 미리보기는 PREVIEW_COUNT개만 읽고 합성을 반복하지 않는다.

## 실행과 결과

GPU 런타임에서 위부터 실행한다. 설정 경로는 알려준 공유 Drive를 기본값으로 사용한다.
원본 읽기, 로컬 합성, ZIP 저장의 진행 상태가 출력된다.
변환 보고서에는 폰트별 클래스/M의 자모 종류 수, 제외된 조합, 선택한 대표 파일을 기록한다.
manifest는 상대 NPZ 경로이며 새 런타임에서도 같은 ZIP을 사용할 수 있다.
캐시 버전은 4이며 이전 교차 폰트 데이터와 구분한다. 이전 데이터 캐시는 재사용하지 않는다.

기본 30 epochs, batch 8이며 GPU 메모리가 부족하면 4로 변경한다.
학습 결과는 Drive의 jamo_run_시간 아래 best.pt, history.json, run_config.json,
learning_curves.png, val_preview 그림과 soft NPZ로 저장한다.
모델 checkpoint 내부 architecture는 기존 구현과 호환성을 유지한다.
기준·입력·정답·예측·자모별 soft mask로 결과를 확인한다.

TEST_CSV는 별도 held-out CSV가 준비되어 있을 때만 사용한다.
[CSV 형식 예시](../../configs/jamo_paths_example.csv).
주석된 픽셀에 한정된 Dice이며 기존 개별 획 분리 Dice와 직접 비교하지 않는다.

## Jamo test: 변형과 폰트 일반화

[Test Colab에서 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/test_jamo.ipynb)

저장된 best.pt를 고정하여 평가한다. test에서 학습하거나 best checkpoint를 고르지 않는다.
기본 checkpoint 경로는 대화에서 확인한 jamo_run_20261014_153935_401612/best.pt다.
다른 학습 결과를 평가할 때는 CHECKPOINT를 수정한다.

| 평가 이름 | 폰트 | 타깃 변형 |
|---|---|---|
| seen_standard | 학습에 사용한 폰트 | 새로운 seed, 학습과 같은 범위 |
| seen_strong | 학습에 사용한 폰트 | 크기 65~115%, 회전 ±15°, 이동 ±20px/128 |
| unseen_standard | train/val 모두에 없는 폰트 | 새로운 seed, 학습과 같은 범위 |
| unseen_strong | train/val 모두에 없는 폰트 | 강한 변형 |

모든 reference/input은 동일 폰트다. 같은 원본 은행에서 다른 변형을 만드는 것은
변형에 대한 일반화 평가이며, 독립적으로 수집된 실제 손글씨 test는 아니다.
검증 폰트는 best.pt 선택에 사용됐으므로 true unseen test에서 제외한다.
기존 10개 폰트가 train/val에 모두 포함됐다면 새 폰트를 추가해야 unseen 평가를 할 수 있다.
새 폰트가 없으면 unseen 모드만 건너뛰고 이유를 출력한다.

새 폰트는 SOURCE_ROOT에 추가하거나 NEW_FONT_ROOT 아래에 폰트이름_unique_images 폴더를 넣는다.
파일 형식과 클래스 공유 규칙은 기존 원본과 같다. NEW_FONT_ROOT에는 기존 SOURCE_ROOT와
겹치는 폰트 이름을 사용하지 않는다. 새로운 TTF를 직접 렌더링하는 기능은 포함하지 않는다.

기본 폰트·클래스별 최대 30개 조합이며 빠른 확인은 SAMPLES_PER_CLASS=5로 한다.
TEST_SEED는 학습 seed와 달라야 한다. 같은 의미적 글자 조합은 허용하되,
실제로 같은 input 픽셀 배열이 train/val/test에 반복되면 다시 변형하거나 제외한다.
겹침 때문에 제외한 조합과 acceptance_rate도 저장하여 강한 변형의 선별 정도를 확인한다.

학습 폰트와 기존 이미지 목록은 해당 실행의 원본 데이터 ZIP에서 복구한다.
기존 run_config.json의 source/seed/생성 설정에 맞는 ZIP을 자동 식별한다.
후보가 없거나 여러 개면 TRAINING_ARCHIVE에 실제 학습 ZIP 경로를 직접 지정한다.
새 학습 노트북은 training_inventory.json에 실제 train/val 폰트와 이미지 해시를 별도로 저장한다.
선택적 추가 증강을 사용한 기존 실행은 해당 augmented training_manifest를 로컬에서 복구해야
전체 학습 이미지를 검사할 수 있다. 기본 VARIANTS=0인 실행에는 추가 복구가 필요 없다.

지표는 hard argmax 분할의 자모별 macro Dice/F1/IoU/Precision/Recall이다.
Dice와 F1은 같은 정의다. Accuracy는 주석된 전경 픽셀 기준이며 배경 정확도를 포함하지 않는다.
애매한 교차 픽셀은 제외하므로 annotation_coverage를 함께 본다.
Reference baseline은 원본 자모까지의 최근접 거리로 픽셀을 배정한 결과다.
모델이 위치 단서만 사용하는 간단한 방법보다 얼마나 좋은지 비교한다.

생성/평가는 Colab 로컬에서 수행하며 checkpoint 옆 jamo_test_시간 폴더에 저장한다.
각 모드의 metrics.json에는 전체·폰트별·클래스별·샘플별 지표가 있고,
dataset.zip에는 해당 test의 상대 경로 manifest와 NPZ가 있다.
비교 표/차트, 일반 예제와 가장 낮은 Dice 예제, soft mask를 저장한다.
이 구현의 테스트는 작은 synthetic fixture와 작은 backbone으로 검증했으며 실제 A100/checkpoint
성능 점수는 사용자의 Colab 실행에서 확인해야 한다.
