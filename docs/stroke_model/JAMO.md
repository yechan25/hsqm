# Jamo: 굵은 자모 마스크 학습

[Colab에서 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/jamo.ipynb)

기존 ReferenceStrokeModel, 증강, 학습, 평가 코드를 재사용한다.
마스크를 skeleton으로 변환하거나 자동으로 굵게 만들지 않는다.
체크포인트 내부 architecture는 호환성을 위해 기존 값을 유지한다.
실제 Drive 데이터 학습/성능 검증은 Colab 실행 후 확인해야 한다.

## 자동 데이터 준비: CSV 불필요

기본 입력은 공유 Drive의 `2026 자율연구/jamo seperation/unique_images`다.

```text
unique_images/
  폰트이름_unique_images/
    0_0_가.png
    0_1_가.png
    1_0_각.png
    1_1_각.png
    1_2_각.png
```

`N_M_글자.png`에서 N은 폰트 안의 index, M은 0 초성/1 중성/2 종성이다.
받침 없는 글자는 0·1, 받침 있는 글자는 0·1·2가 필요하다.
폰트 간에는 index가 아닌 **글자 이름**으로 대응한다. NFC/NFD 한글 이름을 처리한다.
자모 PNG는 원래 글자와 같은 캔버스에서 해당 부분만 남긴 이미지여야 한다.
캔버스 크기 불일치/필수 자모 누락/중복/빈 자모는 오류로 알려준다.

`stroke_model.prepare_jamo.prepare_jamo`는 다음을 수행한다.

1. 폰트 폴더와 파일명을 검사하고 초성→중성→종성 순으로 마스크를 읽는다.
2. 전체 글자를 자모 잉크의 픽셀별 max로 합친다. 위치/두께는 유지하며 128×128로 resize한다.
3. 기준 폰트 하나를 고정한다. REFERENCE_FONT=None이면 글자 수가 가장 많은 폰트를 자동 선택한다.
   지정할 때는 `_unique_images`를 뺀 이름을 사용한다. 기준 폰트에 없는 글자는 제외하고 보고서에 기록한다.
4. 다른 폰트의 같은 글자를 입력/정답으로 사용한다. 기준 폰트 자체는 타깃 샘플에서 제외한다.
5. 타깃 폰트가 2개 이상이면 약 80/20 폰트 단위 train/val 분할한다.
   전체 폰트가 2개뿐이면 최소 2글자가 있어야 하며 글자 단위 분할로 대체한다.
   이 경우 검증은 미관측 폰트 성능이 아니다. 전체 폰트가 1개면 중단한다.
6. 정답에서 여러 자모가 겹치는 픽셀은 -1로 두고 학습/평가에서 제외한다.

이 데이터는 폰트 자모 합성 데이터다. 실제 손글씨 성능을 검증하는 것은 아니다.
폰트 분할은 폰트 이름 기준이며 서로 다른 폰트 파일의 동일 글리프 복제까지 제거하지 않는다.

## 실행과 결과

GPU 런타임에서 위부터 실행한다. 기본 경로가 맞으면 CSV 작성 없이 실행할 수 있다.
변환 후 실제 기준 자모와 대응 정답, 정답 커버리지, 제외된 샘플 수를 보여준다.
학습 원본에만 기본 20개 증강을 적용하고 30 epochs, batch 8로 학습한다.
폰트 데이터가 많으면 VARIANTS=0으로 먼저 실행하여 규모/속도를 확인할 수 있다.
Swin 사전학습 가중치를 사용하는 새 학습이며 기존 획 체크포인트는 불러오지 않는다.

기본 출력은 `jamo seperation/jamo_training`이다.
data에 변환 캐시, jamo_run_시간에 best.pt, history.json, run_config.json,
learning_curves.png, val_preview 그림과 soft NPZ를 저장한다.
원본 마스크 파일은 수정하지 않는다. 파일 목록/크기/수정시간과 설정이 같으면 캐시를 재사용한다.
같은 파일의 내용을 수정한 뒤 강제 재생성하려면 REBUILD=True를 설정한다.

기준 자모, 입력, 정답, 최종 분할, 자모별 soft mask를 확인할 수 있다.
별도 test CSV가 있을 때만 TEST_CSV를 지정한다. 폴더 기본 경로에서는 train/val만 생성한다.
test CSV 형식은 [jamo_paths_example.csv](../../configs/jamo_paths_example.csv)를 참고한다.
test 이미지를 같은 경로에서 수정하면 새 CSV 파일명으로 지정해 test 캐시를 구분한다.
자모 분리는 개별 획 분리와 평가 단위가 다르므로 Dice 숫자를 직접 비교하지 않는다.
