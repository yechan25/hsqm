# Jamo: 굵은 자모 마스크 학습

[Colab에서 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/jamo.ipynb)

기존 ReferenceStrokeModel, CSV 변환, 증강, 학습, 평가 코드를 재사용한다.
입력 마스크를 skeleton으로 변환하거나 자동으로 굵게 만들지 않는다.
새 노트북과 출력 폴더의 이름은 jamo이며 체크포인트 내부 architecture는 호환성을 위해 기존 값을 유지한다.
이 추가는 실행 경로 제공이며 실제 데이터 학습이나 성능 검증 결과가 아니다.

## 데이터 준비

- TRAIN_CSV에 굵은 자모 마스크를 사용하는 CSV를 지정한다. 기본 Drive 경로는 예시다.
- CSV 예시: [jamo_paths_example.csv](../../configs/jamo_paths_example.csv).
- I_prep: 분리할 전체 글자, I_g: 기준 전체 글자.
- I_G_strokes: 기준 글자에서 자모별로 분리한 이미지 경로 목록.
- input_stroke_paths: 입력 글자의 대응 자모 정답 마스크 목록.
- 두 목록은 같은 개수와 의미적 순서여야 한다. 예: '가'의 [ㄱ, ㅏ].
- 기준 마스크는 기준 글자와, 정답 마스크는 입력 글자와 각각 동일한 캔버스/위치/크기를 사용한다.
  자모만 잘라 중앙에 옮긴 이미지는 사용할 수 없다.
- 검은 잉크/흰 배경 PNG와 투명 PNG를 지원한다. 128×128, 잉크=1로 로드한다.
- 정답 겹침/누락은 -1로 제외한다. 전체 자모 품질을 평가하려면 전경을 충분히 주석해야 한다.
- 여러 획을 한 자모로 묶을 때는 기준과 정답 모두 동일하게 합쳐 준비한다.
  기존 획의 어떤 채널이 어느 자모인지 자동 추정하지 않는다.
- K=1이면 softmax가 항상 1이므로 입력 전체를 반환한다. 자모 분리 학습에는 K>=2인 샘플이 필요하다.
- 최소 2개 필자/원본 그룹이 필요하다. writer_id가 있으면 필자 단위, 없으면 입력 이미지 단위로 80/20 분할한다.
  필자 ID 없이 필자 일반화 평가를 보장하지 않는다.

## 실행과 결과

GPU 런타임에서 위부터 실행한다. 설정 셀에서 TRAIN_CSV, OUTPUT_ROOT를 지정한다.
변환 후 실제 기준 자모와 정답을 미리 보여준다. 학습 원본에만 20개 증강을 적용하며
기본 30 epochs, batch 8이다. VARIANTS=0으로 증강 없이 비교할 수 있다.
Swin 사전학습 가중치를 사용하는 새 학습이며 기존 획 체크포인트는 불러오지 않는다.

OUTPUT_ROOT/data에 변환 캐시, jamo_run_시간에 best.pt, history.json,
run_config.json(코드 커밋/설정), learning_curves.png, val_preview 그림과 soft NPZ를 저장한다.
기준 자모, 입력, 정답, 최종 분할, 자모별 soft mask를 확인할 수 있다.
TEST_CSV 지정 시 별도 test 평가도 실행한다.

같은 경로에서 train 원본 이미지를 수정하면 REBUILD=True로 새 캐시를 만든다.
test 이미지를 수정하면 새 CSV 파일명으로 지정해 test 캐시를 구분한다.
기존 skeleton CSV를 지정하면 skeleton 실험이 되며 굵은 자모 실험으로 자동 변경되지 않는다.
자모 분리는 개별 획 분리와 평가 단위가 다르므로 Dice 숫자를 직접 비교하지 않는다.
