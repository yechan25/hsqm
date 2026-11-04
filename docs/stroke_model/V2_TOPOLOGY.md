# Topo 수정: 기존 V2 유지 + 정션 보조 감독

현재 학습 노트북 `notebooks/train_topology_zips.ipynb`는 기존 `ReferenceStrokeModel`과
`train_reference` 경로를 사용한다. 새 학습 아키텍처 식별자는 `reference_v2_topology_loss_v1`이다.
모델의 계산 그래프와 state_dict는 기존 V2와 동일하다. 학습 목표가 다름을 구분하기 위해
체크포인트 메타데이터만 새 이름으로 저장한다. 실제 학습/성능 개선은 아직 검증 전이다.

## 보존한 것

- 동결 Swin-T, 작은 FPN, 원해상도 CNN, 공통 국소 flow, 기준 획 CNN, 공유 head.
- 유효 획 사이 softmax와 `masks=image*p`. soft 마스크의 합은 원본 잉크다.
- 원본 foreground에서 V2 argmax 배정으로 hard 마스크를 표시한다. 사후 팽창은 하지 않는다.
- 기존 sparse annotation CE/Dice + flow smoothness. 얇은 정답선 밖은 background가 아닌 unknown이다.
- 기존 AMP(BF16/FP16+scaler), TF32, batch 8, worker 2, pin memory/persistent workers,
  로컬 NPZ 로딩과 획 chunk 처리를 유지한다.

독립 sigmoid, 양방향 ConvLSTM, 전역 affine, dense thin-mask BCE/Dice/union,
획 순서 섞기/기준 jitter는 새 경로에 넣지 않는다. 이전 Topo는 비교/로드용으로 남긴다.

## 추가한 것

기본 손실에 `junction_weight * 실제 획의 정션 응답 차이`만 더한다. 실제 입력의
`target_masks`를 사용하며 기준 글자나 기존 모델 출력을 정답으로 쓰지 않는다.
정답에 원래 있던 가지는 동일하게 예측하면 정션 차이가 0이다. unknown 경계와 영상 경계는
골격 연산 지지 영역만큼 제외한다. CSV 획 마스크는 실제 경로의 완전한 주석이라고 가정한다.
얇은 마스크를 전체 두께의 dense 분할 정답으로 쓰지 않는다.

속도를 위해 기본 4배치마다 최대 4개의 실제 주석 획을 순환 선택하고,
한 occupancy threshold(0.5), 한 반경(1), skeleton 12회만 계산한다.
가중치 .05는 적용 배치의 값이며 적용하지 않는 배치에는 기본 V2 손실만 계산한다.
주기/획 제한/가중치는 실험 초기값이며 효과가 검증된 최적값이 아니다.
주기마다 가중치를 배수로 올리는 보정은 하지 않는다.
`JUNCTION_WEIGHT=0`이면 정션 계산 없이 기존 V2 모델/손실로 동작한다.

정션 손실은 정션 개수 그 자체가 아닌 미분 가능한 근사 응답이다. 원본 잉크 보존은
정확한 소유 획이나 개별 획 굵기/연결성을 보장하지 않는다. 실제 `초`/`ㅗ` 및 modified 사례에서
획 혼입과 정션 오검출을 별도로 검증해야 한다. 배타적 hard 분할은 교차점 공유를 표현하지 못한다.

## 속도 차이와 실행

실패한 이전 Topo는 ConvLSTM의 획별 순차 처리와 전역 정렬을 추가했고, 매 배치
3개 threshold × 3개 반경의 정션 응답 및 clDice용 골격 연산을 반복했다.
학습은 FP32/worker 0/batch 4였고 V2의 AMP·병렬 로더가 빠져 있었다.
각 요인의 실제 GPU 시간 기여는 아직 측정하지 않았다. 새 경로의 GPU 가속 배수도 미측정이다.

노트북은 `HSQM/v2_topology_runs/시간`에 결과를 저장한다. 기본은 ImageNet 초기화의 새 학습이다.
`INIT_CHECKPOINT`에는 기존 V2 또는 새 V2+정션 가중치를 지정할 수 있으며 optimizer 재개는 아니다.
실패한 `reference_topology_aligned_v1` 가중치는 구조가 달라 초기화용으로 받지 않는다.
학습 진행률과 로그를 출력하고 epoch마다 samples_per_second/train_seconds/data_wait_seconds를 기록한다.
`junction_cpu_dispatch_seconds`는 보조 손실을 포함한 CPU dispatch 구간이며 GPU kernel 실행시간이 아니다.

기존 V2와 비교할 때 같은 원본/분할/초기화/배치/AMP 설정으로 `JUNCTION_WEIGHT=0`과 `.05`를
비교한다. 전체 loss만 비교하지 말고 실제 획 배정, raw soft 마스크, hard 마스크 및 정션을 확인한다.
test 코랩은 두 아키텍처를 구분해 로드하며 새 모델에서는 soft 합/argmax hard 출력을 표시한다.

검증: 정션 off의 기본 V2 손실 동일성, unknown 픽셀의 비감독, 원본 잉크 보존,
잘못 붙은 가지와 실제 정답 가지 구분, 유한 gradient, 한 epoch 저장/재로드를 소형 backbone으로 확인한다.
