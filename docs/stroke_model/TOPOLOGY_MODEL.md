# 전체 정렬과 실제 획 구조 감독을 추가한 추출기

`TopologyStrokeModel`은 기존 V2를 보존하는 별도 실험 아키텍처
`reference_topology_aligned_v1`이다. 구현과 학습 경로를 제공하며 실제 손글씨로
학습된 새 체크포인트나 개선된 추출 성능을 제공하는 것은 아니다.

## automated_train_dataset에서 Colab 학습

[Topology 학습 코랩 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/train_topology_zips.ipynb)

GPU 런타임에서 준비 → TAR/ZIP 및 CSV 변환과 실제 획 정답 확인 → 학습 → 검증 미리보기 순서로 실행한다.
데이터는 공유 Drive의 `2026 자율연구/HSQM/dataset/automated_train_dataset`에 둔다.
`input_stroke`, `input`, `target_stroke`, `target` 폴더와 하위 폴더의 TAR/ZIP을 모두 검색한다.
파일명·번호·개수 제한 없이 TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2, TAR.XZ/TXZ, ZIP을 읽는다.
각 파일은 독립 압축 파일이어야 한다. 새 파일을 추가하면 다음 준비 실행 때 자동 반영된다.
완료한 TAR은 재실행할 때 재사용한다. 해당 역할의 TAR이 없으면 기존 `역할.zip`도 지원한다.
CSV는 `path (1).csv` 우선, 없으면 `path.csv`/`paths.csv`를 선택한다.
`CSV_PATH`로 정확한 파일을 지정할 수 있다. CSV 자체와 데이터는 GitHub에 올리지 않는다.

`prepare_stroke_zips(..., independent=True)`는 실제 `input_stroke_paths`를 이진
`target_masks`로 저장하며 겹침과 빈 획 슬롯을 보존한다. exclusive labels에서 복원하지 않는다.
전체 마스크가 완전한 정답이라고 가정하며 unknown 주석 영역은 CSV 형식으로 제공되지 않는다.
빈 기준 획(128px 변환 후 최대 잉크값 ≤0.5)이 있는 샘플은 행 전체를 제외하고 계속 진행한다.
제외한 CSV 행 번호·입력 파일·빈 기준 획 파일은 `skipped_samples.csv`와 변환 보고서에 기록하며,
학습 결과 폴더에도 복사한다. 입력 정답의 빈 획은 누락 획 감독을 위해 그대로 보존한다.
writer/group ID가 없으면 같은 기준 이미지의 합성 변형을 묶어 80/20 분리한다.
같은 기준의 변형이 train/val에 섞이지 않지만 별도 필자 일반화나 독립 test를 보장하지 않는다.

기본은 128px, batch 4, 30epoch, lr 2e-4. Swin ImageNet 사전학습에서 새 학습하며,
선택적으로 `INIT_CHECKPOINT`에 V2 가중치를 지정하면 특징 추출부를 초기화한다.
양방향 문맥의 의미 순서를 유지하도록 이 노트북은 획 순서를 섞지 않는다.
결과는 `HSQM/topology_runs/실행시간`에 best.pt, history.json, 설정, CSV 선택 기록,
코드 commit SHA와 검증 그림으로 저장한다. 학습 배치는 Colab 로컬 파일을 사용한다.
best는 present macro Dice로 선택하며 정션 손실과 다른 학습 손실은 history에 함께 기록한다.
실제 Drive ZIP과 GPU 성능은 Colab에서 확인해야 한다.

노트북 재생성: `python tests/build_topology_zip_notebook.py`.

## 구조

입력+기준 글자 → 학습 가능한 전역 affine → 정렬된 기준 글자와 기준 획 →
기존 V2의 공유 국소 flow와 특징 추출 → 양방향 ConvLSTM 획 문맥 → 획별 sigmoid.
전역 정렬은 기준과 기준 획만 이동한다. 실제 입력은 재샘플링하거나 자르지 않는다.
최종 마스크는 `image * sigmoid(logit)`이라 실제 입력 밖의 먹은 생성하지 않는다.
교차점은 여러 획이 함께 가질 수 있다. 미배정 먹은 제공하지만 자동 추가 획 판정은 아니다.

전역 affine은 출력 좌표에서 기준 좌표로의 sampling transform이며 양의 determinant를
갖는 회전/축별 scale/shear/translation 조합이다. identity로 초기화된다.
기존 국소 flow의 각 축 ±12px 제한은 전역 정렬 이후의 잔차 이동에 적용된다.
전역 정렬에 별도 변환 정답은 없으며 실제 획 마스크 손실을 통해 학습된다.
`alignment_weight`는 정렬 정답과의 차이가 아니라 과도한 변환을 억제하는 identity prior다.

## 감독과 손실

`target_masks`는 **입력 손글씨의 실제 획 마스크** B,K,H,W다. 기준 획이나 추론
결과를 정답으로 대체하지 않는다. 실제 교차점은 multi-hot, 누락된 예상 획은 all-zero다.
`valid_pixels`는 공통의 주석 확인 영역 B,1,H,W다.

총손실 = 기존 독립 모델 BCE/Dice/union/clDice + 0.2 일관성 + 0.1 정션 구조 차이
+ 0.05 affine identity prior. 숫자는 조정 가능한 실험 초기값이며 보정된 최적값이 아니다.

- 일관성: 수평/수직/대각 이웃에서 예측 마스크 차분과 실제 정답 차분을 비교한다.
  같은 획을 나눠 가져가는 내부 경계를 벌주면서 실제 획 경계는 유지한다.
  근처의 서로 다른 획을 무조건 평활화하지 않는다. 전체 획 ID 교환을 이것만으로
  판별할 수 없으므로 픽셀별 실제 정답 감독이 여전히 필요하다.
- 정션: 공통 절대 레벨 .3/.5/.7에서 soft skeleton을 계산하고 반경 1/2/3의
  국소 crossing excess를 합친다. 예측과 **실제 정답** 응답 지도의 차이를 비교한다.
  기준 위치/기준 길이/기준 seed는 사용하지 않는다. 실제 정답에 있는 가지는
  그대로 예측할 때 구조 손실 0이다. 정션이 항상 없어야 한다는 강제 제약이 아니다.
- 구조 손실은 실제 정답 skeleton 길이로 정규화한다. unknown 영역과 영향 범위는
  구조 지도 비교에서 제외하며, 작은 이미지나 제한적인 주석에서는 구조 손실의
  유효 영역이 없어질 수 있다. 기본 128 해상도에서 사용한다.

모든 예측 손실은 PyTorch 연산으로 미분되며 min/max/abs의 부분미분을 포함한다.
soft skeleton과 crossing 응답은 정션 개수와 동일하지 않으며 곡선/굵기/해상도에
민감한 근사값이다. 외부 그래프 판정과 실제 예제로 검증해야 한다.
진짜 분리를 메우는 등 원본 글씨의 결함을 제거하는 것이 목표가 아니다.

## 실행

기존 `IndependentDataset`, group 분리 manifest, 학습/검증 경로를 재사용한다.

```console
python -m stroke_model.train_independent --model topology --manifest data/manifest.json --init-checkpoint old_v2.pt --output outputs/topology --epochs 30 --batch-size 4
```

`--loss-config` JSON 예:

```json
{
  "base": {"bce_weight": 1.0, "dice_weight": 1.0, "union_weight": 0.2, "cldice_weight": 0.1},
  "consistency_weight": 0.2,
  "junction_weight": 0.1,
  "alignment_weight": 0.05,
  "thresholds": [0.3, 0.5, 0.7],
  "temperature": 0.08,
  "radii": [1, 2, 3],
  "skeleton_iterations": 12
}
```

`--alignment-config` JSON 예:

```json
{"translation_fraction": 0.25, "log_scale": 0.4, "rotation_degrees": 20.0, "shear": 0.2}
```

학습 checkpoint에는 architecture, alignment_config, loss_config가 함께 저장된다.
기존 V2는 warm start만 가능하며 학습된 새 checkpoint로 로드할 수 없다.

Colab은 위의 `train_topology_zips.ipynb`를 사용한다. 학습 설정과 손실 가중치는
세 번째 코드 셀에서 바꿀 수 있다. 마지막 셀은 저장한 Topology best.pt를 로드해
실제 검증 획 정답과 soft/hard 예측을 비교한다. 공유 교차점 보존을 위해 argmax로
획 하나에 강제 배정하지 않는다.

## 검증 범위와 다음 실험

소형 가짜 backbone을 사용해 identity/이동 bounds, 원본 먹 보존, 공유 교차점,
손실의 유한 gradient, 잘못 나눠 배정한 경우의 응답, 실제 정션 보존,
unknown 제외, 1epoch 학습→저장→로드를 검사한다. 이것은 실제 손글씨 성능 증거가 아니다.
현재 train/val 체크포인트 선택은 기존 present macro Dice를 재사용한다.
실제 평가는 별도 필자/폰트에서 획 Dice와 hard 정션/분리 오검출을 함께 비교해야 한다.

먼저 같은 자료/분할에서 기존 V2, 독립 모델, 전역 정렬 추가, 일관성 추가,
정션 추가를 각각 비교해야 원인을 분리할 수 있다. 정상 길이·굵기 변형뿐 아니라
실제 누락/추가/분리/정션 사례도 포함해 결함을 미화하지 않는지 확인한다.
