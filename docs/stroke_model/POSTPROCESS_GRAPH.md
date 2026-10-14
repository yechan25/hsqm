# 교차점을 보존하는 graph 후처리

기존 `postprocess.py`는 비교용으로 유지하고, 새 구현은
`stroke_model/graph_postprocess.py`에 추가했다. 기본 출력은 획마다 독립적으로
임계값을 적용하므로 교차점은 여러 획이 공유할 수 있다. 재학습은 필요 없다.

## 방법

입력 잉크 픽셀을 8방향으로 연결한 그래프 위에서 다음 에너지를 최소화한다.

`E(Q) = ||Q-P||² + a ||Q-H||² + λ Σ_(i,j) w_ij ||Q_i-Q_j||²`

- P: sigmoid 모델 출력. 채널별 최댓값으로 나누지 않는다.
- H: 모델이 출력한 warp된 힌트. 없으면 이 항을 생략한다.
- a: 힌트 신뢰도 (`hint_weight`, 기본 0.25).
- λ: 이웃 간 연속성 강도 (`smoothness`, 기본 1.0).
- w: 예측/힌트가 다른 이웃일수록 작은 연결 강도.
  `exp(-max_k |R_ki-R_kj|² / (2*edge_scale²)) / pixel_distance`,
  `R=(P+aH)/(1+a)`, `edge_scale` 기본 0.25.

고정된 그래프에 대해 엄밀히 볼록한 이차식이며, 희소 선형 시스템을 풀어
유일한 최소점을 구한다. 완전한 무가정 방법이나 새로 학습한 모델은 아니다.
면적 순서, seed 선정, skeleton 가지 제거, 타 채널 강제 이동 규칙을 없애고
해석 가능한 세 가지 조절값으로 바꾼 것이다. 기본값의 실제 데이터 성능은
아직 검증되지 않았다. SciPy CPU 추론용이며 autograd는 지원하지 않는다.

## 사용

```python
from stroke_model.graph_postprocess import postprocess_batch_predictions_graph

with torch.no_grad():
    out = model(I_prep, I_g, I_G_strokes)
    post = postprocess_batch_predictions_graph(
        torch.sigmoid(out['logits']), I_prep,
        reference_strokes=I_G_strokes,  # 실제 채널 식별만; 위치에는 사용하지 않음
        warped_hints=out['H'],         # 반드시 입력 이미지 좌표계
        mode='overlap',
        threshold=config.get('hard_threshold', 0.25),
    )
```

`mode='soft'`로 보정 점수, `mode='exclusive'`로 비교용 배타적 마스크를 받는다.
exclusive는 모든 잉크를 강제 배정하므로 낮은 근거에서도 배정하고, 교차점에서
획이 끊길 수 있다. 실제 사용 기본값은 overlap이다. 반환은 입력 예측과 같은
device의 float32 텐서이며, 계산은 CPU에서 수행한다.

## 평가와 한계

- `notebooks/train_bifpn.ipynb` 마지막 비교 셀은 raw/기존/graph/정답을 같은
  잉크 영역에서 비교한다. graph용 위치 정보에 정답을 넣지 않는다.
- 활성 정답 채널의 Dice/IoU를 사용한다. 빈 채널이 점수를 올리지 않게 한다.
- 중복은 오류가 아니다. coverage, 누락 획, 정답 잉크 밖 비율도 별도로 본다.
- 입력 이진화는 기존과 동일하게 유지했다. 임계값 때문에 원래 잉크가 빠지는
  문제는 이 알고리즘으로 복구되지 않는다. `ink_mask`로 명시적 마스크를 줄 수 있다.
- 실제 끊긴 잉크를 배경을 가로질러 잇지 않는다. 같은 획의 여러 조각을 허용한다.
- 모델/warp 힌트가 모두 틀리면 정답 획을 알아낼 수 없다. 모든 채널이 약한
  픽셀은 미배정으로 남을 수 있으며 이를 임의의 채널로 채우지 않는다.
- 입력 증거가 강하게 틀린 경계는 보존할 수 있다. 연속성 보정이 항상 정확도를
  높인다고 가정하지 않는다. 기존보다 개선됐다는 결론은 실데이터 평가 후 내린다.
- 먼저 validation에서 raw, graph(hint_weight=0), graph(+H), legacy를 비교한다.
  λ∈{0,0.5,1,2}, a∈{0,0.1,0.25}, threshold∈{0.2,0.25,0.35,0.5} 등은
  작은 탐색의 예시다. 최종 test는 파라미터 선택에 사용하지 않는다.
- Gaussian soft target의 threshold mask와 실제 원본 획 마스크는 다르다.
  최종 품질 검증에는 원본 획 라벨도 확인해야 한다.

검증: `python -m unittest discover -s tests -v` (hsqm 폴더에서 실행).

관련 근거: [Grady, Random Walks for Image Segmentation](https://www.cs.bu.edu/groups/ivc/pubs/Grady06.pdf)의
그래프 기반 연속성 개념을 참고하되, 여기서는 hard seed random walker를
그대로 구현한 것이 아니라 모든 픽셀의 soft evidence를 유지하는 이차 최적화를 사용한다.
[SciPy sparse solvers](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).
