# 누락·분리·정션·추가 잉크 구조 지표

평가 입력 사진과 파일명, 0~100 점수를 카드로 표시한다. 높을수록 측정 오류가 적다. 학습 없이 정해진 공식으로 계산하며, 사람의 미적 평가에 교정된 점수는 아니다.

[점수 Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/structural_indicator.ipynb)을 열거나 [추가 셀](../../notebooks/structural_indicator_cell.py)을 기존 구조 진단 뒤에 붙인다. `structure_results`를 재사용하거나 첫 셀 `REPORT_PATH`에 기존 `structure_report.json`을 지정한다. 저장된 NPZ가 있으면 추출기 재추론 없이 추가 잉크를 재측정한다. `MODE="values"`는 `VALUES`와 글자 전체 `EXTRA_VALUE`를 직접 넣는 수치 테스트다.

## 다른 이미지 평가

독립 점수 Colab의 기본값은 `MODE="csv"`다. 첫 설정 셀부터 순서대로 실행하면 `dataset/test_dataset/paths.csv`에서 12장을 선택해 기존 체크포인트로 추론한 뒤 **각 입력 사진과 점수**를 표시한다. 이 모드는 이전 `structure_results`에 들어 있던 사진을 재사용하지 않는다. 새 학습은 하지 않는다.

- 다른 무작위 표본: `RANDOM_SEED`를 다른 정수로 변경한다. 같은 CSV와 seed는 같은 행을 고른다.
- 장수 변경: `SAMPLE_COUNT=12`를 원하는 수로 변경한다.
- 파일명 지정: `INPUT_FILES=["실제파일명.png", "다른파일명.png"]`. 이 경우 표본 장수와 seed는 무시한다.
- 전체 평가: `INPUT_FILES=None`, `SAMPLE_COUNT=None`. 전체 데이터는 시간이 걸릴 수 있다.
- 기존 결과 재계산: `MODE="auto"`와 `REPORT_PATH` 사용. 붙여넣는 추가 셀은 기존처럼 auto가 기본이다.

선택한 CSV 행과 파일명을 먼저 출력한다. CSV에는 입력 이미지와 목표 기준 이미지·기준 획 경로가 필요하다. 입력 획 정답 주석은 사용하지 않는다. 점수 공식과 기본 감점 설정은 그대로 유지한다.

## 추가 획을 놓치던 이유와 수정

기존 지표는 목표 글자의 획마다 누락 M, 분리 D, 정션 J를 계산했다. 목표에 없는 획에는 평가 채널이 없으므로, 목표 아를 야로 쓴 경우 남는 가로획이 점수에서 빠질 수 있었다. 모든 픽셀을 기존 획에 배정하는 모델에서는 미배정 잉크만 세는 방식도 충분하지 않다.

이제 **입력 사진 전체**에서 목표 기준의 허용 영역 밖에 있는 잉크를 E로 측정하고, 글자 전체에 감점을 준다. 기존 획에 붙은 추가 획과 떨어진 추가 획 모두 대상이다. 작은 분리 잡음을 D에서 무시하더라도, 기준 밖에 남은 잉크는 E의 허용량을 넘으면 감점된다.

1. 목표 기준 이미지를 이진화하고 중심선을 만든다. 새 진단 NPZ의 `reference_image`를 우선 사용하며, 오래된 파일에서는 `reference`, `reference_geometry`, `reference_strokes` 순서로 찾는다.
2. 중심선까지의 거리 d와 허용 영역 A = sigmoid((반경-d)/전이폭)을 만든다.
3. 각 기준 채널을 ±6px 안에서 2px 간격으로 이동한 후보들을 준비한다. 중심선 주위 Gaussian(폭 2px)과 입력의 일치도에서 이동 제곱 비용을 뺀 뒤, 온도 0.01의 softmax로 후보 가중치를 구한다. 가중 평균 허용 영역들을 1-product(1-A)로 합친다. 기준 한 채널이 전체 목표 이미지라면 전체에 하나의 이동을 적용한다. 입력의 이산 argmax를 사용하지 않는다.
4. 입력 I는 잉크=1, 배경=0으로 두고, 추가 지도 U = I × (1-A)를 계산한다.
5. E = sum(U) / max(sum(I), 1)로 전체 입력 잉크 중 설명되지 않는 비율을 구한다.

E는 **추가 획 개수나 의미를 인식한 결과가 아니다.** 입력과 목표가 같은 좌표계에 정렬되어 있어야 한다. 위치, 크기, 굵기, 폰트 차이도 E를 증가시킬 수 있고, 허용 영역 안에 들어간 짧은 추가 획은 놓칠 수 있다. 전체 목표 이미지가 없는 기존 중심선에는 목표 정보가 부족할 수도 있다. 정상 자료로 허용 반경과 허용량을 정하고 고정한 뒤 비교해야 한다.

## 점수 공식과 맨 위 설정

    rM_k = (1-M_k)^wM
    d_k = wD·D_k/sD,  rD_k = 1-pD·d_k/(1+d_k)
    j_k = wJ·J_k/sJ,  rJ_k = 1-pJ·j_k/(1+j_k)
    q_k = 100 × rM_k × rD_k × rJ_k
    Qmix = (1-alpha) × mean(q_k) + alpha × min(q_k)
    C_M = 100 × min(rM_k)
    C_D = 100 × min(rD_k), C_J = 100 × min(rJ_k)
    C_E = 100 × exp(-wE·max(E-허용량, 0)/sE)
    Q0 = min(Qmix, C_M, C_D, C_J)
    Q = min(Q0, C_E)

`priority_mode=True`가 새 기본값이다. 거의 사라진 획은 C_M으로 전체 점수를 제한하여 정상 획이 많아도 희석되지 않는다. 추가도 C_E로 제한한다. 분리와 정션은 단독 오류일 때 감점 한도를 두며 기본값 pD=0.60, pJ=0.65다. 분리만 있으면 40~100, 정션만 있으면 35~100점이고, 심한 삭제·추가는 그보다 더 낮아질 수 있다. 이는 사용자가 지정한 중요도를 구현한 평가 규칙이다. 오류의 크기가 다르거나 복합 오류라면 모든 사진의 순서가 항상 고정되지는 않는다. 파일명은 계산에 사용하지 않는다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| missing_weight / split_weight / junction_weight | 1 | 획 오류별 중요도 |
| split_scale / junction_scale | 0.2 / 0.01 | 작을수록 해당 오류 감점 증가 |
| worst_stroke_weight | 0.5 | 평균과 최악 획 점수의 혼합 비중 |
| priority_mode | True | 새 중요도 규칙. False이면 이전 지수 곱셈 공식 사용 |
| split_max_penalty / junction_max_penalty | 0.60 / 0.65 | 단독 분리 / 정션 감점 한도 |
| extra_alignment / extra_alignment_step | 6 / 2 | 제한 이동 범위 / 후보 간격. 범위 0이면 이동 없음 |
| extra_alignment_temperature | 0.01 | 이동 후보 softmax 온도 |
| extra_alignment_prior | 0.001 | 이동 제곱 비용 |
| extra_alignment_sigma | 2.0 | 중심선 일치도를 계산하는 폭 |
| extra_weight | 1 | 추가 잉크 감점 가중치. 0이면 명시적으로 검사 제외 |
| extra_scale | 0.025 | 허용량 초과 E가 2.5%p이면 추가 상한 100/e점 |
| extra_allowance | 0.03 | 전체 잉크의 3%까지 추가 감점 유예 |
| extra_radius | 8 | 기준 중심선 주위 허용 반경. 128 해상도 기준 픽셀 |
| extra_transition | 0.5 | 허용 영역 경계의 전이 폭 |

모든 설정은 첫 셀 `SCORE_CONFIG`에 있다. 반경은 해상도에 맞춰 조정한다. 초기값은 학습·최적화한 값이 아니다. E는 글자 전체 항이므로 획 점수 q에는 포함하지 않고 최종 글자 점수 Q에만 반영한다. 목표에서 누락된 획은 active=True로 유지하고 패딩만 제외한다. 파임 W, 빨간 원 개수, 노란 후보 개수, 파일명, 설문은 공식에 사용하지 않는다.

실제 제공된 개발 예제 4장을 재계산하면 삭제 19.14, 추가 25.07, 분리 68.82, 정상 97.20점이다. 분리 감점 한도만 0.30에서 0.60으로 조정하여 분리는 직전 84.41점에서 68.82점으로 내려갔고, 나머지 세 예제 점수는 유지된다. 분리 예제 E는 12.1%에서 2.5%로 감소했다. 겹침은 로컬 ZIP에 원본 배열이 없어 전체 재계산하지 못했다. 사용자가 제공한 M/D/J 수치만으로 계산한 겹침 상한은 51.71점이며, E까지 측정한 최종 점수는 아니다. 이 예제들로 설계를 확인했으므로 독립 검증은 아니다. 정상 변이와 각 오류를 분리한 추가 자료에서 교정해야 한다.

## 사진과 진단 출력

사진은 NPZ의 실제 평가 입력 스냅샷을 우선 사용하고, 없으면 원본 PNG를 읽는다. 기준 이미지를 평가 사진으로 대체하지 않는다. 카드에 사진·점수·추가 잉크 비율·추가 감점 전 점수가 나온다. **추가 잉크 위치 보기**의 빨간색은 기준 허용 영역 밖의 잉크다. 정션 진단의 빨간 원과는 별도 표시다. **오류별 점수 상한 보기**에서 누락/추가/정션/분리의 상한을 확인한다. 복합 오류는 상한보다 더 낮아질 수 있다.

- `images_and_scores.html`: 내장 사진과 점수 카드, 획별 표 및 추가 위치 펼침.
- `images_and_scores_01.png` 등: 사진과 점수 비교 그림.
- `extra_0001.png`, `extra_0001.npz` 등: 추가 위치와 입력/허용 영역/추가 지도.
- `glyph_scores.csv`: 최종 점수, 추가 감점 전 점수, E와 유지율.
- `stroke_scores.csv`: 획별 M/D/J와 획 점수.
- `indicator_report.json`: 설정과 결과, E 계산에 사용한 파일·기준 키.

NPZ가 없고 E도 제공되지 않으면 기본 설정은 오류를 낸다. 추가 측정이 없다는 이유로 E=0으로 가정하지 않는다. 이전 세 항만 비교하려면 `extra_weight=0`을 명시한다. `SHOW_IMAGES`, `IMAGES_PER_ROW`, `IMAGES_PER_PAGE`, `SHOW_STROKE_TABLE`은 표시 설정이다.

## 미분 가능한 사용

```python
from stroke_model.extra_ink import ExtraInkMetric
from stroke_model.structural_indicator import IndicatorConfig, StructuralIndicator

extra_metric = ExtraInkMetric(aligned_target_strokes, radius=8, transition=.5).to(image.device)
extra = extra_metric(image)['extra']  # image: B,1,H,W, 절대 잉크 0..1
metric = StructuralIndicator(IndicatorConfig())
errors = criterion(output['masks'])['per_stroke']
result = metric(errors['missing'], errors['split'], errors['junction'],
                active=output['active'], extra=extra)
loss = result['penalty'].mean()  # 1-Q/100
```

기준의 이진화·스켈레톤·거리 및 이동 후보 계산은 고정 전처리이며 기준에 대한 미분은 제공하지 않는다. 이동 후보의 가중치 선택부터 U와 E는 입력 이미지에 대해 미분 가능하다. 점수는 ReLU와 최솟값 연산에서 부분 기울기를 사용한다. 최솟값 상한 방식은 지배적인 오류에 기울기를 집중하며, 여러 오류를 동시에 줄이는 학습 목적에는 별도의 보조 손실이 필요할 수 있다. 입력부터 Tensor 경로를 유지해야 한다. JSON/NPZ를 읽는 실행 함수는 평가용이며 역확산 그래프와 연결되지 않는다. 실제 scheduler 연결은 포함하지 않는다. 누락 유지율은 eps를 이용해 양 끝을 정규화하여 분수 가중치의 무한 기울기를 피한다.

검사: `python -m unittest discover -s tests -p test_structural_indicator.py -v` 및 `test_extra_ink.py`, `test_priority_indicator.py`. 기존 공식·이미지 대응 검사 외에 붙은/떨어진 추가 획, 입력 기울기, E 누락 시 실패, 기존 NPZ 재측정과 추가 위치 저장을 검사한다. 정션 근사와 기준 정렬의 오류도 최종 점수에 전달되므로 원래 오류값과 지도를 함께 확인한다.
