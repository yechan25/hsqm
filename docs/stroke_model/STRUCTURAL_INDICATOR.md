# 누락·분리·정션 세 값으로 산출하는 구조 지표

설문이나 추가 학습 없이 missing(M), split(D), junction(J)만 사용한다. 학습 파라미터가 없는 PyTorch 모듈이며 사람이 설정한 공식으로 0~100 지표를 산출한다. 높을수록 이 세 측정 오류가 적다. 미적 품질이나 사람 선호에 교정한 점수는 아니다.

## 바로 실행

- [점수 계산 Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/structural_indicator.ipynb)
- [기존 구조 진단 Colab에 붙일 추가 셀](../../notebooks/structural_indicator_cell.py): 전체를 마지막 새 셀에 붙여 실행한다. 같은 런타임의 structure_results를 재사용한다.
- [코드](../../stroke_model/structural_indicator.py)

독립 노트북을 새로 열었다면 첫 셀의 REPORT_PATH에 기존 structure_report.json 경로를 넣는다. MODE="values"는 첫 셀의 VALUES에 숫자를 직접 넣는 테스트다. 저장된 값을 재사용하므로 기존 획 추출기 추론을 반복하지 않는다. 새 모델 가중치나 설문 데이터도 필요하지 않다.

## 획 점수

    q_k = 100 × (1-M_k)^wM × exp(-wD·D_k/sD - wJ·J_k/sJ)

| 입력·설정 | 의미 |
|---|---|
| M | 기존 missing. 0~1의 기준 대비 누락량 |
| D | 기존 split. 기준 경로의 연결 부족량 |
| J | 기존 junction, 즉 J_soft. 분기 반응 근사 |
| wM, wD, wJ | 오류별 중요도. 0으로 두면 해당 항 제외 |
| sD | split_scale. 기본 0.2 |
| sJ | junction_scale. 기본 0.01 |

기본 가중치는 모두 1이다. sD/sJ가 작을수록 같은 오류값에 더 크게 감점한다. 예를 들어 D=sD이고 다른 오류가 0이면 100/e≈36.79점이다. M=0.809만 있으면 해당 획은 19.1점, M=1이면 0점이다. 세 오류가 모두 0이면 100점이다. 이 값들은 공식의 결과이며 실험으로 입증된 등급 기준은 아니다.

점수는 세 항의 유지율을 곱한다. 한 오류를 다른 오류의 좋은 값으로 상쇄하지 않는다. 파임 W, 빨간 원 개수, 노란 후보 개수, 설문, 변형 이름은 공식에 넣지 않는다.

## 글자 점수

    Q = (1-alpha) × 획 점수 평균 + alpha × 최저 획 점수

alpha는 worst_stroke_weight이며 기본 0.5다. 0이면 단순 평균, 1이면 가장 나쁜 획의 점수다. 패딩 채널은 active=False로 제외하고, 목표 글자에서 실제 누락된 획은 active=True로 남겨야 한다.

예를 들어 4획 중 3획이 100점이고 1획이 완전 누락으로 0점이면, 평균은 75점이고 기본 글자 점수는 37.5점이다. 심한 단일 오류가 평균에 묻히는 정도를 alpha로 조절한다.

## 설정과 결과

첫 셀 SCORE_CONFIG에 모든 가중치, 두 scale, 최악 획 비중을 모았다. 초기값은 공학적 시작값이며 최적화하거나 학습한 값이 아니다. 비교 실험에서는 설정과 원래 측정기 설정을 고정한다.

주요 출력은 글자 structural_score와 획별 stroke_score다. 글자 평균/최저 점수와 각 항의 유지율도 함께 저장한다.

- glyph_scores.csv: 글자별 최종 구조 지표와 평균/최저 획 점수.
- stroke_scores.csv: 원래 M/D/J, 획 점수, 세 유지율.
- indicator_report.json: 점수 설정, 원래 측정 설정, 사용한 항목, 결과 전체.

출력 폴더는 원래 결과 폴더 아래 indicator_시간으로 새로 만든다. SAVE_DIR를 직접 지정하면 아직 없는 경로를 사용한다.

## 미분 가능한 사용

```python
from stroke_model.structural_indicator import IndicatorConfig, StructuralIndicator

metric = StructuralIndicator(IndicatorConfig(**SCORE_CONFIG))
errors = criterion(output['masks'])['per_stroke']
result = metric(errors['missing'], errors['split'], errors['junction'], active=output['active'])
score = result['score']       # 높을수록 좋음
loss = result['penalty'].mean()  # 1-score/100, 낮을수록 좋음
```

Tensor 경로를 유지하면 기울기를 전달한다. 최저 획 연산은 거의 모든 곳에서 부분 기울기를 사용하며 완전히 매끄럽지는 않다. JSON을 읽는 실행 함수는 저장된 숫자를 계산하는 평가 경로라 이미지로 기울기를 전달하지 않는다. 실제 역확산 scheduler에 연결하는 코드는 포함하지 않는다.

분수 가중치에서도 누락 1의 기울기가 무한대가 되지 않도록, 누락 유지율은 eps를 더한 거듭제곱을 양 끝 0/1에 맞춰 정규화한다. 기본 가중치 1에서는 위 공식과 같다. 완전 누락의 0점을 유지하면서 기본값에서 누락을 줄이는 기울기가 사라지지 않는지도 검사한다.

## 해석 범위와 검사

정션 측정의 오검출·미확정성과 기준 정렬 오류는 점수에도 전달된다. 특히 J_soft만으로 실제 정션의 유무나 심각도를 확정하지 못한다. 이 합성 지표를 만든 것으로 원본 글자의 객관적 품질을 검증한 것은 아니다. 최종 점수를 볼 때 원래 세 오류값도 함께 확인한다.

`python -m unittest discover -s tests -p test_structural_indicator.py -v`

정상 100점, 전체 누락, 단일 심각 오류, 각 항의 단조성, 가중치 제외, 패딩 채널, 유한 기울기와 유한차분, 잘못된 입력 거부, 다른 측정값이 결과에 영향을 주지 않는지, CSV/JSON 저장을 검사한다.

노트북 재생성: `python notebooks/build_structural_indicator.py`.
