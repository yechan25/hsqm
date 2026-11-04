# Topo 추출과 기존 원본 점수 함께 보기

[통합 코랩](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/topology_with_scores.ipynb)

사용자가 선택한 방식은 원본 점수 + Topo 추출 결과다. Topo의 가중치와 기존 점수 공식은
바꾸지 않고 같은 입력 스냅샷으로 두 경로를 연결한다. 점수 계산에 soft_masks를 사용하지 않는다.
Topo가 원본 획을 다른 채널에 배정하거나 파임을 만들어도 그것만으로 원본 점수가 내려가지 않는다.
원본에 실제 있는 누락/분리/추가 가지/추가 잉크는 기존 source 측정과 공식으로 평가한다.
구조 점수는 사람이 정한 규칙 기반 지표이며 사람 선호에 보정한 학습 점수가 아니다.

## 구성

준비/Drive → 모델/CSV 설정 → modified 이미지 추출+점수 → 선택적 캔버스 추출+점수.
캔버스는 첫 두 셀 후 마지막 셀로 바로 갈 수 있다. 기준 글자 선택, pen 굵기, undo, clear를 제공한다.
입력 획 정답은 읽지 않는다. 기본은 최신 v2_topology_runs best.pt며 CHECKPOINT로 특정 After를 고른다.
checkpoint의 epoch/보조 감독 가중치를 출력한다. best.pt는 읽기만 한다.

입력/기준/배정 색상/soft 획/exclusive 획과 원본 점수를 같은 그림에 표시한다.
누락·분리·정션·추가 잉크 각각의 점수 상한과 원본 측정의 획별 점수 표를 제공한다.
점수 상한은 각 항의 오류값 그 자체가 아니며 기존 공식의 cap 값이다.
별도 굵기 점수를 새로 추가하지 않는다. 점수 설정과 정렬 설정은 SCORE_CONFIG/SOURCE_CONFIG다.

현재 원본 점수는 Topo 채널과 수치적으로 독립이다. 같은 입력/기준/설정에서 추출기만 바꾸면
점수가 변하지 않는 것이 맞다. 이 노트북은 추출 성능을 원본 점수 상승으로 검증하지 않는다.
추출 개선은 soft/exclusive 그림 및 이전 비교 노트북에서 별도로 확인한다.

원본 점수는 기존 bounded affine 정렬과 source 측정을 사용하므로 추론보다 시간이 걸릴 수 있다.
새로운 추출기 학습이나 추가 점수 네트워크 학습을 실행하지 않는다.

## 저장과 검증

checkpoint 옆 topo_scores_시간 폴더에 통합 PNG, source_report.json, indicator_report.json,
combined_report.json, glyph_scores.csv, stroke_scores.csv와 진단 ZIP을 저장한다.
스냅샷은 기존 추론 NPZ를 사용하며 원본 image/full reference/reference_strokes를 필수로 검사한다.
score_structure_report에는 원본 source 모드만 전달한다. 실제 계산하지 않은 추출 오류를 0으로
꾸며 넣지 않고 stroke ID만 메타데이터로 제공한다. sample별 측정 배열과 정렬 정보도 보존한다.

합성 스냅샷에서 Topo 채널만 바뀌어도 원본 점수가 동일한지, 실제 원본 삭제 시 점수가 내려가는지,
캔버스/CSV 결과 연결과 PNG/CSV/ZIP 생성 및 필수 원본 누락 거부를 확인한다.
사용자의 실제 체크포인트/데이터 결과는 Colab 실행에서 확인해야 한다.
