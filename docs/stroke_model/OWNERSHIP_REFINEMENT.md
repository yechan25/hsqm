# V2 구조를 유지한 작은 획 배정 개선 실험

## 수정 범위

Swin/FPN/flow/공유 head/softmax와 기존 주석 CE/Dice는 유지한다. 모델 파라미터,
state_dict 및 추론 연산은 추가하지 않는다. 현재 V2+정션 가중치로 초기화할 수 있다.
독립 sigmoid, 새 backbone, ConvLSTM, 추가 teacher 추론은 넣지 않는다.

목표는 작은 획이 큰 획 영역을 파먹거나 떨어진 다른 획 조각을 소유하는 오류를 줄이는 것이다.
정션 개수/응답만으로는 연결성을 바꾸지 않는 파임과 고립 조각을 충분히 감독할 수 없다.
현재의 얇은 실제 정답선은 sparse ownership annotation으로 취급해야 한다.

## 연구 근거와 이 구현의 차이

- [ScribbleSup, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Lin_ScribbleSup_Scribble-Supervised_Convolutional_CVPR_2016_paper.html):
  scribble에서 미주석 픽셀로 정보를 전파해 학습하는 근거다. 여기서는 논문의 전체 graphical
  model을 재현하지 않고, 실제 입력 정답선/잉크에서 제한적인 보조 영역 주석을 오프라인으로 만든다.
- [PSI, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_Scribble-Supervised_Semantic_Segmentation_Inference_ICCV_2021_paper.html):
  신뢰도에 따라 seed를 확장하고 공간/의미 문맥을 쓰는 근거다. 여기서는 추가 추론 네트워크나
  반복 diffusion 대신 고정된 보수적 기하 기준과 같은 소유 획의 국소 일관성만 사용한다.
- [Boundary loss, MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html):
  불균형한 영역 손실을 경계 거리로 보완하는 근거다. 그러나 현재 중심선은 실제 영역 경계가
  아니므로 이 논문의 signed contour distance loss를 그대로 적용하지 않는다. 작은 획은
  클래스별 정규화한 보조 CE로 큰 획에 묻히지 않도록 한다. 이것은 해당 논문 재현이 아니다.

위 연구는 한글 획 분리의 성능 향상이나 아래 하이퍼파라미터를 보장하지 않는다.

## 두 가지 선택적 학습 항

1. **약한 영역 소유 감독**: 실제 입력 정답선에서 각 획의 seed를 만든다. 입력 잉크의 같은
   연결 성분 안에서만 확장하고, seed 위치의 실제 잉크 반폭을 넘는 원거리 확장은 제한한다.
   첫 후보와 둘째 후보가 가까운 곳, 실제 다중 주석 교차점, 미검수 영역 근처는 제외한다.
   원래 주석 픽셀은 바꾸지 않는다. 기하 기준의 reliability는 보정된 확률이 아니다.
   완전한 영역 정답이 아닌 weak annotation으로 저장하고 작은 가중치로만 감독한다.
2. **같은 소유 획의 이웃 일관성**: 위 보조 주석에서 소유 획이 같은 가로/세로 이웃의
   해당 획 확률이 급변하지 않게 한다. 다른 소유 획/빈 공간/unknown을 가로질러 평활화하지 않는다.
   실제 끊어진 획에 양쪽 seed가 있으면 두 조각을 그대로 감독하고 사이를 연결하지 않는다.

실제 경로에 포함된 가지는 지우지 않으며 기준 폰트 모양으로 입력을 맞추지 않는다.
새로 추가된 미주석 잉크나 seed 없는 성분은 보조 감독에서 제외한다. 추론 시 이를 삭제하지 않는다.
중심선 위치가 틀렸거나 교차 부근이 모호하면 보조 주석도 틀릴 수 있으므로 검수 묶음으로 먼저 확인한다.

## 속도 예산

거리 변환/성분 분석은 학습 전 CPU에서 한 번 계산하고 `(owners int16, reliability uint8)`
두 장의 맵으로 캐시한다. 128px에서 샘플당 비압축 약 48KiB이고, 압축 NPZ로 저장한다.
키에 원본 manifest/파일 크기·mtime/설정을 포함한다. val/test 주석을 확장하지 않는다.
실제 준비 시간은 preparation_report에 별도로 기록하며 epoch 시간에 숨기지 않는다.

배치 중에는 gather, 클래스별 누적, 가로/세로 차분만 수행한다. 전체 픽셀 쌍 비교,
추가 skeleton 반복, CRF/graph 반복, 추가 모델 forward는 없다. 가중치가 0인 항의 계산은 생략한다.
AMP/worker 2/persistent workers/로컬 로딩은 기존과 같다.

기준 epoch 약 30s에 대해 33s(약 +10%)를 비교 목표로 둔다. GPU 실측 전의 예산이며 보장값이 아니다.
같은 GPU·배치·자료·초기 가중치에서 워밍업 제외 3epochs의 시간을 비교한다. 이 기준은
학습·검증을 포함하는 epoch_seconds에 적용하며 전처리 시간은 별도로 보고한다.
--epoch-warning-seconds 33은 경고만 출력하고 설정을 몰래 바꾸거나 학습을 자동 중단하지 않는다.

## 실행과 검증

코랩의 `OWNERSHIP_WEIGHT=0.1`, `AFFINITY_WEIGHT=0.02`는 시험 시작값이다.
기본값은 둘 다 0으로 기존 동작을 보존한다. 켜면 첫 3개 train 그룹에서 보조 주석을 표시하고,
그 뒤 전체 train 준비 및 학습을 수행한다. weak owners가 다른 획을 침범하는지 확인해야 한다.
CLI: `--ownership-weight .1 --affinity-weight .02 --epoch-warning-seconds 33`.
결과 폴더에 소유 감독 설정/손실/epoch 시간을 기록하며 신경망 가중치 형식은 동일하다.

동일한 기존 체크포인트에서 A=현재 Topo, B=소유 감독만, C=소유 감독+이웃 항을 분기해 짧게 비교한다.
학습 선택에 쓰지 않은 고정 검수 묶음의 파임 픽셀, 다른 획 혼입, 누락/끊김/가지 보존을 본다.
현재 sparse val Dice나 설계상 100%인 잉크 커버리지만으로 개선을 판정하지 않는다.
진짜 교차점에서는 exclusive 출력상 소유 경쟁이 남는다. 교차 픽셀을 여러 획이 온전히 공유하는
요구는 출력 표현을 바꾸지 않고는 해결할 수 없으며 이번 실험의 범위가 아니다.

소형 합성 사례에서 경계 침범/원거리 오배정 손실 증가, 교차점 unknown 유지,
실제 분리 보존, 유한 gradient, 캐시/분할 보존, 저장/재로드를 검증한다.
실제 사진 성능 개선과 epoch 30s 대비 GPU 시간은 아직 검증하지 않았다.
