# 작은 획 누락과 잔여 정션의 추가 학습

V2 네트워크/출력/기존 가중치 형식은 유지한다. 영역 소유 감독과 이웃 일관성에서 얻은
효과를 보존하면서, 실제 정답 중심선을 놓치는 작은 획과 다른 획을 가져오는 채널에 집중한다.

## 실제 주석 기반의 획별 focal 보조 항

[Focal Loss, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)의
쉬운 예제 기여를 줄이고 어려운 예제에 집중하는 원칙을 사용한다. 객체 검출기 재현은 아니다.
실제 labels>=0인 픽셀에서 `-(1-p)^gamma * log(p)`를 계산하고 획별 평균을 낸다.
큰 획의 많은 쉬운 픽셀이 작은 획의 실패를 덮지 않도록 한다. gamma=2, weight=.1은 시험 시작값이다.
배경/미검수/정답 없는 누락 획 슬롯에는 양성 목표를 만들지 않는다. 기준 획의 존재를 근거로
입력에 없는 획을 강제로 생성하는 감독이 아니다. 기존 sparse CE/Dice도 그대로 유지한다.

## 같은 예산의 hard + rotating 정션 선택

[OHEM, CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)의
어려운 예제를 우선 사용하는 원칙을 적용한다. detector/ROI OHEM 자체를 재현하지 않는다.
정션 적용은 기존 4배치마다 최대 4개다. hard_fraction=.5일 때 2개는 다른 실제 획의
주석 픽셀을 높은 확률로 침범한 채널, 2개는 나머지 순환 채널로 고른다.
점수 계산은 detach/no_grad로 하지만 선택된 정션 손실의 gradient는 연결된다.
실제 정답의 정션 응답과 비교하므로 정답에 존재하는 가지를 무조건 제거하지 않는다.
hard_fraction=0이면 기존 순환 선택과 같다. 원래 노이즈가 큰 주석에 편향될 가능성 때문에
전체 예산을 hard 예제만으로 채우지 않는 .5를 기본 시험값으로 둔다.

## 비용과 검증

신규 네트워크/추가 forward/전처리는 없다. focal은 gather/log/클래스 누적,
hard 선택은 이미 계산한 확률의 foreign-seed 최대값과 소수 채널 top-k만 사용한다.
골격 횟수/threshold/해상도/처리 획 수는 늘리지 않는다. 실제 GPU 추가 시간은 아직 미측정이며
기존 약 30s/epoch 대비 33s 비교 예산을 유지한다. 예산은 경고값이고 자동 속도 보장이 아니다.

validation은 추가 forward 없이 주석 중심선 회수율이 10% 미만인 획 수와 최소 회수율을 기록한다.
이는 주석에 한정된 진단이며 실제 누락 획 개수나 전체 획 품질 점수가 아니다.
best 선택 기준은 기존 annotated macro Dice를 유지한다.

## 실행

학습 코랩은 `SEED_FOCAL_WEIGHT=.1`, `JUNCTION_HARD_FRACTION=.5`를 제공한다.
CLI 기본은 둘 다 0이므로 기존 호출 동작을 유지한다. 끄면 이전 방식과 비교할 수 있다.
OWNERSHIP_WEIGHT/AFFINITY_WEIGHT=None이면 INIT_CHECKPOINT의 기록 값을 계승하고,
기록 없는 새 학습은 0으로 처리한다. 사용자가 숫자를 지정하면 그 값이 우선한다.
초기 가중치는 로컬에 한 번 복사한 뒤 메타데이터를 읽어 Drive 반복 로드를 줄인다.

현재 개선된 After best.pt를 INIT_CHECKPOINT로 지정하고 새 실행에서 3epochs 정도 시험한다.
미세조정에서는 lr=1e-4를 시작값으로 비교할 수 있다. optimizer 재개가 아니라 가중치 초기화다.
기존 best.pt는 읽기만 하고 새 실행 폴더에 결과를 저장한다. 같은 캔버스/modified를 비교한다.
실제 사진에서 정션/누락이 줄었는지는 아직 검증 전이다. 원본 점수 모델이나 파임 처리는 바꾸지 않는다.
