# 새 오류 위치 탐지기 학습 코랩

[Colab 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/train_error_localizer.ipynb)

## 실행 순서

1. GPU 런타임에서 준비 셀 실행: 최신 GitHub 코드, 의존성, Nanum 폰트 설치와 Drive 연결.
2. CHECKPOINT에 기존 V2/Topo 획 추출기의 best.pt를 선택한다. None이면 기존 v2_topology_runs의 최신 파일을 고른다. 새 오류 탐지기 best.pt를 이 경로에 넣지 않는다. 기존 학습 CSV/압축 자료를 prepare_pilot_data(reference_only=True)로 복구한다.
3. 정상/부착 돌출/분리 잡획과 실제 변경 위치 라벨을 눈으로 확인한다. 문자 ID는 명시적 char_id 또는 NanumGothic_가 같은 파일명 접미사에서만 읽으며 숫자 ID를 추측하지 않는다. 최소 6개의 확실한 문자명이 필요하다.
4. 기본 96문자, 문자·입력 폰트 분리, 문자/폰트당 변형6개, 10에폭, batch4, width32, LR2e-4로 학습한다. 이 값은 시작 설정이며 최적값이나 학습 시간 보장이 아니다. 추출기는 동결하고 오류 탐지기만 학습한다.
5. test 지표, 주석/예측 지도, 학습 곡선을 확인한다. best.pt/last.pt와 설정·지표·예시 그림 ZIP을 받는다.
6. 선택적으로 REAL_IMAGE에 실제 생성물에서 잘라낸 한 문자 crop과 REAL_CHAR를 넣어 오류 지도를 확인한다. 이 셀은 깨끗한 정답 RGB를 사용하지 않는다.

출력은 HSQM/error_localizer_runs/실행시간에 저장된다. 세션 재시작 후의 학습 재개는 아직 지원하지 않으며 새 실행 폴더에서 다시 시작한다.

## 데이터와 평가의 의미

error_synthesis.py는 같은 폰트의 정상 이미지에 선/점의 추가 잉크만 합성한다. kind1은 기존 잉크에서 시작하는 부착 가지, kind2는 기존 잉크와 떨어진 점, kind0은 변화 없는 정상 이미지다. 실제 추가되어 보이는 픽셀만 위치 정답으로 라벨링한다. 연결성·정상 음성·라벨 재현성은 테스트되지만 합성이 모든 실제 오류를 대표한다는 뜻은 아니다.

문자를 먼저 train70%/val15%/test나머지로 나누고 각 split에 서로 다른 설정 폰트를 사용한다. 기본 train은 NanumGothic/굵게, val은 NanumMyeongjo, test는 NanumSquareR 또는 NanumBarunGothic이다. 파일 경로 및 내용 해시의 split 중복은 거부한다. 가족 폰트의 별도 파일이 같은 family인지 자동 판정하지 않으므로 사용자 폰트 그룹을 바꿀 때 family 분리를 지켜야 한다. 기존 추출기 자체가 이 문자/폰트를 사전학습에서 봤을 수 있으므로 전체 시스템의 unseen-character/font 성능으로 주장하지 않는다.

검증 loss가 낮은 head를 best로 저장한다. 화면의 threshold0.5는 미보정 진단값이다. pixel precision/recall/IoU와 clean_false_positive_pixel_rate를 함께 보고, 정상 세리프가 오류로 잡히는지 그림으로 확인한다. 최종 threshold는 validation에서 별도 보정해야 한다. 같은 test를 보며 설정을 반복 조정하면 held-out 의미가 약해진다.

현재 학습 loop는 FP32, worker0으로 재현하기 쉬운 첫 연결이다. 실제 GPU 메모리/속도는 측정하지 않았고 K가 많은 글자는 batch를 줄여야 할 수 있다. clean RGB는 합성·주석/그림용이며 실제 모델 forward에는 넣지 않는다. sparse 중심선 주석을 dense 추가 획 라벨로 오용하지 않는다.

## 구현 범위

오류 탐지기 Gate A만 학습한다. 복원 제안기 학습, 실제 생성물 수동 라벨 미세조정, confidence calibration, 자동 삭제, 역확산 연결은 후속 단계다. 모델을 학습했다고 곧바로 정상 획을 지워도 된다고 해석하지 않는다. 기존 생성/재역확산 노트북의 기본 동작은 바꾸지 않는다.

로컬 검증: 합성 라벨·부착/분리 연결성·숫자 문자명 거부·split 분리·복제 폰트 파일 거부, 작은 CPU에서 2에폭 학습과 checkpoint 저장/로드/평가까지 수행했다. 이는 실제 한글 오류 탐지 정확도의 검증이 아니다.
