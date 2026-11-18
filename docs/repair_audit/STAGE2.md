# 2단계: 삭제 경계 보호 수정 및 동일 조건 비교

현재 위치는 **1단계 진단 완료 → 2단계 수정 구현·비교**다.
1단계에서 정상 개/곁의 말단·접합부가 피팅 허용 영역 밖으로 판단되는 오탐을 확인했다.
원본 초기 삭제 지도 재현 및 경계 반경 반사실 비교에 기반한 실험 수정이다.

## 실행

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage2.ipynb)

1. 새 L4 GPU 세션에서 1번 셀 실행.
2. 2번 셀에서 기존 best.pt 경로 확인. `ARCHIVE=''`이면 업로드 창에서
   `repair_stage1_20261118_054447_077301.zip`을 선택. 원래 cyclic/selectable ZIP도 지원한다.
3. 3→4→5번 셀 실행. stage1 ZIP의 이미 보정된 결과가 아니라 **원래 input.png**를 재사용한다.
4. 결과 ZIP에서 기존 cyclic과 boundary_guarded를 비교한다. 정상 대조군을 제외하지 않는다.

기존보다 재역확산 경로가 하나 늘어 실행 시간이 증가한다. 처음에는 기본 설정을 유지한다.
경계 픽셀 수는 128 모델 좌표이며 576 전체 이미지 좌표의 1픽셀과 다르다.

## 유지한 것

- 동일 획 추출기 및 기준/피팅 경로. 모델 학습 없음.
- 강도 .25, nominal 60 steps(실제15), seed2026, CFG5.
- 동일 초기 VAE mode latent, 잡음, 시간 단계로 일반/기존/수정 경로 비교.
- 기존 gradient step .008, RMS budget .06, backtracks5, restore/settle/erase 횟수.
- 기존 손실 가중치, 경로 복원과 clean-residual 상태 전달.

## 변경한 것

`BoundaryGuardedRepair`는 기존 controller를 상속하고 삭제 지도 생성만 재정의한다.

1. 반경에 `uncertainty_pixels=1`을 더해 작은 경계 차이를 삭제 근거에서 제외한다.
2. seed 팽창 뒤에도 `responsibility > .5` 및 안전 영역 밖 조건을 만족하는 픽셀만 삭제한다.
3. 모델 지도를 원본 해상도로 보간한 후에도 eligibility의 nearest 지도를 적용해 경계 누출을 제한한다.

반경 여유는 추가 획을 그리거나 기준 글씨를 두껍게 변환하는 처리가 아니다.
실제 말단 돌출이 있는 경우까지 보호할 수 있는 무조건적인 말단 금지는 사용하지 않는다.
최적 여유값은 검증되지 않았으며 다른 문자에서 작은 오류를 놓칠 가능성이 있다.

## 출력과 판정

- `comparison.png`: input / VAE / redenoise_only / cyclic_guided / boundary_guarded.
- 같은 step의 erase/restore/preserve PNG와 모든 clean prediction.
- `boundary_initial_maps.npz`, `boundary_initial_diagnostics.json`, `boundary_guarded_map_history.json`.
- 공통 healthy 지표에 더해 전체 원본 전경을 평가한 `all_source_*` 지표.
  정상 끝이 기존 자동 healthy 마스크에서 제외되는 맹점을 보완하지만, 입력 오류도 포함하므로 정오 지표는 아니다.
- 실패 목록 및 사람 검토표. 최종 출력에서 원본으로 몰래 대체하거나 후처리하지 않는다.

먼저 정상 개/곁의 윤곽 손상이 줄어드는지와 생성 가/꺄의 실제 추가 획 제거가 유지되는지 확인한다.
GPU 결과가 나오기 전에는 품질 개선을 주장하지 않는다. 기준 추종, 손글씨 일반화,
일반 재역확산 자체의 정상 영역 변형은 별도 미해결 과제다.

## 로컬 확인

- 경계의 작은 차이 보호와 먼 돌출 유지, 확장/보간 누출 제한, 복원 경로 보호,
  동일 초기 상태의 세 경로 비교, stage1 원본 재사용 등 CPU 검사 9개 통과.
- Colab 내장 소스를 별도 디렉터리로 복원해 같은 9개 검사를 재실행해 통과.
- 제공된 stage1 ZIP의 저장 소속 확률로 초기 삭제 지도만 재계산한 결과,
  정상 개의 검은 원본 픽셀 침범은 123→2, 정상 곁은 262→0으로 감소.
  이 수치는 576 이미지 좌표의 초기 지도 지지 영역(가중치>.01)과 원본 전경의 교집합이다.
  역확산 중의 지도 갱신과 최종 보정 결과는 GPU 비교가 필요하다.
