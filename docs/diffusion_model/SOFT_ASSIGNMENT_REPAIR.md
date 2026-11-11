# Softmax 연속 거리 가이드 및 반복 역확산 보정

주 실행 파일은 `notebooks/soft_assignment_repair.ipynb`이며 [Colab에서 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/soft_assignment_repair.ipynb)를 사용한다. 현재 버전은 `soft-assignment-v2-matched-baseline-all-slots`다. L4에서 3셀을 순서대로 실행한다. 기본 FOCUS_SLOTS는 (0,1,2,3)으로 네 칸 모두 활성화하며 확대 그림도 네 칸을 각각 표시한다. 원본 ZIP baseline을 그대로 재잡음하며 새 text-to-image 생성, Qwen 및 기준 RGB의 VAE 초기화는 없다.

1번 셀은 Google 공식 drive.mount를 먼저 호출한다. 연결돼 있으면 재인증 호출을 생략하며, 깨진 mount를 다시 연결할 때는 FORCE_REMOUNT=True를 사용할 수 있다. 인증/마운트 timeout은 180초다. Google 인증이나 네트워크 장애를 코드로 해결했다고 보장하지 않는다. 마운트 후 HSQM의 guidance_debug (4).zip을 찾고, 없으면 기존 topo_guided_diffusion/20261108_103932_929716 생성 폴더에서 같은 baseline과 기준 배열을 복원한다. 둘 다 없으면 ZIP만 업로드받는다. best.pt는 기록된 경로 및 알려진 당시 학습 폴더에서 찾으며 임의 최신 모델을 선택하지 않고 SHA256을 검사한다. 입력과 모델은 /content로 복사해 계산하며 최종 ZIP만 Drive의 HSQM/soft_assignment_repair_results에 저장한다.

보조 실행 파일 `notebooks/soft_assignment_repair_no_drive.ipynb`는 마운트 없이 ZIP과 best.pt를 업로드하는 경로다. 이 버전의 결과는 세션 임시 저장소이므로 마지막 ZIP을 다운로드해야 한다.

노트북에 기존 실행기와 새 모듈 소스를 포함한다. GitHub에서 기본 코드를 받은 후 해당 세션에 임베드한 `guided_ddim.py`와 `soft_assignment_guidance.py`를 적용하므로 Git 커밋 없이 로컬 수정 버전을 실행한다. 결과 ZIP에는 실제 적용한 소스도 기록한다.

## 손실

기준 획별 거리 D_k를 미리 구한다. 소스의 획 두께를 nearest-stroke 픽셀의 85% 분위수로 추정해 허용 반폭 r_k를 고정한다(기본 최소 3, 최대 6 모델 픽셀). 각 획의 비용은 `1-exp(-relu(D_k-r_k)/falloff)`다. softmax 모드는 픽셀별 `sum_k p_k * cost_k`, 기하 대조군은 `min_k cost_k`를 사용한다. 작은 배정 누설 .05를 뺀 연속값에 현재/소스 잉크를 곱하고 crop에서 전체 이미지로 복원한다. 모든 기준 획의 합집합 바깥만 남기는 최종 게이트를 사용하지 않는다.

이 비용은 기준 획 조건과 정렬을 요구한다. 오류 없는 생성물에 대한 보편적 검출기가 아니고 누락 획을 추가하지 않는다. 획 배정이 거의 nearest-stroke와 같으면 두 가이드가 비슷할 수 있으므로 둘 다 비교한다. 모델이 새로 학습된 것이 아니며 softmax가 의미적 오류 확률이라는 가정도 하지 않는다.

각 활성 역확산 단계에서 frozen extractor로 배정을 다시 추출한다. 그 단계의 모든 내부 반복과 backtracking 동안 확률/압력 지도는 detached 상태로 고정한다. 최초 오류 압력의 25%를 남겨 잔상이 옅어졌다고 지도에서 사라지는 현상을 줄인다. 주변의 밝은 단색 배경색을 소스에서 추정해 RGB MSE + .5 MAE를 연속 가중하며, 정상 영역 평균 보존 손실은 .1로 보조한다. 이전의 20배 국소 peak 손실은 보정 목적에서 제거했다.

## 실제 보정과 채택

기존 DDIM clean 예측을 VAE로 디코딩해 손실을 잠재값까지 미분한다. 소프트 오류 지도를 확장한 잠재 영역으로 gradient를 제한하고 여러 번 수정한다. 기본은 실제 역확산 35단계(100 nominal × .35), 활성 단계당 4회, 마지막 24회다. 한 단계의 총 잠재 이동은 anchor RMS의 .08 이하, 한 번의 제안은 .01 비율이며 backtracking으로 줄인다. 느려지는 비용이 있으며 전체 반복 횟수는 trace에 저장한다.

총손실이 내려갔다는 이유만으로 채택하지 않는다. 동일하게 고정한 압력 지도에서 삭제 손실이 실제로 감소하고 정상 획 보호 검사도 통과해야 한다. 보호 기준은 무가이드 실행의 같은 단계에서 측정해 저장한다. 보호 영역 RMS, 5×5 국소 RMS, 각 정상 획 코어 잉크 손실률을 개별 비교한다. 한도는 max(기본 한도, 같은 단계 baseline 값)+허용 오차다. 기본 한도는 .025/.12/.04, 추가 허용 오차는 .002/.02/.01이다. 따라서 매 내부 반복마다 보호 예산이 늘어나지 않으며, baseline에서 이미 나빠진 다른 획 하나가 모든 획의 보호 기준을 지배하지 않는다. baseline 없는 단독 최적화 테스트에서만 anchor를 대체 기준으로 쓴다. 이 보호 조건은 정상 상태를 새로 달성했다는 보장이 아니며 실제 출력 그림도 확인해야 한다.

v1 실행 ZIP `20261111_043924_692872.zip`에서는 focus_slots가 [0]이었고 보정 시도 280개 중 261개가 healthy_guard로 거절됐다. 최종 단계에서 삭제 손실 .894→.538 후보가 보호 RMS 약 .000011 증가 때문에 거절됐다. v2는 이 수치 회귀 사례를 테스트하고 무가이드 궤적 대비 실제 허용 오차를 둔다. 기존 1e-6 무변화 한도로 정상적인 VAE 결합 변화까지 막던 버그를 수정한 것이며 단계 수 증가로 해결했다고 해석하지 않는다.

채택한 clean residual은 DDIM 다음 상태에 `sqrt(alpha_prev)`로 전달한다. U-Net 방향은 고정하며 최종 RGB를 직접 덮어쓰지 않는다. 모델 및 VAE/U-Net 가중치는 학습하지 않는다.

## 출력 및 해석

같은 소스와 VAE 샘플링/잡음에서 무가이드·softmax 가이드·거리 가이드를 비교한다. 두 paired 실행의 초기 잠재값과 baseline PNG 일치를 검사한다. 전체/가 확대 그림, 중간 clean 예측 GIF, softmax/거리 초기 압력, 최종 압력, 시도별 삭제 손실 및 guard 실패 이유, 실행 소스를 ZIP으로 저장한다. 모든 방법의 진단에는 공통 최초 압력을 사용하며, 이는 독립 정확도 점수로 해석하지 않는다.

로컬 CPU 테스트는 softmax 교체 시 지도 변화, 반복 중 배정 동결, 삭제 손실 감소, 정상 획 손상 검사, 새 잉크 및 잔상에 대한 압력 유지, 실제 DDIM 상태 변경과 VAE latent gradient 연결을 확인한다. 사용자 ZIP의 저장된 softmax로 새 초기 압력을 구성해 확인했다. 실제 SD1.5 + 사용자 checkpoint에서의 최종 삭제량·잔상·정상 획 보존 품질은 코랩 실행으로 검증해야 한다. 로컬에는 GPU가 없다.
