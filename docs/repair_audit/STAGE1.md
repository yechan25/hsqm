# 1단계: 기존 획 보정 진단 실행

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage1.ipynb)하고
**새 L4 GPU 세션에서 1~5번 코드 셀**을 실행한다. 노트북을 직접 업로드해 실행해도 된다.
소스가 내장되어 있으므로 별도 코드 ZIP이나 Git clone이 필요하지 않다.
생성 모델/추출기를 학습하지 않는다. 기존 결과 ZIP을 재사용한다.

## 준비물

- 기존 `cyclic_stroke_repair_*.zip` 또는 `selectable_adaptive_*.zip`.
  `experiment.json`, `cases.json`, `group_*/source_conditions/condition_1~4.npz`,
  `group_*/generation/generation_report.json`과 page PNG가 필요하다.
- 그 실험의 `best.pt`. SHA256이 ZIP의 기록과 일치해야 한다.
- Colab GPU 및 SD1.5 가중치를 내려받을 네트워크 연결.

현재 알려진 예시 입력은 `cyclic_stroke_repair_20261111_113433_878992.zip`이다.
디버그 그림만 든 ZIP이나 이전 `guidance_debug.zip`은 이 입력 형식과 다르므로 지원하지 않는다.
개인 데이터와 체크포인트는 노트북에 포함하지 않았다.

## 사용자 설정과 실행

1. 1번 셀: 패키지 설치, 내장 코드 준비. torch는 교체하지 않는다.
2. 2번 셀: `CHECKPOINT`를 Drive의 best.pt 경로로 지정한다.
   `ARCHIVE`를 비우면 기존 결과 ZIP을 업로드하고, 경로를 넣으면 Drive에서 읽는다.
   `CASE_IDS=None`은 ZIP의 모든 생성 사례다. 정상 입력 대조군은 그룹마다 한 번 실행한다.
   먼저 한 장 4글자와 정상 대조군으로 실행 가능하며, 이는 대규모 검증이 아니다.
3. 3번 셀: 기존 V2 topology checkpoint와 고정 SD1.5 revision을 로드한다.
4. 4번 셀: 원본 / VAE 왕복 / 일반 재역확산 / 기존 cyclic 보정을 비교한다.
5. 5번 셀: 전체 결과 ZIP을 내려받고 MyDrive/HSQM_stage1_results에 복사한다.

기본 보정 설정은 60 nominal steps, strength .25, seed 2026, CFG 5다.
실제 재역확산은 15단계다. 정상 대조군도 같은 방식으로 보정하므로 시간이 추가된다.
오래 걸리면 사례 수를 줄여 pilot을 실행하고, 임의로 step을 줄인 결과를 원래 실험과 섞지 않는다.

## 이번 비교의 정확한 의미

재역확산 두 경로는 동일한 원본 VAE latent와 동일한 Gaussian noise에서 출발한다.
이번에는 float32 VAE posterior **mode**를 공통으로 사용해 VAE 왕복과의 해석을 단순화했다.
기존 노트북의 sampled posterior 초기화와 다르므로 과거 PNG의 비트 단위 재현은 아니다.
기존 cyclic loss/iteration/주기와 clean-residual 상태 전달은 유지한다.
기준 전체 RGB는 진단 표시와 유사도 측정에만 쓰며 편집 목표로 새로 추가하지 않는다.
기존 controller의 기준 경로 기반 조건과 명시적 누락 경로 투영은 그대로 남는다.

선택적 `RUN_REFERENCE_WIDTH_PROBE=True`는 기준 이미지와 각 기준 획을 1모델픽셀 팽창시킨 뒤
같은 초기 latent에서 추가 보정을 수행한다. 이는 **기준 두께 민감도 스트레스 시험**이다.
진짜 다른 폰트가 아니며 추출기 OOD나 연결 변화도 일으킬 수 있다. 다른 폰트의 일반화나
기준 복사 원인의 확정 실험으로 해석하지 않는다. 첫 실행 기본은 False다.

## 결과와 사람이 할 일

- 각 사례 `audit/comparison.png`: 모든 비교 경로를 확인한다.
- `extraction_slot_*.png`, extraction NPZ: 획 소속 오류를 확인한다.
- `diagnostic_regions.png`: 정상으로 추정한 영역에 오류가 섞였는지 검수한다.
- 단계별 clean_prediction/erase/restore/preserve PNG와 trace: 얇아지는 시점과 삭제 지도를 대조한다.
- `human_review.csv`: 목표 글자, 오류 유형, 추출 적절성, 오류 감소, 얇아짐, 새 오류, 스타일 변화를 기록한다.
  yes/no/uncertain을 쓰고 정상 입력의 error_reduced는 n/a로 둔다.
- `automatic_metrics.csv`: .35/.5/.65 임계값별 고정 중심점 두께비, 정상 후보 잉크 감소·유지율·MSE를 저장한다.
- failures.json: 실패 사례를 분모에서 조용히 제외하지 않는다.

두께비는 원본 중심점에서 distance transform을 측정한 **보조 지표**이며 획 이동·안티앨리어싱에도 변한다.
정상 영역은 고정된 휴리스틱이다. 기준 유사도 상승은 오류 감소가 아니다.
코드 검증과 실제 GPU/체크포인트의 시각적 품질 검증을 구분한다.
우선 출력 ZIP과 사람 검토표로 다음에 고칠 문제 한 가지를 결정한다.
