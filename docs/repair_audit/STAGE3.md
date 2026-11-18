# 3단계: 다른 글자에서 보정 효과와 획 형태 확인

[Colab 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage3.ipynb)

2단계의 경계 보호 수정은 기존 네 글자에서 정상 획 손상을 줄이면서 추가 획 제거를 유지했다.
끝점 뭉침·잔상·재확산 자체의 변형은 남아 있다. 다음 수정 전에 다른 글자에서 반복되는지 확인한다.
이 노트북은 형태 보정 알고리즘을 추가하거나 모델을 학습하지 않는다.

## 실행 순서

새 L4 GPU 세션에서 코드 셀 1→7을 실행한다.

1. 패키지 설치와 내장 소스 복원.
2. 기존 best.pt 경로 확인, 원래 학습 자료에서 기준 글자/획 목록 복구.
3. 기존 네 글자를 제외한 12자를 미리 확인. 필요하면 SELECTED_ROWS에 화면의 row 번호 입력.
4. 4자씩 3묶음으로 img2img 오류 이미지 생성, 각 묶음의 정상 입력도 준비.
5. 같은 체크포인트와 SD 모델 로딩.
6. VAE 왕복 / 보정 없는 재확산 / 기존 cyclic / 경계 보호 보정 비교.
7. 결과 ZIP 다운로드와 MyDrive/HSQM_stage3_results 백업.

기존 stage2 ZIP을 업로드할 필요는 없다. 새 문자의 기준 획이 필요하므로 기존 학습 폴더의
run_config.json, selected_paths.csv, conversion_report.json과 기준 글자/획 원본 압축 자료가 필요하다.
기본 경로는 기존 실행과 같다. 자동 복구가 실패하면 2번 셀의 DATA_ROOT/CSV_PATH/MANIFEST를 지정한다.
원본 데이터와 체크포인트는 수정하지 않는다.

## 고정 조건과 선택

- 생성: SD1.5 고정 revision, 576×576 전체 시트, strength .65, seed100, 100 steps, CFG7.5.
- 보정: 60 nominal steps(실제15), strength .25, seed2026, CFG5.
- 2단계와 같은 추출기·피팅·손실·optimizer·경계 보호(1 model pixel, responsibility .5).
- 같은 입력의 세 재확산 경로는 초기 latent·잡음·시간 단계를 공유한다.
- 기본 12자는 목록에 고르게 분포하도록 결정하며 결과 품질로 선택하지 않는다.
  기존 row 1,2,3,5 및 목록에 있는 해당 콘텐츠 중복을 제외한다.
- 생성 전에 그림을 보고 받침/곡선/접합부가 다양하도록 선택할 수 있다.
  선택 목록과 row/char_id를 저장하며 생성 후 성공 사례만 선별하지 않는다.
- 기존 학습 목록의 다른 문자 평가다. 미학습 문자나 다른 폰트의 일반화 검증은 아니다.
- 생성에 오류가 없거나 심해도 포함한다. 생성 실패/필터링은 기록하며 정상 대조군은 유지한다.

## 평가

각 사례의 audit/shape_comparison.png는 고정 위치에서 네 글자를 각각 확대해 비교한다.
입력/VAE/일반/기존/수정 결과가 나란히 배치된다. 윤곽을 반듯하게 만드는 후처리는 없다.

human_review.csv에 다음을 yes/no/uncertain으로 기록할 수 있다.

- input_has_error, error_reduced: 실제 입력 오류와 제거 여부. 정상 입력의 오류 감소는 n/a.
- thinner, new_error, style_changed: 정상 획 및 스타일 손상.
- straight_or_curve_worsened: 직선·곡선 형태가 입력보다 어색해졌는지.
- endpoint_deformed, junction_deformed: 끝점·접합부의 뭉침, 절단, 울퉁불퉁함.
- residual_artifact: 제거 부위의 색 잔상이나 잔여 조각.

보정 없는 재확산 결과와 비교해 우리 보정이 더한 변형을 구분하고 notes에 기록한다.
자동 폭·기준 유사도는 반듯함이나 정확도의 정답 지표가 아니다.
검토표 수동 입력은 7번 저장 셀 재실행 시 보존된다.

검증 결과가 나온 뒤 반복적인 형태 손상과 잔상 중 다음 개선 대상을 정한다.
로컬 CPU 테스트는 선택/입출력/실패 보존/내장 코드 일관성을 확인한다. 실제 GPU 품질은 검증 대기다.
