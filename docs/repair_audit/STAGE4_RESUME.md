# 4단계 재개: 완료 경로 재사용

[재개 Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage4_resume.ipynb)

코드 셀 1→5를 실행한다. 2번에 중단한 4단계 결과 ZIP과 기존 best.pt 경로를 지정한다.
이번 입력은 repair_stage4_20261123_132010_265667.zip이다. 새 글자 생성·학습은 하지 않는다.
보정 강도·시드·횟수·폭 설정과 모델 revision은 ZIP의 manifest에서 복구한다.

## 이번 ZIP에서 재사용할 작업

- group_000_page_001: 네 경로 모두 완료. 사례 전체와 지표를 재사용한다.
- group_001_page_001: redenoise_only 완료. 이 경로를 건너뛰고 cyclic_guided부터 재계산한다.
- 나머지 네 사례: 네 경로 전부 실행한다.

총 24개 재확산 경로 중 완료한 5개를 재사용하고 19개를 실행한다.
미완료 경로는 중간 latent/controller 상태가 없어 1/15부터 재시작한다.
중간 clean prediction PNG는 latent 체크포인트로 사용하지 않는다.

## 중단과 재개

- 경로 완료마다 ZIP을 만들고 MyDrive/HSQM_stage4_resume_results에 백업한다.
  같은 실행의 timestamp가 붙은 checkpoint.zip을 갱신하며 원본 업로드 ZIP은 바꾸지 않는다.
- 4번 셀을 중지하면 KeyboardInterrupt를 처리하고 현재 상태를 백업한다.
  5번 셀로 현재 ZIP을 다운로드할 수 있다.
- 런타임이 살아 있으면 4번 셀을 다시 실행한다. 완료 경로는 건너뛴다.
- 런타임 종료/연결 끊김 후에는 새 세션에서 최신 checkpoint ZIP을 2번에 넣는다.
  강제 종료 시 마지막 경로 완료 시점의 Drive 백업이 복구 기준이다.
- 자동 압축/Drive 복사 시간이 추가된다. 재확산 작업을 다시 하는 것과는 별도의 비용이다.

## 재사용 검사와 기록

- 모델·손실 구현의 소스 해시와 체크포인트 해시를 검사한다. archive 안의 코드는 실행하지 않는다.
- 원래 입력을 복구하고 audit/input.png와 픽셀 동일성을 검사한다.
- 캐시한 기준 획 배열이 현재 조건과 같은지 검사한다.
- 최종 PNG가 열리고 전체 timestep/step trace가 있는 경로만 완료로 인정한다.
- 부분 경로의 기존 프레임은 audit/interrupted_attempts에 보관한 후 다시 계산한다.
- 완성된 사례는 report의 설정과 전체 네 경로의 완료 여부를 검사한다.
- 새 실행에서 초기 latent를 저장해 이후 재개에서도 같은 텐서를 사용한다.

옛 4단계의 완성되지 않은 사례에는 초기 latent hash가 기록되지 않았다.
이번 ZIP의 두 번째 사례에서 과거 baseline과 새 경로의 초기 텐서가 비트 단위로 같았는지는
소급 검증할 수 없다. 해당 보고서는 legacy_initial_unrecorded=true,
paired_initial_latent=false로 표시한다. 조건/시드는 같으며 새 경로끼리는 초기 텐서를 공유한다.
GPU/패키지 변경을 엄밀히 통제한 최종 검증과 이번 작업 재사용은 구분해야 한다.
완성된 첫 사례는 서로 같은 기존 실행의 결과 전체를 그대로 사용한다.

## 검사

- 중단 경로 재시작, 완료 경로 무재계산, 부분 프레임 보존, 설정 변경 거절.
- 기존/새 ZIP의 초기 latent 추적 여부를 보고서에 구분.
- ZIP 경로 탈출·코드 파일 복사 차단, 전체 trace 검사, standalone 셀 문법 검사.
- 제공된 ZIP을 CPU에서 실제 복구해 완료 5개/잔여19개 계획을 확인했다.
- GPU에서 남은 보정을 직접 실행한 결과는 아직 없다.
