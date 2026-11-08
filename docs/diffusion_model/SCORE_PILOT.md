# 실제 확산 출력의 기존 점수 시험

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/test_diffusion_scores.ipynb).
사용자 체크포인트와 학습 ZIP/TAR·CSV는 Drive의 기존 자료를 쓴다. 임시 manifest/NPZ가 없으면 원래 학습 준비 함수를 다시 실행해 자동 생성한다. 실행 코드는 GitHub main에서 받아 사용하고 실제 커밋을 출력·저장한다. 노트북을 갱신하려면 `python notebooks/build_diffusion_score_pilot.py`.

2번 셀은 Drive 전체에서 manifest를 찾지 않는다. 기본 자료 위치는 기존 학습 노트북과 같은
`HSQM/dataset/automated_train_dataset`이다. `CHECKPOINT` 옆 `run_config.json`의 해상도·seed,
`conversion_report.json`의 변환 방식, `selected_paths.csv`의 원래 CSV를 읽어
`prepare_stroke_zips`로 기존 ZIP/TAR를 복사·해제·변환한다. 원래 NPZ도 세션과 함께
사라지므로 JSON 목록만 만드는 것이 아니라 연결된 NPZ까지 준비한다.
현재 세션의 원래 manifest와 NPZ가 유효하면 재사용하고, 재생성도 로컬 완료 캐시를 재사용한다.
재학습하거나 가중치를 변경하지 않는다. 사용자가 없는 manifest 경로를 입력할 필요가 없다.
처음 복사·압축 해제·변환에는 시간이 걸리며 각 단계의 진행률이 표시된다.

## 선택한 실험

- 공개 사전학습 `stable-diffusion-v1-5/stable-diffusion-v1-5`의 img2img + DDIM.
- 학습 목록의 split=train에서만 기준 이미지와 **기존 ordered strokes**를 읽는다. test/val 선택은 거부한다. NPZ의 image/labels/target_masks는 읽지 않는다.
- 원래 체크포인트의 CSV·seed·해상도·변환 방식으로 학습 목록을 재생성한다. CSV를 별도로 지정하면 저장된 원본 CSV와 일치하는지 검사한다. 재생성 후 표본 수·건너뛴 수·그룹 컬럼·감독 방식도 학습 기록과 비교한다. 예전 학습 코드는 원본 이미지 내용의 해시를 저장하지 않았으므로 이후 원본 이미지가 바뀌지 않았다는 사실까지 소급 보장할 수는 없다. 원래 학습 압축 파일을 사용해야 한다.
- 문자 ID가 숫자여도 Unicode로 해석하지 않는다. 프롬프트는 일반적인 한글 인쇄체 설명이며 목표 글자의 시각 조건은 기준 이미지다.
- 512x512 img2img, strength .25/.45/.65, seed 100/101, DDIM 30단계. img2img가 사용하는 실제 역확산 단계 수는 strength에 따라 다르다.
- 128 등 원래 기준 해상도로 내린 절대 잉크 입력을 저장한다. 투명 배경은 흰색 합성, grayscale, bilinear resize, 1-luminance만 사용한다. polarity 자동 반전, crop, 이진화, 획 복원, 좋은 출력만 선택하지 않는다.
- 기준 이미지 자체를 대조군으로 포함한다. 원본 reference와 ordered strokes는 생성기에 맞춰 수정하지 않는다.
- 생성기를 해제한 뒤 동결된 기존 Topo로 획 추출, 기존 `score_topology_results`로 **원본 source 점수**를 산출한다. 점수 공식/가중치는 바꾸지 않는다. 폰트 실험에서 설정을 바꿨다면 SCORE_CONFIG/SOURCE_CONFIG에 같은 값을 넣어야 한다.
- 원본 RGB, 점수 입력 grayscale, 기준/획 배열, 모델 revision, seed/strength/scheduler, soft 획, 점수표, 진단 ZIP 저장. 필터 여부가 있으면 기록하고 해당 출력도 보존한다.

## 해석과 실행 상태

실제 확산 출력을 대상으로 하는 평가 파이프라인 점검이다. 기준에서 출발하므로 text-to-image 글자 생성 성능, 새로운 조건부 DDPM의 품질 또는 역확산 guidance 효과를 측정하지 않는다. 잘못된 글자/추가 텍스트/배경이 생길 수 있고, 낮은 점수는 스타일·배경 오탐과 실제 획 결함을 분리해 눈으로 확인해야 한다. VAE 복원과 resize의 영향도 포함되므로 생성 과정의 어느 구성요소가 오류를 냈는지 단독으로 설명하지 않는다.

로컬 검사는 생성기를 대체한 작은 테스트로 train 필터·원본 보존·기존 scorer 연결을 검증한다. 실제 SD 1.5 가중치의 GPU 생성과 사용자의 checkpoint/Drive 데이터 평가는 Colab 실행이 필요하다. 점수를 실제 산출했거나 정확도를 입증했다고 보고하지 않는다.

검사: `python -m unittest discover -s tests -p test_diffusion_score_pilot.py -v`

데이터 복원 검사: `python -m unittest discover -s tests -p test_diffusion_pilot_data.py -v`
작은 실제 ZIP/CSV로 없는 manifest와 NPZ를 복원하고 원래 train/val·기준 획이 동일한지,
캐시 재사용·변경 CSV 거부·가중치 보존을 확인한다.

참고: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img , https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
