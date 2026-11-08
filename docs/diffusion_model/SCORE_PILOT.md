# 실제 확산 출력의 기존 점수 시험

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/test_diffusion_scores.ipynb).
사용자 체크포인트·학습 manifest/NPZ는 Drive의 기존 자료를 쓴다. 실행 코드는 GitHub main에서 받아 사용하고 실제 커밋을 출력·저장한다. 노트북을 갱신하려면 `python notebooks/build_diffusion_score_pilot.py`.

## 선택한 실험

- 공개 사전학습 `stable-diffusion-v1-5/stable-diffusion-v1-5`의 img2img + DDIM.
- 학습 목록의 split=train에서만 기준 이미지와 **기존 ordered strokes**를 읽는다. test/val 선택은 거부한다. NPZ의 image/labels/target_masks는 읽지 않는다.
- 사용자는 해당 checkpoint의 실제 학습 manifest를 골라야 한다. best.pt에는 학습 글자 목록이 들어 있지 않아서 경로 선택만으로 실제 학습 이력을 증명할 수는 없다. run_config.json 경로가 유효하면 재사용하고, 경로가 Colab 임시 위치라 사라졌다면 Drive의 원본 manifest를 고른다.
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

참고: https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img , https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
