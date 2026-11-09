# 프롬프트만으로 먼저 생성하는 기본 실행 경로

[새 기본 Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/prompt_then_repair.ipynb)

## 첫 생성의 강제 분리

첫 단계는 diffusion_model/prompt_generation.py의 generate_prompt_image다. 함수의 입력에 image/template/font/reference/checkpoint는 없다. QwenImagePipeline(text-to-image)에 prompt, negative_prompt, seeded generator, inference steps와 크기만 전달한다. 문자를 폰트로 그리거나 기준 이미지를 VAE로 인코딩하지 않는다. 이 단계가 끝나기 전에는 획 추출기/오류 탐지기/기준 자료도 불러오지 않는다.

generation_report.json은 phase=prompt_only_generation, input_image_used=False, template_used=False, font_rendering_used=False와 프롬프트·seed·revision을 기록한다. 정답 이미지 대신 생성물 generated.png를 그대로 저장한다. Qwen이 한글을 틀리게 생성하거나 레이아웃을 지키지 못한 경우에도 폰트 이미지로 대체하거나 결과를 골라내지 않는다.

## Qwen으로 생성 및 보정 교체

기본 모델은 Qwen/Qwen-Image-2512, 1024×1024, 50 steps, true_cfg_scale=4다. 생성 모델을 올린 채로 저장된 오류 탐지기를 실행하므로 코랩 A100 40GB 이상을 요구한다. Transformer와 text encoder는 NF4 4비트, VAE는 BF16이며 VAE 타일링을 사용한다. T4 및 SD1.5로 자동 fallback하지 않는다. 다운로드 용량과 RAM/VRAM 사용은 SD1.5보다 크며 첫 실행은 오래 걸릴 수 있다. 양자화 후 한글 정확도 및 GPU 최대 메모리는 아직 실측하지 않았다.

SD1.5용 DDIM 코드를 Qwen에 호출하지 않는다. diffusion_model/qwen_repair.py는 같은 Qwen 컴포넌트로 QwenImageImg2ImgPipeline을 구성하고 네이티브 FlowMatch 스케줄을 사용한다. 첫 생성에는 입력 이미지가 없고, 재역확산 단계에만 방금 생성한 이미지를 전달한다. 두 비교 경로는 같은 seed로 VAE 샘플링과 재잡음을 수행하며 실제 첫 잠재값의 SHA256이 다르면 중단한다.

가이드 단계는 flow 예측으로 clean = sample - sigma × velocity를 계산한다. 기존 탐지기가 고정한 추가 획 지도와 주변 배경으로 만든 손실을 Qwen VAE 디코더를 통해 clean latent에 미분한다. Transformer 및 VAE 가중치는 동결한다. RMS 정규화한 gradient로 후보를 만들고 손실이 내려가는 경우만 채택한다. 다음 flow 상태에는 (1 - next_sigma) × clean_residual을 더한다. 이는 새 실험적 가이드 연결이며 품질 향상을 입증한 결과가 아니다. 최종 RGB를 직접 덮어쓰지 않는다. 기본 보정은 50 steps × strength .3으로 실제 15회이며, 단계별 loss와 clean 예측 중간 PNG를 저장한다. 저장 PNG는 중간 clean 예측이며 최종 출력은 네이티브 pipeline의 마지막 디코딩이다.

설치는 `diffusion_model/requirements_qwen_colab.txt`에서 diffusers 0.37.0, transformers 4.57.6, huggingface-hub 0.36.0으로 고정한다. 이전 Diffusers 개발 커밋은 Hub >=1.32를 요구하지만 Transformers 4.x는 Hub <1을 요구해서 동시에 설치할 수 없었다. 0.37.0 배포 wheel에 Qwen 생성/img2img API가 모두 있는지 확인했으며 Linux CPython 3.13 wheel 대상으로 전체 의존성 pip dry-run이 성공했다. 이는 실제 코랩 설치 및 GPU 추론 검증과 다르다. 첫 셀은 pip 출력을 그대로 표시하고 `qwen_setup.log`에 저장하며, 설치 후 별도 프로세스에서 실제 Qwen import도 검사한다. 이미 import한 패키지가 변경되면 세션 재시작을 요구한다. [Qwen 모델 설명](https://huggingface.co/Qwen/Qwen-Image-2512), [정식 배포판 img2img 코드](https://github.com/huggingface/diffusers/blob/v0.37.0/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py), [양자화 문서](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)를 따른다.

## 실행

1. 코랩의 런타임을 A100으로 선택하고 코드/의존성·Drive 준비. 예전 실행 세션은 재시작한다.
2. PROMPT를 입력하고 실제 text-to-image 생성. 기본 예시는 가나다라가 적힌 표지판 사진이며 문자열도 수정 가능하다. 현재 후속 검출 경로는 네 문자만 지원하므로 EXPECTED_TEXT와 ROI 개수는 네 개다. 이 제한은 첫 생성기가 기준 이미지를 필요로 한다는 뜻이 아니다.
3. 생성이 끝난 뒤에만 저장된 오류 탐지기 및 그때 사용한 획 추출기를 불러온다. 목표 문자에 맞는 표준 획 조건을 선택한다. 이 표준은 뒤쪽 검출기의 조건이며 처음 생성에 전달되지 않는다.
4. 생성 사진에서 실제 네 글자의 bbox를 CHAR_BOXES에 읽기 순서대로 지정한다. 자동 2×2 crop을 사용하지 않는다. 의도한 글자가 전혀 없거나 OCR/ROI 대응이 실패했다면 작은 획 수정으로 해결 가능하다고 주장하지 않는다.
5. 실제 생성물을 재잡음해 무가이드/가이드 재역확산 비교.
6. 최초 생성물과 비교·중간 결과 및 전체 진단 ZIP.

현재 사진 속 글자 ROI 자동 검출은 미구현이다. 사용자가 명시한 bbox로 국소 검출/보정 능력을 분리 평가한다. 기존 오류 마스크 가이드의 밝은 단색 배경 삭제 제약은 남는다. 일반 사진 배경 인페인팅까지 해결했다고 해석하지 않는다. 실제 GPU 결과는 미검증이다.

## 이전 경로 정리

- structure_generalization의 기본 INIT_MODE를 noise, AUTO_FONT_INPUT을 False로 변경했다. 폰트 입력은 명시적인 대조군에만 허용된다.
- generate_first_stage는 noise를 기본으로 하며 external/reference 첫 생성을 거부한다. 순수 첫 생성에는 새 generate_prompt_image를 사용한다.
- detected_extra_diffusion은 과거 stage1 이미지의 자동 선택을 제거했다. 그 별도 추론 노트북에서는 사용자가 FIRST_IMAGE를 직접 지정해야 한다.
- 이전 문서의 명조체/strength0.65 폰트 img2img는 과거 대조 실험 기록이며 현재 주 실행 경로가 아니다.

검증은 첫 생성 호출에 이미지 인자가 없는지, 템플릿 구성 함수가 호출되지 않는지, 생성 셀이 검출 모델 로딩보다 앞서는지, 폰트 렌더링/과거 파일 자동 선택이 새 노트북에 없는지 확인한다. CPU 합성 예제로 flow residual 계산, gradient 손실 감소, 변화가 없을 때 원래 flow 유지, 실패 후 scheduler 복원을 검사한다. 이 검증은 GPU에서 실제 Qwen 체크포인트를 실행한 품질 평가와 구분한다. 기존 탐지기 정확도가 모델 교체로 자동 개선되지는 않는다.
