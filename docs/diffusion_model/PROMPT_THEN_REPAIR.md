# 프롬프트만으로 먼저 생성하는 기본 실행 경로

[새 기본 Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/prompt_then_repair.ipynb)

## 첫 생성의 강제 분리

첫 단계는 diffusion_model/prompt_generation.py의 generate_prompt_image다. 함수의 입력에 image/template/font/reference/checkpoint는 없다. StableDiffusionPipeline(text-to-image)에 prompt, negative_prompt, seeded generator, inference steps와 크기만 전달한다. 문자를 폰트로 그리거나 기준 이미지를 VAE로 인코딩하지 않는다. 이 단계가 끝나기 전에는 획 추출기/오류 탐지기/기준 자료도 불러오지 않는다.

generation_report.json은 phase=prompt_only_generation, input_image_used=False, template_used=False, font_rendering_used=False와 프롬프트·seed·revision을 기록한다. 정답 이미지 대신 생성물 generated.png를 그대로 저장한다. SD1.5가 한글을 틀리게 생성하거나 레이아웃을 지키지 못한 경우에도 폰트 이미지로 대체하거나 결과를 골라내지 않는다.

## 실행

1. 코드/의존성·Drive 준비.
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

검증은 첫 생성 호출에 이미지 인자가 없는지, 템플릿 구성 함수가 호출되지 않는지, 생성 셀이 검출 모델 로딩보다 앞서는지, 폰트 렌더링/과거 파일 자동 선택이 새 노트북에 없는지 확인한다. bbox가 재역확산의 실제 crop 메타데이터에 전달되는지도 paired runner 테스트로 검사한다.
