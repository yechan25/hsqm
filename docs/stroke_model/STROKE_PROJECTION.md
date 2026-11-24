# 획 단위 제약 투영: 생성 제어 실험 분기

현재 상태는 구현 및 CPU 제약 검사 완료, 실제 확산 생성 품질 검증 대기다.
기존 4단계 보정 실험을 변경하지 않고 별도 모듈로 구현했다. 새 Colab 실행은 요구하지 않는다.

## 구조

올바른 목표 획 K개와 Bounding Box → 획별 중심선·두께 → core/allowed 지도 →
역확산의 clean prediction에 직접 픽셀 투영 → VAE 재인코딩 → 다음 역확산 단계.

- `stroke_model/stroke_projection.py`: 지도 생성, 투영, 획별 위반 검사.
- `diffusion_model/stroke_projected_diffusion.py`: 일반 DDIM과 투영 DDIM의 공통 초기 상태 비교.
- `repair_audit/stroke_projection_pilot.py`: 기존 ZIP만 사용하는 CPU 제약 시각화.

## 강제하는 것과 자유롭게 두는 것

- 각 획의 중심부에서 최소 잉크 .7을 요구한다.
- 각 획의 허용 반경은 반두께+경계 여유1.5(목표 마스크 좌표)다.
- 모든 획의 허용 영역 바깥이며 박스 안인 곳에서 잉크 최대 .03을 요구한다.
- 중심부와 바깥 사이 경계는 수정하지 않는다. 박스 밖도 픽셀 투영에서는 그대로 둔다.
- 다른 획의 허용 영역에 있는 픽셀을 특정 획의 배경 규칙으로 지우지 않는다.
- 두께는 획별로 독립 입력하며 획별 core 위반 수를 기록한다.

검은 잉크/흰 바탕의 한정된 실험이다. 일반 장면의 표면색·질감에 바로 적용할 수 없다.
목표 획은 올바르다고 가정한다. 오류 이미지의 추출 결과를 그 자체로 정답 구조로 사용하지 않는다.
CNN을 새로 학습하지 않고 우선 박스 크기 변환만 사용한다. 입력 스타일 정합을 해결한 모듈은 아니다.

## CPU 실행

저장소 루트에서 다음을 실행한다. PyTorch CPU, numpy, scipy, scikit-image, Pillow만 필요하다.

```powershell
python -m repair_audit.stroke_projection_pilot --archive "C:/Users/freeg/Downloads/repair_stage4_20261123_132010_265667.zip" --output "C:/Users/freeg/Desktop/자율연구/output/stroke_projection_cpu" --case group_000_page_001
```

저장된 추출 결과가 없는 사례는 기본적으로 보류한다. 얇은 기준 획을 실제 입력 두께로 오인해서
정상 글씨를 지우는 문제를 피하기 위함이다. 같은 입력을 사용한 이전 ZIP의 추출 결과가 있으면
`--observations-archive ".../repair_stage3_20261118_064305_733955.zip"`으로 제공할 수 있다.
픽셀이 동일한 입력인지 검사한다. `--allow-template-widths`는 얇은 기준 폭의 실패를 관찰할
명시적 스트레스 시험이며 실제 스타일 보존 검증이 아니다.

`comparison.png`, `per_stroke_constraints.png`, `constraints.npz`, `report.json`, 전체 `summary.json`을 저장한다.
픽셀 투영 전후의 missing/extra는 이 제약에 대한 위반 수이며 실제 글자 정오나 OCR 정확도가 아니다.
픽셀 투영이 원래 정상 형태를 잘못 바꾸어도 위반 수는 0이 될 수 있다.

## 나중에 GPU에서 연결할 API

`run_projected_pair(pipe, source, constraints, output)`에 SD1.5 img2img 파이프라인을 넣는다.
DDIM scheduler와 float32 VAE를 사용한다. 모델 평가 모드를 사용하며 학습하지 않는다.
현재 기본값은 `start_from_noise=False`인 원본 기반 img2img다.
`start_from_noise=True`를 명시하면 원본 내용을 사용하지 않고 전체 잡음에서 지정한 획 구조로 생성하는 실험이다.
이때 source는 캔버스 크기만 정하고 strength는 적용하지 않는다. 실제 steps 전체를 실행한다.
원본 배경·스타일을 유지하는 기능이 아니므로 그 보존 성능을 주장하지 않는다.
`start_from_noise=False`일 때만 기존 입력에 strength만큼 잡음을 넣는 img2img 비교가 된다.
역전파 최적화 없이 매 스텝 한 번 VAE 디코딩·재인코딩을 수행한다.
VAE 왕복 비용이 있으므로 실제 속도 개선을 아직 주장하지 않는다.

일반/투영 경로는 초기 latent·잡음·시간 단계를 공유한다. `report.json`에 투영 전,
픽셀 투영 직후, VAE 왕복 후 위반 수를 각각 기록한다. VAE 왕복에서는 제약이 다시 깨질 수 있다.
`stroke_projected.png`는 원시 확산 출력이며,
`stroke_projected_final_pixel_projection.png`는 `save_final_projection=True`를 명시한 경우에만 저장하는 마지막 픽셀 투영 결과다.
후자는 명시적 후처리 결과이며 원시 생성 결과로 보고하지 않는다.
8-bit PNG 저장에는 최대 1/255 수준의 잉크 양자화 오차를 허용한다.

## 선행 연구와 차별점 검증

- [Diff-Text](https://arxiv.org/abs/2312.12232): 스케치와 기존 Canny ControlNet, 위치 제어를 활용한 추가 학습 없는 생성.
- [FreeText](https://arxiv.org/html/2601.00535v1): 잡음 수준에 맞춘 글리프 latent의 주파수 처리 및 국소 주입.

이 실험은 획별 두께/허용 영역/위반 기록을 명시적으로 다루고 픽셀 잉크 제약에 투영한다.
그러나 현재 core와 allowed를 같은 합집합으로 합치면 RGB 투영 결과는 정확히 같다.
획 분해 자체의 성능 향상이나 새 방법의 독창성은 입증되지 않았다. 보고서에 union_equivalent를 기록한다.
향후 획별 수정 선택·서로 다른 제약·연결 관계를 이용하는 것이 전체 글리프 제어보다 유리한지
추가 비교가 필요하다. 단순 렌더링, 전체 글리프 주입, 획별 제약, 마지막 투영 유무를 구분해야 한다.

## CPU 검사

필수 잉크 추가/바깥 잉크 제거/교차점 보호/경계·박스 밖 불변/재투영 안정성,
합집합과의 동등성, 잘못된 조건 거절, 공통 초기 상태와 역전파 없음,
원시 출력과 마지막 투영 결과 분리 등을 검사한다. CPU 검사는 GPU 품질 검증을 대신하지 않는다.
