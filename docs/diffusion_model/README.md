# A100 한 장으로 실행하는 한글 글자 확산 학습

2026-10-31. **구현된 학습 프로세스**이며, 실제 A100 학습 완료나 한글 품질 개선을 주장하지 않는다.

## 이번 구현의 범위

- 128×128 한 글자 흑백 이미지용 **조건부 pixel DDPM을 처음부터 학습**한다.
- 조건은 기준 글자 이미지다. noisy image와 기준 글자를 채널 방향으로 합쳐 U-Net에 넣는다.
- AnyText/Stable Diffusion 미세조정이나 포스터·자연 장면 전체 생성 모델이 아니다. VAE/CLIP/언어모델도 사용하지 않는다.
- 현재 획 모델과 같은 해상도에서 작은 실험을 먼저 할 수 있도록 정한 시작 구현이다.
- 먼저 baseline을 학습하고 같은 checkpoint에서 OCR-only / stroke-only / combined 감독을 비교한다.
- **역확산 도중 guidance는 아직 구현하지 않았다.** 이번 요청의 학습 경로를 구현했으며, 보조 손실은 학습 중 깨끗한 이미지 추정값에 적용한다.
- 획 보조 손실은 정상 target과 생성 추정값의 추출 마스크를 비교하는 **재구성 감독**이다. 검증된 추가·누락·단절·오연결 품질 점수나 오류 헤드는 아직 없다.

## Colab에서 실행

[Colab에서 바로 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/train_diffusion_a100.ipynb)

기존 프로젝트와 동일하게 첫 셀이 GitHub main을 clone하고 필요한 라이브러리를 설치한다. 실행한 커밋 해시를 출력한다. ZIP 업로드는 필요 없다. 같은 세션에서 첫 셀을 다시 실행해도 이전 checkout은 덮어쓰지 않고 최신 코드를 별도 폴더로 받아 사용한다.

순서:

1. 위 Colab 링크를 열고 런타임을 A100 GPU로 설정한다.
2. 첫 코드 셀에서 GitHub 코드를 받는다. Colab의 CUDA torch는 교체하지 않고 나머지 라이브러리를 설치한다.
3. Drive를 연결하고 PROJECT 경로와 데이터 모드를 확인한다.
4. 데이터 미리보기에서 기준·정상 이미지·글자가 맞는지 확인한다.
5. 처음에는 `STAGE='baseline'`, `MAX_STEPS=100`으로 실행 확인한다.
6. 같은 run을 `MAX_STEPS=10000`으로 늘려 재개한다. 이 수치는 최초 실험 예산일 뿐 수렴이나 소요 시간을 보장하지 않는다.
7. 체크포인트와 검증 미리보기를 Drive에서 확인한다.
8. baseline을 고정한 뒤 `STAGE='ocr'`, `'stroke'`, `'ocr_stroke'`를 각각 새로운 run으로 실행한다.

기본값은 batch=8, accumulation=2, 유효 batch 약16, BF16, gradient checkpointing이다. 짧은 마지막 batch는 작을 수 있다. 실제 A100 VRAM을 출력한다. OOM이면 batch를 4 또는 2로 낮추고 새 run을 시작한다. exact resume에서 batch/accumulation을 바꾸지 않는다.

## 데이터: 세 입력 경로

### 1. 기존 CSV 자동 변환: 노트북 기본

기본 CSV:

`/content/drive/Shareddrives/2026 자율연구/HSQM/dataset/train_dataset/paths.csv`

기존 `stroke_model.reference_cache.prepare_cached`를 재사용해 V2 NPZ로 변환한다. 기준/대상 획 ID 대응이 정확해야 한다. 변환이 이미 완료된 경우 재사용한다. 기존 변환 캐시의 원본 이미지가 수정됐다면 그 도구의 rebuild 절차를 사용한다.

원본 CSV에 writer_id/group_id가 없으면 기존 변환기는 이미지 단위로 나눈다. 이 경우 **필자·폰트 일반화 검증이 아니다**. 제대로 된 group_id를 넣거나 그룹 분할된 manifest를 직접 제공한다.

### 2. 이미 준비한 V2 NPZ manifest

`DATA_MODE='manifest'`, `MANIFEST_PATH='.../manifest.json'` 설정.

각 row:

```json
{"path":"samples/0001.npz","split":"train","group_id":"font_or_writer_A","char_id":"가"}
```

NPZ 필드:

| 필드 | 의미 |
|---|---|
| image | 정상 target, float32 [1,H,W], ink=1/background=0 |
| reference | 추론에서도 제공할 기준 글자, 동일 convention |
| strokes | 기준의 K개 획 마스크 [K,H,W], 획 손실을 쓸 때 필요 |

`labels`는 diffusion 학습에서 사용하지 않는다. 원본 획 추출 모델의 지도학습 레이블과 혼동하지 않는다. `strokes`는 **reference의 획**이며 target의 정답 획으로 바꾸지 않는다.

이미지 쌍 manifest도 지원한다. 이 경우 `path` 대신 `image`, `reference` 경로를 넣고 파일은 검정 글씨·흰 배경으로 준비한다. 이 경로는 획 보조 손실을 지원하지 않는다.

### 3. 한글 폰트에서 정상 이미지 합성

`DATA_MODE='fonts'`, `FONT_DIR`와 `CHARACTERS` 설정. 서로 다른 한글 font family 최소 2개가 필요하다. 같은 폰트의 Bold/Regular는 같은 family로 처리한다.

```bash
python -m diffusion_model.prepare_fonts --font-dir /content/fonts --characters 가나다한글 --output /content/font_data
```

- 폰트 cmap을 확인해 없는 문자를 사각형(tofu)으로 렌더링하지 않는다.
- font family를 먼저 train/val로 나누고 변형을 생성한다.
- 기준은 training font의 정상 글자다. validation target 자체를 조건으로 제공하지 않는다.
- 평가 문자마다 train과 val font의 지원 여부를 확인한다.
- 이 데이터에는 개별 획 레이어가 없다. **stroke-weight는 0이어야 한다.**
- 준비된 데이터를 재사용할 때 폰트·문자 목록을 바꾸려면 새 출력 폴더를 사용한다.

데이터는 최초 단계에서 Drive → `/content`로 복사한다. 로컬 복사는 파일 해시로 확인하고, checkpoint에는 manifest와 실제 데이터 파일의 fingerprint를 저장한다. 재개 시 데이터가 달라지면 중단한다. 기준 폰트/필자/source 그룹이 train/val/test에 중복되면 중단한다. test row는 학습·체크포인트 선택에 사용하지 않는다.

## 모델과 학습 손실

U-Net: 입력2채널(noisy target, reference), 출력1채널(epsilon). 채널 폭64/128/128/256, 4단계, 낮은 해상도의 attention, 학습 timestep 1000, cosine beta schedule. 기본 optimizer AdamW(lr1e-4), warmup200 updates, EMA0.999.

기본 DDPM 손실:

`L_noise = mean((predicted_epsilon - true_noise)^2)`

OCR/획 보조 학습:

`L = L_noise + ocr_weight * L_CTC + stroke_weight * L_stroke_reconstruction`

보조 loss는 aux-start 이후, aux-every update마다, timestep이 aux-max-t보다 작은 샘플에서만 계산한다. 초기값은 각각1000/4/200. notebook의 추가 학습 단계에서는 aux-start=0이다. 선택된 샘플이 없는 update에서는 denoising만 학습한다. 따라서 보조 손실의 실제 적용 빈도는 aux-every 및 timestep 선택에 영향을 받는다. 이를 모든 t에서의 균등 목적 함수라고 해석하지 않는다.

깨끗한 이미지 추정은 epsilon 공식으로 계산한다. 보조 모델 입력은 [0,1]로 clamp한다. 범위 밖의 gradient가0이 되는 한계가 있어 저잡음 t에서만 적용하며, 추후 smooth bound/범위 손실과 비교할 수 있다.

### OCR

- EasyOCR1.7.2 한국어 인식기, 파라미터 동결, 디코딩 전 logits → 목표 문자 CTC.
- torch resize 및 normalize로 입력 gradient를 유지한다.
- `readtext()` confidence를 사용하지 않는다.
- 모든 학습·검증 문자가 checkpoint vocabulary에 있어야 한다. 없거나 text가 비어 있거나 두 글자 이상이면 중단한다. 문자표를 늘리고 기존 weights가 자동 대응한다고 가정하지 않는다.
- OCR wrapper의 CuDNN RNN eval backward 제약을 피하기 위해 해당 forward는 native RNN 경로로 실행한다.
- 공개 OCR 모델 가중치는 첫 OCR 실행 시 다운로드한다.

### 획/자모 보조 감독

- image_size가 맞고 architecture가 reference_v2인 **학습 완료 checkpoint**가 필요하다.
- target과 예측 x0에 같은 frozen 추출기를 적용한다. target 쪽은 gradient를 끊고 예측 쪽은 유지한다.
- soft mask Dice 및 면적 차이를 비교한다. 추가 획 검출기나 인간 품질 점수로 해석하지 않는다.
- 추출기가 잘못 나눴다면 이 손실도 잘못된 방향일 수 있다. 검수된 실제 샘플과 독립 평가를 함께 사용한다.
- checkpoint가 자모 단위로 학습됐다면 이 실험도 자모 단위 감독이다. 개별 획 단위라고 보고하지 않는다.

## CLI 사용

```bash
python -m diffusion_model.train --manifest /content/data/manifest.json --output /content/drive/MyDrive/glyph_baseline --steps 10000
python -m diffusion_model.train --manifest /content/data/manifest.json --output /content/drive/MyDrive/glyph_baseline --steps 12000 --resume /content/drive/MyDrive/glyph_baseline/last.pt
python -m diffusion_model.train --manifest /content/data/manifest.json --output /content/drive/MyDrive/glyph_ocr --steps 2000 --init-from /content/drive/MyDrive/glyph_baseline/last.pt --ocr-weight 0.01 --aux-start 0
python -m diffusion_model.train --manifest /content/data/manifest.json --output /content/drive/MyDrive/glyph_both --steps 2000 --init-from /content/drive/MyDrive/glyph_baseline/last.pt --ocr-weight 0.01 --stroke-weight 0.1 --stroke-checkpoint /content/drive/MyDrive/stroke/best.pt --aux-start 0
```

새 run에서 기존 출력 폴더를 덮어쓰지 않는다. `--resume`은 optimizer·RNG·데이터 순서까지 재개하고, `--init-from`은 같은 데이터/크기의 새 실험을 새 optimizer로 시작한다. `steps`는 전체 누적 optimizer update 수다.

가중치0.01/0.1은 시작값이며 최적값이 아니다. OCR-only와 combined 실험 모두 동일한 baseline checkpoint에서 시작한다. checkpoint와 데이터가 같고 같은 환경일 때 재개 순서를 복원하며, 서로 다른 CUDA/라이브러리에서 비트 단위 재현을 보장하지 않는다.

## 결과

- `last.pt`: optimizer/RNG/EMA 포함 재개용. 500 updates마다, 검증 개선 시, 완료 시 저장. step1에도 저장한다.
- `best_weights.pt`: validation의 고정 noise/timestep에 대한 noise MSE로 선택한 EMA weights.
- `preview_*.png`: 기준 / 생성 / 실제 target. 같은 validation glyph·seed로 비교.
- `preview_labels.json`: preview의 목표 글자.
- `metrics.jsonl`: denoising/OCR/획 loss, 실제 보조 적용 샘플 수, gradient norm 등.
- `run_config.json`: 데이터 fingerprint·코드 해시·설정.

중단된 optimizer update는 저장하지 않는다. Colab 세션이 끊기면 마지막 완성 checkpoint부터 재개한다. `best_weights`의 noise MSE가 낮다고 인간이 판단한 획 오류가 가장 적다는 뜻은 아니다.

```bash
python -m diffusion_model.generate --checkpoint /content/run/best_weights.pt --reference /content/reference_ga.png --output /content/generated --steps 50 --count 4 --seed 100
```

이미 만들어진 같은 seed PNG는 덮어쓰지 않는다. 이 inference는 기본 DDIM이며 OCR/획 guidance를 수행하지 않는다.

## 검증과 다음 단계

CPU 검증:

```bash
python -m unittest discover -s tests -p test_diffusion.py -v
```

데이터 그룹 누수, 획 padding, epsilon→x0 gradient, 고정 OCR 모양 stub의 입력 gradient, EMA 복원, 실제 작은 diffusers U-Net의 역전파·DDIM 생성을 검사한다. 별도 smoke script는 실제 tiny CPU 학습·체크포인트 재개·생성까지 확인한다. 공개 EasyOCR/획 checkpoint의 실제 품질 및 A100 VRAM 검증과는 다르다.

주 실험은 baseline / OCR-only / stroke-only / combined 네 군을 동일한 데이터·공통 초기 checkpoint에서 비교한다. 단순 전체 글자 재구성 손실도 기준선으로 추가해 획 추출의 추가 가치를 검증한다. 폰트·필자 분할과 별도 사람 평가를 사용한다. 기준 이미지를 복사하는 현상, 스타일 다양성 붕괴, OCR만 속이는 패턴도 확인한다.

2026-10-31 로컬 CPU 확인 결과:

- 단위 검사 8개 통과. 실제 작은 Diffusers U-Net 역전파·DDIM 생성 포함.
- 실제 tiny DDPM 1 update → checkpoint 재개 → 2 update와, 중단 없는 2 update의 model/EMA가 비트 단위로 일치.
- checkpoint 로드 후 CLI 이미지 생성, preview·로그 저장 확인.
- 실제 두 font family의 렌더링 → 그룹 분할 → NPZ 읽기 확인. 이 실행 확인에는 로컬 라틴 폰트를 사용했으며 한글 생성 품질 검증이 아니다.
- Colab notebook 코드 셀 구문 및 배포 ZIP의 파일 해시·가중치/데이터 미포함 확인.
- 실제 A100 CUDA/BF16 실행, 공개 EasyOCR 가중치 및 사용자 획 checkpoint를 연결한 학습은 아직 미실행. OCR/획 손실의 입력 gradient 단위 검사는 작은 대체 모델로 수행했다.

이 글자 crop 실험이 작동하면 AnyText 등의 장면 생성기 미세조정 또는 원래 계획의 역확산 guidance로 확장한다. 이번 checkpoint가 그 대형 모델에 직접 로드되지는 않는다.

## 참고

- DDPM: https://arxiv.org/abs/2006.11239
- Diffusers U-Net API: https://huggingface.co/docs/diffusers/api/models/unet2d
- DDPM scheduler: https://huggingface.co/docs/diffusers/api/schedulers/ddpm
- EasyOCR 공식 소스: https://github.com/JaidedAI/EasyOCR
- AnyText의 OCR 기반 이미지 감독: https://arxiv.org/abs/2311.03054
