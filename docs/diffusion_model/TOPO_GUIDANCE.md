# 4글자 공동 역확산: 추출 획 기반 guidance

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/topo_guided_diffusion.ipynb).

고정 조건은 기존 SD1.5 epsilon U-Net, DDIM, 4개 기준의 576x576 공동 이미지,
strength=0.65, seed=100, nominal steps=30이다. Diffusers가 strength에 따라 선택하는
실제 timestep을 기록한다(기본 설정에서 19개). 같은 VAE 샘플링·잡음 추가 결과를 한 번
만들어 baseline/uniform_guided/guided에 복사하며, 초기 잠재값 SHA256을 저장한다. 별도로 seed만
맞춰 두 번 인코딩하는 방식이 아니다. 모든 모델은 동결하고 optimizer를 만들지 않는다.

## 주 신호를 원래 추출 획 경로로 연결

`stroke_model.diffusion_guidance.TopoGuidance`:

- 예상 완성 이미지의 4개 고정 crop을 torch bilinear로 checkpoint 해상도로 내린다.
- 실제 동결 Topo에 넣어 **soft masks**를 얻고, 기존 `StrokeStructureLoss`를 직접 호출한다.
- 각 획의 `l_k=missing+log1p(split/.2)+log1p(junction/.01)`를 유지한다.
  구조 설정의 가중치와 width 설정도 기존대로 적용한다.
- 집중 보정은 `w_k=min(3,1+2*(1-stroke_score_k/100))`를 detach해 정한다.
  `L_stroke=sum(w_k*l_k)/sum(w_k)`를 글자별로 구한 뒤 네 글자를 평균한다.
  정상 획도 양의 비중을 유지하며 한 획의 비중이 무제한 커지지 않는다.
- 각 단계의 원래 예상 완성 이미지에서 정한 가중치를 모든 backtracking 후보에
  고정한다. 후보마다 가중치를 다시 계산해 서로 다른 목적함수를 비교하지 않는다.
  다음 timestep에서는 새 예상 완성 이미지로 우선순위를 다시 정한다.
- 균등 비교 경로는 `focus_strength=0`으로 모든 가중치를 1로 두며, 보정 예산과
  일정, 채택 기준은 집중 보정과 같다. `compare_uniform=False`로 비교 경로를 생략할 수 있다.
- 주 손실은 원본 직접 측정 점수로 대체하지 않는다. 소량의 원본 주변 잉크 보조항
  `.05*sum(foreground_ink*(1-reference_support))/reference_ink_mass`와 범위 밖 RGB
  손실 `.01*mean(relu(-rgb)^2+relu(rgb-1)^2)`만 별도로 더한다.
- 기준 자료의 준비만 CPU/SciPy이며 live image -> soft masks -> 구조 손실 경로에는
  PIL/NumPy/argmax/연결 성분 분리를 넣지 않는다. 등록/기준은 생성 중 움직이지 않는다.
- 기존 구조 측정 기본값을 사용한다. grayscale reconstruction이 수렴하지 않으면
  잘린 측정으로 진행하지 않고 중단한다. 사용자의 실제 가중치에서 비용/메모리는 실행 검증이 필요하다.

최종 **주 점수**도 추출 soft masks의 missing/split/junction을 `StructuralIndicator`로
집계한다. `extra_weight=0`을 명시한 순수 획 구조 점수이며, 추가 잉크를 측정한 척 0으로
넣지 않는다. extra 보조항은 독립 열로 기록한다. 기존 원본 source 점수는 5번 셀의
선택적 `RUN_INDEPENDENT_SOURCE_AUDIT`에서만 별도 검증한다.

## 잡음 예측 보정

`diffusion_model.guided_ddim.denoise`는 Diffusers DDIM step의 `pred_original_sample`을 사용한다.
처음 구현은 예상 완성 잠재값 z0를 leaf로 만들고 VAE -> Topo -> 획 손실을 미분하는
**backward clean-latent guidance**다. U-Net을 통과해 zt까지 미분하지 않는다.
실제 생성 모델은 그대로 epsilon을 예측하며, z0 기울기를 epsilon 보정으로 변환한다.

`g0 = grad(L, predicted_z0)`

`epsilon' = epsilon + positive_scale * g0`

같은 zt에서 이 보정은 `z0' = z0 - sqrt((1-alpha_bar)/alpha_bar)*positive_scale*g0`가 되어
예상 완성 잠재값의 손실 하강 방향이다. 현재 notebook 기본값은
`desired=.01*ramp*RMS(predicted_z0)*g0/RMS(g0)`를 정하고,
`epsilon'=epsilon+sqrt(alpha_bar/(1-alpha_bar))*desired`로 변환한다.
따라서 후반 잡음이 작아져도 예상 완성 잠재값의 수정 예산이 함께 사라지지 않는다.
진행률 .30 이후부터 마지막 단계까지 매번 검사하며, 초기에는 보정량을 점차 늘린다.
기존 `noise_ratio=.05` 모드도 `clean_step_ratio=None`으로 재현할 수 있다.
이전보다 검사 횟수가 늘어 실행 시간이 길어질 수 있다.

수정한 epsilon으로 DDIM의 실제 clean estimate를 다시 구하고 평가한다. 같은 시점의
전체 손실이 낮아지고 4글자 평균 추출 획 점수가 낮아지지 않으며 실제 다음 상태가 변할 때만 채택한다.
추가로 **보정 전 90점 이상인 각 획**이 1점 넘게 나빠지는 후보는 거부한다.
`healthy_stroke_score`, `max_healthy_stroke_drop`으로 임계값을 명시한다.
이는 같은 시점 후보 사이의 보호 조건이며, 최종 결과에서 누적 악화가 없다는 보장은 아니다.
그렇지 않으면 보정량을 절반씩 줄여 최대 3번 검사하고, 모두 실패하면 기본 DDIM step을 쓴다.
모든 채택/거부·기울기 크기·보정 크기·글자별 점수·손실을 기록한다.
각 단계에서 그래프를 끊어 전체 trajectory를 역전파하지 않는다.

DDIM 상태와 보정은 두 경로 모두 fp32로 누적하고 U-Net 입력만 모델 정밀도로 변환한다.
VAE 인코딩은 기존 fp16 설정으로 한 번 수행하고, 이후 두 경로 모두 VAE decode를 fp32로
진행해 입력 gradient 안정성을 높인다. 따라서 과거 fp16 PNG와 비트 단위 일치가 아니라
이 실험의 baseline/guided 쌍을 비교한다. 이미지 범위 clamp 바깥에서는 구조 gradient가
0이 될 수 있어 별도 범위 손실을 둔다. epsilon prediction, clip_sample=False,
thresholding=False만 지원한다. safety checker도 두 경로에 동일하게 적용하고 결과를 보존한다.

## 코랩과 결과

1. GitHub 코드·라이브러리·Drive 연결.
2. 기존 checkpoint와 기준 전용 준비(전체 학습 NPZ 변환 생략).
3. 학습 범위의 기준을 **정확히 4개** 선택.
4. reference/baseline/uniform_guided/guided 전체 이미지, 단계별 예상 완성 이미지, trace JSONL 저장.
5. 획별 점수·가중치·결함 지도·RGB 손실 기울기와 세 경로 점수 비교, 진단 ZIP 출력.

`stroke_scores.csv`는 method/slot/stroke별 점수, 가중치, 비중, 개별 손실과 결함값을 담는다.
`stroke_comparison.csv`는 동일 기준 획별 최종 점수와 guided-minus-baseline/uniform을 담는다.
원래 `topo_scores.csv`의 글자 점수 공식은 바꾸지 않아 비교가 가능하다.
`stroke_guidance_trace.csv`는 실제 guided 실행의 단계별 고정 가중치와 채택 전후 획 점수다.
PNG/NPZ에는 누락·분리·정션 위치 지도와 각 획의 가중 손실 기여분을 RGB 입력에 미분한
기울기를 저장한다. 기울기는 빨강=RGB를 낮춰 어둡게, 파랑=RGB를 높여 밝게 하는 하강
방향이다. 그림마다 색 범위를 확대하므로 색 강도를 획 사이에서 직접 비교하지 않는다.
이는 입력 이미지의 기울기 진단으로, VAE를 통과한 실제 잠재값 보정 이미지와는 다르다.
최종 이미지에서 다시 산출한 가중치와 생성 당시 단계별 가중치를 구분한다.
진단은 획별 역전파가 추가되므로 5번 셀도 기존보다 느릴 수 있다.

4/5번 셀은 `diagnose_pair(OUTPUT)`도 실행한다. 기존 결과를 다시 생성하지 않고도
PNG의 평균/최대 차이와 바뀐 픽셀 비율, 채택 단계의 획 손실/주변 잉크 보조항 감소를
나눠 확인한다. 새 실행에서는 실제 다음 상태 변화 RMS와 최종 잠재값 차이도 기록한다.
`paired_pixel_difference.png`는 차이를 확대 표시하며 개선 여부를 뜻하지 않는다.
진단 수치는 `guidance_diagnostics.json`, 글자별 차이는 `pixel_differences.csv`에 저장한다.
채택 횟수만으로 개선을 판단하지 않는다. 획 손실 변화가 작고 보조항만 줄었다면
현재 추출 획 목적이 눈에 보이는 결함을 충분히 반영하는지 먼저 확인해야 한다.

`generation_report.json`에는 timestep, initial latent hash, checkpoint hash/epoch,
model revision, sampler/손실 설정, 채택 단계 수가 포함된다. 실행 실패 시 상태와 오류를
기록하며, 채택이 0이어도 baseline/guided 이미지를 보존한다. 0회 채택을 개선이라고
보고하지 않는다. 모델/데이터/가중치는 GitHub에 업로드하지 않는다.

## 검증과 한계

`python -m unittest discover -s tests -p test_topo_guided_ddim.py -v`

실제 Diffusers DDIM의 0-guidance 동일성, 보정 부호/손실 하강, 주 점수가 낮아지는 보정
거부, 동결 가중치, 네 칸 입력 gradient, 동일 원본에서 Topo 배정만 바꿔도 손실이
달라지는지 검사한다. 작은 실제 Diffusers U-Net/VAE로도 gradient 경로를 검사한다.
문제 획의 비중 증가/상한/가중치 detach, 후보 가중치 고정, 평균 점수가 올라도 정상
획이 악화되면 거부하는지, 동일 초기값의 세 경로와 획별 CSV/지도/gradient도 검사한다.
작은 테스트용 네트워크와 사용자의 학습된 checkpoint에 대한 품질 검증은 구분한다.

실제 pretrained SD1.5 + 사용자 Topo의 Colab GPU 실행/VRAM/최종 획 개선은 실행 후
확인해야 한다. 같은 시점의 loss/점수 조건을 만족해도 이후 denoising이나 최종 PNG
변환 뒤의 점수가 단조롭게 높아진다는 보장은 없다. 추출기의 배정만 유리해지는 현상,
기준 스타일 복사, 원본 글자 결함을 시각/독립 원본 검사로 함께 확인해야 한다.
최근 추가한 이진 윤곽/성분 진단은 이 guidance 손실에 포함하지 않는다.

참고: [Universal Guidance](https://arxiv.org/abs/2302.07121),
[Diffusers 0.35.1 img2img](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py),
[DDIM](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py).
