# 4글자 공동 역확산: 추출 획 기반 guidance

[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/topo_guided_diffusion.ipynb).

고정 조건은 기존 SD1.5 epsilon U-Net, DDIM, 4개 기준의 576x576 공동 이미지,
strength=0.65, seed=100, nominal steps=30이다. Diffusers가 strength에 따라 선택하는
실제 timestep을 기록한다(기본 설정에서 19개). 같은 VAE 샘플링·잡음 추가 결과를 한 번
만들어 baseline/structure_guided/guided에 복사하며, 초기 잠재값 SHA256을 저장한다. 별도로 seed만
맞춰 두 번 인코딩하는 방식이 아니다. 모든 모델은 동결하고 optimizer를 만들지 않는다.

## 주 신호를 원래 추출 획 경로로 연결

`stroke_model.diffusion_guidance.TopoGuidance`:

- 예상 완성 이미지의 4개 고정 crop을 torch bilinear로 checkpoint 해상도로 내린다.
- 실제 동결 Topo에 넣어 **soft masks**를 얻고, 기존 `StrokeStructureLoss`를 직접 호출한다.
- 각 획의 `l_k=missing+log1p(split/.2)+log1p(junction/.01)`를 유지한다.
  구조 설정의 가중치와 width 설정도 기존대로 적용한다.
- 기존 구조 전용 집중 보정은 `w_k=min(3,1+2*(1-stroke_score_k/100))`를 detach해 정한다.
  `L_stroke=sum(w_k*l_k)/sum(w_k)`를 글자별로 구한 뒤 네 글자를 평균한다.
  정상 획도 양의 비중을 유지하며 한 획의 비중이 무제한 커지지 않는다.
- 각 단계의 원래 예상 완성 이미지에서 정한 가중치를 모든 backtracking 후보에
  고정한다. 후보마다 가중치를 다시 계산해 서로 다른 목적함수를 비교하지 않는다.
  다음 timestep에서는 새 예상 완성 이미지로 우선순위를 다시 정한다.
- 구조 전용 비교 경로 `structure_guided`는 아래 윤곽 보조 손실을 끈다.
  보정 예산·일정·전달 방식·채택 기준·초기 잠재값은 guided와 같다.
  균등 구조 보정도 필요하면 `compare_uniform=True`로 한 경로 더 생성한다.
- 주 손실은 원본 직접 측정 점수로 대체하지 않는다. 소량의 원본 주변 잉크 보조항
  `.05*sum(foreground_ink*(1-reference_support))/reference_ink_mass`와 범위 밖 RGB
  손실 `.01*mean(relu(-rgb)^2+relu(rgb-1)^2)`를 별도로 더한다.
- 기준 자료의 준비만 CPU/SciPy이며 live image -> soft masks -> 구조 손실 경로에는
  PIL/NumPy/argmax/연결 성분 분리를 넣지 않는다. 등록/기준은 생성 중 움직이지 않는다.
- 기존 구조 측정 기본값을 사용한다. grayscale reconstruction이 수렴하지 않으면
  잘린 측정으로 진행하지 않고 중단한다. 사용자의 실제 가중치에서 비용/메모리는 실행 검증이 필요하다.

최종 **주 점수**도 추출 soft masks의 missing/split/junction을 `StructuralIndicator`로
집계한다. `extra_weight=0`을 명시한 순수 획 구조 점수이며, 추가 잉크를 측정한 척 0으로
넣지 않는다. extra 보조항은 독립 열로 기록한다. 기존 원본 source 점수는 5번 셀의
선택적 `RUN_INDEPENDENT_SOURCE_AUDIT`에서만 별도 검증한다.

## 미분 가능한 획 윤곽·국소 잉크량 보조 손실

`stroke_model.stroke_appearance.StrokeAppearanceLoss`를 실제 예측 soft 획에 적용한다.
원본 전체 이미지의 점수로 추출 획 경로를 대신하지 않으며, 이진화·argmax·CPU 윤곽
추출을 live 경로에 넣지 않는다. 현재 기준은 **정상 reference sheet를 같은 crop/resize와
동결 Topo에 넣어 얻은 실제 soft masks**다. 생성 전에 한 번 측정하고 고정하며, 모든
경로가 같은 목표를 공유한다. reference의 appearance loss가 0인지 기록한다.

이전 방식은 기준 stroke mask의 중첩을 정규화해 목표로 썼지만, 실제 학습된 Topo는
정상 입력에서도 그와 다른 soft 배정을 했다. 제공된 ZIP에서는 정상 reference의
appearance loss가 .39~.54였고, 대부분 획의 집중 비중이 포화했다. 스칼라 오차를
빼는 방법 대신 목표 soft masks 자체를 정상 추출 응답으로 맞춘다. 이를 통해 추출기의
평상시 배정 차이에 보정 예산을 소모하는 일을 줄인다. 구조 손실의 기준 획과 점수
공식은 그대로다. `appearance_reference_masks`를 평가 NPZ에 저장한다.
`appearance_target='partition_reference'`로 이전 목표 방식을 재현할 수 있으며,
이 필드가 없는 과거 보고서도 이전 방식으로 평가한다.

- `local_mass`: 3x3/5x5 평균 잉크량이 기준과 다른 정도. 너무 굵거나 가늘어지는
  경우 양쪽에 반응하며, 물리적 두께 추정값은 아니다.
- `contour`: `mask-avgpool3(mask)`의 국소 경계 응답을 같은 기준 응답과 비교한다.
  기준 자체의 셰리프·모서리는 비교 대상에 들어 있어 일치할 때 손실이 0이다.
- 차이의 절댓값에서 작은 contrast 허용량을 빼고 양수 부분을 제곱한다.
  기본 local_mass .03, contour .05의 허용량은 채널 수 K로 나눠,
  많은 채널에 오류 잉크를 나누어 허용량 아래로 숨기는 현상을 줄인다.
- 각 획의 오차 합을 그 획의 기준 잉크량으로 나눈다. 이미지 전체 면적으로 나눠
  작은 획 결함의 신호가 희석되지 않게 한다. 해상도는 Topo checkpoint 해상도다.

`a_k=local_mass_k+.5*contour_k`, 새 per-stroke 손실은 `l_k+.5*a_k`다.
`.5`인 `appearance_weight`는 실험 시작값이며 학습/보정된 계수가 아니다.
집중 비중은 구조 결함과 `1-exp(-a_k/.1)` 중 큰 쪽으로 정해, 구조 점수가 100인
획에서도 윤곽 결함에 집중할 수 있게 한다. 비중은 detach하며 후보 간 고정한다.
기존 구조 점수는 변경하지 않는다. 새 목적 손실은 `guidance_stroke_loss`, 구조 부분은
`stroke_loss`, 보조 오차는 `appearance_loss`/`contour`/`local_mass`에 따로 기록한다.
`appearance_loss`는 비교를 위해 가중치와 무관한 획 평균으로 표시한다.

기본 세 경로는 `baseline`(보정 없음), `structure_guided`(기존 획별 구조 보정),
`guided`(새 보조 손실 추가)다. `appearance_weight=0`이면 새 보조 손실과 그 집중
우선순위를 끄며 중복 structure_guided 경로를 생성하지 않는다. 오래된 generation
report의 평가도 보조 손실 가중치 0으로 처리한다.

이 보조 손실은 기준과의 형태 차이를 벌점으로 삼으므로 생성 글씨체의 다양성을
줄이거나 기준 폰트를 복사하는 방향으로 작용할 수 있다. 위치 오차에도 반응한다.
reference-relative appearance 개선과 보편적 미관/글자 품질 개선을 구분하고,
구조 점수·원본 그림·보조 손실을 함께 비교해야 한다.

## 잡음 예측 모델과 실제 상태 보정

`diffusion_model.guided_ddim.denoise`는 Diffusers DDIM step의 `pred_original_sample`을 사용한다.
처음 구현은 예상 완성 잠재값 z0를 leaf로 만들고 VAE -> Topo -> 획 손실을 미분하는
**backward clean-latent guidance**다. U-Net을 통과해 zt까지 미분하지 않는다.
실제 생성 모델은 그대로 epsilon을 예측한다. 아래 두 전달 방식을 구분한다.

기존 `state_update='epsilon'`은 z0 기울기를 epsilon 보정으로 변환한다.

`g0 = grad(L, predicted_z0)`

`epsilon' = epsilon + positive_scale * g0`

같은 zt에서 이 보정은 `z0' = z0 - sqrt((1-alpha_bar)/alpha_bar)*positive_scale*g0`가 되어
예상 완성 잠재값의 손실 하강 방향이다. clean 보정 예산은
`desired=.01*ramp*RMS(predicted_z0)*g0/RMS(g0)`를 정하고,
`epsilon'=epsilon+sqrt(alpha_bar/(1-alpha_bar))*desired`로 변환할 수 있다.
그러나 이를 보통 DDIM step에 넣으면 x0를 바꾸는 항과 바뀐 epsilon 방향 항이
부분 상쇄된다. 실제 ZIP에서는 clean 보정 대비 다음 상태 변화의 RMS 비율이 평균
.188, 가장 이른 채택 단계에서 .072였다. 이는 단계 사이 변화량의 비율이며,
최종 이미지 개선 비율이나 기울기 전체의 보존율을 뜻하지 않는다.

현재 notebook의 `state_update='clean_residual'`은 **잡음 방향을 고정한 guided mean
update**다. 보정 후보 `z0'=z0-desired`를 평가한 뒤, 실제 다음 상태는
`z_prev'=z_prev-sqrt(alpha_bar_prev)*desired`로 갱신한다.
즉 DDIM의 기존 잔차 방향에 새 clean estimate를 합성해 보정량 상쇄를 줄인다.
이는 수정한 epsilon만 넣은 표준 DDIM step과 다른 조건부 평균 이동이다. U-Net과
가중치 학습은 바꾸지 않으며 eta=0을 유지한다. 로그의 `correction_rms`는 이 모드에서
clean 후보에 대응하는 **등가 epsilon 변화량**이고, 실제 적용량은 `state_delta_rms`다.
`state_to_clean_transfer_ratio`도 단계별로 저장한다. 마지막 단계에는 scheduler의
`final_alpha_cumprod`를 써서 steps_offset/set_alpha_to_one 설정을 보존한다.

진행률 .30 이후부터 마지막 단계까지 매번 검사하며, 초기에는 보정량을 점차 늘린다.
기존 `noise_ratio=.05` 모드도 `clean_step_ratio=None`으로 재현할 수 있다.
이전보다 검사 횟수가 늘어 실행 시간이 길어질 수 있다.

각 모드의 수정된 clean estimate를 다시 평가한다. 같은 시점의
전체 손실이 낮아지고 4글자 평균 추출 획 점수가 낮아지지 않으며 실제 다음 상태가 변할 때만 채택한다.
추가로 **보정 전 90점 이상인 각 획**이 1점 넘게 나빠지는 후보는 거부한다.
`healthy_stroke_score`, `max_healthy_stroke_drop`으로 임계값을 명시한다.
이는 같은 시점 후보 사이의 보호 조건이며, 최종 결과에서 누적 악화가 없다는 보장은 아니다.
그렇지 않으면 보정량을 절반씩 줄인다. notebook 기본 backtracks=5로 최대 6번
검사하고, 모두 실패하면 기본 DDIM step을 쓴다. 초기 예산은 기존과 같은 1%이며,
추가 backtracking은 마지막 단계처럼 작은 보정만 통과하는 경우를 검사하기 위한 것이다.
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
4. reference/baseline/structure_guided/guided 전체 이미지, 단계별 예상 완성 이미지, trace JSONL 저장.
5. 획별 점수·가중치·결함 지도·RGB 손실 기울기와 세 경로 점수 비교, 진단 ZIP 출력.

`stroke_scores.csv`는 method/slot/stroke별 점수, 가중치, 비중, 개별 손실과 결함값을 담는다.
`stroke_comparison.csv`는 동일 기준 획별 최종 점수와 guided-minus-baseline/structure를 담는다.
선택한 경우 uniform 비교도 포함한다.
`appearance_comparison.csv`는 동일 기준 획별 보조 오차를 비교하며
`baseline_minus_guided`/`structure_minus_guided`가 양수이면 기준 대비 오차가 줄었다.
원래 `topo_scores.csv`의 글자 점수 공식은 바꾸지 않아 비교가 가능하다.
`stroke_guidance_trace.csv`는 실제 guided 실행의 단계별 고정 가중치와 채택 전후 획 점수다.
PNG/NPZ에는 누락·분리·정션·윤곽·국소 잉크량 오차 지도와 각 획의 가중 손실 기여분을 RGB 입력에 미분한
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
단순히 1채널값만 달라진 배경까지 센 비율과 구분하도록 8/255 넘게 달라진 픽셀 비율도 출력한다.
채택 횟수만으로 개선을 판단하지 않는다. 획 손실 변화가 작고 보조항만 줄었다면
현재 추출 획 목적이 눈에 보이는 결함을 충분히 반영하는지 먼저 확인해야 한다.

`generation_report.json`에는 timestep, initial latent hash, checkpoint hash/epoch,
model revision, sampler/손실 설정, 채택 단계 수가 포함된다. 실행 실패 시 상태와 오류를
기록하며, 채택이 0이어도 baseline/guided 이미지를 보존한다. 0회 채택을 개선이라고
보고하지 않는다. 모델/데이터/가중치는 GitHub에 업로드하지 않는다.

## 검증과 한계

`python -m unittest discover -s tests -p test_topo_guided_ddim.py -v`

`python -m unittest discover -s tests -p test_stroke_appearance.py -v`

실제 Diffusers DDIM의 0-guidance 동일성, 보정 부호/손실 하강, 주 점수가 낮아지는 보정
거부, 동결 가중치, 네 칸 입력 gradient, 동일 원본에서 Topo 배정만 바꿔도 손실이
달라지는지 검사한다. 작은 실제 Diffusers U-Net/VAE로도 gradient 경로를 검사한다.
문제 획의 비중 증가/상한/가중치 detach, 후보 가중치 고정, 평균 점수가 올라도 정상
획이 악화되면 거부하는지, 동일 초기값의 세 경로와 획별 CSV/지도/gradient도 검사한다.
추가로 돌출을 지우고 패임을 채우는 기울기 부호, 모양 오차 감소, 일치하는 셰리프의
0 손실, 겹치는 기준 획의 attainable target, 구조 점수가 높은 경우에도 새 보조
신호가 실제 DDIM을 통과해 최종 상태에 남고 보조 오차가 줄어드는지 검사한다.
정상 추출 기준의 0 appearance loss/고정 target, 새 평균 이동에서 기대한
sqrt(alpha_prev)만큼 실제 상태가 바뀌는지, offset=1/final alpha 처리도 검사한다.
작은 테스트용 네트워크와 사용자의 학습된 checkpoint에 대한 품질 검증은 구분한다.

실제 pretrained SD1.5 + 사용자 Topo의 Colab GPU 실행/VRAM/최종 획 개선은 실행 후
확인해야 한다. 같은 시점의 loss/점수 조건을 만족해도 이후 denoising이나 최종 PNG
변환 뒤의 점수가 단조롭게 높아진다는 보장은 없다. 추출기의 배정만 유리해지는 현상,
기준 스타일 복사, 원본 글자 결함을 시각/독립 원본 검사로 함께 확인해야 한다.
최근 추가한 이진 윤곽/성분 진단은 이 guidance 손실에 포함하지 않는다.

### 보정이 약할 때: 손실의 민감도부터 구분

`python tests/diagnose_guidance_sensitivity.py`는 실제 기본 구조 손실에 이상적인
단일 획 마스크를 넣어, 원형/두께 3배/작은 돌출/작은 패임/끊김을 비교한다.
이 고정 예시에서는 원형뿐 아니라 두께 3배·작은 돌출·작은 패임도 주 점수 100,
손실 0, 마스크 기울기 0이다. 반면 끊김에는 양의 손실과 기울기가 생긴다.
출력은 `outputs/guidance_sensitivity/sensitivity.json`과 PNG다.
이는 추출을 완벽하게 해도 구조 손실 자체에 검출하지 못하는 변형이 있음을 보여주며,
사용자의 실제 생성 결과에서 같은 원인이 발생했다는 판정은 아니다.
주변 잉크 보조항을 뺀 주 신호만 분리해 검사한다. 전체 guidance의 기울기가 항상
0이라는 의미도 아니다. 검출되지 않는 결함에는 획 가중치를 높여도 새 수정 방향이
생기지 않는다. 실제 로그의 획 손실·보조항 감소와 채택/거부 사유를 함께 보고,
필요하면 미분 가능한 획 윤곽/두께 보조 손실을 별도 실험해야 한다.
현재 진단은 새 보조 손실의 기울기와 25회 합성 마스크 직접 보정 전후도 함께 저장한다.
이는 실제 확산 결과가 아닌 soft-mask 수준의 방향 검증이며, 기존 주 점수의 공식은 그대로다.

ZIP만으로 이전 결과를 재분석하려면 다음을 실행한다:
`python -m diffusion_model.inspect_guidance_archive guidance_debug.zip --output outputs/guidance_archive_audit`.
저장된 정상 reference soft masks를 목표로 다시 계산한 값과 원래 값을 CSV/JSON에
나란히 저장한다. 모델 가중치나 생성 재실행이 필요 없다. 이는 이전 이미지의
재측정이며 새 전달 방식으로 생성한 결과가 아니다.

참고: [Universal Guidance](https://arxiv.org/abs/2302.07121),
[Diffusers 0.35.1 img2img](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py),
[DDIM](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py).
