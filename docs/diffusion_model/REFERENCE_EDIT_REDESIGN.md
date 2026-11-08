# 원본 획 기반 국소 편집 재설계

상태: 설계안. 새 모델 학습 또는 실제 SD 생성 개선을 검증한 결과가 아니다.

## 이번 실패의 확인된 원인

guidance_debug (3).zip에서 direct_reference=True임을 확인했다. 활성 13단계 중 채택 1단계, 후보 거부는 new_local_ink 75회, loss_not_lower 1회다. 마지막 단계에서는 손실 0.164912→0.120337, 추출 점수 87.878→90.971인 후보도 국소 잉크 증가로 거부됐다. 가장 작은 후보도 국소 증가 0.003712가 허용량 0.002를 넘었다. 이 수치는 해당 후보가 시각적으로 올바르다는 뜻이 아니다.

## 기존 모델의 역할

기존 대화와 reference_model.py를 다시 확인했다. V2/Topo 경로는 Swin 특징 추출, FPN, 기준 정렬 flow, 공유 획 헤드, 획 간 softmax, masks=image*probabilities 구조다. 입력 잉크를 어느 획에 배정할지 학습한다. 모든 잉크를 분배하므로 잘못 생긴 왼쪽 돌출도 어떤 획에 귀속된다. 이것이 직접적인 삭제 예측은 아니다. infer_topology.py는 독립 마스크 구조도 지원하므로 실제 checkpoint architecture를 실행 보고서에 기록해야 한다.

현재 guidance는 추출 구조/윤곽/원본 오차를 가중합하고 한 번 미분한 전체 잠재값 방향을 축소하며 시도한다. 다른 위치의 손상 때문에 거부되면 목표 돌출도 수정하지 못한다. 큰 가중치나 보호 조건 해제만으로 해결하지 않는다.

## 새 구조: 진단과 편집 분리

1. 기존 추출기는 고정하여 획 식별과 정상 획 보존 평가에 사용한다.
2. 기준 획과 입력의 정렬은 편집 전에 확정한다. 현재 결함을 따라 기준이 변형되는 것을 막기 위해 편집 중 정렬과 관심 영역을 고정한다. 큰 위치 오차는 먼저 보고하고 단순 차이로 삭제하지 않는다.
3. 원본 해상도에서 지울 영역 E, 채울 영역 M, 보호 영역 P를 별도로 만든다. 가의 사례에서는 세로획 왼쪽 돌출은 E, 세로획과 정상 오른쪽 가로획은 P다. 획 마스크 전체를 지우는 방식은 사용하지 않는다. 다른 정상 획과의 교차 픽셀은 삭제 후보에서 제외한다.
4. 첫 버전은 정렬된 기준의 차이에서 E/M을 계산한다. 학습 없이 검증 가능한 대조군이다. 폰트 변형 대응용 학습 버전은 별도 편집 헤드가 입력 잉크·정렬 기준·기준 획·추출 확률을 받아 erase/fill/keep 지도를 예측한다. 기존 softmax에 extra 채널 하나만 붙여 해결했다고 간주하지 않는다.
5. 학습 버전은 정상 글자에 가로획의 반대편 연장, 접합부 돌기, 결손, 떨어진 점을 합성하고 실제 변경 영역으로 지도한다. 깨끗한 글자는 no-edit 정답이다. 글자와 폰트를 분리한 검증 묶음으로 정렬 오차·세리프 오탐을 점검한다. 현재 희소 중심선 주석만으로 dense 삭제 정답을 만들지 않는다.

## 역확산 적용

먼저 E의 잉크를 낮추는 손실, M의 결손을 줄이는 손실, P의 원래 모습을 유지하는 손실을 분리한다. 각 결함 영역의 면적으로 정규화하여 작은 돌출이 네 글자 전체 평균에 묻히지 않게 한다. 목표 이미지는 보호 영역을 기존 생성값으로 유지하고 E/M에서만 바뀌게 만든다.

예상 완성 잠재값을 고정된 편집 지도에 대해 여러 번 최적화한다. 한 번의 전체 기울기를 계속 반으로 줄이는 현재 방식과 달리, 매 반복에서 보호 영역 손상을 다시 계산해 수정 방향을 바꾼다. 잠재값 변화량도 제한한다. 수정한 clean latent는 기존 clean_residual 방식으로 다음 단계에 전달한다.

채택 조건은 대상 결함 감소와 보호 영역 허용 오차다. 전체 점수 상승을 필수 조건으로 삼지 않는다. 보호 영역 밖 새 점은 별도로 감시하고, 기존 전 영역 단일 최대값 거부 조건을 무조건 제거하지 않는다. 실패하면 거부 사유와 남은 결함을 명시한다. VAE의 공간적 결합 때문에 보존과 삭제를 동시에 달성하지 못할 수 있다.

## 실행 순서와 통과 기준

A. 이번 baseline과 기준만으로 E/M/P 지도를 먼저 저장한다. 왼쪽 돌출이 E이고 세로획이 P인지 눈으로 검수한다.
B. 확산 반복을 빼고 동일 VAE의 고정 잠재값에서 국소 편집이 가능한지 검사한다. 픽셀 직접 편집은 지도 진단용 대조군이며 모델 성능으로 보고하지 않는다.
C. B가 통과하면 역확산 중간/후반에 연결한다. 동일 초기 잠재값으로 baseline, 기존 guidance, 새 국소 편집을 비교한다.
D. 주 지표는 해당 돌출의 잔여 잉크/길이, 세로획 연결성, 정상 가로획 유지, 보호 영역 변화와 새 점 발생이다. 총점은 참고값이다. 여러 seed와 다른 글자에서도 확인한다.

이번 첨부 결과는 실패 사례로 보존한다. 기존 생성 노트북을 다시 실행하라고 하기 전에 A/B 결과를 제시한다.


## 구현된 실험 경로

local_reference_edit.py와 GuidanceConfig(local_edit=True)를 추가했다. 현재는 학습 헤드가 아닌 정렬 기준 차이 방식이다. 임계값 0.2, 기준 주변 1픽셀 여유로 erase/fill 지도를 만들고 첫 활성 단계 이후 고정한다. 각 단계 보호 목표는 해당 단계 보정 전 RGB로 갱신하여 U-Net 자체 변화를 오류로 세지 않는다. 24회 최적화 중 보호 영역 RMS 0.03, 5×5 국소 RMS 최대 0.12 이내이며 편집 오차가 가장 낮은 후보를 채택한다. 이동 예산은 해당 clean latent RMS의 8%, 반복 스텝은 1%다. 수치는 실험 초기값으로 최적화된 값이 아니다.

현재 erase/fill 각각의 전체 영역 면적으로 정규화한다. 결함 연결 성분별 정규화와 학습된 편집 헤드는 아직 구현하지 않았다. 자동 정렬도 추가하지 않았으므로 위치가 다른 폰트에는 주의가 필요하다. 범위 밖 decoder RGB에서도 미분이 사라지지 않도록 편집 손실에는 clamp를 적용하지 않는다.

첨부 guidance_debug (3).zip에 대한 지도는 outputs/local_edit_audit/ga_edit_maps.png에 저장했다. 왼쪽 돌출을 삭제 대상으로 포착하나 ㄱ 끝과 획 길이 차이도 감지한다. 지도 시각 검수와 합성/대체 네트워크 테스트는 실제 SD VAE 보정 성공과 다르다. 실제 GPU/사용자 체크포인트 실행은 검증하지 않았다.


## Round 4 refinement

Archive guidance_debug (4).zip: 7/7 active steps accepted. In the largest erase component of the first glyph (511 pixels), mean ink fell from 0.80324 to 0.21437 and pixels above 0.5 ink fell from 427 to 12. The final native excess-ink guard still failed (0.003706 versus 0.002); this is not a complete success. Later fixed-size optimizer candidates repeatedly exceeded the protected local peak bound.

The optimizer now backtracks up to six halvings from its last feasible state. Both edit error and the protected objective must decrease, within the unchanged protection/latent limits. Rejected states are no longer used as the next iterate. Logs include attempt size and backtrack index. This is a targeted experimental refinement; no GPU comparison against round 4 has been performed locally.


## Extended denoising preset

The guided notebook now sets STEPS=60 (previously 30). With strength=0.65, Diffusers uses 39 actual DDIM timesteps instead of 19. Keeping start_fraction=0.65 and every=1 yields 14 active edit opportunities instead of 7; each retains its 24-iteration maximum, backtracking, and unchanged protection limits. Seed remains 100, and baseline/guided share the same initial latent within the new run. Different timestep grids can change the start noise level, so old 30-step outputs are not an exact matched-start ablation. The notebook prints the actual step count; generation_report.json retains exact timesteps and the configuration. More denoising provides more opportunities and costs more time; improved removal is not yet GPU-validated.


## Residual removal preset

After the 60-step screenshot still showed a pale left spur, the notebook preset was increased to nominal STEPS=100: 65 actual DDIM steps, 23 late editing opportunities, and up to 48 local iterations per opportunity. This replaces 39 actual steps, 14 opportunities, and 24 local iterations. The strength, seed, map threshold, step size, latent budget, and protection bounds are unchanged. Inner optimization still stops early when it cannot find a feasible improving step; more iterations do not guarantee further removal. Maximum local iteration opportunities rise from 336 to 1104, so runtime can increase substantially. The screenshot alone cannot establish whether the residual is inside the fixed edit map or limited by protection/latent constraints; the next trace and maps are needed to distinguish those cases. No new GPU validation has been performed.


## Faint boundary cleanup
The notebook uses local_halo_radius=2: expand detected erase regions into nearby reference background only (dilated reference ink <0.05). Faint pixels below the original 0.2 detection threshold can then be optimized instead of protected. This is an inference from the screenshot, not a confirmed diagnosis from a new trace. The last denoising step permits up to 96 inner iterations (48 elsewhere), within the same per-step latent/protection bounds. The first fixed maps include the halo and are exported as before. Defaults remain backward compatible. More iterations may still stop early at the protection or latent constraint. GPU improvement remains unverified.


## Residual-sensitive erase objective and stop diagnostics

The notebook now sets local_erase_l1_weight=0.5 (legacy API default 0). L_edit = mean_E((RGB-1)^2) + 0.5 mean_E(abs(RGB-1)) + mean_M((RGB-reference)^2). The existing 20×protected mean/peak squared penalties and hard limits remain. The added absolute error retains an erase derivative as a pale residual approaches white; previously its squared error derivative shrank linearly. This does not prove that gradient magnitude was the cause of the supplied screenshot: normalized latent directions and VAE coupling still matter.

Trace summaries now record stop_reason (iteration_limit, zero_gradient, protection_limit, no_feasible_descent), residual mean/peak and latent budget fraction. A budget fraction near one is evidence of a trust-region bound, not automatic permission to increase it. The notebook displays the final summary. No pixel painting or thresholding of the exported generated RGB is applied. This is still a known-reference control experiment and its actual SD improvement needs GPU validation with the next debug archive.
