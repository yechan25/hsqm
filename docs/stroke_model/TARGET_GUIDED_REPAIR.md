# 2×2 생성물에서 관측 두께·정답 획 목표를 이용한 재역확산

이 경로는 이전의 삭제/복원 사이클과 별도로, 사용자가 정한 다음 순서를 연결한다.

1. 기존 multiglyph_pilot/generate_cases로 2×2 한 장을 생성한다.
2. 기존 ReferenceStrokeModel로 각 칸의 획 마스크를 추출하고 두께를 측정한다.
3. 정답 획의 스켈레톤에 관측 두께를 입혀 목표 획 이미지를 구성한다.
4. 글자마다 하나의 제한된 affine 변환을 적용해 생성 영역에 정합한다.
5. 획별 평가로 수정 범위·강도를 정하고 원래 생성물에서 재역확산한다.

현재 구현·CPU 확인 완료, 실제 SD1.5 GPU 품질 검증 전이다. 추출 모델의 학습 가중치는 바꾸지 않는다.

## 변경한 핵심

`stroke_model/target_stroke_plan.py`의 `build_target_plan`이 목표 이미지와 편집 계획을 만든다.

- 추출 마스크의 중심선에서 분기점 등을 제외한 안정적인 표본의 거리 변환 중앙값으로 반두께를 측정한다.
- 작은/누락 획처럼 두께를 측정하기 어려운 경우 같은 글자의 측정 가능한 획 중앙값을 빌린다.
  한 글자에서 어느 획도 측정할 수 없으면 기준의 얇은 두께를 쓰지 않고 보류한다.
- 목표 스켈레톤은 정답 획에서 구한다. 오류 입력의 스켈레톤을 목표로 따라 그리지 않는다.
- 열린 획과 폐곡선 모두 동일하게 관측 두께로 렌더링한다. 기존 폐곡선 thin fallback을 사용하지 않는다.
- 두께를 입힌 후 기존 `register_reference`의 제한된 affine 정렬을 이용한다.
  모든 획에 동일한 변환을 적용하고 획별 독립 피팅으로 내부 구조를 바꾸지 않는다.
- 획 색은 해당 추출 영역의 중앙값, 배경은 보이는 배경 픽셀의 최근접 보간으로 임시 구성한다.
  복잡한 장면의 배경 복원/재질 추정을 해결한 방법은 아니다.

`TargetImageGuide`는 완성된 목표 RGB와 획별 edit-strength 지도를 받는다.
역확산의 clean prediction을 해당 지도 안에서 목표 RGB 쪽으로 이동시킨다.
그 결과를 VAE로 인코딩해 다음 DDIM 단계에 반영한다. 역전파 최적화는 없다.
VAE 왕복으로 제약이 달라지거나 지역 밖으로 변화가 번질 수 있으며 비용도 발생한다.

`repair_audit.target_guided_repair.repair_generated_sheet`는 위 과정과 기존 추출기·샘플러를 연결한다.
기본은 기존 생성물에 잡음을 넣는 img2img이고, 새 잡음에서 재생성하지 않는다.
마지막 결과를 목표 이미지로 덮어쓰지 않으며 최종 pixel-projection PNG도 만들지 않는다.

## 점수의 역할과 구현 범위

정상과 기준 일치도 점수(0~100)를 획 ID 순서로 받을 수 있다.
90 이상이면 해당 획의 수정 강도는 0, 40 이하면 1이며 중간은 선형이다.
한 스텝의 RGB 이동에는 추가로 최대 blend .8을 적용한다. 이 값은 미검증 시작값이며 조정 가능하다.
점수는 미분 목적함수로 최대화하지 않고, 원본에서 한 번 계산한 편집 계획을 고정하는 데 사용한다.
두께는 관측 값을 보존하므로 기준 두께와 다른 것만으로 기준 두께로 되돌리지 않는다.

기존 ZIP에는 논문식 네 속성 점수가 없어 기본 기하 진단 점수 `geometry_proxy_v1`도 제공한다.
길이 비율, 주축 방향, 경로 길이/외곽 span의 굽힘 근사, 몸체 Dice를 사용한다.
원래 측정값과 방향 차이를 저장하지만 이는 Li & Ren 논문의 정규화 공식을 재현한 것이 아니다.
서예 미관·글자 정오·추출 신뢰도에 교정된 점수도 아니다. 정상 스타일 차이에도 반응할 수 있다.
외부 점수를 제공하면 편집 강도에는 외부 점수를 사용하고 기하 측정은 진단으로만 남긴다.
논문의 0~1 점수를 넣으려면 명시적으로 100을 곱해 변환한다.

외부 점수 JSON 형식: `{"case_id": [[획1점수, ...], [둘째칸...], [셋째칸...], [넷째칸...]]}`.
`--scores`로 전달한다. 획 순서는 저장된 condition_N.npz의 strokes 순서와 같아야 한다.

## 실행 방법

### Colab 실행

`notebooks/repair_audit/target_guided_repair.ipynb`를 Colab GPU 런타임에 열고 코드 셀 1→6을 실행한다.
이전 stage 결과 ZIP은 필요 없다. 기존 best.pt와 원래 기준 글자/획 자료를 Drive에서 읽는다.
1번 셀은 이전 세션 모듈 캐시를 비우고 포함된 소스를 다시 불러온다. 업데이트 후 import 오류가 나면
최신 노트북을 다시 열고 1번부터 실행한다. 단순 파일 재작성만으로는 이미 import한 옛 모듈이 갱신되지 않는다.
2번 셀의 CHECKPOINT, DATA_ROOT를 확인한다. best.pt 옆 학습 설정·selected_paths.csv·conversion_report를 사용해
기존 기준 목록을 준비하며 전체 학습 NPZ 변환은 생략한다. 유효한 TRAIN_MANIFEST를 직접 지정할 수도 있다.

3번 셀에서 생성용 입력 폰트 이미지와 고정 기준 획을 분리해 지정한다.
선택 화면은 왼쪽의 활성 칸/입력·기준 선택과 오른쪽 썸네일 목록을 나란히 표시한다.
그림 바로 아래 선택 버튼을 누르면 활성 칸에 반영된다. 목록은 8개씩 페이지/검색으로 이동한다.
네 칸의 입력 미리보기가 상단에 유지되며 row 직접 입력은 접힌 선택 사항으로 둔다.
이 변경은 3번 선택 셀만 바꾸며 모델·학습 가중치·추출·생성·보정 로직이나 설정은 바꾸지 않는다.
처음에는 한 기준 폰트에서 글자별 획 한 종류를 골라 글자 이름(예: 가)으로 잠근다.
PROTOTYPE_BANK에 reference/strokes와 내용 해시를 저장하고 같은 글자 키의 다른 획으로 덮어쓰기를 거부한다.
다음 입력 폰트에서는 이미 잠긴 글자 이름을 그대로 사용한다. 폰트 선택으로 기준을 바꾸지 않는다.
네 칸의 입력 폰트가 서로 달라도 같은 글자 키라면 추출기와 보정기의 reference/strokes는 동일하다.
입력 폰트의 전체 글자 이미지만 생성에 사용하며 그 폰트의 획 주석은 보정 조건으로 전달하지 않는다.
입력 폰트와 고정 기준을 각각 미리보기로 표시한다. 입력과 기준의 문자 대응은 사용자가 확인한다.
문자/폰트 메타데이터가 없는 자료에서 같은 문자나 동일 폰트인지 자동 추정하지 않는다.
서로 다른 글자의 기준을 같은 폰트에서 고르는 것은 사용자가 확인해야 하며, 현재 자료로 폰트 ID를 자동 검증하지 않는다.
글자마다 한 기준을 고정하는 것은 동일 문자의 입력 폰트 변화에 대한 시험 조건이다. 모든 글자에 같은 획 세트를 쓰는 뜻은 아니다.
미리보기는 왼쪽 위/오른쪽 위/왼쪽 아래/오른쪽 아래 순서이며 같은 글자 반복 선택도 허용한다.
4번은 모델 로딩, 5번은 현재 선택을 고정해 새 생성·추출·보정, 6번은 가장 최근 결과 다운로드다.
다른 글자를 시험할 때는 선택을 바꾸고 5번만 다시 실행한다. 모델 재로딩이나 stage 결과가 필요 없다.

기본은 strength .65, seeds100/101, nominal60 steps로 생성 2장 + 정상 1장이다.
GENERATION_STRENGTHS=[.45,.65]처럼 강도 비교를 늘릴 수 있다. 모든 생성물을 보존하며 OCR로 거르지 않는다.
2×2 전체를 공동 img2img 생성하고, 생성에는 보정 가이드를 사용하지 않는다.
이후 두 보정 arm은 동일 초기 latent를 사용한다. 생성 fp16 VAE, 보정 fp32 VAE를 구분한다.
실행마다 고유 폴더를 만들고 선택 row·기준 획·체크포인트 해시·생성 설정·원시 결과·trace를 저장한다.
실제 사례 결과는 fresh_inputs/<case_id>/, 선택 내용은 selected_reference/selection.json에 있다.
selection.json에 입력과 고정 기준의 일치 여부도 기록한다. 같으면 교차 폰트 일반화 사례로 간주하지 않는다.
체크포인트는 기존 것을 사용하며, 기준 고정만으로 학습에 없던 폰트에 대한 일반화를 입증하지 않는다.
실제 폰트 일반화 주장은 학습과 시험 폰트를 분리한 평가가 추가로 필요하다.
생성 시트 저장 및 보정 사례 완료/실패/수동 중단 시 Drive에 ZIP을 백업한다.
5번 재실행은 새 실험이며 중간 latent/세션 간 재개는 지원하지 않는다. 실제 T4 속도·품질은 미검증이다.

노트북 재생성: `python notebooks/repair_audit/build_selectable_target_repair.py`.
이전 ZIP 기반 실행은 별도 `target_guided_repair_archive.ipynb`에 보존한다.

### CPU CLI

```powershell
python -m repair_audit.target_guided_repair --archive "C:/Users/freeg/Downloads/repair_stage4_20261123_132010_265667.zip" --output "C:/Users/freeg/Desktop/자율연구/output/target_repair_cpu" --case group_000_page_001
```

해당 생성 사례와 정상 대조군을 함께 복구한다. 저장된 추출 결과가 없는 사례는 보류한다.
`--observations-archive`에 같은 픽셀 입력을 사용한 이전 stage3 ZIP을 지정해 추출 마스크를 재사용할 수 있다.
체크포인트와 모델을 로딩하거나 확산을 실행하지 않는다.

- target_plan/aligned_target.png: 관측 두께로 렌더링하고 정합한 완성 목표 이미지.
- edit_strength.png: 점수가 수정 범위·강도로 변환된 지도.
- glyph_N_target_strokes.npz: 두께 렌더링 직후와 정합 후의 획별 alpha.
- plan.json: 관측 두께, 점수 출처, 측정값, 편집 강도, affine/Dice와 한계.
- cpu_blend_diagnostic_only.png: 입력에 가이드를 한 번 적용한 CPU 진단이며 확산 결과가 아니다.
- comparison.png: 입력/목표/편집 지도/CPU 진단 비교.

첫 실제 CPU 예에서는 affine Dice가 약 .91~.95였고 기하 지표가 24획 중 23획을 수정 대상으로 잡았다.
이는 대부분이 진짜 오류라는 뜻이 아니다. 기본 지표가 보수적인 정상 보존 판별기로 검증되지 않았음을 보여준다.
정상 대조군, 외부 점수, 변경 범위와 실제 글자 형태를 함께 확인해야 한다.

## 기존 생성 코드에 연결

```python
from repair_audit.target_guided_repair import repair_generated_sheet
guide, report = repair_generated_sheet(
    pipe, extractor, generated_sheet, conditions, output,
    scores=per_glyph_scores,  # None이면 공개된 미검증 기하 진단 점수
    steps=60, strength=.25, seed=2026,
    prompt=prompt, negative_prompt=negative_prompt,
)
```

conditions는 reference [1,H,W], strokes [K,H,W]를 가진 네 사전이다.
기존 고정 체크포인트의 추출기와 SD1.5/DDIM pipeline, float32 VAE를 사용한다.
saved masks가 있으면 cached_masks에 넣어 추출 재실행을 생략할 수 있다.
새 학습이나 새 CNN 정합기는 추가하지 않았다.

CPU 검사는 폐곡선 두께 렌더링, 글자 공통 affine, 점수별 편집 선택,
관측 두께 불변, 동일 초기 latent, 원본 사용, 최종 대체 없음과 전체 API 연결을 포함한다.
실제 GPU 생성 품질·속도·배경 조화·논문 대비 성능 향상은 검증되지 않았다.
