# HSQM

한글 이미지 연구 코드와 실험을 관리하는 공용 저장소입니다. 모델 업데이트는 이 저장소의 커밋 이력에서 확인합니다.

## 폴더 안내

| 경로 | 용도 |
|---|---|
| [stroke_model/](stroke_model/) | 획 분리 모델, 학습·추론·후처리 및 관련 데이터 처리 코드 |
| [font_test/](font_test/) | 폰트 관련 실험 코드와 설정을 위한 공간 |
| [configs/](configs/) | 모델 학습 설정 |
| [notebooks/](notebooks/) | Colab 실행 및 결과 확인 노트북 |
| [tests/](tests/) | 코드 검증 및 배포 묶음 생성 도구 |
| [docs/stroke_model/](docs/stroke_model/) | 획 분리 모델 설명 및 실행 안내 |
| [diagnostics/](diagnostics/) | 버전 관리하는 소규모 진단 예시 |

## 획 분리 모델

- 굵은 자모 마스크 학습: [Jamo Colab](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/jamo.ipynb) — [데이터 준비와 실행](docs/stroke_model/JAMO.md)
- 최신 구현: **기준 획 조건부 V2 (`ReferenceStrokeModel`)** — [구조와 데이터 설계](docs/stroke_model/REFERENCE_V2.md)
- 기존 모델: **Swin + Warp Hint + Cross-Attention + BiFPN (`SwinWarpHintStrokeModel`)**
- [학습·추론·Colab 실행 안내](docs/stroke_model/README.md)
- [그래프 후처리 설명](docs/stroke_model/POSTPROCESS_GRAPH.md)

최신 구현과 최고 성능 체크포인트는 별도로 구분합니다. 학습 결과를 기록할 때 코드 커밋, 설정, 데이터 버전, 평가 지표와 체크포인트 위치를 함께 남깁니다.

## 업데이트 원칙

1. 획 분리 모델 관련 실행 코드는 `stroke_model/`에 저장합니다. 새 연구 항목은 `font_test/`처럼 별도 폴더로 구분합니다.
2. 동작이나 구조를 바꿀 때 관련 문서·설정도 함께 갱신합니다. 노트북은 저장소의 코드를 가져와 실행하도록 유지합니다.
3. 변경 내용을 확인하고 필요한 검증을 실행한 후 커밋하고 GitHub에 push합니다. GitHub에서 해당 커밋이 확인되어야 업로드 완료입니다.
4. 원본 데이터, 모델 가중치, 대용량 생성 결과와 비밀 키는 커밋하지 않습니다. 데이터와 체크포인트는 Drive 등 외부 저장소에서 관리합니다.
5. 재현할 실험은 실행 당시 Git 커밋을 기록합니다. 코드 수정만으로 학습 완료나 성능 향상을 주장하지 않습니다.

```bash
git pull --ff-only
# 파일 수정 및 필요한 검증
git status
git diff
git add <변경한 파일 경로>
git commit -m "Describe the change"
git push origin main
```
