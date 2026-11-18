# 획 보정 진단·수정 비교

기존 획 추출기와 생성 결과를 그대로 사용해 VAE 왕복, 일반 재역확산,
복원·정리·삭제 보정의 차이를 확인하는 독립 진단 모듈입니다. 새 학습은 하지 않습니다.

현재 단계: **1단계 원인 진단 완료 → 2단계 삭제 경계 보호 수정 비교**.

- **[2단계 Colab 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage2.ipynb)**
- [2단계 변경 사항·실행 안내](../docs/repair_audit/STAGE2.md)
- 입력: 방금 실행한 stage1 결과 ZIP과 동일 best.pt. 기존·수정 보정을 같은 강도/잡음으로 비교합니다.

- **[Colab에서 실행](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage1.ipynb)**
- [실행 안내와 결과 해석](../docs/repair_audit/STAGE1.md)
- [실행 노트북](../notebooks/repair_audit/stage1.ipynb)
- [진단 코드](stage1.py)
- [검증 코드](../tests/test_repair_stage1_audit.py)

## 실행

1. 새 L4 GPU Colab에서 노트북을 엽니다.
2. 코드 셀 1을 실행하고, 셀 2에서 Drive의 `CHECKPOINT` 경로를 확인합니다.
3. `ARCHIVE`를 비워두면 기존 cyclic/selectable 실험 결과 ZIP을 업로드할 수 있습니다.
4. 코드 셀 2→3→4→5를 실행합니다. 정상 입력 대조군도 포함됩니다.
5. 결과 ZIP과 `human_review.csv`를 확인합니다.

노트북에는 필요한 실행 소스가 내장되어 있습니다. GitHub 저장소를 clone하거나
코드 ZIP을 추가 업로드할 필요가 없습니다. 실험 데이터·모델 가중치는 저장소에 포함하지 않습니다.

자동 두께 지표와 기준 유사도는 정확도 판정이 아닙니다. 실제 얇아짐과 오류 감소는
비교 이미지로 검수합니다. CPU 검증 통과와 실제 GPU 품질 검증은 구분합니다.

## 다음 단계

- [2단계: 삭제 경계 보호 비교](../docs/repair_audit/STAGE2.md)
- **[3단계 Colab: 다른 글자 12자·획 형태 검증](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/repair_audit/stage3.ipynb)**
- [3단계 자료 준비와 평가 안내](../docs/repair_audit/STAGE3.md)

## 개발

저장소 루트에서 실행합니다.

```bash
python notebooks/repair_audit/build_stage1.py
python notebooks/repair_audit/build_stage2.py
python notebooks/repair_audit/build_stage3.py
python -m unittest discover -s tests -p test_repair_stage1_audit.py -v
```

runtime은 이 폴더, 실행 노트북은 `notebooks/repair_audit/`,
문서는 `docs/repair_audit/`로 구분했습니다. 기존 보정 실행 경로는 바꾸지 않습니다.
