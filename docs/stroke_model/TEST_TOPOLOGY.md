# Topo 추출 모델 테스트

[테스트 코랩 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/test_topology.ipynb)

준비/Drive 연결 → 모델과 테스트 CSV 선택 → 추론 순으로 실행한다. 재학습하지 않는다.
기본 체크포인트는 `HSQM/topology_runs/시간/best.pt` 중 가장 최근 실행이며 경로를 출력한다.
현재는 `HSQM/v2_topology_runs/시간/best.pt`를 먼저 찾고 없으면 이전 폴더를 찾는다.
특정 실행을 확인하려면 `CHECKPOINT`를 직접 지정한다. Topo architecture만 허용한다.
Swin 가중치를 포함한 학습 결과 전체를 로드하며 ImageNet 가중치를 다시 다운로드하지 않는다.
새 V2+정션 모델은 모델 자체가 V2이며 soft 합은 원본, hard 출력은 foreground 내 argmax다.
이전 독립 Topo는 기존 sigmoid/threshold 방식으로 따로 표시한다.

기본 데이터는 `HSQM/dataset/test_dataset`의 CSV와 이미지다. 다른 폴더는 `DATA_ROOT`,
정확한 CSV는 `CSV_PATH`, 이미지가 폴더 밖에 있으면 `DATASET_ROOT`를 지정한다.
CSV에는 입력 글자, 기준 글자, 기준 획 경로만 필요하다. 입력 획 정답은 읽지 않는다.
기본은 기준 글자별로 골고루 최대 8장. `INPUT_FILES` 또는 1부터 시작하는 `SAMPLE_ROWS`로
특정 이미지를 고를 수 있고, 둘 다 None에 `PREVIEW_COUNT=None`이면 CSV 전체를 처리한다.
빈 기준 획 샘플은 제외 기록을 남기고 진행한다. 다른 경로/형식 오류는 중단한다.

원본/기준/soft 합집합/hard 합집합/색상별 추출/미추출 원본 잉크를 나란히 표시한다.
획별 기준과 soft/hard 예측도 표시하며 출력은 0~1 고정 범위다. 독립 출력의 겹침은
자홍색으로 표시한다. 두껍게 만들기, 잉크 강제 배정, 연결 복구 등의 후처리는 하지 않는다.
이전 독립 Topo의 hard 기준은 기본 0.5이고 `THRESHOLD`로 바꿀 수 있다.
새 V2+정션 모델은 기존 V2 argmax 분할이므로 이 threshold를 사용하지 않는다.
새 모델의 커버리지 100%는 설계상 잉크 보존의 결과이며 올바른 획 배정의 증거가 아니다.
soft 마스크 원본은 그대로 저장한다.

체크포인트 옆 `topo_test_시간/`에 원시 NPZ(확률/마스크/flow/정렬 정보), 비교 PNG,
`inference.json`, `ink_coverage.csv`를 저장한다. 원본 잉크 커버리지와 중복률은
두께/미추출 확인용 진단이며 획 정확도나 기존 품질 평가 모델 점수가 아니다.
학습 이미지나 검증 이미지를 다시 사용하면 독립 test 성능이라고 해석하지 않는다.

소형 backbone으로 체크포인트 로드→정답 없는 CSV 추론→PNG/NPZ 저장, 빈 기준 제외,
V2 거부, 얇은 마스크 미추출 진단을 검증한다. 실제 체크포인트/GPU 평가는 Colab에서 수행한다.
노트북 재생성: `python tests/build_topology_test_notebook.py`.
