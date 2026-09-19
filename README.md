# Hangul Stroke Extraction

## 새 연구 모델: 기준 획 조건부 V2

[구조·데이터 구축·미분 가능 지표 설계](REFERENCE_V2.md), [Colab 학습 노트북](notebooks/train_reference_v2.ipynb).
[V2 학습을 Colab에서 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/train_reference_v2.ipynb)
첫 셀에서 GitHub 코드를 받고, 두 번째 셀에서 기존 train_dataset/paths.csv를 자동 변환합니다.
이후 train 증강 → 학습 → 검증 결과 표시까지 5개 셀로 실행합니다.
완료된 변환·동일 설정의 증강 데이터는 Drive에서 재사용합니다. 이미지 내용을 같은 경로에서 수정했다면 두 번째 셀의 `rebuild=True`로 다시 만드세요.
동결 Swin + 공유 획 추출 헤드. 실제 데이터 학습 전이며 기존 체크포인트와 호환되지 않습니다.
아래 셀 3개 노트북은 기존 FPN/BiFPN 모델용입니다.

Swin Transformer + Warp Hint + Cross-Attention + BiFPN 기반 한글 획 분리 모델.

## 가장 쉬운 실행: 셀 3개

[Colab에서 간단 실행 노트북 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/easy_postprocess.ipynb)

1. **준비하기** 실행
2. **Drive 연결하고 기존 모델 불러오기** 실행 (모델이 여러 개면 번호 선택)
3. **결과 보기** 실행

코드는 자동으로 받고, 기존 FPN/BiFPN 체크포인트는 구조를 확인해 불러옵니다.
기록된 공유 Drive의 HSQM 폴더 또는 MyDrive/HSQM에서 best.pt와 테스트
데이터를 찾습니다. 못 찾는 경우에만 파일 경로를 묻습니다.
3번 셀에서 이미지 번호와 `mode`를 바꿀 수 있습니다. 기본 `exclusive`는
모든 글자 픽셀을 한 획에만 배정하며, `overlap`은 중복을 허용합니다.
결과 아래에 미배정·중복·배경 침범을 표시합니다. 모드 변경 후에는 3번 셀만
다시 실행하세요. 재학습하지 않습니다.

## 교차점 공유 후처리

`stroke_model/graph_postprocess.py`에 교차점의 중복을 허용하는 graph 기반
후처리를 추가했습니다. 큰 획 우선/분기점 제거 규칙 대신 모델 출력과 정렬된
힌트, 이웃 간 연속성을 함께 최적화합니다. 재학습 없이 사용 가능합니다.
사용법과 한계는 [POSTPROCESS_GRAPH.md](POSTPROCESS_GRAPH.md)를 참고하세요.
실제 데이터에서의 성능 개선은 아직 검증되지 않았습니다.

[Colab에서 학습/후처리 비교 노트북 열기](https://colab.research.google.com/github/yechan25/hsqm/blob/main/notebooks/train_bifpn.ipynb)

기존 체크포인트만 평가할 경우: 코드 받기, Drive 마운트, config 로드 셀을
실행하고 `from stroke_model.utils import get_device; device = get_device()`로
device를 준비한 뒤 8절의 체크포인트 로드와 9절의 후처리 비교를 실행합니다.
6절의 학습 셀은 실행할 필요가 없습니다. `config['out_dir']`는 기존
`checkpoints/best.pt`가 들어 있는 출력 폴더로 지정하세요.

## 구조

```
stroke_model/
    model.py       # SwinWarpHintStrokeModel (decoder: BiFPN + CoordAttention)
    losses.py       # dice/bce/fp/coverage/flow-smooth/hint-aux loss
    data.py          # StrokeDataset, augmentation, path resolution
    train.py         # 학습 루프 (run_training)
configs/
    base.yaml        # 학습 설정
```

## 사용법 (Colab)

```python
!git clone https://github.com/<아이디>/<레포이름>.git
%cd <레포이름>
!pip install -q -r requirements.txt

from stroke_model.utils import load_config
from stroke_model.train import run_training

config = load_config("configs/base.yaml")
config["dataset_root"] = "..."  # 필요시 경로 override
model, history = run_training(config)
```

## 변경 이력

- v1: `01_Training.ipynb` / `02_Use.ipynb`에서 코드 분리, decoder를 FPN -> BiFPN + Coordinate Attention으로 교체
