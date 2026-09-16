# Hangul Stroke Extraction

Swin Transformer + Warp Hint + Cross-Attention + BiFPN 기반 한글 획 분리 모델.

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
