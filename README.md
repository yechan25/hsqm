# Hangul Stroke Extraction

Swin Transformer + Warp Hint + Cross-Attention + BiFPN 기반 한글 획 분리 모델.

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
