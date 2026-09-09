"""공용 유틸리티: seed, device, checkpoint, config 로딩."""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def average_logs(logs: List[Dict[str, float]]) -> Dict[str, float]:
    if not logs:
        return {}
    keys = logs[0].keys()
    return {k: float(np.mean([x[k] for x in logs])) for k in keys}


def safe_json_dump(obj: Any, path):
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def save_checkpoint(path, model: nn.Module, optimizer, scheduler, epoch: int,
                     best_score: float, history: List[Dict[str, Any]], config: Dict[str, Any]):
    path = Path(path)
    ensure_dir(path.parent)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "history": history,
        "config": config,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
