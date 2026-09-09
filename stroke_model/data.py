"""데이터셋 로딩 / augmentation. 01_Training.ipynb와 동일 (변경 없음)."""
import ast
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

COL_ALIASES = {
    "I_prep": ["I_prep", "i_prep", "input", "input_image", "input_path",
               "input_image_path", "prep", "prep_path", "source_image", "source_path"],
    "I_g": ["I_g", "i_g", "target_image", "target", "target_path",
            "target_image_path", "glyph", "glyph_path", "reference_image", "reference_path"],
    "I_G_strokes": ["I_G_strokes", "i_g_strokes", "target_strokes", "target_stroke_paths",
                     "template_strokes", "template_stroke_paths", "reference_strokes",
                     "reference_stroke_paths", "g_stroke_paths"],
    "target_strokes": ["input_stroke_paths", "input_strokes", "input_strokes_paths",
                        "prep_stroke_paths", "prep_strokes", "gt_stroke_paths", "gt_strokes",
                        "answer_stroke_paths", "stroke_paths"],
    "char_id": ["char", "character", "char_id", "glyph_id", "text", "letter"],
}

_PATH_INDEX_CACHE: Dict[str, Dict[str, List[Path]]] = {}


def _build_path_index(root) -> Dict[str, List[Path]]:
    root_path = Path(root)
    key = str(root_path.resolve()) if root_path.exists() else str(root_path)
    if key in _PATH_INDEX_CACHE:
        return _PATH_INDEX_CACHE[key]

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    index: Dict[str, List[Path]] = {}
    if root_path.exists():
        for q in root_path.rglob("*"):
            if q.is_file() and q.suffix.lower() in exts:
                index.setdefault(q.name, []).append(q)
    _PATH_INDEX_CACHE[key] = index
    return index


def _choose_path(candidates: List[Path], raw_value: str) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    raw_norm = raw_value.replace("\\", "/").strip("/").lower()
    parts = [x for x in raw_norm.split("/")[:-1] if x]
    if parts:
        scored = []
        for c in candidates:
            cn = str(c).replace("\\", "/").lower()
            score = sum(1 for part in parts if part in cn)
            scored.append((score, len(str(c)), c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]
    return sorted(candidates, key=lambda x: (len(str(x)), str(x)))[0]


def resolve_path(root, value: Any) -> Optional[Path]:
    if value is None:
        return None
    s = str(value).strip().strip("'").strip('"')
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    p = Path(s)
    if p.is_absolute():
        if p.exists():
            return p
        name = p.name
    else:
        direct = Path(root) / p
        if direct.exists():
            return direct
        name = p.name

    index = _build_path_index(root)
    candidates = index.get(name, [])
    if candidates:
        return _choose_path(candidates, s)

    low = name.lower()
    lower_matches: List[Path] = []
    for fname, paths in index.items():
        if fname.lower() == low:
            lower_matches.extend(paths)
    if lower_matches:
        return _choose_path(lower_matches, s)

    return Path(root) / p


def parse_path_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    for sep in [";", "|", "\n", ","]:
        if sep in s:
            return [x.strip().strip("'").strip('"') for x in s.split(sep) if x.strip()]
    return [s.strip().strip("'").strip('"')]


def find_column(df: pd.DataFrame, aliases: Sequence[str], required: bool = True,
                 exclude: Optional[set] = None) -> Optional[str]:
    exclude = exclude or set()
    exclude_lower = {x.lower() for x in exclude}
    lower_to_col = {c.lower(): c for c in df.columns if c.lower() not in exclude_lower}
    for a in aliases:
        if a.lower() in lower_to_col:
            return lower_to_col[a.lower()]
    for c in df.columns:
        if c.lower() in exclude_lower:
            continue
        cl = c.lower()
        for a in aliases:
            if a.lower() in cl or cl in a.lower():
                return c
    if required:
        raise KeyError(f"CSV에서 필요한 컬럼을 찾지 못했습니다. aliases={aliases}, columns={list(df.columns)}")
    return None


def load_grayscale(path: Path, image_size: int, auto_invert: bool = True) -> torch.Tensor:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    img = Image.open(path).convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if auto_invert and arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr)[None, :, :]


def binarize_tensor(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (x > threshold).float()


def gaussian_soften(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = int(6 * sigma + 1) | 1
    y = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])
    flat = y.flatten(1)
    mx = flat.max(dim=1).values.clamp_min(1e-6).view(-1, 1, 1)
    return (y / mx).clamp(0, 1)


def load_stroke_stack(root, value: Any, image_size: int, max_strokes: int,
                       sigma: float, auto_invert: bool = True) -> torch.Tensor:
    paths = parse_path_list(value)
    chans: List[torch.Tensor] = []
    for s in paths[:max_strokes]:
        p = resolve_path(root, s)
        if p is None or not p.exists():
            chans.append(torch.zeros(1, image_size, image_size))
            continue
        x = load_grayscale(p, image_size=image_size, auto_invert=auto_invert)
        x = binarize_tensor(x, 0.5)
        x = gaussian_soften(x, sigma=sigma)
        chans.append(x)
    while len(chans) < max_strokes:
        chans.append(torch.zeros(1, image_size, image_size))
    return torch.cat(chans, dim=0)


def detect_dataset_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {
        "I_prep": find_column(df, COL_ALIASES["I_prep"], required=True),
        "I_g": find_column(df, COL_ALIASES["I_g"], required=True),
        "I_G_strokes": find_column(df, COL_ALIASES["I_G_strokes"], required=True),
        "target_strokes": find_column(df, COL_ALIASES["target_strokes"], required=True),
        "char_id": find_column(df, COL_ALIASES["char_id"], required=False),
    }
    print("[columns]", cols)
    return cols


def load_dataset_sample(row: pd.Series, root: str, cols: Dict[str, Optional[str]],
                         config: Dict[str, Any]) -> Dict[str, Any]:
    size, K = config["image_size"], config["max_strokes"]
    auto_inv = config["auto_invert"]

    p_prep = resolve_path(root, row[cols["I_prep"]])
    p_g = resolve_path(root, row[cols["I_g"]])

    return {
        "I_prep": load_grayscale(p_prep, size, auto_inv),
        "I_g": load_grayscale(p_g, size, auto_inv),
        "I_g_vis": load_grayscale(p_g, size, auto_invert=False),
        "I_G_strokes": load_stroke_stack(root, row[cols["I_G_strokes"]], size, K,
                                          sigma=config["template_sigma"], auto_invert=auto_inv),
        "target_strokes": load_stroke_stack(root, row[cols["target_strokes"]], size, K,
                                             sigma=config["target_sigma"], auto_invert=auto_inv),
        "char_id": str(row[cols["char_id"]]) if cols["char_id"] is not None else "",
        "prep_path": str(p_prep),
        "g_path": str(p_g),
    }


def apply_joint_augmentation(item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    I_prep, I_g = item["I_prep"], item["I_g"]
    I_G_strokes, target = item["I_G_strokes"], item["target_strokes"]
    tensors = [I_prep, I_g, I_G_strokes, target]

    if random.random() < config["affine_p"]:
        angle = random.uniform(-config["rotation_deg"], config["rotation_deg"])
        max_trans = config["translate_frac"] * config["image_size"]
        translate = [int(random.uniform(-max_trans, max_trans)), int(random.uniform(-max_trans, max_trans))]
        scale = random.uniform(config["scale_min"], config["scale_max"])
        new_tensors = [
            TF.affine(t, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0],
                      interpolation=InterpolationMode.BILINEAR, fill=0.0, center=None).clamp(0, 1)
            for t in tensors
        ]
        I_prep, I_g, I_G_strokes, target = new_tensors

    if random.random() < config["hflip_p"]:
        I_prep, I_g = TF.hflip(I_prep), TF.hflip(I_g)
        I_G_strokes, target = TF.hflip(I_G_strokes), TF.hflip(target)

    if random.random() < config["brightness_contrast_p"]:
        b, c = random.uniform(0.85, 1.15), random.uniform(0.85, 1.15)
        I_prep = TF.adjust_contrast(TF.adjust_brightness(I_prep, b), c).clamp(0, 1)
        I_g = TF.adjust_contrast(TF.adjust_brightness(I_g, b), c).clamp(0, 1)

    if random.random() < config["noise_p"]:
        I_prep = (I_prep + torch.randn_like(I_prep) * config["noise_std"]).clamp(0, 1)

    if random.random() < config["stroke_cutout_p"]:
        s = config["stroke_cutout_size"]
        H, W = I_prep.shape[-2:]
        y, x = random.randint(0, max(0, H - s)), random.randint(0, max(0, W - s))
        I_prep[:, y:y + s, x:x + s] = 0.0

    item = dict(item)
    item.update(I_prep=I_prep, I_g=I_g, I_G_strokes=I_G_strokes, target_strokes=target)
    return item


class StrokeDataset(Dataset):
    def __init__(self, csv_path: str, root: str, train: bool, config: Dict[str, Any]):
        self.root = root
        self.train = train
        self.config = config
        self.df = pd.read_csv(csv_path)
        self.cols = detect_dataset_columns(self.df)
        print(f"[Dataset] {csv_path} | rows={len(self.df)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = load_dataset_sample(self.df.iloc[idx], self.root, self.cols, self.config)
        if self.train and self.config["augment"]:
            item = apply_joint_augmentation(item, self.config)
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ["I_prep", "I_g", "I_g_vis", "I_G_strokes", "target_strokes"]:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    out["char_id"] = [b["char_id"] for b in batch]
    out["prep_path"] = [b["prep_path"] for b in batch]
    out["g_path"] = [b["g_path"] for b in batch]
    return out


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    for key in ["I_prep", "I_g", "I_g_vis", "I_G_strokes", "target_strokes"]:
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def maybe_apply_mixup(batch: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    if random.random() >= config["mixup_p"]:
        return batch
    alpha = config["mixup_alpha"]
    if alpha <= 0:
        return batch
    B = batch["I_prep"].shape[0]
    if B < 2:
        return batch
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(B, device=batch["I_prep"].device)
    mixed = dict(batch)
    for key in ["I_prep", "I_g", "I_G_strokes", "target_strokes"]:
        mixed[key] = lam * batch[key] + (1.0 - lam) * batch[key][perm]
    return mixed
