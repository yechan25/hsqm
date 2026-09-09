"""손실 함수. 01_Training.ipynb의 loss 정의와 동일 (변경 없음)."""
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_weights_from_target(target: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    """target: B,K,H,W. 짧은 획일수록 큰 가중치."""
    if power <= 0:
        return torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)

    area = target.flatten(2).sum(dim=2)
    valid = (area > 1e-6).float()
    mean_area = area[area > 1e-6].mean().clamp_min(1.0) if (area > 1e-6).any() else torch.tensor(1.0, device=target.device)
    w = (mean_area / area.clamp_min(1.0)).pow(power)
    return w.clamp(0.5, 5.0) * valid + (1.0 - valid)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                    weights: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (pred * target).sum(dim=dims)
    den = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (den + eps)
    loss = 1.0 - dice
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def dilate_soft_target(target: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return target
    pad = kernel_size // 2
    return F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=pad).clamp(0, 1)


def bce_loss(logits: torch.Tensor, target: torch.Tensor, config: Dict[str, Any],
             weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    k = int(config.get("bce_target_dilate_kernel", 1))
    bce_target = dilate_soft_target(target, k)
    pos_weight = torch.tensor(config["pos_weight"], device=logits.device, dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, bce_target, pos_weight=pos_weight, reduction="none")
    if weights is not None:
        raw = raw * weights[:, :, None, None]
    return raw.mean()


def false_positive_loss(logits: torch.Tensor, target: torch.Tensor, config: Dict[str, Any],
                         weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    k = int(config.get("fp_ignore_dilate_kernel", 1))
    ignore_near_stroke = dilate_soft_target(target, k)
    bg = (1.0 - ignore_near_stroke).clamp(0, 1)
    raw = pred * bg
    if weights is not None:
        raw = raw * weights[:, :, None, None]
    return raw.mean()


def coverage_loss(logits: torch.Tensor, target: torch.Tensor, config: Dict[str, Any],
                   weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    wide_target = dilate_soft_target(target, int(config.get("bce_target_dilate_kernel", 3)))
    target_area = wide_target.flatten(2).sum(dim=2).clamp_min(1.0)
    pred_inside = (pred * wide_target).flatten(2).sum(dim=2)
    loss = F.relu(0.70 * target_area - pred_inside) / target_area
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def flow_smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


def hint_aux_loss(H: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (H * target).sum(dim=(2, 3))
    den = H.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return (1.0 - dice).mean()


def total_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor, epoch: int,
               config: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits, flow, H = outputs["logits"], outputs["flow"], outputs["H"]

    w = channel_weights_from_target(target, config["short_stroke_weight_power"])

    L_dice = soft_dice_loss(logits, target, w)
    L_bce = bce_loss(logits, target, config, w)
    L_fp = false_positive_loss(logits, target, config, w)
    L_cov = coverage_loss(logits, target, config, w)
    L_smooth = flow_smoothness_loss(flow)

    base_hint_w = config["hint_aux_weight"]
    decay_epoch = max(1, int(config["hint_aux_decay_epoch"]))
    hint_coef = base_hint_w * max(0.0, 1.0 - epoch / decay_epoch)
    L_hint = hint_aux_loss(H, target)

    loss = (
        config["dice_weight"] * L_dice
        + config["bce_weight"] * L_bce
        + config["false_positive_weight"] * L_fp
        + config["coverage_weight"] * L_cov
        + config["warp_smooth_weight"] * L_smooth
        + hint_coef * L_hint
    )

    logs = {
        "loss": float(loss.detach().cpu()),
        "dice_loss": float(L_dice.detach().cpu()),
        "bce": float(L_bce.detach().cpu()),
        "fp": float(L_fp.detach().cpu()),
        "coverage": float(L_cov.detach().cpu()),
        "flow_smooth": float(L_smooth.detach().cpu()),
        "hint_aux": float(L_hint.detach().cpu()),
        "hint_coef": float(hint_coef),
    }
    return loss, logs


def compute_metrics(logits: torch.Tensor, target: torch.Tensor, config: Dict[str, Any]) -> Dict[str, float]:
    pred = torch.sigmoid(logits)
    eps = 1e-6

    inter = (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    soft_dice = ((2 * inter + eps) / (den + eps)).mean()

    th = config["hard_threshold"]
    pred_h = (pred > th).float()
    tgt_h = (target > th).float()

    inter_h = (pred_h * tgt_h).sum(dim=(2, 3))
    union_h = ((pred_h + tgt_h) > 0).float().sum(dim=(2, 3))
    den_h = pred_h.sum(dim=(2, 3)) + tgt_h.sum(dim=(2, 3))

    hard_dice = ((2 * inter_h + eps) / (den_h + eps)).mean()
    hard_iou = ((inter_h + eps) / (union_h + eps)).mean()

    return {
        "soft_dice": float(soft_dice.cpu()),
        "hard_dice": float(hard_dice.cpu()),
        "hard_iou": float(hard_iou.cpu()),
    }
