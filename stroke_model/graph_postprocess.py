"""Overlap-preserving refinement by a strictly convex graph energy.

On the 8-connected input-ink graph, minimize, independently for every stroke:
    ||q - p||^2 + hint_weight * ||q - h||^2
        + smoothness * sum_{(i,j)} w_ij * (q_i - q_j)^2.

The hint term is optional; h MUST be the warped hint in input coordinates.
Edges use the fused evidence to avoid smoothing across a predicted boundary.
No seed selection, channelwise peak normalization, area ranking, skeleton
cutting, or forced single-component reassignment is performed. Channels may
overlap. Scores are membership evidence, not calibrated probabilities.

This CPU/SciPy implementation is for inference, not differentiable training.
"""
from typing import Optional, Sequence

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu

from .postprocess import make_input_ink_binary


def _unit_array(value, shape, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return arr


def _active_indices(reference_strokes, active_channels, shape):
    if reference_strokes is None and active_channels is None:
        raise ValueError("Supply reference_strokes or explicit active_channels; do not infer stroke count from noise.")
    if active_channels is None:
        ref = _unit_array(reference_strokes, shape, "reference_strokes")
        # This dataset uses exactly zero padding; use the unwarped reference
        # only to identify real channels, never as a spatial prior.
        return np.flatnonzero(ref.reshape(shape[0], -1).sum(axis=1) > 0)
    ids = np.asarray(list(active_channels))
    if ids.size == 0:
        return np.empty(0, dtype=np.int64)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("active_channels must be a sequence of integer indices")
    if (ids < 0).any() or (ids >= shape[0]).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("active_channels contains duplicates or out-of-range indices")
    return np.sort(ids)


def _ink_laplacian(ink, evidence, edge_scale):
    """Sparse 8-neighbor graph; background never becomes a graph node."""
    height, width = ink.shape
    count = int(ink.sum())
    node = np.full(ink.shape, -1, dtype=np.int64)
    node[ink] = np.arange(count)
    rows, cols, weights = [], [], []
    for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        a = (slice(y0, y1), slice(x0, x1))
        b = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
        valid = ink[a] & ink[b]
        u, v = node[a][valid], node[b][valid]
        if not len(u):
            continue
        # Max over channels makes edge strength insensitive to zero padding.
        delta = np.max(np.abs(evidence[:, a[0], a[1]] - evidence[:, b[0], b[1]]), axis=0)[valid]
        weight = np.exp(-0.5 * (delta / edge_scale) ** 2) / np.hypot(dy, dx)
        rows.extend([u, v])
        cols.extend([v, u])
        weights.extend([weight, weight])
    if not rows:
        return sparse.csc_matrix((count, count), dtype=np.float64)
    adjacency = sparse.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count, count),
    ).tocsc()
    return sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel(), format="csc") - adjacency


def refine_strokes_graph(
    pred_soft: np.ndarray,
    input_img: np.ndarray,
    *,
    reference_strokes: Optional[np.ndarray] = None,
    warped_hints: Optional[np.ndarray] = None,
    active_channels: Optional[Sequence[int]] = None,
    ink_mask: Optional[np.ndarray] = None,
    smoothness: float = 1.0,
    hint_weight: float = 0.25,
    edge_scale: float = 0.25,
) -> np.ndarray:
    """Return K,H,W refined scores in [0,1], zero outside ink/inactive channels.

    Existing ink extraction is retained for a fair legacy comparison. Supply
    ink_mask explicitly if your application already has a trusted binary mask.
    smoothness=0 and hint_weight=0 is the masked raw-prediction baseline.
    Defaults are starting points, not parameters validated on real handwriting.
    """
    pred_soft = np.asarray(pred_soft)
    if pred_soft.ndim != 3 or min(pred_soft.shape) < 1:
        raise ValueError("pred_soft must have nonempty shape K,H,W")
    shape = pred_soft.shape
    pred = _unit_array(pred_soft, shape, "pred_soft")
    img = _unit_array(input_img, shape[1:], "input_img")
    for name, value in [("smoothness", smoothness), ("hint_weight", hint_weight), ("edge_scale", edge_scale)]:
        if not np.isfinite(value) or value < 0 or (name == "edge_scale" and value == 0):
            raise ValueError(f"Invalid {name}: {value}")
    ids = _active_indices(reference_strokes, active_channels, shape)
    if ink_mask is None:
        ink = make_input_ink_binary(img)
    else:
        mask = _unit_array(ink_mask, shape[1:], "ink_mask")
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("ink_mask must be binary")
        ink = mask.astype(bool)
    hints = None if warped_hints is None else _unit_array(warped_hints, shape, "warped_hints")
    result = np.zeros(shape, dtype=np.float32)
    if not ink.any():
        return result
    if not len(ids):
        raise ValueError("Input has ink but no active reference strokes; check the reference/labels.")

    # Without a supplied aligned hint, there is no hint penalty toward zero.
    alpha = hint_weight if hints is not None else 0.0
    evidence = pred[ids].copy()
    if alpha:
        evidence = (evidence + alpha * hints[ids]) / (1.0 + alpha)
    rhs = evidence[:, ink].T
    if smoothness:
        laplacian = _ink_laplacian(ink, evidence, edge_scale)
        system = sparse.eye(int(ink.sum()), format="csc") + smoothness / (1.0 + alpha) * laplacian
        # One factorization, all stroke right-hand sides. Identity makes the
        # system nonsingular even for disconnected ink or isolated pixels.
        refined = splu(system).solve(rhs)
    else:
        refined = rhs
    for column, channel in enumerate(ids):
        result[channel, ink] = np.clip(refined[:, column], 0.0, 1.0)
    return result


def postprocess_batch_predictions_graph(
    pred_soft: torch.Tensor,
    input_imgs: torch.Tensor,
    reference_strokes: torch.Tensor,
    warped_hints: Optional[torch.Tensor] = None,
    *,
    mode: str = "overlap",
    threshold: float = 0.25,
    **kwargs,
) -> torch.Tensor:
    """Batched inference wrapper; returns a float32 tensor on pred_soft.device.

    mode='overlap' thresholds each stroke independently (recommended).
    mode='soft' returns scores for later evaluation/threshold selection.
    mode='exclusive' is an optional comparison; ties go to the lowest channel.
    Overlap output does NOT promise full ink coverage or a connected stroke.
    """
    if mode not in {"soft", "overlap", "exclusive"}:
        raise ValueError("mode must be soft, overlap, or exclusive")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if pred_soft.ndim != 4 or pred_soft.shape[0] < 1:
        raise ValueError("pred_soft must have nonempty shape B,K,H,W")
    batch, _, height, width = pred_soft.shape
    if input_imgs.shape != (batch, 1, height, width) or reference_strokes.shape != pred_soft.shape:
        raise ValueError("input_imgs/reference_strokes shapes do not match pred_soft")
    if warped_hints is not None and warped_hints.shape != pred_soft.shape:
        raise ValueError("warped_hints shape does not match pred_soft")
    pred = pred_soft.detach().float().cpu().numpy()
    images = input_imgs.detach().float().cpu().numpy()
    refs = reference_strokes.detach().float().cpu().numpy()
    hints = None if warped_hints is None else warped_hints.detach().float().cpu().numpy()
    output = np.zeros_like(pred)
    for b in range(batch):
        scores = refine_strokes_graph(pred[b], images[b, 0], reference_strokes=refs[b],
                                     warped_hints=None if hints is None else hints[b], **kwargs)
        if mode == "soft":
            output[b] = scores
        elif mode == "overlap":
            output[b] = scores > threshold
        else:
            ids = _active_indices(refs[b], kwargs.get("active_channels"), pred[b].shape)
            ink = np.asarray(kwargs["ink_mask"], dtype=bool) if kwargs.get("ink_mask") is not None else make_input_ink_binary(images[b, 0])
            if len(ids):
                winner = ids[np.argmax(scores[ids], axis=0)]
                for k in ids:
                    output[b, k] = (winner == k) & ink
    return torch.from_numpy(output).to(device=pred_soft.device)
