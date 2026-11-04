"""V2 unchanged; optional junction supervision from actual input annotations.

No new model modules, affine, recurrent context, or independent-mask BCE.
Thin annotations supervise ownership via the ORIGINAL sparse V2 labels.
They never label unannotated input ink as background.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .reference_model import partition_loss
from .stroke_structure import soft_skeleton, soft_occupancy, _crossings

ARCHITECTURE = 'reference_v2_topology_loss_v1'


@dataclass(frozen=True)
class JunctionConfig:
    weight: float = .05
    every: int = 4
    max_strokes: int = 4
    threshold: float = .5
    temperature: float = .08
    skeleton_iterations: int = 12
    radius: int = 1

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('junction weight must be finite and nonnegative')
        if not 0 < self.threshold < 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Invalid junction occupancy settings')
        if any(not isinstance(v, int) or v < 1 for v in
               (self.every, self.max_strokes, self.skeleton_iterations, self.radius)):
            raise ValueError('Junction intervals, limits, iterations and radius must be positive integers')


class TopologyDataset(Dataset):
    """Original V2 sample plus ACTUAL masks, used only for auxiliary topology."""
    def __init__(self, rows, size):
        from .train_reference import StrokeDataset
        self.base = StrokeDataset(rows, size, include_topology=True)
        self.rows = rows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        base = self.base[index]
        target, valid = base[4:6]
        if target.shape != base[2].shape or valid.shape != base[0].shape:
            raise ValueError('Actual topology masks/valid region shapes do not match V2 sample')
        for tensor in (target, valid):
            if not torch.isfinite(tensor).all() or (tensor < 0).any() or (tensor > 1).any():
                raise ValueError('Actual topology masks must be finite in [0,1]')
        return base


def collate_topology(samples):
    from .train_reference import collate
    base = collate([sample[:4] for sample in samples])
    count = base[2].shape[1]
    targets = torch.stack([F.pad(s[4], (0, 0, 0, 0, 0, count-len(s[4]))) for s in samples])
    return (*base, targets, torch.stack([s[5] for s in samples]))


def junction_map(masks, cfg):
    occupancy = soft_occupancy(masks, cfg.threshold, cfg.temperature)
    skeleton = soft_skeleton(occupancy, cfg.skeleton_iterations)
    response = F.relu(_crossings(skeleton, cfg.radius)-2).square()
    return F.avg_pool2d(response, 3, 1, 1)


def junction_discrepancy(output, target_masks, valid_pixels, cfg, step=0):
    """Compare actual branch evidence; never force all strokes to have no branches.

    Rotate through at most max_strokes present annotations per application.
    No per-image contrast normalization or dense thin-mask matching is used.
    """
    pred = output['masks'].float()
    target = target_masks.to(pred).detach()
    valid = valid_pixels.to(pred).detach()
    if target.shape != pred.shape or valid.shape != pred[:, :1].shape:
        raise ValueError('Junction targets/valid region must match prediction shapes')
    eligible = output['active'] & ((target*valid).sum((-2,-1)) > 0)
    indices = eligible.nonzero(as_tuple=False)
    if not len(indices):
        return pred.sum()*0
    start = ((step//cfg.every)*cfg.max_strokes) % len(indices)
    selected = indices[(torch.arange(min(cfg.max_strokes,len(indices)),device=pred.device)+start) % len(indices)]
    b, k = selected.unbind(1)
    p, t, v = pred[b,k,None], target[b,k,None], valid[b]
    # Ignore unknown-region boundaries and image edges at the operator's support.
    margin = cfg.skeleton_iterations + cfg.radius + 2
    known = 1-F.max_pool2d(F.pad(1-v,(margin,)*4,value=1),2*margin+1,1)
    pmap = junction_map(p*v, cfg)
    with torch.no_grad():
        tmap = junction_map(t*v, cfg)
        length = soft_skeleton(t*v,cfg.skeleton_iterations).sum((1,2,3)).clamp_min(1)
    return (((pmap-tmap).abs()*known).sum((1,2,3))/length).mean()


def v2_topology_loss(output, labels, target_masks=None, valid_pixels=None, config=None, step=0):
    cfg = config or JunctionConfig()
    base = partition_loss(output, labels)
    zero = output['masks'].sum()*0
    if cfg.weight == 0 or step % cfg.every:
        return {**base, 'junction': zero, 'junction_applied': False}
    if target_masks is None or valid_pixels is None:
        raise ValueError('Actual input masks and known-region metadata are required')
    # FP32 topology operators for stable differences, including under AMP.
    with torch.autocast(device_type=output['masks'].device.type, enabled=False):
        junction = junction_discrepancy(output, target_masks, valid_pixels, cfg, step)
    return {**base, 'loss': base['loss']+cfg.weight*junction,
            'junction': junction, 'junction_applied': True}
