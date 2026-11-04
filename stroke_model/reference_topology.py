"""Experimental globally aligned, topology-supervised independent extraction.

Topology is matched to ACTUAL input annotations, never to template geometry or
an unconditional zero-branch target. This module requires new training weights.
"""
from dataclasses import dataclass, asdict, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from .reference_independent import IndependentStrokeModel, IndependentLossConfig, independent_stroke_loss
from .reference_model import block
from .stroke_structure import soft_skeleton, soft_occupancy, _crossings


@dataclass(frozen=True)
class AlignmentConfig:
    translation_fraction: float = .25  # fraction of image width/height
    log_scale: float = .4
    rotation_degrees: float = 20.
    shear: float = .2

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Alignment bounds must be finite and nonnegative')


class TopologyStrokeModel(IndependentStrokeModel):
    architecture = 'reference_topology_aligned_v1'

    def __init__(self, *args, alignment_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignment_config = alignment_config or AlignmentConfig()
        self.global_aligner = nn.Sequential(block(2, 16, 2), block(16, 32, 2),
                                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 6))
        nn.init.zeros_(self.global_aligner[-1].weight)
        nn.init.zeros_(self.global_aligner[-1].bias)

    def warm_start(self, state):
        # Accept only the complete V2 base. New modules remain explicitly untrained.
        expected = self.state_dict()
        required = {k for k in expected if not k.startswith(('context_', 'global_aligner.'))}
        if set(state) != required:
            raise ValueError('Warm start requires a complete compatible V2 state')
        self.load_state_dict(state, strict=False)
        self.independent_checkpoint_loaded = False

    def affine_matrix(self, image, reference):
        cfg = self.alignment_config
        tx, ty, sx, sy, angle, shear = self.global_aligner(torch.cat((reference, image), 1)).tanh().unbind(1)
        sx, sy = (sx * cfg.log_scale).exp(), (sy * cfg.log_scale).exp()
        angle = angle * (cfg.rotation_degrees * math.pi / 180)
        sh = shear * cfg.shear
        c, s = angle.cos(), angle.sin()
        # Sampling transform: output/input canvas coordinate -> reference coordinate.
        return torch.stack((c*sx, c*sh-s*sy, tx*2*cfg.translation_fraction,
                            s*sx, s*sh+c*sy, ty*2*cfg.translation_fraction), 1).reshape(-1, 2, 3)

    def forward(self, image, reference, strokes, active=None, *, allow_untrained=False):
        if not self.training and not self.independent_checkpoint_loaded and not allow_untrained:
            raise RuntimeError('Topology extraction requires a trained topology checkpoint')
        if active is None:
            active = strokes.flatten(2).sum(-1) > 0
        theta = self.affine_matrix(image, reference)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        aligned = F.grid_sample(torch.cat((reference, strokes), 1), grid, align_corners=False)
        out = super().forward(image, aligned[:, :1], aligned[:, 1:], active,
                              allow_untrained=allow_untrained)
        return {**out, 'alignment_theta': theta, 'aligned_reference': aligned[:, :1],
                'aligned_strokes': aligned[:, 1:]}


@dataclass(frozen=True)
class TopologyLossConfig:
    base: IndependentLossConfig = field(default_factory=IndependentLossConfig)
    consistency_weight: float = .2
    junction_weight: float = .1
    alignment_weight: float = .05
    # Shared absolute levels; no per-stroke contrast normalization.
    thresholds: tuple = (.3, .5, .7)
    temperature: float = .08
    radii: tuple = (1, 2, 3)
    skeleton_iterations: int = 12

    def __post_init__(self):
        if isinstance(self.base, dict):
            object.__setattr__(self, 'base', IndependentLossConfig(**self.base))
        if not isinstance(self.base, IndependentLossConfig):
            raise ValueError('base must be IndependentLossConfig or a configuration dictionary')
        if any(not math.isfinite(v) or v < 0 for v in
               (self.consistency_weight, self.junction_weight, self.alignment_weight)):
            raise ValueError('Loss weights must be finite and nonnegative')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('temperature must be positive')
        if not self.thresholds or any(not math.isfinite(t) or not 0 < t < 1 for t in self.thresholds):
            raise ValueError('thresholds must lie in (0,1)')
        if not self.radii or any(not isinstance(r, int) or r < 1 for r in self.radii):
            raise ValueError('radii must be positive integers')
        if not isinstance(self.skeleton_iterations, int) or self.skeleton_iterations < 1:
            raise ValueError('skeleton_iterations must be a positive integer')


def junction_signature(masks, config):
    """Soft local branch evidence, not a count or an exact topology guarantee.

    Uses prediction-only geometry: no reference seeds, centerline, or length.
    A matched defective target receives zero discrepancy and is preserved.
    """
    shape = masks.shape
    masks = masks.reshape(-1, 1, *shape[-2:])
    maps = []
    for level in config.thresholds:
        occ = soft_occupancy(masks, level, config.temperature)
        skel = soft_skeleton(occ, config.skeleton_iterations)
        # Neighborhood evidence avoids relying on one skeleton-center pixel.
        response = torch.stack([F.relu(_crossings(skel, r)-2).square()
                                for r in config.radii]).mean(0)
        maps.append(F.avg_pool2d(response, 3, 1, 1))
    return torch.stack(maps).mean(0).reshape(shape)


def topology_stroke_loss(output, target_masks, image, valid_pixels=None, config=None):
    cfg = config or TopologyLossConfig()
    base = independent_stroke_loss(output, target_masks, image, valid_pixels, cfg.base)
    pred = output['masks']
    target = torch.as_tensor(target_masks, device=pred.device, dtype=pred.dtype)
    valid = torch.ones_like(image) if valid_pixels is None else valid_pixels.to(pred)
    active = output['active'][:, :, None, None].to(pred)
    weight = valid * active
    consistency = pred.sum()*0
    # Match signed label transitions, including true stroke boundaries. Merely
    # smoothing nearby ink would merge touching strokes and hide real breaks.
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = (slice(None, -dy), slice(dy, None)) if dy else (slice(None), slice(None))
        x0, x1 = (slice(None, -dx), slice(dx, None)) if dx > 0 else (
            (slice(-dx, None), slice(None, dx)) if dx < 0 else (slice(None), slice(None)))
        a, b = (..., y0, x0), (..., y1, x1)
        support = weight[a]*weight[b]*torch.maximum(target[a], target[b])
        error = ((pred[a]-pred[b])-(target[a]-target[b])).square()
        consistency = consistency + (error*support).sum()/support.sum().clamp_min(1)
    consistency = consistency/4
    # Erode the known region so unknown annotation boundaries cannot fabricate
    # branch evidence; account for skeleton/ring/pooling receptive fields.
    margin = cfg.skeleton_iterations + max(cfg.radii) + 2
    known = 1-F.max_pool2d(F.pad(1-valid, (margin,)*4, value=1), 2*margin+1, 1)
    structural_weight = known*active
    pmap = junction_signature(pred*valid, cfg)
    tmap = junction_signature(target*valid, cfg)
    # Normalize per annotated stroke skeleton length instead of image area.
    length = soft_skeleton(target*valid, cfg.skeleton_iterations).sum((-2,-1)).clamp_min(1)
    discrepancy = ((pmap-tmap).abs()*structural_weight).sum((-2,-1))/length
    junction = (discrepancy*output['active']).sum()/output['active'].sum().clamp_min(1)
    alignment = pred.sum()*0
    if 'alignment_theta' in output:
        identity = pred.new_tensor([[1,0,0],[0,1,0]])
        alignment = (output['alignment_theta']-identity).square().mean()
    return {**base, 'consistency': consistency, 'junction': junction, 'alignment': alignment,
            'loss': base['loss']+cfg.consistency_weight*consistency+
                    cfg.junction_weight*junction+cfg.alignment_weight*alignment}
