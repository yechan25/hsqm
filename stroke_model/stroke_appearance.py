"""Differentiable reference-relative edge/local-ink auxiliaries for soft strokes.

These are appearance losses, not topology scores or physical width estimates.
They deliberately bias generation toward the reference's shape and style.
"""
from dataclasses import dataclass,asdict
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AppearanceConfig:
    mass_radii: tuple=(1,2)
    mass_allowance: float=.03
    contour_allowance: float=.05
    contour_weight: float=.5
    priority_scale: float=.1

    def __post_init__(self):
        if not self.mass_radii or any(type(r) is not int or r<1 for r in self.mass_radii):
            raise ValueError('Appearance mass radii must be positive integers')
        for name,value in asdict(self).items():
            if name!='mass_radii' and (not math.isfinite(value) or value<0):
                raise ValueError('Appearance settings must be finite and nonnegative')
        if self.priority_scale<=0:raise ValueError('Appearance priority scale must be positive')


def _average(value,radius):
    return F.avg_pool2d(value,2*radius+1,1,radius,count_include_pad=False)


class StrokeAppearanceLoss(nn.Module):
    def __init__(self,reference,config=None):
        super().__init__()
        self.config=config or AppearanceConfig()
        reference=torch.as_tensor(reference).detach().float()
        if (reference.ndim!=3 or not torch.isfinite(reference).all()
                or (reference<0).any() or (reference>1).any() or (reference.sum((-1,-2))<=0).any()):
            raise ValueError('Appearance reference needs nonempty K,H,W soft strokes in [0,1]')
        self.register_buffer('reference',reference[None])
        self.register_buffer('mass',reference.sum((-1,-2))[None].clamp_min(1))

    def forward(self,masks,return_maps=False):
        if masks.ndim==3:masks=masks[None]
        if masks.ndim!=4 or masks.shape[1:]!=self.reference.shape[1:]:
            raise ValueError('Appearance mask shape must match the reference')
        if not torch.isfinite(masks).all() or (masks<0).any() or (masks>1).any():
            raise ValueError('Appearance masks must be finite and in [0,1]')
        cfg=self.config
        # Divide the contrast allowance by K so a many-channel partition cannot
        # hide a visible artifact simply by spreading it across stroke channels.
        channels=masks.shape[1]
        local=torch.stack([F.relu((_average(masks,r)-_average(self.reference,r)).abs()
                                 -cfg.mass_allowance/channels).square() for r in cfg.mass_radii]).mean(0)
        edge=masks-_average(masks,1)
        ref_edge=self.reference-_average(self.reference,1)
        contour=F.relu((edge-ref_edge).abs()-cfg.contour_allowance/channels).square()
        local_value=local.sum((-1,-2))/self.mass
        contour_value=contour.sum((-1,-2))/self.mass
        result=dict(loss=local_value+cfg.contour_weight*contour_value,
                    local_mass=local_value,contour=contour_value)
        if return_maps:result.update(local_mass_map=local.detach(),contour_map=contour.detach())
        return result
