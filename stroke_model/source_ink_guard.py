"""Native-resolution excess ink checks independent of Topo channel ownership."""
import torch
from torch import nn
from torch.nn import functional as F

from .source_quality_v5 import foreground_evidence


class SourceInkGuard(nn.Module):
    def __init__(self,reference_rgb,support_radius=1,patch_radius=2, direct=False):
        super().__init__()
        if any(type(r) is not int or r<0 for r in (support_radius,patch_radius)):
            raise ValueError('Ink guard radii must be nonnegative integers')
        self.support_radius=support_radius;self.patch_radius=patch_radius
        self.direct=direct
        ink=self.ink(reference_rgb).detach()
        self.register_buffer('reference_ink',ink.clone())
        self.register_buffer('support',F.max_pool2d(ink,2*support_radius+1,1,support_radius))

    @staticmethod
    def ink(rgb):
        if rgb.ndim!=4 or rgb.shape[:2]!=(1,3):raise ValueError('Expected one RGB sheet')
        rgb=rgb.clamp(0,1)
        return 1-(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)

    def forward(self,rgb):
        ink=self.ink(rgb)
        if ink.shape!=self.reference_ink.shape:raise ValueError('Ink guard sheet size changed')
        extra=F.relu(ink-self.support)
        missing=F.relu(self.reference_ink-ink)
        excess=(extra if self.direct else F.relu(foreground_evidence(ink)-self.support)).square()
        local=F.avg_pool2d(excess,2*self.patch_radius+1,1,self.patch_radius)
        denom=self.reference_ink.sum().clamp_min(1)
        loss=excess.sum()/denom
        if self.direct:
            loss=loss+missing.square().sum()/denom
        return dict(loss=loss,local_excess=local,extra_map=extra,missing_map=missing)


def local_ink_change(before,after,anchor,allowance):
    """Do not let many individually small guidance steps accumulate new blobs.

    Tighten the anchor with the current estimate. Changes inherited from U-Net
    may remain above the anchor; guidance cannot increase those further.
    """
    anchor=before.detach() if anchor is None else torch.minimum(anchor,before.detach())
    limit=torch.maximum(before.detach(),anchor+allowance)
    violation=float((after.detach()-limit).amax().clamp_min(0))
    increase=float((after.detach()-before.detach()).amax().clamp_min(0))
    return violation<=1e-6,increase,violation,anchor
