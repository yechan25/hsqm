"""Local repair proposal and fixed editing objective; no inference clean target.

These are untrained building blocks, not a validated diffusion integration.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .error_localizer import block, _unit


class LocalRepairProposal(nn.Module):
    architecture = 'local_repair_proposal_v1'

    def __init__(self, feature_width=32):
        super().__init__()
        if feature_width < 8 or feature_width % 4:
            raise ValueError('Feature width must be a multiple of 4, at least 8')
        self.feature_width = feature_width
        self.refine = nn.Sequential(block(feature_width + 4, feature_width),
                                    block(feature_width, feature_width))
        self.residual = nn.Conv2d(feature_width, 3, 1)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)

    def forward(self, rgb, features, error_probability):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('RGB must be B,3,H,W')
        if features.shape != (rgb.shape[0], self.feature_width, *rgb.shape[-2:]):
            raise ValueError('Invalid localizer features')
        if error_probability.shape != rgb[:, :1].shape:
            raise ValueError('Invalid edit probability shape')
        _unit(rgb, 'RGB'); _unit(error_probability, 'Error probability')
        if not torch.isfinite(features).all():raise ValueError('Nonfinite features')
        # Do not train the detector to hide errors to reduce reconstruction loss.
        x = self.refine(torch.cat((rgb, features.detach(), error_probability.detach()), 1))
        return (rgb + self.residual(x).tanh()).clamp(0, 1)


def _masked_mean(error, mask):
    return (error.mean(1, keepdim=True) * mask).sum() / mask.sum().clamp_min(1)


def proposal_loss(proposal, corrupted, clean, edit_target, valid, identity_weight=1.):
    """Same-style CLEAN image is training supervision only, never test input."""
    if proposal.shape != corrupted.shape or clean.shape != proposal.shape:
        raise ValueError('Matching training RGB images required')
    if edit_target.shape != proposal[:, :1].shape or valid.shape != edit_target.shape:
        raise ValueError('Matching B,1,H,W masks required')
    if not math.isfinite(identity_weight) or identity_weight < 0:raise ValueError('Invalid identity weight')
    for value, name in ((clean, 'Clean RGB'), (corrupted, 'Corrupted RGB'),
                        (edit_target, 'Edit labels'), (valid, 'Validity')):
        _unit(value, name)
    repaired = _masked_mean((proposal - clean).abs(), edit_target * valid)
    identity = _masked_mean((proposal - corrupted).abs(), (1 - edit_target) * valid)
    return dict(loss=repaired + identity_weight * identity, repair=repaired, identity=identity)


class FixedRepairPlan(nn.Module):
    """Freeze predicted edit area and proposal for a downstream latent optimizer.

    Caller supplies a validated/calibrated mask. No automatic confidence or
    arbitrary threshold is claimed. Newly initialized models must not enable it.
    """
    def __init__(self, generated_rgb, predicted_proposal, edit_mask, keep_weight=1.):
        super().__init__()
        if generated_rgb.ndim != 4 or generated_rgb.shape[1] != 3 or generated_rgb.shape != predicted_proposal.shape:
            raise ValueError('Matching B,3,H,W source/proposal required')
        if edit_mask.shape != generated_rgb[:, :1].shape:
            raise ValueError('Edit mask must be B,1,H,W')
        if not math.isfinite(keep_weight) or keep_weight < 0:raise ValueError('Invalid keep weight')
        for value, name in ((generated_rgb, 'Generated RGB'), (predicted_proposal, 'Proposal'), (edit_mask, 'Edit mask')):
            _unit(value, name)
        self.keep_weight = keep_weight
        self.register_buffer('source', generated_rgb.detach().clone())
        self.register_buffer('target', predicted_proposal.detach().clone())
        self.register_buffer('edit_mask', edit_mask.detach().clone())

    def forward(self, decoded_rgb):
        if decoded_rgb.shape != self.source.shape:raise ValueError('Decoder shape changed')
        # Do not clamp decoder outputs: overshoot must retain a corrective gradient.
        edit = _masked_mean(F.smooth_l1_loss(decoded_rgb, self.target, reduction='none'), self.edit_mask)
        keep = _masked_mean(F.smooth_l1_loss(decoded_rgb, self.source, reduction='none'), 1 - self.edit_mask)
        return dict(loss=edit + self.keep_weight * keep, edit=edit, keep=keep)

    def composite(self, edited_rgb):
        """Explicit RGB compositing, not proof of preservation by diffusion."""
        if edited_rgb.shape != self.source.shape:raise ValueError('Edited shape changed')
        return self.source * (1 - self.edit_mask) + edited_rgb * self.edit_mask
