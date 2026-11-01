"""Reference-aware unexplained ink, measured on the INPUT, not assigned masks.

Reference centerlines define tolerant support regions with bounded translations. This
detects both attached and detached added ink even if every input pixel was
assigned to an existing stroke by softmax. It is not an extra-stroke counter.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, shift
from skimage.morphology import skeletonize
import torch
from torch import nn


class ExtraInkMetric(nn.Module):
    def __init__(self, aligned_reference_strokes, radius=8., transition=0.5,
                 min_ink_mass=1., reference_threshold=0.5, alignment=6,
                 alignment_step=2, alignment_temperature=.01, alignment_prior=.001,
                 alignment_sigma=2.):
        super().__init__()
        if not all(math.isfinite(x) for x in (radius, transition, min_ink_mass, reference_threshold)):
            raise ValueError('Extra ink settings must be finite')
        if radius < 0 or transition <= 0 or min_ink_mass <= 0 or not 0 < reference_threshold < 1:
            raise ValueError('Invalid extra ink radius, transition, mass or threshold')
        if (not isinstance(alignment, int) or alignment < 0 or not isinstance(alignment_step, int)
                or alignment_step < 1 or not math.isfinite(alignment_temperature)
                or alignment_temperature <= 0 or not math.isfinite(alignment_prior) or alignment_prior < 0
                or not math.isfinite(alignment_sigma) or alignment_sigma <= 0):
            raise ValueError('Invalid bounded alignment settings')
        reference = torch.as_tensor(aligned_reference_strokes).detach().float().cpu().numpy()
        if reference.ndim != 3 or not np.isfinite(reference).all() or (reference < 0).any() or (reference > 1).any():
            raise ValueError('Reference must be finite K,H,W ink in [0,1]')
        if not len(reference) or any(not (s > reference_threshold).any() for s in reference):
            raise ValueError('Keep every expected reference stroke; empty reference strokes are invalid')
        # Fixed translation banks; prediction-dependent weights stay in torch.
        offsets = sorted(set(range(-alignment, alignment+1, alignment_step)) | {0, alignment})
        coords = [(dy, dx) for dy in offsets for dx in offsets]
        matches, supports = [], []
        for stroke in reference:
            distance = distance_transform_edt(~skeletonize(stroke > reference_threshold))
            match = np.exp(-.5*(distance/alignment_sigma)**2)
            support = 1/(1+np.exp(np.clip((distance-radius)/transition, -80, 80)))
            match_bank, support_bank = [], []
            for dy, dx in coords:
                moved = shift(match, (dy, dx), order=0, mode='constant')
                match_bank.append(moved/max(moved.sum(), 1e-8))
                support_bank.append(shift(support, (dy, dx), order=0, mode='constant'))
            matches.append(match_bank)
            supports.append(support_bank)
        self.register_buffer('matches', torch.tensor(np.asarray(matches), dtype=torch.float32))
        self.register_buffer('supports', torch.tensor(np.asarray(supports), dtype=torch.float32))
        self.register_buffer('shift_cost', torch.tensor([alignment_prior*(dy*dy+dx*dx)
                                                       for dy, dx in coords], dtype=torch.float32))
        self.alignment_temperature = alignment_temperature
        self.min_ink_mass = min_ink_mass

    def forward(self, image):
        image = torch.as_tensor(image)
        if image.ndim == 2:
            image = image[None, None]
        elif image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[1] != 1 or image.shape[-2:] != self.supports.shape[-2:]:
            raise ValueError('image must have shape H,W or 1,H,W or B,1,H,W matching the reference')
        if not image.is_floating_point():
            image = image.float()
        if image.device != self.supports.device:
            raise ValueError('Move ExtraInkMetric to the image device')
        if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
            raise ValueError('Image must be finite absolute ink in [0,1]')
        matches = self.matches.to(dtype=image.dtype)
        supports = self.supports.to(dtype=image.dtype)
        alignment_scores = torch.einsum('bchw,knhw->bkn', image, matches)-self.shift_cost
        weights = torch.softmax(alignment_scores/self.alignment_temperature, dim=-1)
        per_stroke_support = torch.einsum('bkn,knhw->bkhw', weights, supports)
        support = 1-(1-per_stroke_support).prod(1, keepdim=True)
        extra_map = image * (1-support)
        extra_mass = extra_map.flatten(1).sum(1)
        fraction = extra_mass / image.flatten(1).sum(1).clamp_min(self.min_ink_mass)
        return {'extra': fraction, 'extra_map': extra_map, 'support': support, 'alignment_weights': weights,
                'extra_mass': extra_mass}
