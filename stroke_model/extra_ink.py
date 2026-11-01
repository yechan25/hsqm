"""Reference-aware unexplained ink, measured on the INPUT, not assigned masks.

Fixed aligned reference centerlines define a tolerant support region. This
detects both attached and detached added ink even if every input pixel was
assigned to an existing stroke by softmax. It is not an extra-stroke counter.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
from torch import nn


class ExtraInkMetric(nn.Module):
    def __init__(self, aligned_reference_strokes, radius=8., transition=0.5,
                 min_ink_mass=1., reference_threshold=0.5):
        super().__init__()
        if not all(math.isfinite(x) for x in (radius, transition, min_ink_mass, reference_threshold)):
            raise ValueError('Extra ink settings must be finite')
        if radius < 0 or transition <= 0 or min_ink_mass <= 0 or not 0 < reference_threshold < 1:
            raise ValueError('Invalid extra ink radius, transition, mass or threshold')
        reference = torch.as_tensor(aligned_reference_strokes).detach().float().cpu().numpy()
        if reference.ndim != 3 or not np.isfinite(reference).all() or (reference < 0).any() or (reference > 1).any():
            raise ValueError('Reference must be finite K,H,W ink in [0,1]')
        if not len(reference) or any(not (s > reference_threshold).any() for s in reference):
            raise ValueError('Keep every expected reference stroke; empty reference strokes are invalid')
        centerline = np.logical_or.reduce([skeletonize(s > reference_threshold) for s in reference])
        distance = distance_transform_edt(~centerline)
        distance = torch.tensor(distance, dtype=torch.float32)[None, None]
        self.register_buffer('support', torch.sigmoid((radius-distance)/transition))
        self.min_ink_mass = min_ink_mass

    def forward(self, image):
        image = torch.as_tensor(image)
        if image.ndim == 2:
            image = image[None, None]
        elif image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[1:] != self.support.shape[1:]:
            raise ValueError('image must have shape H,W or 1,H,W or B,1,H,W matching the reference')
        if not image.is_floating_point():
            image = image.float()
        if image.device != self.support.device:
            raise ValueError('Move ExtraInkMetric to the image device')
        if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
            raise ValueError('Image must be finite absolute ink in [0,1]')
        extra_map = image * (1-self.support)
        extra_mass = extra_map.flatten(1).sum(1)
        fraction = extra_mass / image.flatten(1).sum(1).clamp_min(self.min_ink_mass)
        return {'extra': fraction, 'extra_map': extra_map, 'support': self.support,
                'extra_mass': extra_mass}
