"""Experimental boundary uncertainty guard for the existing cyclic repair.

Only deletion-map construction changes. Extractor, fitted paths, restoration,
loss weights, gradient step and DDIM transfer stay with RestoreSettleErase.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import binary_dilation

from .restore_settle_erase import RestoreSettleErase


@dataclass(frozen=True)
class BoundaryGuardConfig:
    uncertainty_pixels: float = 1.
    responsibility_floor: float = .5

    def __post_init__(self):
        if not math.isfinite(self.uncertainty_pixels) or self.uncertainty_pixels < 0:
            raise ValueError('uncertainty_pixels must be finite and nonnegative')
        if not math.isfinite(self.responsibility_floor) or not 0 <= self.responsibility_floor < 1:
            raise ValueError('responsibility_floor must be in [0,1)')


class BoundaryGuardedRepair(RestoreSettleErase):
    def __init__(self, *args, boundary_config=None, **kwargs):
        self.boundary_config = boundary_config or BoundaryGuardConfig()
        self.boundary_history = []
        super().__init__(*args, **kwargs)
        self.metadata.update(kind='restore_settle_erase_boundary_guard_v1',
                             boundary_guard=asdict(self.boundary_config))

    @torch.no_grad()
    def refresh_erase(self, image):
        erase = torch.zeros_like(self.erase)
        rows = []
        cfg = self.boundary_config
        for slot, r in enumerate(self.records, 1):
            ink = self.crop(self.ink(image).clamp(0, 1), r['box'], r['size'])
            p = self.extractor(ink, r['reference'], r['strokes'])['probabilities'][0].float().cpu().numpy()
            if not np.isfinite(p).all() or p.shape != r['distances'].shape:
                raise ValueError('Invalid stroke extraction')
            # The paths and their calibrated radii stay fixed. This is an
            # uncertainty allowance, not a new target thickness or wider stroke.
            radius = r['radii'] + cfg.uncertainty_pixels
            outside = np.clip((r['distances'] - radius - 1.5) / 1.5, 0, 1)
            safe = (r['distances'] <= radius + 1.).any(0)
            responsibility = (p * outside).sum(0)
            observed = ink[0, 0].cpu().numpy()
            eligible = (responsibility > cfg.responsibility_floor) & ~safe
            seeds = (observed > .15) & eligible
            grown = binary_dilation(seeds, iterations=1)
            # Recheck every grown pixel. A faint boundary seed must not turn an
            # otherwise ineligible neighboring body pixel into a deletion target.
            weight = grown * responsibility * eligible
            patch = self.tensor(weight)
            gate = self.tensor(eligible.astype(np.float32))
            x0, y0, x1, y1 = r['box']
            shape = (y1-y0, x1-x0)
            projected = F.interpolate(patch, shape, mode='bilinear', align_corners=False)
            gate = F.interpolate(gate, shape, mode='nearest') > .5
            erase[..., y0:y1, x0:x1] = projected * gate
            rows.append(dict(slot=slot, seeds=int(seeds.sum()),
                black_seeds=int((seeds & (observed > .5)).sum()),
                grown_rejected=int((grown & ~eligible).sum()),
                deletion_pixels_model=int((weight > .01).sum())))
        self.erase = erase * (self.restore <= 0).float() * (1-self.path_union)
        self.update_support()
        self.boundary_history.append(dict(refresh=len(self.boundary_history), slots=rows,
            sheet_deletion_pixels=int((self.erase > .01).sum())))
