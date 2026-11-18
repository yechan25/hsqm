"""Calibrate deletion corridors of closed fallback strokes from original ink.

Only deletion radii change. Paths, restoration, fitting and the image itself do
not change. Robust samples near the expected path limit the influence of spurs.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .boundary_guarded_repair import BoundaryGuardedRepair


@dataclass(frozen=True)
class ObservedWidthConfig:
    minimum_samples: int = 12
    minimum_path_coverage: float = .7
    maximum_increase: float = 4.
    maximum_half_width: float = 8.

    def __post_init__(self):
        if type(self.minimum_samples) is not int or self.minimum_samples < 3:
            raise ValueError('minimum_samples must be an integer >=3')
        if not math.isfinite(self.minimum_path_coverage) or not 0 < self.minimum_path_coverage <= 1:
            raise ValueError('minimum_path_coverage must be in (0,1]')
        for value in (self.maximum_increase, self.maximum_half_width):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Width limits must be finite and positive')


class ObservedWidthRepair(BoundaryGuardedRepair):
    def __init__(self, source, extractor, records, *, width_config=None, **kwargs):
        self.width_config = width_config or ObservedWidthConfig()
        super().__init__(source, extractor, records, **kwargs)
        cfg = self.width_config
        self.width_calibration = []
        # Calibrate once from the untouched source, never from later predictions.
        for slot, (record, raw) in enumerate(zip(self.records, records), 1):
            for k, info in enumerate(raw.get('fit_metadata', [{} for _ in raw['strokes']])):
                old = float(record['radii'][k, 0, 0])
                row = dict(slot=slot, stroke=k+1, old_radius=old, new_radius=old,
                           method=info.get('method'), reason='not_closed_fallback')
                if (info.get('reference_path', {}).get('closed')
                        and 'fallback' in info.get('method', '')):
                    path = record['distances'][k] == 0
                    observed = raw['extracted_masks'][k] > .35
                    coverage = int((path & observed).sum()) / max(int(path.sum()), 1)
                    # Measure at observed centers, anchored near the expected loop.
                    # Sampling on a slightly offset prototype underestimates width.
                    centers = skeletonize(observed)
                    supported = centers & (record['distances'][k] <= old+cfg.maximum_increase)
                    count = int(supported.sum())
                    row.update(samples=count, path_coverage=coverage)
                    if count >= cfg.minimum_samples and coverage >= cfg.minimum_path_coverage:
                        values = distance_transform_edt(observed)[supported]
                        estimate = float(np.median(values))
                        new = max(old, min(estimate, old+cfg.maximum_increase, cfg.maximum_half_width))
                        record['radii'][k, 0, 0] = new
                        row.update(observed_half_width=estimate, new_radius=new,
                                   reason='calibrated' if new > old else 'no_increase_needed')
                    else:
                        row['reason'] = 'insufficient_visible_path_support'
                self.width_calibration.append(row)
        self.metadata.update(kind='closed_fallback_observed_width_v1',
                             observed_width_config=asdict(cfg), width_calibration=self.width_calibration)
