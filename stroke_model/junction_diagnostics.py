"""Non-differentiable branch verification on absolute level-set skeletons.

This independent diagnostic must never be added to a training/guidance loss.
It checks the differentiable surrogate against actual thinned binary masks.
"""
import numpy as np
import torch
from scipy.ndimage import label, center_of_mass
from skimage.morphology import skeletonize

from .stroke_structure import _crossings


def diagnose_junction(mask, seed_region, thresholds):
    ink = np.asarray(mask, dtype=np.float32)
    seed = np.asarray(seed_region, dtype=bool)
    if ink.ndim != 2 or seed.shape != ink.shape:
        raise ValueError('mask and seed_region must have matching H,W shapes')
    levels, artifacts = [], []
    for threshold in thresholds:
        foreground = ink >= threshold
        components, _ = label(foreground, structure=np.ones((3, 3)))
        marked = np.unique(components[foreground & seed])
        marked = marked[marked > 0]
        attached = np.isin(components, marked) if len(marked) else np.zeros_like(foreground)
        skeleton = skeletonize(attached)
        crossing = _crossings(torch.tensor(skeleton, dtype=torch.float32)[None, None], 1)[0, 0].numpy()
        branches = skeleton & (crossing >= 3)
        branch_labels, count = label(branches, structure=np.ones((3, 3)))
        centers = [tuple(float(x) for x in point) for point in
                   center_of_mass(branches, branch_labels, range(1, count + 1))] if count else []
        levels.append({'threshold': float(threshold), 'branch_regions': int(count),
                       'branch_pixels': int(branches.sum()), 'skeleton_pixels': int(skeleton.sum())})
        artifacts.append({'skeleton': skeleton, 'branches': branches, 'centers': centers})
    # Select the clearest branch-containing level; preserve the full level table.
    selected = max(range(len(levels)), key=lambda i: (levels[i]['branch_regions'],
                                                     levels[i]['skeleton_pixels']))
    hits = sum(item['branch_regions'] > 0 for item in levels)
    return {'branch_regions_max': max(item['branch_regions'] for item in levels),
            'levels_with_branches': hits, 'tested_levels': len(levels),
            'selected_threshold': levels[selected]['threshold'], 'levels': levels,
            **artifacts[selected]}
