"""Non-differentiable branch verification on absolute level-set skeletons.

This independent diagnostic must never be added to a training/guidance loss.
It checks the differentiable surrogate against actual thinned binary masks.
"""
import heapq
import math
import numpy as np
from scipy.ndimage import label, center_of_mass, distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize

def skeleton_graph(skeleton):
    """8-neighbor graph, omitting diagonal shortcuts around occupied corners."""
    pixels = set(map(tuple, np.argwhere(skeleton)))
    graph = {}
    for y, x in pixels:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y+dy, x+dx)
                if (not dy and not dx) or q not in pixels:
                    continue
                if dy and dx and ((y+dy, x) in pixels or (y, x+dx) in pixels):
                    continue
                neighbors.append(q)
        graph[y, x] = neighbors
    return graph


def resolved_arms(graph, root, radius):
    """Count independent arms that leave a local junction core, not pixel tips.

    Paths joining each other inside the core count once. Endpoints inside the
    core are unresolved; this does NOT establish that a short feature is normal.
    """
    distances = {root: 0.}
    queue = [(0., root)]
    exits = set()
    while queue:
        distance, p = heapq.heappop(queue)
        if distance != distances[p]:
            continue
        for q in graph[p]:
            step = math.hypot(q[0]-p[0], q[1]-p[1])
            value = distance + step
            if distances.get(q, math.inf) <= value:
                continue
            if value > radius:
                exits.add(p)
            elif value < distances.get(q, math.inf):
                distances[q] = value
                heapq.heappush(queue, (value, q))
    remaining = set(distances) - {root}
    count = 0
    while remaining:
        start = remaining.pop()
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in graph[p]:
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        reaches = bool(component & exits) or any(distances[p] >= radius-1e-6 for p in component)
        if reaches and any(p in component for p in graph[root]):
            count += 1
    return count


def _regions(candidates, merge_radius):
    if not candidates.any():
        return 0, []
    if merge_radius:
        yy, xx = np.mgrid[-merge_radius:merge_radius+1, -merge_radius:merge_radius+1]
        expanded = binary_dilation(candidates, (xx*xx+yy*yy) <= merge_radius**2)
    else:
        expanded = candidates
    regions, count = label(expanded, structure=np.ones((3, 3)))
    centers = [tuple(float(v) for v in point) for point in
               center_of_mass(candidates, regions, range(1, count+1))]
    return int(count), centers


def diagnose_junction(mask, seed_region, thresholds, *, arm_scale=1.25,
                      arm_offset=1., min_arm_length=2., merge_radius=1,
                      match_radius=2, min_levels=2):
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
        graph = skeleton_graph(skeleton)
        raw = np.zeros_like(skeleton)
        branches = np.zeros_like(skeleton)
        radius_field = distance_transform_edt(attached)
        for p in graph:
            if len(graph[p]) >= 3:
                raw[p] = True
                radius = max(min_arm_length, arm_scale*radius_field[p] + arm_offset)
                if resolved_arms(graph, p, radius) >= 3:
                    branches[p] = True
        count, centers = _regions(branches, merge_radius)
        uncertain = raw & ~branches
        uncertain_count, uncertain_centers = _regions(uncertain, merge_radius)
        levels.append({'threshold': float(threshold), 'branch_regions': int(count),
                       'unresolved_regions': uncertain_count,
                       'branch_pixels': int(branches.sum()), 'skeleton_pixels': int(skeleton.sum())})
        artifacts.append({'skeleton': skeleton, 'branches': branches, 'raw': raw, 'centers': centers,
                          'uncertain': uncertain, 'uncertain_centers': uncertain_centers})
    yy, xx = np.mgrid[-match_radius:match_radius+1, -match_radius:match_radius+1]
    footprint = (xx*xx+yy*yy) <= match_radius**2
    votes = sum(binary_dilation(a['branches'], footprint).astype(np.int16) for a in artifacts)
    stable_region = votes >= min_levels
    candidate_max = max(item['branch_regions'] for item in levels)
    for item, artifact in zip(levels, artifacts):
        artifact['branches'] = artifact['branches'] & stable_region
        artifact['uncertain'] = artifact['raw'] & ~artifact['branches']
        count, centers = _regions(artifact['branches'], merge_radius)
        uncertain_count, uncertain_centers = _regions(artifact['uncertain'], merge_radius)
        item['branch_regions'] = count
        item['unresolved_regions'] = uncertain_count
        item['branch_pixels'] = int(artifact['branches'].sum())
        artifact['centers'], artifact['uncertain_centers'] = centers, uncertain_centers
    # Select the clearest branch-containing level; preserve the full level table.
    selected = max(range(len(levels)), key=lambda i: (levels[i]['branch_regions'],
                                                     levels[i]['skeleton_pixels']))
    hits = sum(item['branch_regions'] > 0 for item in levels)
    return {'branch_regions_max': max(item['branch_regions'] for item in levels),
            'candidate_regions_max': candidate_max,
            'unresolved_regions_max': max(item['unresolved_regions'] for item in levels),
            'levels_with_branches': hits, 'tested_levels': len(levels),
            'selected_threshold': levels[selected]['threshold'], 'levels': levels,
            **artifacts[selected]}
