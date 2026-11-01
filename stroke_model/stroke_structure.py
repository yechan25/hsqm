"""Reference-aware experimental stroke losses; inputs are absolute ink in [0, 1].

Reference preparation is discrete and cached. All prediction-dependent losses use
PyTorch operations, including min/max subgradients. No predicted components are
discarded. Scores are uncalibrated errors, not human-quality probabilities.
"""
from dataclasses import dataclass, fields
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class StructureConfig:
    # All distances are pixels at the inference resolution (normally 128).
    reference_threshold: float = 0.5
    coverage_radius: int = 2
    required_ink: float = 0.7
    anchor_spacing: int = 10
    anchor_radius: int = 1
    corridor_radius: int = 3
    pair_spans: tuple = (1, 2)
    max_pairs: int = 32
    connectivity_steps: int | None = None  # None: crop area, with convergence check
    skeleton_iterations: int = 12
    junction_radii: tuple = (1, 2, 3)
    junction_margin: float = 0.0  # No font/short-branch exemption by default
    junction_seed_radius: int = 2
    attachment_steps: int | None = None  # None: H*W, with convergence check
    expected_width: float = 5.0  # Optional normal-width prior, NOT skeleton width
    width_radius: int = 4
    width_tolerance: float = 0.25
    missing_weight: float = 1.0
    split_weight: float = 1.0
    junction_weight: float = 1.0
    width_weight: float = 0.0  # Enable only after normal-width calibration
    eps: float = 1e-6

    def __post_init__(self):
        for name in ('coverage_radius', 'anchor_radius', 'corridor_radius',
                     'junction_seed_radius', 'width_radius'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('anchor_spacing', 'max_pairs', 'skeleton_iterations'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.anchor_spacing <= 2 * self.anchor_radius + 1:
            raise ValueError('anchor_spacing must exceed the anchor diameters')
        if not self.pair_spans or any(not isinstance(v, int) or v < 1 for v in self.pair_spans):
            raise ValueError('pair_spans must contain positive integers')
        if not self.junction_radii or any(not isinstance(v, int) or v < 1 for v in self.junction_radii):
            raise ValueError('junction_radii must contain integers >= 1')
        for name in ('connectivity_steps', 'attachment_steps'):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or v < 1):
                raise ValueError(f'{name} must be None or a positive integer')
        if not 0 < self.required_ink <= 1 or not 0 < self.reference_threshold < 1:
            raise ValueError('required_ink must be in (0,1]; reference_threshold in (0,1)')
        if self.expected_width <= 0 or self.eps <= 0 or not 0 <= self.width_tolerance < 1:
            raise ValueError('Invalid width or eps')
        if self.junction_margin < 0 or any(getattr(self, n) < 0 for n in
                ('missing_weight', 'split_weight', 'junction_weight', 'width_weight')):
            raise ValueError('Weights and junction_margin must be nonnegative')
        if any(not math.isfinite(float(getattr(self, f.name))) for f in fields(self)
               if isinstance(getattr(self, f.name), (float, int))):
            raise ValueError('Configuration values must be finite')


def _pool(x, radius, mean=False):
    k = 2 * radius + 1
    return (F.avg_pool2d(x, k, 1, radius) if mean else F.max_pool2d(x, k, 1, radius))


def soft_skeleton(x, iterations):
    """Pooling skeleton from soft-clDice; a surrogate, not exact graph thinning."""
    def erode(v):
        return torch.minimum(-F.max_pool2d(-v, (3, 1), 1, (1, 0)),
                             -F.max_pool2d(-v, (1, 3), 1, (0, 1)))
    def opening(v):
        return F.max_pool2d(erode(v), 3, 1, 1)
    skel = F.relu(x - opening(x))
    for _ in range(iterations):
        x = erode(x)
        delta = F.relu(x - opening(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def reconstruct(mask, seed, steps=None):
    """8-connected grayscale reconstruction; no dilation/closing of mask.

    Returns (reachable strength, converged). None allows H*W iterations, a
    conservative finite-grid bound. Early stopping reads detached values only;
    the returned tensor retains the graph. Explicit small step caps can truncate.
    """
    reached = torch.minimum(mask, seed)
    limit = mask.shape[-2] * mask.shape[-1] if steps is None else steps
    converged = False
    for _ in range(limit):
        updated = torch.minimum(mask, F.max_pool2d(reached, 3, 1, 1))
        if torch.equal(updated.detach(), reached.detach()):
            reached = updated
            converged = True
            break
        reached = updated
    return reached, converged


def _ordered_centerline(binary):
    skel = skeletonize(binary)
    _, count = label(skel, structure=np.ones((3, 3)))
    if count != 1:
        raise ValueError('Reference stroke must have one connected centerline; clean the reference')
    points = [tuple(v) for v in np.argwhere(skel)]
    lookup = set(points)
    neighbors = {}
    for y, x in points:
        adjacent = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if (not dy and not dx) or q not in lookup:
                    continue
                # Avoid triangular shortcut edges at a normal digital corner.
                if dy and dx and ((y + dy, x) in lookup or (y, x + dx) in lookup):
                    continue
                adjacent.append(q)
        neighbors[(y, x)] = adjacent
    if any(len(v) > 2 for v in neighbors.values()):
        raise ValueError('Reference skeleton branches; provide a clean unbranched reference stroke')
    endpoints = sorted(p for p, n in neighbors.items() if len(n) == 1)
    closed = len(points) > 2 and not endpoints
    if len(points) < 5 or (not closed and len(endpoints) != 2):
        raise ValueError('Reference centerline must be a nontrivial open curve or closed ring')
    start = endpoints[0] if endpoints else min(points)
    order, visited, current = [], set(), start
    while current not in visited:
        order.append(current)
        visited.add(current)
        choices = sorted(q for q in neighbors[current] if q not in visited)
        if not choices:
            break
        current = choices[0]
    if len(order) != len(points):
        raise ValueError('Cannot order reference centerline without losing pixels')
    return skel, np.asarray(order), closed


def _disk_region(shape, point, radius):
    yy, xx = np.indices(shape)
    return ((yy - point[0]) ** 2 + (xx - point[1]) ** 2 <= radius ** 2)


class _Pair(nn.Module):
    def __init__(self, corridor, a, b, slices):
        super().__init__()
        for name, value in [('corridor', corridor), ('a', a), ('b', b)]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32)[None, None])
        self.slices = slices


class _Stroke(nn.Module):
    def __init__(self, reference, config):
        super().__init__()
        skel, order, self.closed = _ordered_centerline(reference > config.reference_threshold)
        self.length = len(order)
        self.register_buffer('skeleton', torch.tensor(skel, dtype=torch.float32)[None, None])
        self.register_buffer('ys', torch.tensor(order[:, 0], dtype=torch.long))
        self.register_buffer('xs', torch.tensor(order[:, 1], dtype=torch.long))
        distance = distance_transform_edt(~skel)
        self.register_buffer('seeds', torch.tensor(distance <= config.junction_seed_radius,
                                                   dtype=torch.float32)[None, None])
        normal = distance <= max(0.5, config.expected_width / 2)
        self.register_buffer('normal', torch.tensor(normal, dtype=torch.float32)[None, None])
        anchors = list(range(0, len(order), config.anchor_spacing))
        if not self.closed and anchors[-1] != len(order) - 1:
            anchors.append(len(order) - 1)
        raw_pairs = []
        for span in config.pair_spans:
            for i in range(len(anchors)):
                j = i + span
                if not self.closed and j >= len(anchors):
                    continue
                if self.closed and span >= len(anchors):
                    continue
                begin, end = anchors[i], anchors[j % len(anchors)]
                path = (order[begin:end + 1] if end > begin else
                        np.concatenate((order[begin:], order[:end + 1])))
                if len(path) <= 2 * config.anchor_radius + 2:
                    continue
                # Keep anchor patches disjoint, including on tight curves.
                if np.linalg.norm(order[begin] - order[end]) <= 2 * config.anchor_radius + 1:
                    continue
                segment = np.zeros_like(skel)
                segment[path[:, 0], path[:, 1]] = True
                corridor = distance_transform_edt(~segment) <= config.corridor_radius
                a = _disk_region(skel.shape, order[begin], config.anchor_radius) & corridor
                b = _disk_region(skel.shape, order[end], config.anchor_radius) & corridor
                yy, xx = np.where(corridor)
                slices = (slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
                raw_pairs.append((corridor[slices], a[slices], b[slices], slices))
        if len(raw_pairs) > config.max_pairs:
            ids = np.linspace(0, len(raw_pairs) - 1, config.max_pairs).round().astype(int)
            raw_pairs = [raw_pairs[i] for i in ids]
        self.pairs = nn.ModuleList([_Pair(*args) for args in raw_pairs])
        if not self.pairs:
            raise ValueError('No valid anchor pairs: reduce anchor_spacing/radius for this reference')


def _ring_offsets(radius):
    # Sample a digital square perimeter in cyclic order, with no duplicates.
    return ([( -radius, x) for x in range(-radius, radius)] +
            [(y, radius) for y in range(-radius, radius)] +
            [(radius, x) for x in range(radius, -radius, -1)] +
            [(y, -radius) for y in range(radius, -radius, -1)])


def _crossings(skeleton, radius):
    h, w = skeleton.shape[-2:]
    padded = F.pad(skeleton, (radius,) * 4)
    samples = [padded[..., radius + y:radius + y + h, radius + x:radius + x + w]
               for y, x in _ring_offsets(radius)]
    return sum((a - b).abs() for a, b in zip(samples, samples[1:] + samples[:1])) / 2


class StrokeStructureLoss(nn.Module):
    """Prepare once on fixed aligned reference strokes, then call on live masks.

    reference_strokes: K,H,W. forward: K,H,W or B,K,H,W, floating [0,1].
    Returned per_stroke values are B,K. loss is an EXTRACTOR structural loss,
    not a source-image quality score. Reference deficits are returned separately.
    Closed curves are ordered
    cyclically; local corridors prevent bypassing a gap via the rest of a ring.
    No alignment is estimated here. Geometry must already be in input coordinates.
    """
    def __init__(self, reference_strokes, config=None):
        super().__init__()
        self.config = config or StructureConfig()
        ref = torch.as_tensor(reference_strokes).detach().float().cpu().numpy()
        if ref.ndim != 3 or not np.isfinite(ref).all() or (ref < 0).any() or (ref > 1).any():
            raise ValueError('reference_strokes must be finite K,H,W in [0,1]')
        self.shape = ref.shape
        prepared = []
        for k, stroke in enumerate(ref):
            try:
                prepared.append(_Stroke(stroke, self.config))
            except ValueError as exc:
                raise ValueError(f'Reference stroke {k + 1}: {exc}') from exc
        if not prepared:
            raise ValueError('At least one reference stroke is required')
        self.strokes = nn.ModuleList(prepared)

    def forward(self, masks, return_maps=False):
        if masks.ndim == 3:
            masks = masks[None]
        if masks.ndim != 4 or tuple(masks.shape[1:]) != tuple(self.shape):
            raise ValueError(f'masks must have trailing shape {self.shape}')
        if not masks.is_floating_point():
            raise ValueError('masks must be floating point')
        if not bool(torch.isfinite(masks).all()) or bool((masks < 0).any()) or bool((masks > 1).any()):
            raise ValueError('masks must be finite in [0,1]; do not normalize each stroke')
        if next(self.buffers()).device != masks.device:
            raise ValueError('Move the prepared loss to masks.device with .to(device)')
        cfg = self.config
        metrics = {name: [] for name in ('missing', 'split', 'junction', 'width')}
        all_maps, convergence = [], []
        for k, stroke in enumerate(self.strokes):
            m = masks[:, k:k + 1].float()
            coverage = _pool(m, cfg.coverage_radius)[:, 0, stroke.ys, stroke.xs]
            deficits = F.relu(1 - coverage / cfg.required_ink).square()
            missing = deficits.mean(-1)
            local_mass = _pool(m, cfg.width_radius, mean=True)[:, 0, stroke.ys, stroke.xs]
            expected = _pool(stroke.normal, cfg.width_radius, mean=True)[:, 0, stroke.ys, stroke.xs]
            width = F.relu(1 - local_mass / (expected * (1 - cfg.width_tolerance) + cfg.eps)).square().mean(-1)
            split_values, split_maps, pair_converged = [], [], []
            for pair in stroke.pairs:
                f = m[..., pair.slices[0], pair.slices[1]] * pair.corridor
                reached, converged = reconstruct(f, f * pair.a, cfg.connectivity_steps)
                u = (f * pair.a).flatten(1).amax(1)
                v = (f * pair.b).flatten(1).amax(1)
                c = (reached * pair.b).flatten(1).amax(1)
                strength = torch.minimum(u, v)
                # clamp avoids the tiny nonzero epsilon penalty on perfect paths.
                gap = F.relu((strength - c) / strength.clamp_min(cfg.eps)).square()
                split_values.append(u * v * gap)
                pair_converged.append(converged)
                if return_maps:
                    split_maps.append((pair, F.relu(f - reached)))
            split = torch.stack(split_values, -1).mean(-1)
            skeleton = soft_skeleton(m, cfg.skeleton_iterations)
            attached, attach_converged = reconstruct(m, m * stroke.seeds, cfg.attachment_steps)
            responses = [F.relu(_crossings(skeleton, r) - 2 - cfg.junction_margin).square()
                         for r in cfg.junction_radii]
            junction_map = torch.stack(responses).mean(0) * skeleton * attached
            junction = junction_map.flatten(1).sum(-1) / stroke.length
            for name, value in [('missing', missing), ('split', split), ('junction', junction), ('width', width)]:
                metrics[name].append(value)
            convergence.append({'stroke': k + 1, 'pairs': all(pair_converged),
                                'attachment': attach_converged, 'closed': stroke.closed})
            if return_maps:
                missing_map = torch.zeros_like(m)
                missing_map[:, 0, stroke.ys, stroke.xs] = deficits
                split_map = torch.zeros_like(m)
                for pair, area in split_maps:
                    # Maps are diagnostics, not the differentiable loss path.
                    with torch.no_grad():
                        target = split_map[..., pair.slices[0], pair.slices[1]]
                        target.copy_(torch.maximum(target, area.detach()))
                all_maps.append({'mask': m.detach(), 'centerline': stroke.skeleton.detach(),
                                 'missing': missing_map.detach(), 'split': split_map.detach(),
                                 'junction': junction_map.detach(), 'attached': attached.detach()})
        per_stroke = {key: torch.stack(value, -1) for key, value in metrics.items()}
        extraction = cfg.split_weight * per_stroke['split'] + cfg.junction_weight * per_stroke['junction']
        reference = cfg.missing_weight * per_stroke['missing'] + cfg.width_weight * per_stroke['width']
        result = {'loss': extraction.mean(), 'reference_loss': reference.mean(),
                  'per_stroke': {**per_stroke, 'extraction_error': extraction,
                                 'reference_deficit': reference},
                  'source_quality': {'assessed': False,
                      'reason': 'Extracted-mask defects do not identify source-image defects.'},
                  'convergence': convergence}
        if return_maps:
            result['maps'] = all_maps
        return result
