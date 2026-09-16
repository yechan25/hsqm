"""후처리: soft 예측을 배타적(exclusive) 획 partition으로 변환.
02_Use.ipynb의 bigstroke_protected 파이프라인을 그대로 옮긴 것 (변경 없음).
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize


def make_input_ink_binary(input_img: np.ndarray, threshold_mode: str = "mean",
                           fixed_threshold: float = 0.08, mean_scale: float = 0.85,
                           min_threshold: float = 0.04, close_iter: int = 0) -> np.ndarray:
    x = input_img.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    if float(x.max()) <= 1e-08:
        return np.zeros_like(x, dtype=bool)
    nz = x[x > min_threshold]
    if threshold_mode == "mean" and nz.size > 0:
        th = max(min_threshold, float(nz.mean()) * mean_scale)
    else:
        th = fixed_threshold
    ink = x >= th
    if close_iter > 0:
        structure = np.ones((3, 3), dtype=bool)
        ink = ndimage.binary_closing(ink, structure=structure, iterations=close_iter)
    return ink.astype(bool)


def get_active_channels(pred_soft_b: np.ndarray, target_hint_b: Optional[np.ndarray] = None,
                         min_hint_mass: float = 1.0, min_pred_peak: float = 0.04,
                         min_pred_mass: float = 0.15) -> List[int]:
    K = pred_soft_b.shape[0]
    active: List[int] = []
    if target_hint_b is not None:
        for k in range(K):
            if float(target_hint_b[k].sum()) >= min_hint_mass:
                active.append(k)
    if len(active) > 0:
        return active
    for k in range(K):
        p = pred_soft_b[k]
        if float(p.max()) >= min_pred_peak and float(p.sum()) >= min_pred_mass:
            active.append(k)
    if len(active) == 0:
        active = [int(np.argmax([pred_soft_b[k].sum() for k in range(K)]))]
    return active


def smooth_score_maps(pred_soft_b: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    K = pred_soft_b.shape[0]
    scores = np.zeros_like(pred_soft_b, dtype=np.float32)
    for k in range(K):
        p = pred_soft_b[k].astype(np.float32)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.clip(p, 0.0, 1.0)
        if sigma > 0:
            p = ndimage.gaussian_filter(p, sigma=sigma)
        scores[k] = p
    return scores


def remove_small_components_bool(mask: np.ndarray, min_area: int = 2) -> np.ndarray:
    lab, n = ndimage.label(mask > 0)
    if n <= 0:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for comp_id in range(1, n + 1):
        comp = lab == comp_id
        if int(comp.sum()) >= min_area:
            out |= comp
    return out


def make_seed_cores(scores: np.ndarray, input_ink: np.ndarray, active_channels: List[int],
                     target_hint_b: Optional[np.ndarray] = None, mean_gain: float = 1.15,
                     peak_ratio: float = 0.35, min_core_area: int = 2) -> Dict[int, np.ndarray]:
    cores: Dict[int, np.ndarray] = {}
    for k in active_channels:
        s = scores[k]
        vals = s[input_ink]
        if vals.size == 0:
            cores[k] = np.zeros_like(input_ink, dtype=bool)
            continue
        peak = float(s.max())
        mean_val = float(vals.mean())
        th = max(mean_val * mean_gain, peak * peak_ratio)
        core = (s >= th) & input_ink
        if target_hint_b is not None and float(target_hint_b[k].sum()) > 0:
            hint = target_hint_b[k] > 0.1
            hint = ndimage.binary_dilation(hint, structure=np.ones((3, 3), dtype=bool), iterations=2)
            core = core | (s >= max(mean_val, peak * 0.22)) & input_ink & hint
        core = remove_small_components_bool(core, min_area=min_core_area)
        if not core.any() and peak > 1e-08:
            fallback = (s >= peak * 0.5) & input_ink
            core = remove_small_components_bool(fallback, min_area=1)
        cores[k] = core.astype(bool)
    return cores


def resolve_core_conflicts(cores: Dict[int, np.ndarray], scores: np.ndarray,
                            active_channels: List[int], input_ink: np.ndarray) -> Dict[int, np.ndarray]:
    if not active_channels:
        return cores
    core_stack = np.stack([cores[k] for k in active_channels], axis=0)
    overlap = core_stack.sum(axis=0) > 1
    resolved = {k: cores[k].copy() for k in active_channels}
    if overlap.any():
        score_stack = np.stack([scores[k] for k in active_channels], axis=0)
        winner_local = np.argmax(score_stack, axis=0)
        active_arr = np.array(active_channels, dtype=np.int32)
        winner = active_arr[winner_local]
        for k in active_channels:
            lose = overlap & (winner != k)
            resolved[k][lose] = False
    for k in active_channels:
        resolved[k] = resolved[k] & input_ink
    return resolved


def make_distance_penalized_label_map(scores: np.ndarray, input_ink: np.ndarray, active_channels: List[int],
                                       cores: Dict[int, np.ndarray], distance_weight: float = 0.85,
                                       no_core_distance_weight: float = 1.35, core_bonus: float = 0.35) -> np.ndarray:
    H, W = input_ink.shape
    label_map = np.full((H, W), -1, dtype=np.int32)
    if not input_ink.any():
        return label_map
    score_list = []
    max_dist = float(np.sqrt(H * H + W * W))
    for k in active_channels:
        s = scores[k].astype(np.float32)
        peak = float(s.max())
        s_norm = s / peak if peak > 1e-08 else np.zeros_like(s)
        core = cores.get(k, np.zeros_like(input_ink, dtype=bool))
        if core.any():
            dist = ndimage.distance_transform_edt(~core)
            dist_norm = np.clip(dist / max_dist, 0.0, 1.0)
            combined = s_norm - distance_weight * dist_norm + core_bonus * core.astype(np.float32)
        else:
            pseudo = s_norm >= max(float(s_norm[input_ink].mean()) if input_ink.any() else 0.0, 0.25)
            if pseudo.any():
                dist = ndimage.distance_transform_edt(~pseudo)
                dist_norm = np.clip(dist / max_dist, 0.0, 1.0)
            else:
                dist_norm = np.ones_like(s_norm)
            combined = s_norm - no_core_distance_weight * dist_norm
        score_list.append(combined)
    score_stack = np.stack(score_list, axis=0)
    winner_local = np.argmax(score_stack, axis=0)
    active_arr = np.array(active_channels, dtype=np.int32)
    winner = active_arr[winner_local]
    label_map[input_ink] = winner[input_ink]
    for k in active_channels:
        core = cores.get(k, None)
        if core is not None and core.any():
            label_map[core & input_ink] = k
    label_map[~input_ink] = -1
    return label_map


def absorb_tiny_islands_seeded(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                active_channels: List[int], cores: Dict[int, np.ndarray],
                                min_component_area: int = 3) -> np.ndarray:
    lm = label_map.copy()
    for k in active_channels:
        mask = lm == k
        lab, n = ndimage.label(mask)
        for comp_id in range(1, n + 1):
            comp = lab == comp_id
            area = int(comp.sum())
            if area >= min_component_area:
                continue
            core = cores.get(k, np.zeros_like(input_ink, dtype=bool))
            if (comp & core).any():
                continue
            border = ndimage.binary_dilation(comp, structure=np.ones((3, 3), dtype=bool), iterations=1) & input_ink & ~comp
            neighbor_labels = sorted(set(int(v) for v in lm[border].flatten() if int(v) in active_channels and int(v) != k))
            ys, xs = np.where(comp)
            for y, x in zip(ys, xs):
                if neighbor_labels:
                    best_k = max(neighbor_labels, key=lambda c: float(scores[c, y, x]))
                else:
                    best_k = max(active_channels, key=lambda c: float(scores[c, y, x]))
                lm[y, x] = best_k
    lm[~input_ink] = -1
    return lm


def prevent_small_stealing_big_components(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                           active_channels: List[int], cores: Dict[int, np.ndarray],
                                           small_area_ratio: float = 0.18) -> np.ndarray:
    lm = label_map.copy()
    areas = {k: int((lm == k).sum()) for k in active_channels}
    for small_k in active_channels:
        small_mask = lm == small_k
        lab, n = ndimage.label(small_mask)
        for comp_id in range(1, n + 1):
            comp = lab == comp_id
            comp_area = int(comp.sum())
            if comp_area == 0:
                continue
            if (comp & cores.get(small_k, np.zeros_like(input_ink, dtype=bool))).any():
                continue
            border = ndimage.binary_dilation(comp, structure=np.ones((3, 3), dtype=bool), iterations=1) & input_ink & ~comp
            neighbor_labels = [int(v) for v in np.unique(lm[border]) if int(v) in active_channels and int(v) != small_k]
            if not neighbor_labels:
                continue
            big_k = max(neighbor_labels, key=lambda c: areas.get(c, 0))
            if areas.get(big_k, 0) <= 0:
                continue
            if comp_area <= max(2, int(areas[big_k] * small_area_ratio)):
                small_score = float((scores[small_k] * comp).mean())
                big_score = float((scores[big_k] * comp).mean())
                if small_score < big_score * 1.35:
                    lm[comp] = big_k
    lm[~input_ink] = -1
    return lm


def component_score_for_label(comp: np.ndarray, label_k: int, scores: np.ndarray,
                               cores: Dict[int, np.ndarray]) -> float:
    area = int(comp.sum())
    if area <= 0:
        return -1e18
    prob_sum = float((scores[label_k] * comp).sum())
    prob_mean = prob_sum / max(area, 1)
    core_hit = int((comp & cores.get(label_k, np.zeros_like(comp, dtype=bool))).sum())
    return prob_sum + 0.4 * prob_mean + 3.0 * core_hit + 0.05 * area


def get_neighbor_labels(region: np.ndarray, label_map: np.ndarray, active_channels: List[int],
                         exclude_label: Optional[int] = None) -> List[int]:
    dil = ndimage.binary_dilation(region, structure=np.ones((3, 3), dtype=bool), iterations=1)
    border = dil & ~region
    labels = []
    for v in np.unique(label_map[border]):
        iv = int(v)
        if iv in active_channels and (exclude_label is None or iv != exclude_label):
            labels.append(iv)
    return labels


def assign_region_to_best_neighbor(region: np.ndarray, old_label: int, label_map: np.ndarray,
                                    scores: np.ndarray, active_channels: List[int],
                                    cores: Dict[int, np.ndarray]) -> int:
    neighbor_labels = get_neighbor_labels(region=region, label_map=label_map, active_channels=active_channels, exclude_label=old_label)
    candidate_labels = neighbor_labels if neighbor_labels else [k for k in active_channels if k != old_label]
    if not candidate_labels:
        return old_label
    best_label = candidate_labels[0]
    best_score = -1e18
    area = max(int(region.sum()), 1)
    for k in candidate_labels:
        prob_sum = float((scores[k] * region).sum())
        prob_mean = prob_sum / area
        core = cores.get(k, np.zeros_like(region, dtype=bool))
        if core.any():
            dist = ndimage.distance_transform_edt(~core)
            dist_bonus = -0.025 * float(dist[region].mean())
        else:
            dist_bonus = 0.0
        touch_bonus = 0.35 if k in neighbor_labels else 0.0
        score = prob_sum + 0.5 * prob_mean + dist_bonus + touch_bonus
        if score > best_score:
            best_score = score
            best_label = k
    return int(best_label)


def enforce_each_label_connected_preserve_pixels(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                                  active_channels: List[int], cores: Dict[int, np.ndarray],
                                                  max_passes: int = 8) -> np.ndarray:
    lm = label_map.copy()
    lm[~input_ink] = -1
    missing = input_ink & (lm < 0)
    if missing.any():
        active_scores = np.stack([scores[k] for k in active_channels], axis=0)
        local_idx = np.argmax(active_scores, axis=0)
        active_arr = np.array(active_channels, dtype=np.int32)
        chosen = active_arr[local_idx]
        lm[missing] = chosen[missing]
    for _ in range(max_passes):
        changed = False
        label_order = sorted(active_channels, key=lambda k: int((lm == k).sum()), reverse=True)
        for k in label_order:
            mask = lm == k
            lab, n = ndimage.label(mask)
            if n <= 1:
                continue
            comps = []
            for comp_id in range(1, n + 1):
                comp = lab == comp_id
                score = component_score_for_label(comp, k, scores, cores)
                comps.append((score, comp_id, comp, int(comp.sum())))
            comps.sort(key=lambda x: x[0], reverse=True)
            for score, comp_id, comp, area in comps[1:]:
                if area <= 0:
                    continue
                new_label = assign_region_to_best_neighbor(region=comp, old_label=k, label_map=lm, scores=scores, active_channels=active_channels, cores=cores)
                if new_label != k:
                    lm[comp] = new_label
                    changed = True
        lm[~input_ink] = -1
        if not changed:
            break
    for k in active_channels:
        mask = lm == k
        lab, n = ndimage.label(mask)
        if n <= 1:
            continue
        comps = []
        for comp_id in range(1, n + 1):
            comp = lab == comp_id
            score = component_score_for_label(comp, k, scores, cores)
            comps.append((score, comp_id, comp, int(comp.sum())))
        comps.sort(key=lambda x: x[0], reverse=True)
        for score, comp_id, comp, area in comps[1:]:
            new_label = assign_region_to_best_neighbor(region=comp, old_label=k, label_map=lm, scores=scores, active_channels=active_channels, cores=cores)
            lm[comp] = new_label
    lm[~input_ink] = -1
    missing = input_ink & (lm < 0)
    if missing.any():
        active_scores = np.stack([scores[k] for k in active_channels], axis=0)
        local_idx = np.argmax(active_scores, axis=0)
        active_arr = np.array(active_channels, dtype=np.int32)
        chosen = active_arr[local_idx]
        lm[missing] = chosen[missing]
    return lm


def get_label_areas_from_label_map(label_map: np.ndarray, active_channels: List[int]) -> Dict[int, int]:
    return {int(k): int((label_map == k).sum()) for k in active_channels}


def build_big_stroke_protection_zones(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                       active_channels: List[int], cores: Dict[int, np.ndarray],
                                       big_area_ratio_threshold: float = 0.7, protect_core_dilate_iter: int = 3,
                                       protect_mask_dilate_iter: int = 1) -> Dict[int, np.ndarray]:
    areas = get_label_areas_from_label_map(label_map, active_channels)
    if not areas:
        return {}
    max_area = max(areas.values())
    if max_area <= 0:
        return {}
    structure = np.ones((3, 3), dtype=bool)
    zones: Dict[int, np.ndarray] = {}
    for k in active_channels:
        area = areas.get(int(k), 0)
        if area < max(3, int(max_area * big_area_ratio_threshold)):
            continue
        mask_k = label_map == k
        core_k = cores.get(k, np.zeros_like(input_ink, dtype=bool))
        zone = np.zeros_like(input_ink, dtype=bool)
        if core_k.any():
            zone |= ndimage.binary_dilation(core_k, structure=structure, iterations=protect_core_dilate_iter)
        if mask_k.any():
            zone |= ndimage.binary_dilation(mask_k, structure=structure, iterations=protect_mask_dilate_iter)
        zones[int(k)] = zone & input_ink
    return zones


def protect_big_strokes_from_small_invaders(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                             active_channels: List[int], cores: Dict[int, np.ndarray],
                                             big_area_ratio_threshold: float = 0.7, protect_core_dilate_iter: int = 3,
                                             protect_mask_dilate_iter: int = 1, steal_area_ratio_threshold: float = 0.22,
                                             overlap_ratio_threshold: float = 0.55, keep_if_own_core_overlap_ratio: float = 0.3,
                                             small_score_margin: float = 1.45) -> np.ndarray:
    lm = label_map.copy()
    lm[~input_ink] = -1
    areas = get_label_areas_from_label_map(lm, active_channels)
    if not areas:
        return lm
    protection_zones = build_big_stroke_protection_zones(label_map=lm, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, big_area_ratio_threshold=big_area_ratio_threshold, protect_core_dilate_iter=protect_core_dilate_iter, protect_mask_dilate_iter=protect_mask_dilate_iter)
    if not protection_zones:
        return lm
    big_channels = sorted(protection_zones.keys(), key=lambda k: areas.get(k, 0), reverse=True)
    for big_k in big_channels:
        zone = protection_zones[big_k]
        for small_k in active_channels:
            small_k = int(small_k)
            if small_k == big_k:
                continue
            small_mask = lm == small_k
            if not small_mask.any():
                continue
            lab, n = ndimage.label(small_mask)
            for comp_id in range(1, n + 1):
                comp = lab == comp_id
                comp_area = int(comp.sum())
                if comp_area <= 0:
                    continue
                overlap = comp & zone
                overlap_area = int(overlap.sum())
                if overlap_area <= 0:
                    continue
                overlap_ratio = overlap_area / max(comp_area, 1)
                own_core = cores.get(small_k, np.zeros_like(input_ink, dtype=bool))
                own_core_overlap = int((comp & own_core).sum())
                own_core_ratio = own_core_overlap / max(comp_area, 1)
                big_area = max(areas.get(big_k, 0), 1)
                is_small_vs_big = comp_area <= max(2, int(big_area * steal_area_ratio_threshold))
                strongly_inside_big = overlap_ratio >= overlap_ratio_threshold
                weak_own_evidence = own_core_ratio < keep_if_own_core_overlap_ratio
                small_score_mean = float(scores[small_k][comp].mean())
                big_score_mean = float(scores[big_k][comp].mean())
                if (is_small_vs_big or strongly_inside_big) and weak_own_evidence:
                    if small_score_mean <= big_score_mean * small_score_margin:
                        lm[comp] = big_k
    lm[~input_ink] = -1
    return lm


def prevent_small_masks_containing_big_parts(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                              active_channels: List[int], cores: Dict[int, np.ndarray],
                                              contain_overlap_ratio_threshold: float = 0.28,
                                              contain_big_core_overlap_threshold: int = 1,
                                              small_score_margin: float = 1.5) -> np.ndarray:
    lm = label_map.copy()
    lm[~input_ink] = -1
    areas = get_label_areas_from_label_map(lm, active_channels)
    if not areas:
        return lm
    sorted_labels = sorted([int(k) for k in active_channels], key=lambda k: areas.get(k, 0), reverse=True)
    structure = np.ones((3, 3), dtype=bool)
    for i, big_k in enumerate(sorted_labels):
        big_mask = lm == big_k
        if not big_mask.any():
            continue
        big_core = cores.get(big_k, np.zeros_like(input_ink, dtype=bool))
        big_zone = ndimage.binary_dilation(big_mask | big_core, structure=structure, iterations=1) & input_ink
        for small_k in sorted_labels[i + 1:]:
            small_mask = lm == small_k
            if not small_mask.any():
                continue
            lab, n = ndimage.label(small_mask)
            for comp_id in range(1, n + 1):
                comp = lab == comp_id
                comp_area = int(comp.sum())
                if comp_area <= 0:
                    continue
                overlap_big_zone = int((comp & big_zone).sum())
                overlap_ratio = overlap_big_zone / max(comp_area, 1)
                big_core_hit = int((comp & big_core).sum())
                own_core = cores.get(small_k, np.zeros_like(input_ink, dtype=bool))
                own_core_hit = int((comp & own_core).sum())
                if overlap_ratio >= contain_overlap_ratio_threshold and big_core_hit >= contain_big_core_overlap_threshold and own_core_hit == 0:
                    small_score_mean = float(scores[small_k][comp].mean())
                    big_score_mean = float(scores[big_k][comp].mean())
                    if small_score_mean <= big_score_mean * small_score_margin:
                        lm[comp] = big_k
    lm[~input_ink] = -1
    return lm


def fill_big_stroke_bites(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                           active_channels: List[int], cores: Dict[int, np.ndarray],
                           big_area_ratio_threshold: float = 0.7, bite_dilate_iter: int = 1,
                           max_bite_area_ratio: float = 0.18, small_score_margin: float = 1.35) -> np.ndarray:
    lm = label_map.copy()
    lm[~input_ink] = -1
    areas = get_label_areas_from_label_map(lm, active_channels)
    if not areas:
        return lm
    max_area = max(areas.values())
    if max_area <= 0:
        return lm
    structure = np.ones((3, 3), dtype=bool)
    big_channels = [int(k) for k in active_channels if areas.get(int(k), 0) >= max(3, int(max_area * big_area_ratio_threshold))]
    big_channels.sort(key=lambda k: areas.get(k, 0), reverse=True)
    for big_k in big_channels:
        big_mask = lm == big_k
        if not big_mask.any():
            continue
        big_near = ndimage.binary_dilation(big_mask | cores.get(big_k, np.zeros_like(input_ink, dtype=bool)), structure=structure, iterations=bite_dilate_iter) & input_ink
        big_area = max(areas.get(big_k, 0), 1)
        for other_k in active_channels:
            other_k = int(other_k)
            if other_k == big_k:
                continue
            other_mask = (lm == other_k) & big_near
            if not other_mask.any():
                continue
            lab, n = ndimage.label(other_mask)
            for comp_id in range(1, n + 1):
                comp = lab == comp_id
                comp_area = int(comp.sum())
                if comp_area <= 0 or comp_area > max(2, int(big_area * max_bite_area_ratio)):
                    continue
                own_core = cores.get(other_k, np.zeros_like(input_ink, dtype=bool))
                if (comp & own_core).any():
                    continue
                other_score = float(scores[other_k][comp].mean())
                big_score = float(scores[big_k][comp].mean())
                if other_score <= big_score * small_score_margin:
                    lm[comp] = big_k
    lm[~input_ink] = -1
    return lm


def skeleton_neighbor_count(binary: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.int32)
    kernel[1, 1] = 0
    return ndimage.convolve(binary.astype(np.int32), kernel, mode="constant", cval=0)


def skeleton_branch_points(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = mask.astype(bool)
    if not m.any():
        skel = np.zeros_like(m, dtype=bool)
        branch = np.zeros_like(m, dtype=bool)
        return skel, branch
    skel = skeletonize(m)
    deg = skeleton_neighbor_count(skel)
    branch = skel & (deg >= 3)
    return skel.astype(bool), branch.astype(bool)


def reassign_region_preserve_pixels(label_map: np.ndarray, region: np.ndarray, from_label: int,
                                     scores: np.ndarray, active_channels: List[int],
                                     cores: Dict[int, np.ndarray]) -> np.ndarray:
    lm = label_map.copy()
    region = region.astype(bool)
    if not region.any():
        return lm
    neighbor_labels = get_neighbor_labels(region=region, label_map=lm, active_channels=active_channels, exclude_label=from_label)
    candidates = neighbor_labels if neighbor_labels else [k for k in active_channels if k != from_label]
    if not candidates:
        return lm
    ys, xs = np.where(region)
    for y, x in zip(ys, xs):
        best_k = None
        best_score = -1e18
        for k in candidates:
            raw = float(scores[k, y, x])
            core = cores.get(k, None)
            if core is not None and core.any():
                dist = ndimage.distance_transform_edt(~core)
                dist_penalty = 0.025 * float(dist[y, x])
            else:
                dist_penalty = 0.0
            touch_bonus = 0.25 if k in neighbor_labels else 0.0
            score = raw - dist_penalty + touch_bonus
            if score > best_score:
                best_score = score
                best_k = k
        if best_k is None:
            others = [k for k in active_channels if k != from_label]
            best_k = max(others, key=lambda k: float(scores[k, y, x])) if others else from_label
        lm[y, x] = int(best_k)
    return lm


def component_score_bigfirst_no_tjunction(comp: np.ndarray, label_k: int, scores: np.ndarray,
                                           cores: Dict[int, np.ndarray]) -> float:
    area = int(comp.sum())
    if area <= 0:
        return -1e18
    prob_sum = float((scores[label_k] * comp).sum())
    prob_mean = prob_sum / max(area, 1)
    core_hit = int((comp & cores.get(label_k, np.zeros_like(comp, dtype=bool))).sum())
    return prob_sum + 0.4 * prob_mean + 3.0 * core_hit + 0.12 * area


def enforce_no_tjunction_big_first_preserve_pixels(label_map: np.ndarray, scores: np.ndarray, input_ink: np.ndarray,
                                                    active_channels: List[int], cores: Dict[int, np.ndarray],
                                                    max_passes: int = 8, cut_dilate_iter: int = 1) -> np.ndarray:
    lm = label_map.copy()
    lm[~input_ink] = -1
    for _ in range(max_passes):
        changed = False
        label_order = sorted(active_channels, key=lambda k: int((lm == k).sum()), reverse=True)
        for k in label_order:
            mask = lm == k
            if not mask.any():
                continue
            skel, branch = skeleton_branch_points(mask)
            if not branch.any():
                continue
            cut_zone = ndimage.binary_dilation(branch, structure=np.ones((3, 3), dtype=bool), iterations=cut_dilate_iter)
            remainder = mask & ~cut_zone
            lab, n = ndimage.label(remainder)
            if n <= 0:
                continue
            comps = []
            for comp_id in range(1, n + 1):
                comp = lab == comp_id
                score = component_score_bigfirst_no_tjunction(comp, k, scores, cores)
                comps.append((score, comp_id, comp, int(comp.sum())))
            if len(comps) <= 1:
                region_to_move = cut_zone & mask
                if region_to_move.any():
                    new_lm = reassign_region_preserve_pixels(label_map=lm, region=region_to_move, from_label=k, scores=scores, active_channels=active_channels, cores=cores)
                    if not np.array_equal(new_lm, lm):
                        lm = new_lm
                        changed = True
                continue
            comps.sort(key=lambda x: x[0], reverse=True)
            region_to_move = np.zeros_like(mask, dtype=bool)
            for score, comp_id, comp, area in comps[1:]:
                region_to_move |= comp
            region_to_move |= cut_zone & mask
            if region_to_move.any():
                new_lm = reassign_region_preserve_pixels(label_map=lm, region=region_to_move, from_label=k, scores=scores, active_channels=active_channels, cores=cores)
                if not np.array_equal(new_lm, lm):
                    lm = new_lm
                    changed = True
        lm[~input_ink] = -1
        missing = input_ink & (lm < 0)
        if missing.any():
            active_scores = np.stack([scores[k] for k in active_channels], axis=0)
            local_idx = np.argmax(active_scores, axis=0)
            active_arr = np.array(active_channels, dtype=np.int32)
            chosen = active_arr[local_idx]
            lm[missing] = chosen[missing]
            changed = True
        if not changed:
            break
    return lm


def connectivity_report_from_stack(post_stack: np.ndarray, active_channels: Optional[List[int]] = None) -> Dict[str, Any]:
    K = post_stack.shape[0]
    if active_channels is None:
        active_channels = list(range(K))
    component_counts = {}
    bad_channels = []
    for k in active_channels:
        mask = post_stack[k] > 0.5
        lab, n = ndimage.label(mask)
        component_counts[int(k)] = int(n)
        if n > 1:
            bad_channels.append(int(k))
    return {"component_counts": component_counts, "bad_channels": bad_channels, "num_bad_channels": len(bad_channels)}


def label_map_to_stack(label_map: np.ndarray, K: int) -> np.ndarray:
    H, W = label_map.shape
    out = np.zeros((K, H, W), dtype=np.float32)
    for k in range(K):
        out[k] = (label_map == k).astype(np.float32)
    return out


def label_has_tjunction(mask: np.ndarray) -> bool:
    _, branch = skeleton_branch_points(mask)
    return bool(branch.any())


def branchy_channels_from_stack(post_stack: np.ndarray, active_channels: Optional[List[int]] = None) -> List[int]:
    K = post_stack.shape[0]
    if active_channels is None:
        active_channels = list(range(K))
    return [k for k in active_channels if label_has_tjunction(post_stack[k] > 0.5)]


def postprocess_sample_seeded_partition(pred_soft_b: np.ndarray, input_img: np.ndarray,
                                         target_hint_b: Optional[np.ndarray] = None, blur_sigma: float = 1.0,
                                         distance_weight: float = 0.85, core_bonus: float = 0.35,
                                         island_min_area: int = 3) -> np.ndarray:
    K, H, W = pred_soft_b.shape
    input_ink = make_input_ink_binary(input_img, threshold_mode="mean", mean_scale=0.85, min_threshold=0.04, close_iter=0)
    if not input_ink.any():
        return np.zeros((K, H, W), dtype=np.float32)
    active_channels = get_active_channels(pred_soft_b, target_hint_b=target_hint_b, min_hint_mass=1.0, min_pred_peak=0.04, min_pred_mass=0.15)
    scores = smooth_score_maps(pred_soft_b, sigma=blur_sigma)
    cores = make_seed_cores(scores, input_ink=input_ink, active_channels=active_channels, target_hint_b=target_hint_b, mean_gain=1.15, peak_ratio=0.35, min_core_area=2)
    cores = resolve_core_conflicts(cores, scores=scores, active_channels=active_channels, input_ink=input_ink)
    label_map = make_distance_penalized_label_map(scores, input_ink=input_ink, active_channels=active_channels, cores=cores, distance_weight=distance_weight, no_core_distance_weight=1.35, core_bonus=core_bonus)
    label_map = absorb_tiny_islands_seeded(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, min_component_area=island_min_area)
    label_map = prevent_small_stealing_big_components(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, small_area_ratio=0.18)
    label_map = protect_big_strokes_from_small_invaders(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, big_area_ratio_threshold=0.7, protect_core_dilate_iter=3, protect_mask_dilate_iter=1, steal_area_ratio_threshold=0.22, overlap_ratio_threshold=0.55, keep_if_own_core_overlap_ratio=0.3, small_score_margin=1.45)
    label_map = prevent_small_masks_containing_big_parts(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, contain_overlap_ratio_threshold=0.28, contain_big_core_overlap_threshold=1, small_score_margin=1.5)
    label_map = fill_big_stroke_bites(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, big_area_ratio_threshold=0.7, bite_dilate_iter=1, max_bite_area_ratio=0.18, small_score_margin=1.35)
    label_map = enforce_no_tjunction_big_first_preserve_pixels(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, max_passes=8, cut_dilate_iter=1)
    label_map = enforce_each_label_connected_preserve_pixels(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, max_passes=8)
    label_map = protect_big_strokes_from_small_invaders(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, big_area_ratio_threshold=0.7, protect_core_dilate_iter=2, protect_mask_dilate_iter=1, steal_area_ratio_threshold=0.2, overlap_ratio_threshold=0.5, keep_if_own_core_overlap_ratio=0.25, small_score_margin=1.4)
    label_map = fill_big_stroke_bites(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, big_area_ratio_threshold=0.7, bite_dilate_iter=1, max_bite_area_ratio=0.16, small_score_margin=1.3)
    label_map = enforce_no_tjunction_big_first_preserve_pixels(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, max_passes=4, cut_dilate_iter=1)
    label_map = enforce_each_label_connected_preserve_pixels(label_map, scores=scores, input_ink=input_ink, active_channels=active_channels, cores=cores, max_passes=4)
    bad = input_ink & (label_map < 0)
    if bad.any():
        active_scores = np.stack([scores[k] for k in active_channels], axis=0)
        local_idx = np.argmax(active_scores, axis=0)
        chosen = np.array(active_channels, dtype=np.int32)[local_idx]
        label_map[bad] = chosen[bad]
    label_map[~input_ink] = -1
    return label_map_to_stack(label_map, K)


def postprocess_batch_predictions_seeded_partition(pred_soft: torch.Tensor, input_imgs: torch.Tensor,
                                                     target_hints: Optional[torch.Tensor] = None, blur_sigma: float = 1.0,
                                                     distance_weight: float = 0.85, core_bonus: float = 0.35,
                                                     island_min_area: int = 3) -> torch.Tensor:
    pred_np = pred_soft.detach().float().cpu().numpy()
    input_np = input_imgs.detach().float().cpu().numpy()
    hint_np = target_hints.detach().float().cpu().numpy() if target_hints is not None else None
    B, K, H, W = pred_np.shape
    out = np.zeros_like(pred_np, dtype=np.float32)
    for b in range(B):
        out[b] = postprocess_sample_seeded_partition(pred_soft_b=pred_np[b], input_img=input_np[b, 0], target_hint_b=hint_np[b] if hint_np is not None else None, blur_sigma=blur_sigma, distance_weight=distance_weight, core_bonus=core_bonus, island_min_area=island_min_area)
    return torch.from_numpy(out)


def check_partition_integrity(pred_post_b: torch.Tensor, input_img: torch.Tensor) -> Dict[str, float]:
    post = pred_post_b.detach().float().cpu().numpy()
    img = input_img.detach().float().cpu().numpy()
    if img.ndim == 3:
        img = img[0]
    input_ink = make_input_ink_binary(img, threshold_mode="mean", mean_scale=0.85, min_threshold=0.04, close_iter=0)
    union = post.sum(axis=0) > 0.5
    overlap_count = np.maximum(post.sum(axis=0) - 1.0, 0.0).sum()
    missing = input_ink & ~union
    extra = union & ~input_ink
    return {"input_ink_pixels": float(input_ink.sum()), "union_pixels": float(union.sum()),
            "missing_pixels": float(missing.sum()), "extra_pixels": float(extra.sum()),
            "overlap_excess_pixels": float(overlap_count)}
