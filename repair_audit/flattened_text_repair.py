"""Repair a single flattened 2x2 image using local deletion inpainting."""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, grey_closing
import torch

from stroke_model.source_structure import SourceConfig
from stroke_model.target_stroke_plan import TargetImageGuide, TargetPlanConfig, build_target_plan
from .target_guided_repair import extract_sheet_masks


@dataclass
class FlattenedPlan:
    source: Image.Image
    proxy: Image.Image
    opacity: np.ndarray
    aligned_alpha: np.ndarray
    erase: np.ndarray
    add: np.ndarray
    report: dict


def prepare_flattened_plan(source, conditions, extractor):
    """Estimate dark ink from local contrast, then fit the usual reference glyphs."""
    source = source.convert('RGB')
    if source.width != source.height or source.width % 16 or len(conditions) != 4:
        raise ValueError('Use a square 2x2 sheet with four references and dimensions divisible by 16')
    gray = np.asarray(source, np.float32) @ np.array([.299, .587, .114], np.float32) / 255
    window = max(9, round(source.width / 12) | 1)
    local_light = grey_closing(gray, size=(window, window))
    opacity = np.clip((local_light-gray)/np.maximum(local_light, .15), 0, 1)
    opacity = np.where(opacity >= .06, opacity, 0).astype(np.float32)
    proxy = Image.fromarray(np.rint((1-opacity)*255).astype(np.uint8)).convert('RGB')
    observed = extract_sheet_masks(proxy, conditions, extractor)
    aligned, rendered = build_target_plan(proxy, conditions, observed,
        registration_config=SourceConfig(rotation=0., shear=0.))
    alpha = np.zeros_like(opacity)
    ink = np.zeros_like(opacity, dtype=bool)
    for glyph, target, masks in zip(aligned.report['glyphs'], rendered, observed, strict=True):
        x0,y0,x1,y1 = glyph['bbox']
        box = np.s_[y0:y1,x0:x1]
        size = (x1-x0,y1-y0)
        alpha[box] = np.asarray(Image.fromarray(target['aligned_strokes'].max(0)).resize(size, Image.Resampling.BILINEAR))
        support = np.asarray(Image.fromarray(masks.max(0)).resize(size, Image.Resampling.BILINEAR)) > .22
        ink[box] = binary_dilation(support, iterations=3) & (opacity[box] > .12)
    # Never open up a hole inside a target stroke; include a small edge margin for residual ink.
    erase = binary_dilation(ink, iterations=2) & (alpha <= .03)
    add = (alpha - opacity) > .06
    report = dict(aligned.report, input_mode='single_flattened_image',
                  erase_pixels=int(erase.sum()), add_pixels=int(add.sum()))
    return FlattenedPlan(source, proxy, opacity, alpha, erase, add, report)


def inpaint_deleted_ink(plan, pipe, *, steps=25, seed=2026):
    if not plan.erase.any():
        return plan.source.copy()
    mask = Image.fromarray(plan.erase.astype(np.uint8)*255)
    result = pipe(prompt='continuous background matching the surrounding color and texture',
        negative_prompt='text, glyphs, letters, lines of writing', image=plan.source, mask_image=mask,
        width=plan.source.width, height=plan.source.height, strength=1., num_inference_steps=steps,
        guidance_scale=5., generator=torch.Generator('cpu').manual_seed(seed))
    if getattr(result,'nsfw_content_detected',None) and result.nsfw_content_detected[0]:
        raise ValueError('인페인팅 결과가 필터링됐습니다.')
    patch = result.images[0].convert('RGB')
    if patch.size!=plan.source.size:
        raise ValueError('Inpaint output must match the source dimensions')
    return Image.composite(patch, plan.source, mask)


class FlattenedTextGuide(TargetImageGuide):
    def __init__(self, plan, filled):
        source = np.asarray(plan.source, np.float32)/255
        target = np.asarray(filled.convert('RGB'), np.float32)/255
        if target.shape!=source.shape:
            raise ValueError('Inpaint output must match the source dimensions')
        weights = (plan.erase | plan.add).astype(np.float32)
        if not weights.any():
            raise ValueError('추가하거나 지울 획 영역이 잡히지 않았습니다.')
        # Add only the missing opacity. Avoid painting the full target on top of ink already there.
        delta = (plan.aligned_alpha-plan.opacity)/np.maximum(1-plan.opacity, .01)
        target[plan.add] *= 1-np.clip(delta[plan.add,None],0,1)
        target[weights==0] = source[weights==0]
        super().__init__(source, target, weights, [], dict(plan.report,strokes=[]),
                         TargetPlanConfig(maximum_step_blend=1.))
        self.metadata.update(kind='flattened_local_inpaint', outside_roi='original_pixels',
                             inside_roi='raw_guided_rgb', score_role='none')

    @torch.no_grad()
    def finish(self, rgb):
        source = torch.as_tensor(self.source,device=rgb.device).to(rgb).permute(2,0,1)[None]
        mask = torch.as_tensor(self.roi,device=rgb.device)[None,None]
        return torch.where(mask,rgb,source)
