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


def white_background_proxy(source):
    """Keep dark neutral ink and nearby antialiasing in the same pixel coordinates."""
    rgb = np.asarray(source.convert('RGB'), np.float32)/255
    gray = rgb @ np.array([.299, .587, .114], np.float32)
    chroma = np.ptp(rgb, axis=-1)
    core = (gray <= .30) & (chroma <= .12)
    edge = (gray <= .58) & (chroma <= .23) & binary_dilation(core, iterations=2)
    window = max(9, round(source.width / 12) | 1)
    light = grey_closing(gray, size=(window, window))
    opacity = np.clip((light-gray)/np.maximum(light, .15), 0, 1)
    opacity[~(core | edge)] = 0
    proxy = Image.fromarray(np.rint((1-opacity)*255).astype(np.uint8)).convert('RGB')
    return proxy, opacity


def prepare_flattened_plan(source, conditions, extractor):
    """Extract strokes from the white proxy, retaining the full-color image for repair."""
    source = source.convert('RGB')
    if source.width != source.height or source.width % 16 or len(conditions) != 4:
        raise ValueError('Use a square 2x2 sheet with four references and dimensions divisible by 16')
    proxy, opacity = white_background_proxy(source)
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
                  extraction_input='neutral_ink_on_white',
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


class WhiteStrokeGuide(TargetImageGuide):
    def __init__(self, plan):
        source = np.asarray(plan.proxy, np.float32)/255
        target = source.copy()
        weights = (plan.erase | plan.add).astype(np.float32)
        if not weights.any():
            raise ValueError('추가하거나 지울 획 영역이 잡히지 않았습니다.')
        target[plan.erase] = 1
        target[plan.add] = (1-plan.aligned_alpha[plan.add])[:,None]
        super().__init__(source, target, weights, [], dict(plan.report,strokes=[]),
                         TargetPlanConfig(maximum_step_blend=1.))
        self.metadata.update(kind='white_stroke_layer', outside_roi='original_proxy_pixels',
                             inside_roi='grayscale_guided_ink', score_role='none')

    @torch.no_grad()
    def finish(self, rgb):
        source = torch.as_tensor(self.source,device=rgb.device).to(rgb).permute(2,0,1)[None]
        mask = torch.as_tensor(self.roi,device=rgb.device)[None,None]
        gray = rgb.mean(dim=1,keepdim=True).expand_as(rgb)
        return torch.where(mask,gray,source)


def composite_repaired_ink(plan, repaired_white, filled):
    """Transfer new ink opacity and the inpainted deletion patch onto the original RGB."""
    if repaired_white.size!=plan.source.size or filled.size!=plan.source.size:
        raise ValueError('Repair layers must match the source dimensions')
    source = np.asarray(plan.source, np.uint8)
    finished = source.copy()
    finished[plan.erase] = np.asarray(filled.convert('RGB'))[plan.erase]
    before = np.asarray(plan.proxy.convert('L'), np.float32)/255
    after = np.asarray(repaired_white.convert('L'), np.float32)/255
    # The input already has the old ink. Apply only the extra darkening.
    transmission = np.clip(after/np.maximum(before, 1/255), 0, 1)
    mask = plan.add & ~plan.erase
    finished[mask] = np.rint(source[mask]*transmission[mask,None]).astype(np.uint8)
    return Image.fromarray(finished)
