"""Project the aligned glyph silhouette inside a local ink mask."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import torch


def finish_ink_rgb(rgb, source, mask, background=None):
    if background is None:
        clean = rgb.mean(1, keepdim=True).expand_as(rgb)
    else:
        # Recover the black ink opacity along the known background RGB ray.
        energy = background.square().sum(1, keepdim=True)
        transmission = ((rgb*background).sum(1, keepdim=True)/energy.clamp_min(1e-6)).clamp(0,1)
        clean = background*transmission
    return torch.where(mask, clean, source)


class InkTargetGuide:
    def __init__(self, aligned_guide, rendered, background_image=None):
        self.source = aligned_guide.source.copy()
        self.target = self.source.copy()
        self.background = (np.asarray(background_image.convert('RGB'),np.float32)/255
                           if background_image is not None else None)
        if self.background is not None and self.background.shape!=self.source.shape:
            raise ValueError('Background must match the sheet size')
        self.roi = np.zeros(self.source.shape[:2], bool)
        self.metadata = dict(kind='glyph_ink_projection', final_target_replacement=False,
                             outside_roi='original_pixels', inside_roi='grayscale_output')
        if self.background is not None:
            self.metadata['inside_roi']='black_ink_over_known_background'
        self.report = dict(aligned_guide.report, ink_projection=self.metadata)
        for glyph, strokes in zip(aligned_guide.report['glyphs'], rendered, strict=True):
            x0, y0, x1, y1 = glyph['bbox']
            region = np.s_[y0:y1, x0:x1]
            crop = aligned_guide.source[region]
            gray = crop @ np.array([.299, .587, .114], np.float32)
            light = float(np.percentile(gray, 90))
            if light < .65:
                raise ValueError('글자 마스크 가이드는 밝은 배경의 어두운 글자용입니다.')
            alpha = np.asarray(Image.fromarray(strokes['aligned_strokes'].max(0)).resize(
                (x1-x0, y1-y0), Image.Resampling.BILINEAR)).clip(0, 1)
            # Exclude the faint edge of the old ink from background samples.
            old_ink = gray < light - .025
            excluded = binary_dilation(old_ink, iterations=2)
            if excluded.all():
                raise ValueError('글자 주변에서 배경 샘플을 찾지 못했습니다.')
            indices = distance_transform_edt(excluded, return_distances=False, return_indices=True)
            background = gray[tuple(indices)]
            edit = excluded | (alpha > .005)
            ink_samples = gray[gray < light - .35]
            ink_level = float(np.median(ink_samples)) if ink_samples.size else 0.
            composed = background*(1-alpha) + ink_level*alpha
            target_crop = self.target[region]
            target_crop[edit] = composed[edit, None]
            if self.background is not None:
                background_crop=self.background[region]
                opacity=np.where(excluded, np.clip(1-gray/np.maximum(background, .01),0,1), 0)
                self.source[region]=background_crop*(1-opacity[...,None])
                target_crop[:]=self.source[region]
                target_crop[edit]=(background_crop*(1-alpha[...,None]))[edit]
            self.roi[region] = edit
        if not self.roi.any():
            raise ValueError('보정할 글자 영역이 없습니다.')
        self.weights = self.roi.astype(np.float32)
        self._mask = torch.from_numpy(self.roi)[None, None]
        self._target = torch.from_numpy(self.target).permute(2,0,1)[None]
        self._source = torch.from_numpy(self.source).permute(2,0,1)[None]
        self._background = (torch.from_numpy(self.background).permute(2,0,1)[None]
                            if self.background is not None else None)

    @torch.no_grad()
    def project(self, rgb):
        # The same silhouette sets both the ink to add and the ink to remove.
        return torch.where(self._mask.to(rgb.device), self._target.to(rgb), rgb)

    @torch.no_grad()
    def diagnostics(self, rgb):
        error = (rgb.mean(1, keepdim=True)-self._target.to(rgb).mean(1, keepdim=True))[self._mask.to(rgb.device)]
        return dict(missing_ink=float(error.clamp_min(0).mean()),
                    extra_ink=float((-error).clamp_min(0).mean()))

    @torch.no_grad()
    def finish(self, rgb):
        background=self._background.to(rgb) if self._background is not None else None
        return finish_ink_rgb(rgb,self._source.to(rgb),self._mask.to(rgb.device),background)
