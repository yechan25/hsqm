"""Per-stroke ink constraints for black text on a white background.

No classifier or learned correctness claim. Correct target strokes and boxes are
external inputs. Pixel projection is exact; VAE round-trips need rechecking.
"""
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch


@dataclass(frozen=True)
class StrokeProjectionConfig:
    core_fraction: float = .25
    minimum_core_radius: float = .5
    boundary_margin: float = 1.5
    minimum_ink: float = .7
    maximum_background_ink: float = .03

    def __post_init__(self):
        if any(not math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError('All parameters must be finite')
        if not 0 < self.core_fraction <= 1 or self.minimum_core_radius < 0 or self.boundary_margin < 0:
            raise ValueError('Invalid spatial parameters')
        if not 0 <= self.maximum_background_ink < self.minimum_ink <= 1:
            raise ValueError('Require background ink < minimum stroke ink in [0,1]')


def ink(rgb):
    return 1-(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)


class StrokeProjection:
    def __init__(self, cores, allowed, roi, identities, config=None):
        self.config=config or StrokeProjectionConfig()
        self.cores=np.asarray(cores,dtype=bool)
        self.allowed=np.asarray(allowed,dtype=bool)
        self.roi=np.asarray(roi,dtype=bool)
        if (self.cores.ndim!=3 or not len(self.cores) or self.allowed.shape!=self.cores.shape
                or self.roi.shape!=self.cores.shape[1:] or len(identities)!=len(self.cores)):
            raise ValueError('Require K matching core/allowed masks, ROI and identities')
        if np.any(self.cores & ~self.allowed) or np.any(self.allowed & ~self.roi):
            raise ValueError('Require core subset of allowed subset of ROI')
        if not self.cores.reshape(len(self.cores),-1).any(1).all():
            raise ValueError('Every stroke needs a nonempty core')
        self.identities=identities
        self.foreground=self.cores.any(0)
        # A stroke may never erase a different stroke at crossings or junctions.
        self.background=self.roi & ~self.allowed.any(0)
        self.free=self.roi & ~self.foreground & ~self.background

    @classmethod
    def from_glyphs(cls, glyphs, canvas_size, config=None):
        """glyph: strokes KxHxW, bbox x0,y0,x1,y1, optional half_widths.

        Widths and config distances use the target-mask coordinate system.
        Overlapping glyph boxes are rejected; within-glyph stroke overlap is valid.
        """
        cfg=config or StrokeProjectionConfig()
        width,height=canvas_size
        cores=[];allowed=[];identities=[];roi=np.zeros((height,width),bool)
        for gi,glyph in enumerate(glyphs):
            x0,y0,x1,y1=glyph['bbox']
            if any(type(v) is not int for v in (x0,y0,x1,y1)) or not 0<=x0<x1<=width or not 0<=y0<y1<=height:
                raise ValueError('Invalid box')
            if roi[y0:y1,x0:x1].any():raise ValueError('Glyph boxes must not overlap')
            roi[y0:y1,x0:x1]=True
            strokes=np.asarray(glyph['strokes'])
            if strokes.ndim!=3 or not len(strokes) or not np.isfinite(strokes).all() or strokes.min()<0 or strokes.max()>1:
                raise ValueError('Target strokes must be finite KxHxW ink in [0,1]')
            widths=glyph.get('half_widths')
            if widths is not None and (len(widths)!=len(strokes) or any(not math.isfinite(v) or v<=0 for v in widths)):
                raise ValueError('Invalid per-stroke widths')
            for k,stroke in enumerate(strokes):
                binary=stroke>.35;path=skeletonize(binary)
                if not path.any():raise ValueError('Empty target stroke')
                half=float(np.median(distance_transform_edt(binary)[path])) if widths is None else widths[k]
                distance=distance_transform_edt(~path)
                core=distance<=min(half,max(cfg.minimum_core_radius,half*cfg.core_fraction))
                permit=distance<=half+cfg.boundary_margin
                def place(mask):
                    full=np.zeros_like(roi)
                    resized=np.asarray(Image.fromarray(mask).resize((x1-x0,y1-y0),Image.Resampling.NEAREST))
                    full[y0:y1,x0:x1]=resized
                    return full
                core_canvas=place(core)
                if not core_canvas.any():raise ValueError('Box too small to retain this stroke core')
                cores.append(core_canvas);allowed.append(place(permit))
                identities.append(dict(glyph=gi+1,stroke=k+1,half_width=float(half)))
        return cls(cores,allowed,roi,identities,cfg)

    def _masks(self,rgb):
        if rgb.ndim!=4 or rgb.shape[0:2]!=(1,3) or tuple(rgb.shape[-2:])!=self.roi.shape:
            raise ValueError('Expected one RGB tensor matching the constraint canvas')
        if not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
            raise ValueError('Projection expects finite RGB values in [0,1]')
        return [torch.as_tensor(a.copy(),device=rgb.device)[None,None] for a in (self.foreground,self.background)]

    @torch.no_grad()
    def project(self,rgb):
        fg,bg=self._masks(rgb)
        original=ink(rgb);target=original.clone()
        target=torch.where(fg,target.clamp_min(self.config.minimum_ink),target)
        target=torch.where(bg,target.clamp_max(self.config.maximum_background_ink),target)
        # Change luminance along the black/white ray, preserving chromatic ratios
        # while darkening. Unconstrained pixels are bitwise unchanged.
        luminance=1-original;desired=1-target
        darker=rgb*(desired/luminance.clamp_min(1e-8))
        lighter=rgb+(1-rgb)*((desired-luminance)/(1-luminance).clamp_min(1e-8))
        changed=torch.where((desired<luminance).expand_as(rgb),darker,lighter).clamp(0,1)
        return torch.where((fg|bg).expand_as(rgb),changed,rgb)

    @torch.no_grad()
    def diagnostics(self,rgb):
        fg,bg=self._masks(rgb);value=ink(rgb);eps=1e-6
        rows=[]
        for identity,core in zip(self.identities,self.cores):
            selected=torch.as_tensor(core.copy(),device=rgb.device)
            v=value[0,0][selected]
            rows.append(dict(**identity,core_pixels=int(core.sum()),
                missing_pixels=int((v<self.config.minimum_ink-eps).sum()),minimum_ink=float(v.min())))
        return dict(strokes=rows,missing_pixels=int(((value<self.config.minimum_ink-eps)&fg).sum()),
            extra_pixels=int(((value>self.config.maximum_background_ink+eps)&bg).sum()),
            free_boundary_pixels=int(self.free.sum()),correctness_score=False)
