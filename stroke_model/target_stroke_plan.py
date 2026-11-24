"""Observed widths -> canonical strokes -> one glyph-level affine -> edit plan.

Scores select edit strength; they are not differentiated through a generator.
The built-in geometry scores are explicit uncalibrated engineering proxies, NOT
a reproduction of Li & Ren's calligraphy evaluator or extraction confidence.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation, convolve
from skimage.morphology import skeletonize
import torch

from .source_structure import SourceConfig, register_reference, affine_warp


@dataclass(frozen=True)
class TargetPlanConfig:
    mask_threshold: float = .35
    keep_score: float = 90.
    full_edit_score: float = 40.
    maximum_step_blend: float = .8
    edit_margin: int = 2
    width_min_samples: int = 6

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):raise ValueError('Nonfinite config')
        if not 0<self.mask_threshold<1 or not 0<=self.full_edit_score<self.keep_score<=100:
            raise ValueError('Invalid threshold or score range')
        if not 0<self.maximum_step_blend<=1 or type(self.edit_margin) is not int or self.edit_margin<0:
            raise ValueError('Invalid blend/margin')
        if type(self.width_min_samples) is not int or self.width_min_samples<1:raise ValueError('Invalid sample count')


def _features(mask):
    binary=mask>.35;path=skeletonize(binary)
    if not path.any():return dict(present=False)
    # Undirected 8-neighbour edge length, counting each edge once.
    length=0.
    for dy,dx,cost in [(0,1,1.),(1,0,1.),(1,1,2**.5),(1,-1,2**.5)]:
        a=path[:path.shape[0]-dy or None, max(0,-dx):path.shape[1]-max(0,dx)]
        b=path[dy:,max(0,dx):path.shape[1]-max(0,-dx)]
        length+=float((a&b).sum())*cost
    xy=np.argwhere(path)[:,::-1].astype(float)
    center=xy.mean(0);cov=np.cov(xy.T) if len(xy)>2 else np.eye(2)
    values,vectors=np.linalg.eigh(cov);axis=vectors[:,-1]
    ambiguous=bool(values[-1]<1.15*max(values[0],1e-6))
    angle=float(np.arctan2(axis[1],axis[0]))
    # Bend is a dimensionless path/span proxy, not circle-fit curvature.
    span=max(float(np.linalg.norm(xy.max(0)-xy.min(0))),1.)
    return dict(present=True,length=length,width=float(2*np.median(distance_transform_edt(binary)[path])),
        angle=angle,angle_ambiguous=ambiguous,bend=length/span,
        center=center.tolist(),area=int(binary.sum()))


def evaluate_stroke(observed,target):
    a,b=_features(observed),_features(target)
    if not a['present']:
        return dict(score=0.,source='geometry_proxy_v1',observed=a,target=b,reason='absent_observed_stroke')
    length_ratio=1. if a['length']<1 and b['length']<1 else a['length']/max(b['length'],1.)
    width_ratio=a['width']/max(b['width'],1e-6)
    angle_delta=float((a['angle']-b['angle']+np.pi/2)%np.pi-np.pi/2)
    angle_used=not (a['angle_ambiguous'] or b['angle_ambiguous'])
    bend_error=abs(a['bend']-b['bend'])
    ma,mb=observed>.35,target>.35
    dice=float(2*(ma&mb).sum()/max(int(ma.sum()+mb.sum()),1))
    # Width is measured and reported, but NOT forced back to template width.
    error=(abs(math.log(max(length_ratio,1e-6)))+
           (abs(angle_delta)/(np.pi/4) if angle_used else 0.)+.5*bend_error+(1-dice))
    return dict(score=float(100*np.exp(-error)),source='geometry_proxy_v1',observed=a,target=b,
        length_ratio=length_ratio,width_ratio=width_ratio,angle_delta_radians=angle_delta,
        angle_used=angle_used,bend_error=bend_error,body_dice=dice)


def measure_widths(observed,config):
    estimates=[]
    for mask in observed:
        binary=mask>config.mask_threshold;path=skeletonize(binary)
        degree=convolve(path.astype(int),np.ones((3,3),int),mode='constant')-path
        stable=path&(degree==2)
        values=distance_transform_edt(binary)[stable]
        estimate=float(np.median(values)) if len(values)>=config.width_min_samples else None
        estimates.append(estimate)
    reliable=[w for w in estimates if w is not None and w>0]
    if not reliable:raise ValueError('No measurable observed stroke widths; defer this glyph')
    median=float(np.median(reliable))
    return [dict(half_width=w if w is not None else median,
                 source='observed_stroke' if w is not None else 'observed_glyph_median') for w in estimates]


def _background(rgb,foreground):
    if foreground.all():raise ValueError('No background pixels available')
    _,indices=distance_transform_edt(foreground,return_indices=True)
    return rgb[indices[0],indices[1]]


class TargetImageGuide:
    def __init__(self,source,target,weights,stroke_regions,report,config):
        self.source=np.asarray(source,np.float32);self.target=np.asarray(target,np.float32)
        self.weights=np.asarray(weights,np.float32)
        self.stroke_regions=stroke_regions;self.report=report;self.config=config
        self.roi=self.weights>0
        self.metadata=dict(kind='observed_width_aligned_target_image',config=asdict(config),
            score_role='frozen edit selection and blend strength; not a scalar reward gradient',
            final_pixel_replacement=False)

    @torch.no_grad()
    def project(self,rgb):
        if rgb.shape!=(1,3,*self.weights.shape):raise ValueError('Expected matching RGB sheet')
        target=torch.as_tensor(self.target.copy(),device=rgb.device).permute(2,0,1)[None].to(rgb)
        weight=torch.as_tensor(self.weights.copy(),device=rgb.device)[None,None].to(rgb)
        return rgb+weight*(target-rgb)

    @torch.no_grad()
    def diagnostics(self,rgb):
        target=torch.as_tensor(self.target.copy(),device=rgb.device).permute(2,0,1)[None].to(rgb)
        error=(rgb-target).square().mean(1)[0]
        rows=[]
        for region,item in zip(self.stroke_regions,self.report['strokes']):
            mask=torch.as_tensor(region.copy(),device=rgb.device)
            rows.append(dict(glyph=item['glyph'],stroke=item['stroke'],original_score=item['score'],
                edit_strength=item['edit_strength'],target_rgb_mse=float(error[mask].mean()) if mask.any() else 0.))
        return dict(strokes=rows,target_rgb_mse=float(error.mean()),correctness_score=False)


def build_target_plan(source,conditions,observed_masks,*,scores=None,config=None,registration_config=None):
    """Build a full target IMAGE before deriving local, score-controlled edits.

    Optional scores: four lists of 0..100 consistency scores, in stroke-ID order.
    Omit scores to use the disclosed geometry proxy. Correct extraction/canonical
    stroke correspondence is assumed; extraction reliability is not predicted.
    """
    cfg=config or TargetPlanConfig();source=np.asarray(source.convert('RGB'),np.float32)/255
    h,w=source.shape[:2]
    if len(conditions)!=4 or len(observed_masks)!=4 or h!=w or h%2:raise ValueError('Expected a 2x2 sheet')
    if scores is not None and len(scores)!=4:raise ValueError('Need four per-glyph score lists')
    target=source.copy();weights=np.zeros((h,w),np.float32);regions=[];rows=[];glyph_reports=[];rendered=[]
    for gi,(condition,observed) in enumerate(zip(conditions,observed_masks)):
        strokes=np.asarray(condition['strokes'],np.float32);observed=np.asarray(observed,np.float32)
        if observed.shape!=strokes.shape or strokes.ndim!=3 or not len(strokes):raise ValueError('Stroke masks must correspond')
        if any(not np.isfinite(a).all() or a.min()<0 or a.max()>1 for a in (strokes,observed)):
            raise ValueError('Expected finite ink masks in [0,1]')
        size=strokes.shape[-2:];side=w//2;x0=(gi%2)*side;y0=(gi//2)*side
        crop=source[y0:y0+side,x0:x0+side];luma=crop@np.array([.299,.587,.114])
        widths=measure_widths(observed,cfg)
        masks=[]
        for stroke,width in zip(strokes,widths):
            path=skeletonize(stroke>.35)
            if not path.any():raise ValueError('Empty canonical stroke')
            # Render EVERY skeleton, including closed curves, at measured width.
            masks.append(np.clip(width['half_width']+.5-distance_transform_edt(~path),0,1))
        masks=np.asarray(masks,np.float32)
        source_ink=np.asarray(Image.fromarray((1-luma).astype(np.float32)).resize(size[::-1],Image.Resampling.BILINEAR))
        _,_,registration=register_reference(source_ink,masks.max(0),masks,registration_config or SourceConfig())
        aligned=np.stack([affine_warp(mask,registration['parameters']) for mask in masks]).clip(0,1)
        supplied=None if scores is None else scores[gi]
        if supplied is not None and (len(supplied)!=len(strokes) or any(not math.isfinite(s) or not 0<=s<=100 for s in supplied)):
            raise ValueError('Scores must match stroke IDs and be in [0,100]')
        foreground=(1-luma)>.35
        background=_background(crop,foreground)
        patch=background.copy()
        glyph_weights=np.zeros(size,np.float32)
        for k,(alpha,actual,width) in enumerate(zip(aligned,observed,widths)):
            evaluation=evaluate_stroke(actual,alpha)
            score=evaluation['score'] if supplied is None else float(supplied[k])
            strength=float(np.clip((cfg.keep_score-score)/(cfg.keep_score-cfg.full_edit_score),0,1))
            region=(alpha>.03)|(actual>.15)
            if cfg.edit_margin:region=binary_dilation(region,iterations=cfg.edit_margin)
            glyph_weights=np.maximum(glyph_weights,region*strength*cfg.maximum_step_blend)
            resized=np.asarray(Image.fromarray(alpha).resize((side,side),Image.Resampling.BICUBIC)).clip(0,1)
            observation=np.asarray(Image.fromarray(actual).resize((side,side),Image.Resampling.BILINEAR))
            samples=crop[observation>.5]
            if not len(samples):samples=crop[foreground]
            color=np.median(samples,axis=0) if len(samples) else np.zeros(3)
            patch=patch*(1-resized[...,None])+color*resized[...,None]
            full=np.zeros((h,w),bool)
            full[y0:y0+side,x0:x0+side]=np.asarray(Image.fromarray(region).resize((side,side),Image.Resampling.NEAREST))
            regions.append(full)
            rows.append(dict(glyph=gi+1,stroke=k+1,score=score,
                score_source=evaluation['source'] if supplied is None else 'provided_per_stroke_score',
                edit_strength=strength,width=width,measurements=evaluation))
        target[y0:y0+side,x0:x0+side]=patch
        weights[y0:y0+side,x0:x0+side]=np.asarray(Image.fromarray(glyph_weights).resize((side,side),Image.Resampling.BILINEAR))
        glyph_reports.append(dict(glyph=gi+1,bbox=[x0,y0,x0+side,y0+side],registration=registration))
        rendered.append(dict(thick_strokes=masks,aligned_strokes=aligned))
    report=dict(strokes=rows,glyphs=glyph_reports,target_built_from='canonical skeletons + observed widths, then shared glyph affine',
        score_calibration='uncalibrated geometry proxy unless supplied; not paper reproduction',
        width_used_for='appearance preservation, not forced to reference thickness',
        limitations=['Canonical stroke correspondence is assumed correct.','Geometry proxy is not a correctness probability.',
            'Nearest visible background is only a provisional target background, not scene inpainting.',
            'Registration may fail on severe errors; inspect parameters and Dice before interpreting results.'])
    return TargetImageGuide(source,target,weights,regions,report,cfg),rendered
