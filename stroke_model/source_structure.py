"""Source-image errors after ONE bounded affine registration of the target.

Extractor channel mistakes are diagnostics, not evidence of missing source ink.
Registration is CPU evaluation preprocessing, not differentiable guidance.
PreparedSourceMetric retains image gradients with the registration held fixed.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt
from scipy.optimize import minimize
import torch
from torch import nn
from torch.nn import functional as F
from .stroke_structure import (StructureConfig, _Stroke, _pool, reconstruct,
    soft_occupancy, soft_skeleton, soft_distance_to_background, occupancy_ring_arms)


@dataclass(frozen=True)
class SourceConfig:
    translation: float = 12.
    rotation: float = 15.
    scale_min: float = .85
    scale_max: float = 1.2
    shear: float = .2
    registration_iterations: int = 45
    registration_prior: float = .00001
    extra_radius: float = 2.
    extra_transition: float = .5
    junction_tolerance: float = .5
    junction_position_tolerance: int = 2
    coverage_radius: int = 2
    corridor_radius: int = 3

    def __post_init__(self):
        if any(not math.isfinite(x) for x in asdict(self).values()):
            raise ValueError('Source settings must be finite')
        if (self.translation <= 0 or self.rotation <= 0 or self.shear <= 0
                or not 0 < self.scale_min <= 1 <= self.scale_max or self.scale_min == self.scale_max
                or self.registration_iterations < 1 or self.registration_prior < 0
                or self.extra_radius < 0 or self.extra_transition <= 0 or self.junction_tolerance < 0):
            raise ValueError('Invalid source settings')
        for key in ('registration_iterations','junction_position_tolerance','coverage_radius','corridor_radius'):
            if not isinstance(getattr(self,key), int) or getattr(self,key)<0:
                raise ValueError('Source iteration counts and radii must be nonnegative integers')


def affine_warp(array, parameters):
    tx, ty, angle, sx, sy, shear = parameters
    angle = np.deg2rad(angle)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = rotation @ np.array([[sy, shear], [0, sx]])  # y,x convention
    inverse = np.linalg.inv(matrix)
    center = (np.asarray(array.shape)-1)/2
    offset = center-inverse@(center+np.array([ty,tx]))
    return affine_transform(array, inverse, offset, order=1, mode='constant')


def register_reference(image, reference, strokes, config=None):
    cfg = config or SourceConfig()
    image, reference, strokes = [np.asarray(x, dtype=np.float32) for x in (image,reference,strokes)]
    image, reference = image.squeeze(), reference.squeeze()
    if (image.ndim!=2 or reference.shape!=image.shape or strokes.ndim!=3
            or strokes.shape[1:]!=image.shape or not len(strokes)):
        raise ValueError('Need matching input H,W, full target H,W and target strokes K,H,W')
    for value in (image,reference,strokes):
        if not np.isfinite(value).all() or (value<0).any() or (value>1).any():
            raise ValueError('Source arrays must be finite absolute ink in [0,1]')
    if reference.max()<=.5 or any(s.max()<=.5 for s in strokes):
        raise ValueError('Target and all expected target strokes must be nonempty')
    def objective(parameters):
        moved = affine_warp(reference, parameters)
        dice = 2*(image*moved).sum()/(np.square(image).sum()+np.square(moved).sum()+1e-6)
        return 1-dice+cfg.registration_prior*sum(v*v for v in parameters[:3])
    bounds=[(-cfg.translation,cfg.translation),(-cfg.translation,cfg.translation),
            (-cfg.rotation,cfg.rotation),(cfg.scale_min,cfg.scale_max),
            (cfg.scale_min,cfg.scale_max),(-cfg.shear,cfg.shear)]
    fit=minimize(objective,[0,0,0,1,1,0],method='Powell',bounds=bounds,
                 options={'maxiter':cfg.registration_iterations,'xtol':.001,'ftol':.00001})
    # An optimizer result may be worse than identity; preserve the best of both.
    parameters = fit.x if objective(fit.x)<objective([0,0,0,1,1,0]) else np.array([0,0,0,1,1,0])
    target=affine_warp(reference,parameters)
    dice=2*(image*target).sum()/(np.square(image).sum()+np.square(target).sum()+1e-6)
    # Keep reference centerline topology intact; sample the live input into
    # reference coordinates instead of re-skeletonizing a raster-warped target.
    return reference, strokes, {'parameters':parameters.tolist(),'dice':float(dice),
                            'optimizer_converged':bool(fit.success),'fixed_for_gradients':True}


class PreparedSourceMetric(nn.Module):
    def __init__(self, aligned_target, aligned_strokes, config=None, parameters=None):
        super().__init__()
        self.config=config or SourceConfig()
        self.structure_config=StructureConfig(coverage_radius=self.config.coverage_radius,
            corridor_radius=self.config.corridor_radius, reference_threshold=.5)
        self.strokes=nn.ModuleList([_Stroke(s,self.structure_config) for s in aligned_strokes])
        self.register_buffer('target',torch.tensor(aligned_target,dtype=torch.float32)[None,None])
        distance=distance_transform_edt(np.asarray(aligned_target)<.3)
        support=1/(1+np.exp(np.clip((distance-self.config.extra_radius)/self.config.extra_transition,-80,80)))
        # Exact support=1 inside target ink; avoid charging even an identical
        # image for the sigmoid's nonzero interior tail.
        support=np.minimum(support/(1/(1+np.exp(-self.config.extra_radius/self.config.extra_transition))),1)
        self.register_buffer('support',torch.tensor(support,dtype=torch.float32)[None,None])
        pose=parameters if parameters is not None else [0,0,0,1,1,0]
        source_support=affine_warp(support,pose)
        self.register_buffer('source_support',torch.tensor(source_support,dtype=torch.float32)[None,None])
        weights=np.stack([np.exp(-distance_transform_edt(~(s>.15))/8.) for s in aligned_strokes])
        self.register_buffer('junction_weights',torch.tensor(weights,dtype=torch.float32))
        h,w=self.target.shape[-2:]
        tx,ty,angle,sx,sy,shear=parameters if parameters is not None else [0,0,0,1,1,0]
        angle=np.deg2rad(angle)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        matrix=rotation@np.array([[sy,shear],[0,sx]])
        yy,xx=np.indices((h,w));center=(np.array([h,w])-1)/2
        positions=matrix@np.stack([yy-center[0],xx-center[1]]).reshape(2,-1)
        positions+= (center+np.array([ty,tx]))[:,None]
        grid=np.stack([(positions[1].reshape(h,w)+.5)*2/w-1,
                       (positions[0].reshape(h,w)+.5)*2/h-1],axis=-1)
        self.register_buffer('sampling_grid',torch.tensor(grid,dtype=torch.float32)[None])

    def forward(self,image):
        image=torch.as_tensor(image)
        if image.ndim==2:image=image[None,None]
        elif image.ndim==3:image=image[None]
        if image.ndim!=4 or image.shape[1:]!=self.target.shape[1:]:
            raise ValueError('Image must be H,W, 1,H,W or B,1,H,W matching the prepared target')
        if not image.is_floating_point():image=image.float()
        if image.device!=self.target.device or not bool(torch.isfinite(image).all()) or bool(((image<0)|(image>1)).any()):
            raise ValueError('Image must be finite ink [0,1] on the metric device')
        source_image=image
        image=F.grid_sample(image,self.sampling_grid.to(image).expand(image.shape[0],-1,-1,-1),align_corners=False)
        cfg=self.structure_config
        missing,split=[],[]
        for stroke in self.strokes:
            coverage=_pool(image,cfg.coverage_radius)[:,0,stroke.ys,stroke.xs]
            missing.append(F.relu(1-coverage/cfg.required_ink).square().mean(-1))
            pairs=[]
            for pair in stroke.pairs:
                ink=image[...,pair.slices[0],pair.slices[1]]*pair.corridor
                reached,_=reconstruct(ink,ink*pair.a)
                u=(ink*pair.a).flatten(1).amax(1)
                v=(ink*pair.b).flatten(1).amax(1)
                c=(reached*pair.b).flatten(1).amax(1)
                pairs.append(u*v*F.relu((torch.minimum(u,v)-c)/torch.minimum(u,v).clamp_min(cfg.eps)).square())
            split.append(torch.stack(pairs,-1).mean(-1))
        levels=[]
        for threshold in cfg.junction_thresholds:
            occupancy=soft_occupancy(image,threshold,cfg.junction_temperature)
            expected=soft_occupancy(self.target.to(image),threshold,cfg.junction_temperature).expand_as(image)
            radius=cfg.junction_probe_scale*soft_distance_to_background(occupancy,cfg.junction_distance_steps)+cfg.junction_probe_offset
            arms=occupancy_ring_arms(occupancy,radius,cfg.junction_angular_samples,cfg.junction_angular_smoothing)
            target_arms=occupancy_ring_arms(expected,radius,cfg.junction_angular_samples,cfg.junction_angular_smoothing)
            allowed=_pool(target_arms,self.config.junction_position_tolerance).clamp_min(2)
            excess=F.relu(arms-allowed-self.config.junction_tolerance).square()
            levels.append(excess*soft_skeleton(occupancy,cfg.skeleton_iterations))
        junction_map=torch.stack(levels).mean(0)
        junction=torch.stack([(junction_map[:,0]*w).flatten(1).sum(1)/s.length
                              for w,s in zip(self.junction_weights,self.strokes)],-1)
        # Inspect EVERY original pixel, including margins outside the registered
        # canonical view; an affine crop must never hide an added stroke.
        extra_map=source_image*(1-self.source_support)
        extra=extra_map.flatten(1).sum(1)/source_image.flatten(1).sum(1).clamp_min(1)
        return {'missing':torch.stack(missing,-1),'split':torch.stack(split,-1),'junction':junction,
                'extra':extra,'extra_map':extra_map,'junction_map':junction_map,'support':self.source_support,
                'registered_image':image,'source_image':source_image}
