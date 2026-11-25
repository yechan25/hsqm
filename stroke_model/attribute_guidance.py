"""Straighten ONE observed centerline, retaining endpoint direction and width profile."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,distance_transform_edt,map_coordinates
from scipy.spatial import cKDTree
from .stroke_reference_fit import principal_path
from .target_stroke_plan import TargetImageGuide,TargetPlanConfig


def straightness_plan(source,masks,glyph,stroke,*,amount=.6,blend=.4):
    if not 0<=amount<=1 or not 0<blend<=1:raise ValueError('Invalid correction amount')
    rgb=np.asarray(source.convert('RGB'),np.float32)/255
    if len(masks)!=4 or rgb.shape[0]!=rgb.shape[1] or rgb.shape[0]%2:raise ValueError('Expected a 2x2 sheet')
    if not 0<=glyph<4:raise ValueError('Invalid glyph index')
    observed=np.asarray(masks[glyph],np.float32)
    if observed.ndim!=3 or not 0<=stroke<len(observed):raise ValueError('Invalid stroke index')
    if not np.isfinite(observed).all() or observed.min()<0 or observed.max()>1:raise ValueError('Invalid extracted masks')
    binary=observed[stroke]>.35
    path,info=principal_path(binary,n=128)
    if info['closed'] or info['path_coverage']<.8:raise ValueError('폐곡선 또는 분기가 많은 획입니다. 다른 직선 획을 고르세요.')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    chord=np.linalg.norm(path[-1]-path[0])
    if chord<3 or arc[-1]/chord>1.4:raise ValueError('짧거나 꺾인 획입니다. 한 방향의 직선 획을 고르세요.')
    t=arc/arc[-1];line=path[0]+t[:,None]*(path[-1]-path[0])
    changed=(1-amount)*path+amount*line
    radius=map_coordinates(distance_transform_edt(binary),[path[:,1],path[:,0]],order=1,mode='nearest')
    dense_t=np.linspace(0,1,1024)
    dense=np.stack([np.interp(dense_t,t,changed[:,axis]) for axis in range(2)],axis=1)
    radii=np.interp(dense_t,t,radius)
    yy,xx=np.indices(binary.shape);distance,near=cKDTree(dense).query(np.stack([xx,yy],-1))
    alpha=np.clip(radii[near]+.5-distance,0,1).astype(np.float32)
    others=np.delete(observed,stroke,axis=0)
    protected=binary_dilation(others.max(0)>.3,iterations=1) if len(others) else np.zeros_like(binary)
    region=binary_dilation(binary|(alpha>.03),iterations=2)&~protected
    side=rgb.shape[0]//2;y=(glyph//2)*side;x=(glyph%2)*side
    crop=rgb[y:y+side,x:x+side]
    resize=lambda a,mode:np.asarray(Image.fromarray(a).resize((side,side),mode))
    a=resize(alpha,Image.Resampling.BICUBIC).clip(0,1)
    edit=resize(region,Image.Resampling.NEAREST).astype(bool)
    if not edit.any():raise ValueError('다른 획과 겹쳐 수정할 영역이 없습니다. 다른 획을 고르세요.')
    old=resize(observed[stroke],Image.Resampling.BILINEAR)
    # White/plain background experiment: exclude faint old edges before sampling background.
    foreground=binary_dilation((1-crop.mean(-1))>.02,iterations=2)
    if foreground.all():raise ValueError('No clean background available')
    _,indices=distance_transform_edt(foreground,return_indices=True)
    background=crop[indices[0],indices[1]]
    samples=crop[old>.5]
    if not len(samples):raise ValueError('Selected stroke has insufficient visible ink')
    color=np.median(samples,axis=0)
    patch=background*(1-a[...,None])+color*a[...,None]
    target=rgb.copy();target_crop=target[y:y+side,x:x+side];target_crop[edit]=patch[edit]
    weights=np.zeros(rgb.shape[:2],np.float32);weights[y:y+side,x:x+side]=edit*blend
    if amount==0:target=rgb.copy();weights[:]=0
    full_region=weights>0
    new_length=float(np.linalg.norm(np.diff(changed,axis=0),axis=1).sum())
    report=dict(attribute='straightness',glyph=glyph+1,stroke=stroke+1,amount=amount,
        endpoint_positions_before=path[[0,-1]].tolist(),endpoint_positions_after=changed[[0,-1]].tolist(),
        chord_length=float(chord),path_length_before=float(arc[-1]),path_length_after=new_length,
        tortuosity_before=float(arc[-1]/chord),tortuosity_after=float(new_length/chord),
        width_profile_percentiles=(2*np.percentile(radius,[10,50,90])).tolist(),
        direction_degrees=float(np.rad2deg(np.arctan2(*(path[-1]-path[0])[::-1]))),
        strokes=[dict(glyph=glyph+1,stroke=stroke+1,score=None,edit_strength=amount)],
        limitations=['Only user-selected straight strokes; not a correctness detector.',
            'Rasterized width/endcaps can differ despite retaining the sampled width profile.',
            'Protected crossings may prevent full correction. Plain background only.',
            'Diffusion/VAE can change unselected areas; inspect raw results.'])
    guide=TargetImageGuide(rgb,target,weights,[full_region],report,TargetPlanConfig(maximum_step_blend=blend))
    guide.metadata.update(kind='single_stroke_straightness',score_role='none; user-selected attribute',
                          whole_glyph_affine=False,canonical_shape_replacement=False)
    return guide
