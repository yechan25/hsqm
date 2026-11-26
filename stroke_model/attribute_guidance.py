"""Map one canonical stroke to its observed position, direction and width."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,distance_transform_edt,gaussian_filter1d,label,map_coordinates
from scipy.spatial import cKDTree
from .stroke_reference_fit import principal_path
from .target_stroke_plan import TargetImageGuide,TargetPlanConfig


def _open_path(binary):
    labels,count=label(binary,np.ones((3,3)))
    if not count:raise ValueError('추출된 획이 비어 있습니다.')
    largest=np.argmax(np.bincount(labels[binary]))
    component=labels==largest
    path,info=principal_path(component,n=128)
    if info['closed'] or info['path_coverage']<.8 or np.linalg.norm(path[-1]-path[0])<3:
        raise ValueError('두 끝점을 안정적으로 찾기 어렵습니다. 다른 획을 고르세요.')
    return path,component


def mapped_stroke_plan(source,masks,conditions,glyph,stroke,*,blend=.4):
    if not 0<blend<=1:raise ValueError('Invalid blend')
    rgb=np.asarray(source.convert('RGB'),np.float32)/255
    if len(masks)!=4 or rgb.shape[0]!=rgb.shape[1] or rgb.shape[0]%2:raise ValueError('Expected a 2x2 sheet')
    if not 0<=glyph<4:raise ValueError('Invalid glyph index')
    observed=np.asarray(masks[glyph],np.float32)
    if observed.ndim!=3 or not 0<=stroke<len(observed):raise ValueError('Invalid stroke index')
    if not np.isfinite(observed).all() or observed.min()<0 or observed.max()>1:raise ValueError('Invalid extracted masks')
    canonical=np.asarray(conditions[glyph]['strokes'],np.float32)
    if canonical.shape!=observed.shape:raise ValueError('기준 획과 추출 획의 대응이 맞지 않습니다.')
    path,binary=_open_path(observed[stroke]>.35)
    template,_=_open_path(canonical[stroke]>.35)
    observed_axis=path[-1]-path[0]
    if (template[-1]-template[0])@observed_axis<0:template=template[::-1]
    # Smooth the canonical raster path lightly and pin its two ends.
    smooth=gaussian_filter1d(template,2.,axis=0)
    t=np.linspace(0,1,len(template))[:,None]
    smooth+=(1-t)*(template[0]-smooth[0])+t*(template[-1]-smooth[-1])
    axis=template[-1]-template[0]
    basis=lambda v: np.stack([v,[-v[1],v[0]]],axis=1)
    transform=basis(observed_axis)@np.linalg.inv(basis(axis))
    mapped=(smooth-template[0])@transform.T+path[0]
    middle=path[len(path)//5:-len(path)//5]
    radius=float(np.median(map_coordinates(distance_transform_edt(binary),
        [middle[:,1],middle[:,0]],order=1)))
    dense=np.stack([np.interp(np.linspace(0,1,1024),t[:,0],mapped[:,i]) for i in range(2)],axis=1)
    others=np.delete(observed,stroke,axis=0)
    protected=binary_dilation(others.max(0)>.3,iterations=1) if len(others) else np.zeros_like(binary)
    side=rgb.shape[0]//2;y=(glyph//2)*side;x=(glyph%2)*side
    crop=rgb[y:y+side,x:x+side]
    resize=lambda a,mode:np.asarray(Image.fromarray(a).resize((side,side),mode))
    # Render directly at output resolution to avoid resampling a jagged mask.
    sy,sx=np.array([side,side])/binary.shape
    yy,xx=np.indices((side,side));points=np.stack([(xx+.5)/sx-.5,(yy+.5)/sy-.5],-1)
    distance,_=cKDTree(dense).query(points)
    a=np.clip((radius-distance)*sx+.5,0,1).astype(np.float32)
    region=binary_dilation(resize(binary,Image.Resampling.NEAREST).astype(bool)|(a>.03),
        iterations=max(1,round(2*sx))) & ~resize(protected,Image.Resampling.NEAREST).astype(bool)
    edit=region
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
    full_region=weights>0
    report=dict(attribute='canonical_stroke_mapping',glyph=glyph+1,stroke=stroke+1,
        observed_endpoints=path[[0,-1]].tolist(),mapped_endpoints=mapped[[0,-1]].tolist(),
        chord_length=float(np.linalg.norm(observed_axis)),representative_width=float(2*radius),
        direction_degrees=float(np.rad2deg(np.arctan2(*observed_axis[::-1]))),
        strokes=[dict(glyph=glyph+1,stroke=stroke+1,score=None,edit_strength=blend)],
        limitations=['Requires correct stroke correspondence and usable endpoints.',
            'Protected crossings may prevent full correction. Plain background only.',
            'Diffusion/VAE can change unselected areas; inspect raw results.'])
    guide=TargetImageGuide(rgb,target,weights,[full_region],report,TargetPlanConfig(maximum_step_blend=blend))
    guide.metadata.update(kind='single_canonical_stroke_mapping',score_role='none; user-selected stroke',
                          whole_glyph_affine=False,canonical_shape_replacement=True)
    return guide
