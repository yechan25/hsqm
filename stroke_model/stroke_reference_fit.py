"""Fit valid reference strokes to extracted stroke positions, not their defects.

Inference-only geometry: principal skeleton paths discard short side branches;
straight references remain straight, curves receive a regularized deformation.
The original reference centerline is deformed and re-rendered at the observed
width. Output is a candidate for inspection, not a ground-truth reconstruction.
"""
from dataclasses import dataclass, asdict
import heapq
import numpy as np
from scipy.ndimage import label, distance_transform_edt, map_coordinates, affine_transform,gaussian_filter1d
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class ReferenceFitConfig:
    points: int=64
    smoothness: float=12.
    shape_prior: float=.15
    max_displacement: float=16.
    max_width: float=10.
    min_width: float=1.
    min_length_ratio: float=.65
    max_length_ratio: float=1.6
    junction_radius: float=2.5
    junction_weight: float=8.
    max_junction_shift: float=2.

    def __post_init__(self):
        if type(self.points) is not int or self.points<8:raise ValueError('At least eight path samples required')
        if any(not np.isfinite(v) or v<0 for v in asdict(self).values()):raise ValueError('Invalid fit configuration')
        if not self.min_width or self.max_width<self.min_width or not 0<self.min_length_ratio<self.max_length_ratio:
            raise ValueError('Invalid width/length limits')


def _unit(array):
    array=np.asarray(array,np.float32)
    if array.ndim!=2 or not np.isfinite(array).all() or array.min()<0 or array.max()>1:
        raise ValueError('Expected a finite 2D ink mask in [0,1]')
    return array


def _component(mask,hint=None):
    threshold=max(.1,min(.4,float(mask.max())*.4))
    binary=mask>=threshold
    labels,count=label(binary,np.ones((3,3)))
    if not count:raise ValueError('Empty/low-confidence extracted stroke')
    distance=None
    if hint is not None and hint.max()>.1:
        distance=distance_transform_edt(hint<max(.1,float(hint.max())*.3))
    scores=[]
    for k in range(1,count+1):
        region=labels==k;mass=float(mask[region].sum())
        proximity=1. if distance is None else np.exp(-float(np.median(distance[region]))/20.)
        scores.append(mass*proximity)
    chosen=labels==(int(np.argmax(scores))+1)
    return chosen,dict(components=count,selected_mass_fraction=float(mask[chosen].sum()/max(float(mask.sum()),1e-6)))


def _resample(path,n):
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    if arc[-1]<1e-6:raise ValueError('Degenerate stroke path')
    keep=np.r_[True,np.diff(arc)>1e-8];arc=arc[keep];path=path[keep]
    t=np.linspace(0,arc[-1],n)
    return np.stack([np.interp(t,arc,path[:,i]) for i in range(2)],1)


def principal_path(binary,n=64):
    """Longest geodesic path of the principal component; never reject spurs."""
    skeleton=skeletonize(binary)
    coordinates=np.argwhere(skeleton)
    if len(coordinates)<2:raise ValueError('Insufficient centerline support')
    indices={tuple(yx):i for i,yx in enumerate(coordinates)}
    graph=[[] for _ in coordinates]
    for i,(y,x) in enumerate(coordinates):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if not (dx or dy):continue
                j=indices.get((y+dy,x+dx))
                if j is not None:graph[i].append((j,float(np.hypot(dx,dy))))
    endpoints=[i for i,edges in enumerate(graph) if len(edges)==1]
    def farthest(start):
        distances=np.full(len(graph),np.inf);distances[start]=0.;parent=np.full(len(graph),-1,int)
        queue=[(0.,start)]
        while queue:
            cost,i=heapq.heappop(queue)
            if cost>distances[i]+1e-8:continue
            for j,weight in graph[i]:
                proposed=cost+weight
                if proposed<distances[j]-1e-8:
                    distances[j]=proposed;parent[j]=i;heapq.heappush(queue,(proposed,j))
        candidates=endpoints if endpoints else list(range(len(graph)))
        end=max(candidates,key=lambda j:distances[j] if np.isfinite(distances[j]) else -1)
        return end,parent
    first,_=farthest(endpoints[0] if endpoints else 0)
    last,parent=farthest(first);order=[last]
    while order[-1]!=first:
        previous=parent[order[-1]]
        if previous<0:raise ValueError('Disconnected centerline')
        order.append(int(previous))
    path=coordinates[order,::-1].astype(float)
    return _resample(path,n),dict(endpoints=len(endpoints),path_coverage=len(order)/len(coordinates),closed=len(endpoints)==0)


def _width(binary,path):
    radius=distance_transform_edt(binary)
    samples=map_coordinates(radius,[path[:,1],path[:,0]],order=1,mode='nearest')
    middle=samples[len(samples)//10:len(samples)-len(samples)//10]
    return max(1.,float(np.median(2*middle-1)))


def render_reference(reference,original_path,target_path,width_scale):
    """Render the deformed original centerline with the estimated stroke width.

    Raster thin hints must not be stretched directly: their one-pixel staircase
    and interpolation profile create serrated, gray thick strokes.
    """
    reference=_unit(reference);h,w=reference.shape
    # Dense samples reduce nearest-segment seams while preserving reference caps.
    # Keep normalized arc correspondence even when relative segment lengths change.
    target=np.stack([np.interp(np.linspace(0,1,512),np.linspace(0,1,len(target_path)),target_path[:,i]) for i in range(2)],1)
    yy,xx=np.mgrid[:h,:w];pixels=np.stack((xx.ravel(),yy.ravel()),1)
    binary=reference>=max(.1,float(reference.max())*.3)
    width=_width(binary,original_path)*width_scale
    distance,_=cKDTree(target).query(pixels)
    return np.clip(.5+(width/2-distance),0,1).reshape(h,w).astype(np.float32)


def _affine_fallback(reference,observed,config):
    """Preserve a loop/branch prototype rather than cutting it into a path."""
    def moments(mask):
        y,x=np.nonzero(mask);xy=np.stack((x,y),1).astype(float)
        center=xy.mean(0);extent=np.percentile(xy,95,axis=0)-np.percentile(xy,5,axis=0)
        return center,np.maximum(extent,1)
    source=reference>max(.1,float(reference.max())*.3)
    rc,re=moments(source);tc,te=moments(observed)
    shift=tc-rc;shift*=min(1.,config.max_displacement/max(np.linalg.norm(shift),1e-6))
    scale=np.clip(te/re,.75,1.35)
    # scipy uses y,x and maps output to input.
    inverse=np.diag(1/scale[::-1]);offset=rc[::-1]-inverse@(rc+shift)[::-1]
    result=affine_transform(reference,inverse,offset=offset,output_shape=reference.shape,order=1,mode='constant')
    return result.clip(0,1).astype(np.float32),dict(translation=shift.tolist(),scale=scale.tolist())


def fit_reference_stroke(reference,extracted,*,hint=None,config=None):
    config=config or ReferenceFitConfig();reference=_unit(reference);extracted=_unit(extracted)
    if reference.shape!=extracted.shape:raise ValueError('Reference/extraction shape mismatch')
    hint=reference if hint is None else _unit(hint)
    if hint.shape!=reference.shape:raise ValueError('Hint shape mismatch')
    observed,component_info=_component(extracted,hint)
    canonical,_=_component(reference)
    original,path_info=principal_path(canonical,config.points)
    metadata=dict(component_info,reference_path=path_info,warnings=[])
    if path_info['closed'] or path_info['path_coverage']<.65 or (path_info['endpoints']>2 and path_info['path_coverage']<.9):
        warped,transform=_affine_fallback(reference,observed,config)
        metadata.update(method='affine_topology_preserving_fallback',transform=transform)
        metadata['warnings'].append('Closed/branched prototype: affine fallback, no open-curve fit')
        return dict(warped=warped,hint=hint,reference=reference,extracted=extracted,
            reference_path=None,target_path=None,hint_path=None,width_scale=1.,metadata=metadata)
    try:
        hint_binary,_=_component(hint)
        initial,_=principal_path(hint_binary,config.points)
        if np.linalg.norm(initial[[0,-1]]-original[[0,-1]])>np.linalg.norm(initial[::-1][[0,-1]]-original[[0,-1]]):initial=initial[::-1]
        if np.max(np.linalg.norm(initial-original,axis=1))>config.max_displacement:
            initial=original.copy();metadata['warnings'].append('Learned hint displacement too large; initialized from reference')
    except ValueError:
        initial=original.copy();metadata['warnings'].append('No usable learned hint; initialized from reference')
    target,target_info=principal_path(observed,config.points)
    if np.linalg.norm(target[[0,-1]]-initial[[0,-1]])>np.linalg.norm(target[::-1][[0,-1]]-initial[[0,-1]]):target=target[::-1]
    metadata['extracted_path']=target_info
    if target_info['closed']:
        target=initial.copy();metadata['warnings'].append('Closed extracted path for open reference: retained initialization')
    if target_info['path_coverage']<.75:metadata['warnings'].append('Side branches excluded by principal-path fit')
    if component_info['selected_mass_fraction']<.65:metadata['warnings'].append('Large disconnected extraction mass excluded')
    chord=original[-1]-original[0];length=np.linalg.norm(np.diff(original,axis=0),axis=1).sum()
    normal=np.array([-chord[1],chord[0]])/max(np.linalg.norm(chord),1e-6)
    deviation=np.max(np.abs((original-original[0])@normal))
    straight=deviation<=1.1 or deviation/max(length,1)<.025
    if straight:
        # Robust TLS: trim off-axis centerline points before estimating direction.
        points=target.copy()
        for _ in range(3):
            center=points.mean(0);_,_,v=np.linalg.svd(points-center,full_matrices=False);axis=v[0]
            residual=np.abs((target-center)@np.array([-axis[1],axis[0]]))
            points=target[residual<=max(.75,float(np.quantile(residual,.85)))]
        projection=(target-center)@axis
        if axis@(initial[-1]-initial[0])<0:axis=-axis;projection=-projection
        endpoints=np.quantile(projection,[.01,.99])
        target=center+np.linspace(*endpoints,config.points)[:,None]*axis
        metadata['method']='robust_straight_reference_fit'
    else:metadata['method']='regularized_curved_reference_fit'
    target_length=np.linalg.norm(np.diff(target,axis=0),axis=1).sum()
    ratio=target_length/max(length,1e-6)
    if not config.min_length_ratio<=ratio<=config.max_length_ratio:
        metadata['warnings'].append('Extracted path length outside allowed range; retained bounded initialization')
        target=initial.copy()
        if straight:target=np.linspace(target[0],target[-1],config.points)
    n=config.points;second=np.diff(np.eye(n),n=2,axis=0)
    system=(1+config.shape_prior)*np.eye(n)+config.smoothness*(second.T@second)
    if straight:
        fitted=target
    else:
        # Smooth the deformation, rather than tracing every observed bend/spur.
        fitted=initial+np.linalg.solve(system,target-initial)
        endpoints=fitted[[0,-1]].copy()
        fitted=gaussian_filter1d(fitted,.8,axis=0,mode='nearest')
        fitted[0],fitted[-1]=endpoints
    displacement=fitted-original
    norm=np.linalg.norm(displacement,axis=1,keepdims=True)
    limit=min(1.,config.max_displacement/max(float(norm.max()),1e-6)) if straight else np.minimum(1,config.max_displacement/np.maximum(norm,1e-6))
    fitted=original+displacement*limit
    ref_width=_width(canonical,original)
    target_width=float(np.clip(_width(observed,target),config.min_width,config.max_width))
    width_scale=float(np.clip(target_width/ref_width,.5,8.))
    warped=render_reference(reference,original,fitted,width_scale)
    metadata.update(width_reference=ref_width,width_observed=target_width,width_scale=width_scale,
        extracted_length_ratio=float(ratio),max_displacement=float(np.linalg.norm(fitted-original,axis=1).max()))
    return dict(warped=warped,hint=hint,reference=reference,extracted=extracted,
        reference_path=original,target_path=fitted,hint_path=initial,width_scale=width_scale,metadata=metadata)


def fit_reference_glyph(strokes,extracted,*,hints=None,glyph_reference=None,config=None):
    config=config or ReferenceFitConfig()
    strokes=np.asarray(strokes,np.float32);extracted=np.asarray(extracted,np.float32)
    if strokes.shape!=extracted.shape or strokes.ndim!=3:raise ValueError('K matching reference/extracted masks required')
    hints=strokes if hints is None else np.asarray(hints,np.float32)
    if hints.shape!=strokes.shape:raise ValueError('Hint shape mismatch')
    fitted=[]
    for k in range(len(strokes)):
        try:item=fit_reference_stroke(strokes[k],extracted[k],hint=hints[k],config=config)
        except ValueError as exc:
            item=dict(warped=strokes[k].copy(),hint=hints[k],reference=strokes[k],extracted=extracted[k],
                reference_path=None,target_path=None,hint_path=None,width_scale=1.,
                metadata=dict(method='unreliable_reference_fallback',warnings=[str(exc)]))
        fitted.append(item)
    # Joint bounded translations keep reference contacts from independently drifting.
    matrix=np.eye(len(strokes));rhs=np.zeros((len(strokes),2));contacts=[]
    for i,a in enumerate(fitted):
        if a['reference_path'] is None:continue
        for j in range(i+1,len(fitted)):
            b=fitted[j]
            if b['reference_path'] is None:continue
            distance=np.linalg.norm(a['reference_path'][:,None]-b['reference_path'][None],axis=2)
            u,v=np.unravel_index(distance.argmin(),distance.shape)
            connected=distance[u,v]<=config.junction_radius
            if not connected and glyph_reference is not None and distance[u,v]<=10:
                line=np.linspace(a['reference_path'][u],b['reference_path'][v],32)
                ink=map_coordinates(_unit(np.asarray(glyph_reference).squeeze()),[line[:,1],line[:,0]],order=1,mode='constant')
                connected=float((ink>.25).mean())>.85
            if not connected:continue
            # Sparse annotations often omit the segment inside a junction. If
            # the full reference confirms a connection, extend an endpoint to
            # the other stroke instead of translating its whole visible tip.
            if glyph_reference is not None:
                ia=0 if u<4 else -1 if u>=len(a['reference_path'])-4 else None
                ib=0 if v<4 else -1 if v>=len(b['reference_path'])-4 else None
                if (ia is None)!=(ib is None):
                    end_item,end_index,other,other_index=(a,ia,b,v) if ia is not None else (b,ib,a,u)
                    difference=other['target_path'][other_index]-end_item['target_path'][end_index]
                    gap=float(np.linalg.norm(difference))
                    if gap<=config.max_displacement:
                        t=np.linspace(0,1,len(end_item['target_path']))
                        factor=(1-t)**3 if end_index==0 else t**3
                        if end_item['metadata'].get('method')=='robust_straight_reference_fit':factor=(1-t) if end_index==0 else t
                        end_item['target_path']=end_item['target_path']+factor[:,None]*difference
                        end_item['metadata'].setdefault('endpoint_extensions',[]).append(gap)
            difference=a['target_path'][u]-b['target_path'][v]
            weight=config.junction_weight
            matrix[i,i]+=weight;matrix[j,j]+=weight;matrix[i,j]-=weight;matrix[j,i]-=weight
            rhs[i]-=weight*difference;rhs[j]+=weight*difference
            contacts.append(dict(strokes=[i,j],indices=[int(u),int(v)],gap_before=float(np.linalg.norm(difference))))
    shift=np.linalg.solve(matrix,rhs)
    shift*=np.minimum(1,config.max_junction_shift/np.maximum(np.linalg.norm(shift,axis=1,keepdims=True),1e-6))
    for i,item in enumerate(fitted):
        if item['target_path'] is None:continue
        item['target_path']=item['target_path']+shift[i]
        displacement=item['target_path']-item['reference_path']
        maximum=float(np.linalg.norm(displacement,axis=1).max())
        if maximum>config.max_displacement:
            item['target_path']=item['reference_path']+displacement*(config.max_displacement/maximum)
            item['metadata']['warnings'].append('Joint displacement clipped; inspect contact consistency')
        item['warped']=render_reference(item['reference'],item['reference_path'],item['target_path'],item['width_scale'])
        item['metadata']['junction_translation']=shift[i].tolist()
        item['metadata']['final_max_displacement']=float(np.linalg.norm(item['target_path']-item['reference_path'],axis=1).max())
    for contact in contacts:
        i,j=contact['strokes'];u,v=contact['indices']
        contact['gap_after']=float(np.linalg.norm(fitted[i]['target_path'][u]-fitted[j]['target_path'][v]))
    masks=np.stack([item['warped'] for item in fitted])
    return dict(strokes=masks,candidate=masks.max(0),fits=fitted,contacts=contacts,config=asdict(config))
