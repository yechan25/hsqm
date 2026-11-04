"""Training-only, tangent-weighted sparse affinity supervision.

Inspired by AAF (ECCV 2018), not its adversarial radius-learning algorithm.
Actual input seed geometry supplies local directions; reference conditioning
remains in the unchanged V2 head. No canonical-pixel matching or thinning.
"""
import math
import torch
from torch.nn import functional as F


@torch.no_grad()
def seed_directions(labels, count, window=7):
    """Local axial PCA: cos(2 theta), sin(2 theta), coherence at known seeds.

    Corners/circles with ambiguous local orientation automatically lose weight.
    One-pixel and absent annotations do not invent a direction.
    """
    seeds=F.one_hot(labels.clamp_min(0),count).permute(0,3,1,2).float()
    seeds=seeds*(labels[:,None]>=0)
    h,w=labels.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=labels.device,dtype=torch.float32),
                       torch.arange(w,device=labels.device,dtype=torch.float32),indexing='ij')
    fields=torch.stack((seeds,seeds*x,seeds*y,seeds*x*x,seeds*x*y,seeds*y*y),2)
    b,k=seeds.shape[:2]
    moments=F.avg_pool2d(fields.reshape(b*k,6,h,w),window,1,window//2).reshape(b,k,6,h,w)
    mass=moments[:,:,0];denom=mass.clamp_min(1e-8)
    cx,cy=moments[:,:,1]/denom,moments[:,:,2]/denom
    xx=(moments[:,:,3]/denom-cx.square()).clamp_min(0)
    xy=moments[:,:,4]/denom-cx*cy
    yy=(moments[:,:,5]/denom-cy.square()).clamp_min(0)
    delta=xx-yy;twoxy=2*xy
    magnitude=torch.sqrt(delta.square()+twoxy.square()).clamp_min(1e-6)
    coherence=(magnitude/(xx+yy).clamp_min(1e-6)).clamp(0,1)
    coherence=coherence*((mass*window*window)>=2)*(labels[:,None]>=0)
    ids=labels.clamp_min(0)[:,None]
    return tuple(v.gather(1,ids).squeeze(1) for v in
                 (delta/magnitude,twoxy/magnitude,coherence))


def directional_affinity_loss(output,labels,owners=None,reliability=None,radii=(1,3,5),margin=1.):
    """Symmetric KL for same owners, margin separation for different owners.

    Evaluate four axes at three fixed radii. Tangential same-owner pairs and
    transverse different-owner pairs receive more weight. Balance by anchor
    stroke AND same/different relation, so large strokes cannot dominate.
    Unknown pixels/overlapping seeds never become negative targets. Optional
    weak ownership is used only where the existing conservative cache trusts it.
    """
    if not radii or any(type(r) is not int or r<1 for r in radii):
        raise ValueError('radii must be positive integers')
    if not math.isfinite(margin) or margin<=0:raise ValueError('margin must be positive')
    p=output['probabilities'].float()
    if labels.shape!=p.shape[:1]+p.shape[2:]:raise ValueError('labels must be B,H,W')
    ids=labels if owners is None else owners
    weights=(ids>=0).float() if reliability is None else reliability.float()*(ids>=0)
    if ids.shape!=labels.shape or weights.shape!=labels.shape:raise ValueError('Invalid owner maps')
    # Even when weak regions are enabled, ambiguity in true seed overlaps is not
    # filled here. The cache's original policy is authoritative.
    with torch.no_grad():
        c,s,coherence=seed_directions(labels,p.shape[1])
    safe=p.clamp_min(1e-6);logs=safe.log()
    totals=p.new_zeros((p.shape[0],2*p.shape[1]))+p.sum()*0;masses=torch.zeros_like(totals)
    h,w=labels.shape[-2:]
    for radius in radii:
        for dy,dx in ((0,radius),(radius,0),(radius,radius),(radius,-radius)):
            if abs(dy)>=h or abs(dx)>=w:continue
            ya=slice(max(0,-dy),min(h,h-dy));yb=slice(max(0,dy),min(h,h+dy))
            xa=slice(max(0,-dx),min(w,w-dx));xb=slice(max(0,dx),min(w,w+dx))
            a=(...,ya,xa);b=(...,yb,xb)
            same=ids[a]==ids[b]
            valid=(ids[a]>=0)&(ids[b]>=0)
            kl=.5*((p[a]-p[b])*(logs[a]-logs[b])).sum(1)
            # Axial orientation makes endpoint order irrelevant.
            norm=dx*dx+dy*dy
            align_a=.5*(1+c[a]*(dx*dx-dy*dy)/norm+s[a]*(2*dx*dy)/norm)
            align_b=.5*(1+c[b]*(dx*dx-dy*dy)/norm+s[b]*(2*dx*dy)/norm)
            align=(align_a+align_b)*.5
            confidence=(coherence[a]+coherence[b])*.5
            direction=torch.where(same,align,1-align).clamp(0,1)
            weight=torch.minimum(weights[a],weights[b])*valid*(1+confidence*direction)
            values=torch.where(same,kl,F.relu(margin-kl))
            # Both endpoints get equal credit: swapping stroke ordering cannot
            # change the loss or make a small anchor disappear in a large class.
            for endpoint in (ids[a],ids[b]):
                category=endpoint.clamp_min(0)+same.long()*p.shape[1]
                totals=totals.scatter_add(1,category.flatten(1),(values*weight).flatten(1))
                masses=masses.scatter_add(1,category.flatten(1),weight.flatten(1))
    supported=masses>0
    return ((totals/masses.clamp_min(1e-6))*supported).sum()/supported.sum().clamp_min(1)
