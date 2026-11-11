"""Reference-conditioned extra ink from frozen extractor ownership.

This is an aligned-reference demonstration, not reference-free error detection.
Supported reference strokes are protected. For partition extractors, summing
outside all stroke supports reduces to a geometric spill map. The notebook
checks this limitation explicitly rather than attributing that effect to learning.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def extracted_extra_map(model,rgb,slots,*,focus_slots=(0,),tolerance=2):
    if type(tolerance) is not int or tolerance<0:raise ValueError('Nonnegative integer tolerance required')
    if not focus_slots or any(i not in range(len(slots)) for i in focus_slots):raise ValueError('Invalid focus slots')
    probability=torch.zeros_like(rgb[:,:1]);geometry=torch.zeros_like(probability)
    details=[]
    for i,slot in enumerate(slots):
        x0,y0,x1,y1=slot['bbox']
        reference=torch.as_tensor(slot['reference'],device=rgb.device,dtype=torch.float32)[None]
        strokes=torch.as_tensor(slot['strokes'],device=rgb.device,dtype=torch.float32)[None]
        crop=F.interpolate(rgb[...,y0:y1,x0:x1],reference.shape[-2:],mode='bilinear',align_corners=False)
        ink=1-(crop*crop.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        prediction=model(ink,reference,strokes)
        masks=prediction['masks']
        support=F.max_pool2d(strokes,2*tolerance+1,1,tolerance)
        assignment_extra=(masks*(1-support)).sum(1,keepdim=True).clamp(0,1)
        # A different stroke may legitimately own ink here. Never erase inside
        # ANY supported reference stroke merely because assignment differs.
        outside=(support.amax(1,keepdim=True)<.1).float()
        extra=assignment_extra*outside
        union=ink*outside
        if i in focus_slots:
            probability[...,y0:y1,x0:x1]=F.interpolate(extra,(y1-y0,x1-x0),mode='bilinear',align_corners=False)
            geometry[...,y0:y1,x0:x1]=F.interpolate(union,(y1-y0,x1-x0),mode='bilinear',align_corners=False)
        details.append(dict(slot=i,bbox=list(slot['bbox']),masks=masks[0].cpu().numpy(),
            reference_strokes=strokes[0].cpu().numpy(),extra=extra[0,0].cpu().numpy(),
            assignment_extra=assignment_extra[0,0].cpu().numpy()))
    return probability,geometry,details


def fixed_region_metrics(rgb,objective):
    """Optimization-region diagnostics, not an independent quality score."""
    with torch.no_grad():
        error=(rgb-objective.target).abs().mean(1,keepdim=True)
        mask=objective.mask
        outside=1-mask
        return dict(edit_region_mae=float((error*mask).sum()/mask.sum().clamp_min(1)),
            protected_region_mae=float(((rgb-objective.source).abs().mean(1,keepdim=True)*outside).sum()/outside.sum().clamp_min(1)),
            protected_peak_channel_change=float(((rgb-objective.source).abs()*outside).amax()),
            edit_pixels=int(mask.sum()))
