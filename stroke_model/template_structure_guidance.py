"""Template-conditioned normalized stroke geometry; no target-pixel loss.

Experimental moments/relative-attachment proxy, not a complete topology metric.
"""
import torch
from torch import nn
from torch.nn import functional as F


def stroke_descriptor(masks):
    b,k,h,w=masks.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=masks.device,dtype=masks.dtype),
                         torch.arange(w,device=masks.device,dtype=masks.dtype),indexing='ij')
    xy=torch.stack((xx,yy),-1)
    mass=masks.sum((-2,-1)).clamp_min(1e-6)
    center=torch.einsum('bkhw,hwd->bkd',masks,xy)/mass[...,None]
    delta=xy[None,None]-center[:,:,None,None]
    var=(masks[...,None]*delta.square()).sum((2,3))/mass[...,None]
    cross=(masks*delta[...,0]*delta[...,1]).sum((2,3))/mass
    scale=var.sum(-1).clamp_min(1e-4).sqrt()
    orient=torch.stack((var[...,0]-var[...,1],2*cross),-1)/scale.square()[...,None]
    skew=(masks[...,None]*(delta/scale[:,:,None,None,None]).pow(3)).sum((2,3))/mass[...,None]
    glyph_mass=mass.sum(1,keepdim=True)
    glyph_center=(center*mass[...,None]).sum(1,keepdim=True)/glyph_mass[...,None]
    glyph_scale=((mass*((center-glyph_center).square().sum(-1)+var.sum(-1))).sum(1)/glyph_mass[:,0]).clamp_min(1e-4).sqrt()
    centers=(center-glyph_center)/glyph_scale[:,None,None]
    # Dilation allows adjacent exclusive strokes to register a contact.
    near=F.avg_pool2d(masks,5,1,2)
    pairs=[]
    for i in range(k):
        for j in range(i+1,k):
            contact=near[:,i]*near[:,j]
            raw_cm=contact.sum((-2,-1));cm=raw_cm.clamp_min(1e-6)
            reliability=(raw_cm/(.01*torch.minimum(mass[:,i],mass[:,j]))).clamp(max=1)
            point=torch.einsum('bhw,hwd->bd',contact,xy)/cm[:,None]
            # Position of contact along each stroke, normalized by stroke size.
            for index in (i,j):
                pairs.append((point-center[:,index])/scale[:,index,None]*reliability[:,None])
            pairs.append((cm/torch.minimum(mass[:,i],mass[:,j])).unsqueeze(-1))
    flat=[orient.flatten(1),skew.flatten(1),centers.flatten(1),
          (scale/glyph_scale[:,None]).flatten(1),(mass/glyph_mass).flatten(1)]
    if pairs:flat.extend(pairs)
    return torch.cat(flat,1)


class TemplateStructureGuidance(nn.Module):
    def __init__(self,model,slots):
        super().__init__();self.model=model.eval().requires_grad_(False);self.slots=slots
        for i,slot in enumerate(slots):
            device=next(model.parameters()).device
            ref=torch.as_tensor(slot['reference'],dtype=torch.float32,device=device)[None]
            strokes=torch.as_tensor(slot['strokes'],dtype=torch.float32,device=device)[None]
            self.register_buffer(f'ref_{i}',ref);self.register_buffer(f'strokes_{i}',strokes)
            with torch.no_grad():target=stroke_descriptor(self.model(ref,ref,strokes)['masks'])
            self.register_buffer(f'target_{i}',target)

    def forward(self,rgb):
        luminance=(rgb.clamp(0,1)*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        rows=[];losses=[]
        for i,slot in enumerate(self.slots):
            ref=getattr(self,f'ref_{i}');strokes=getattr(self,f'strokes_{i}')
            x0,y0,x1,y1=slot['bbox']
            ink=F.interpolate(1-luminance[...,y0:y1,x0:x1],ref.shape[-2:],mode='bilinear',align_corners=False)
            with torch.no_grad():pred=self.model(ink,ref,strokes)
            masks=pred['masks'].detach()+(ink-ink.detach())*pred['probabilities'].detach()
            error=F.smooth_l1_loss(stroke_descriptor(masks),getattr(self,f'target_{i}'))
            losses.append(error)
            rows.append(dict(slot=i,descriptor_loss=error,per_stroke=[]))
        bound=(F.relu(-rgb).square()+F.relu(rgb-1).square()).mean()
        return torch.stack(losses).mean()+.01*bound,dict(per_glyph=rows)
