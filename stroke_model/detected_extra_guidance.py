"""Inference-only fixed extra-ink maps -> actual VAE/DDIM editing objective.

No clean target or template-pixel matching. Flat light-background crops only;
photo background reconstruction needs a separate inpainting model.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import label, binary_dilation
from .error_localizer import ARCHITECTURE, ErrorLocalizerConfig, StrokeErrorLocalizer


def load_error_head(checkpoint, extractor_checkpoint, device='cpu'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('architecture')!=ARCHITECTURE or state.get('epoch',0)<1:
        raise ValueError('A trained error-localizer checkpoint is required; no random head fallback')
    expected=state.get('metadata',{}).get('extractor_sha256')
    actual=hashlib.sha256(Path(extractor_checkpoint).read_bytes()).hexdigest()
    if not expected or expected!=actual:
        raise ValueError('Use the same frozen extractor that trained this error detector')
    head=StrokeErrorLocalizer(ErrorLocalizerConfig(**state['config']))
    head.load_state_dict(state['model'],strict=True)
    return head.to(device).eval().requires_grad_(False),dict(epoch=state['epoch'],
        threshold_calibrated=bool(state.get('threshold_calibrated',False)),
        detector_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())


@torch.no_grad()
def predict_extra_sheet(head,extractor,rgb,slots,characters):
    if len(slots)!=len(characters):raise ValueError('Character count must match supplied slots')
    result=torch.zeros_like(rgb[:,:1])
    for slot,char in zip(slots,characters):
        if len(char)!=1:raise ValueError('One explicit character per slot required')
        x0,y0,x1,y1=slot['bbox']
        reference=torch.as_tensor(slot['reference'],device=rgb.device,dtype=torch.float32)[None]
        strokes=torch.as_tensor(slot['strokes'],device=rgb.device,dtype=torch.float32)[None]
        crop=F.interpolate(rgb[...,y0:y1,x0:x1],reference.shape[-2:],mode='bilinear',align_corners=False)
        ink=1-(crop*crop.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        active=strokes.sum((-2,-1))>0
        p=extractor(ink,reference,strokes,active=active)['probabilities']
        out=head(crop,p,torch.tensor([ord(char)],device=rgb.device),active)
        result[...,y0:y1,x0:x1]=F.interpolate(out['extra_probability'],(y1-y0,x1-x0),mode='bilinear',align_corners=False)
    return result


class DetectedExtraGuidance(nn.Module):
    def __init__(self,generated_rgb,error_probability,slots,threshold=.5,keep_weight=20.):
        super().__init__()
        if generated_rgb.ndim!=4 or generated_rgb.shape[:2]!=(1,3) or error_probability.shape!=generated_rgb[:,:1].shape:
            raise ValueError('One RGB sheet and matching extra probability map required')
        if not 0<threshold<1 or not np.isfinite(keep_weight) or keep_weight<0:
            raise ValueError('Invalid mask threshold/keep weight')
        for value in (generated_rgb,error_probability):
            if not torch.isfinite(value).all() or (value<0).any() or (value>1).any():
                raise ValueError('Images/probabilities must be in [0,1]')
        self.slots=slots;self.threshold=threshold;self.keep_weight=keep_weight
        mask=(error_probability>=threshold).detach()
        if not mask.any():raise ValueError('No detected edit pixels at this threshold; inspect the map first')
        rgb=generated_rgb.detach().cpu().numpy()[0].transpose(1,2,0)
        target=rgb.copy();m=mask.cpu().numpy()[0,0]
        # Each proposed region gets local background from the GENERATED image.
        # Never paint the final output; this is only a fixed latent-loss target.
        labels,count=label(m)
        for index in range(1,count+1):
            region=labels==index
            ring=binary_dilation(region,iterations=6)&~binary_dilation(m,iterations=1)
            if not ring.any():raise ValueError('No observed background around an edit region')
            pixels=rgb[ring];light=pixels.mean(1)
            bright=pixels[light>=np.quantile(light,.75)]
            color=np.median(bright,axis=0)
            if color.mean()<.7 or np.median(abs(bright-color))>.08:
                raise ValueError('Non-flat/dark background: this deletion prototype needs an inpainting target')
            target[region]=color
        self.register_buffer('source',generated_rgb.detach().clone())
        self.register_buffer('probability',error_probability.detach().clone())
        self.register_buffer('mask',mask.float())
        self.register_buffer('target',torch.from_numpy(target.transpose(2,0,1).copy())[None].to(generated_rgb))

    def forward(self,rgb):
        if rgb.shape!=self.source.shape:raise ValueError('Generated sheet shape changed')
        absolute=(rgb-self.target).abs().mean(1,keepdim=True)
        squared=(rgb-self.target).square().mean(1,keepdim=True)
        edit=((squared+.5*absolute)*self.mask).sum()/self.mask.sum().clamp_min(1)
        keep_map=(rgb-self.source).square().mean(1,keepdim=True)*(1-self.mask)
        keep=keep_map.sum()/(1-self.mask).sum().clamp_min(1)
        peak=F.avg_pool2d(keep_map,5,1,2).amax()
        rows=[]
        for i,slot in enumerate(self.slots):
            x0,y0,x1,y1=slot['bbox'];m=self.mask[...,y0:y1,x0:x1]
            e=absolute[...,y0:y1,x0:x1]
            rows.append(dict(slot=i,edit_loss=(e*m).sum()/m.sum().clamp_min(1),per_stroke=[]))
        return edit+self.keep_weight*(keep+peak),dict(per_glyph=rows,
            edit_loss=edit,structure_loss=edit,preservation_loss=keep,protected_local_peak_squared=peak)
