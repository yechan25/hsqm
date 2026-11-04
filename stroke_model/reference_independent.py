"""Experimental independent stroke masks with bidirectional ConvLSTM context.

ReferenceStrokeModel is a warm-start feature extractor, not a calibrated
independent-mask checkpoint. Actual target_masks (not template strokes) are
required for training. This is inspired by BiConvExtractNet, not its reproduction.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .reference_model import ReferenceStrokeModel, resize
from .stroke_structure import soft_skeleton


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, hidden):
        super().__init__()
        self.hidden=hidden
        self.gates=nn.Conv2d(cin+hidden,4*hidden,3,padding=1)

    def forward(self,x,state):
        h,c=state
        i,f,o,g=self.gates(torch.cat((x,h),1)).chunk(4,1)
        c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c),c


class IndependentStrokeModel(ReferenceStrokeModel):
    architecture='reference_independent_context_v1'

    def __init__(self,*args,context_width=16,**kwargs):
        super().__init__(*args,**kwargs)
        self.context_width=context_width
        self.context_forward=ConvLSTMCell(3,context_width)
        self.context_backward=ConvLSTMCell(3,context_width)
        self.context_head=nn.Conv2d(2*context_width,1,1)
        nn.init.zeros_(self.context_head.weight)
        nn.init.zeros_(self.context_head.bias)
        self.independent_checkpoint_loaded=False

    def warm_start(self,state):
        """Copy a V2 feature initializer; this does NOT certify new predictions."""
        expected=ReferenceStrokeModel.state_dict(self)
        base={k:v for k,v in state.items() if not k.startswith('context_')}
        required={k for k in expected if not k.startswith('context_')}
        if set(base)!=required:
            raise ValueError('Warm start must contain every compatible base parameter')
        self.load_state_dict(base,strict=False)
        self.independent_checkpoint_loaded=False

    def load_trained(self,checkpoint):
        if checkpoint.get('architecture')!=self.architecture:
            raise ValueError('An old partition checkpoint requires warm_start and retraining')
        self.load_state_dict(checkpoint['model'],strict=True)
        self.independent_checkpoint_loaded=True

    def forward(self,image,reference,strokes,active=None,*,allow_untrained=False):
        if not self.training and not self.independent_checkpoint_loaded and not allow_untrained:
            raise RuntimeError('Independent masks need a trained independent checkpoint; old logits are uncalibrated')
        base=super().forward(image,reference,strokes,active)
        valid=base['active']
        logits=base['logits'].masked_fill(~valid[:,:,None,None],0)
        b,k,h,w=logits.shape
        sequence=torch.stack((image[:,0,None].expand(-1,k,-1,-1),base['hints'],logits.clamp(-12,12)),2)
        small=F.avg_pool2d(sequence.reshape(b*k,3,h,w),4).reshape(b,k,3,h//4,w//4)
        contexts=[]
        for cell,indices in ((self.context_forward,range(k)),(self.context_backward,range(k-1,-1,-1))):
            zero=small.new_zeros(b,self.context_width,h//4,w//4)
            state=(zero,zero.clone()); outputs={}
            for i in indices:
                updated=cell(small[:,i],state)
                mask=valid[:,i,None,None,None]
                state=tuple(torch.where(mask,n,o) for n,o in zip(updated,state))
                outputs[i]=state[0]*mask
            contexts.append(torch.stack([outputs[i] for i in range(k)],1))
        features=torch.cat(contexts,2).reshape(b*k,2*self.context_width,h//4,w//4)
        correction=resize(self.context_head(features),(h,w)).reshape(b,k,h,w)
        logits=(logits+correction).masked_fill(~valid[:,:,None,None],-torch.inf)
        probabilities=torch.sigmoid(logits)*valid[:,:,None,None]
        union=1-(1-probabilities).prod(1,keepdim=True)
        return {**base,'logits':logits,'probabilities':probabilities,'masks':image*probabilities,
                'union':image*union,'unassigned':image*(1-union),
                'output_mode':'independent','requires_retraining':not self.independent_checkpoint_loaded}


@dataclass(frozen=True)
class IndependentLossConfig:
    bce_weight: float=1.
    dice_weight: float=1.
    union_weight: float=.2
    cldice_weight: float=.1
    max_area_weight: float=4.
    skeleton_iterations: int=12
    eps: float=1e-6

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Loss settings must be finite')
        if min(self.bce_weight,self.dice_weight,self.union_weight,self.cldice_weight)<0 or self.max_area_weight<1:
            raise ValueError('Invalid independent loss weights')
        if not isinstance(self.skeleton_iterations,int) or self.skeleton_iterations<1 or self.eps<=0:
            raise ValueError('Invalid skeleton iterations or epsilon')


def independent_stroke_loss(output,target_masks,image,valid_pixels=None,config=None):
    """Multi-hot labels allow shared crossings and all-zero absent strokes.

    target_masks must describe ACTUAL input strokes in semantic slot order.
    Unknown annotations are excluded with valid_pixels; never invented from
    template distance or the old extractor's outputs.
    """
    cfg=config or IndependentLossConfig()
    logits=output['logits']; active=output['active']
    target=torch.as_tensor(target_masks,device=logits.device,dtype=logits.dtype)
    if target.shape!=logits.shape or not bool(torch.isfinite(target).all()) or bool(((target<0)|(target>1)).any()):
        raise ValueError('Actual target_masks must match B,K,H,W and lie in [0,1]')
    if image.shape!=logits[:,:1].shape or not bool(torch.isfinite(image).all()) or bool(((image<0)|(image>1)).any()):
        raise ValueError('Image must be B,1,H,W ink in [0,1]')
    if bool((target.masked_select(~active[:,:,None,None].expand_as(target))>0).any()):
        raise ValueError('Padded/inactive template slots cannot have a target stroke')
    valid=torch.ones_like(image) if valid_pixels is None else torch.as_tensor(valid_pixels,device=logits.device,dtype=logits.dtype)
    if valid.shape!=image.shape or not bool(torch.isfinite(valid).all()) or bool(((valid<0)|(valid>1)).any()):
        raise ValueError('valid_pixels must be B,1,H,W in [0,1]')
    if not bool((valid.flatten(1).sum(1)>0).all()):
        raise ValueError('Every sample needs supervised pixels')
    safe=logits.masked_fill(~active[:,:,None,None],0)
    pred=torch.sigmoid(safe)
    mass=(target*valid).sum((-2,-1)); area=valid.sum((-2,-1)).expand_as(mass)
    present=(mass>cfg.eps)&active
    weights=torch.ones_like(mass)
    for i in range(len(mass)):
        if bool(present[i].any()):
            average=mass[i,present[i]].mean()
            weights[i]=torch.where(present[i],(average/mass[i].clamp_min(1)).clamp(1/cfg.max_area_weight,cfg.max_area_weight),weights[i])
    bce=(F.binary_cross_entropy_with_logits(safe,target,reduction='none')*valid).sum((-2,-1))/area.clamp_min(1)
    bce=(bce*weights*active).sum()/active.sum().clamp_min(1)
    p=pred*valid;t=target*valid
    dice=1-(2*(p*t).sum((-2,-1))+cfg.eps)/(p.sum((-2,-1))+t.sum((-2,-1))+cfg.eps)
    dice=dice[present].mean() if bool(present.any()) else p.sum()*0
    ps=soft_skeleton(p,cfg.skeleton_iterations);ts=soft_skeleton(t,cfg.skeleton_iterations)
    precision=(ps*t).sum((-2,-1))/ps.sum((-2,-1)).clamp_min(cfg.eps)
    sensitivity=(ts*p).sum((-2,-1))/ts.sum((-2,-1)).clamp_min(cfg.eps)
    cldice=1-2*precision*sensitivity/(precision+sensitivity).clamp_min(cfg.eps)
    cldice=cldice[present].mean() if bool(present.any()) else p.sum()*0
    union=1-(1-pred*active[:,:,None,None]).prod(1,keepdim=True)
    union_target=target.amax(1,keepdim=True)
    union_loss=(F.binary_cross_entropy(union.clamp(cfg.eps,1-cfg.eps),union_target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    loss=cfg.bce_weight*bce+cfg.dice_weight*dice+cfg.union_weight*union_loss+cfg.cldice_weight*cldice
    return dict(loss=loss,bce=bce,dice=dice,cldice=cldice,union=union_loss)
