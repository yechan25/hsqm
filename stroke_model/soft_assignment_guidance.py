"""Reference-conditioned soft ownership guidance with constrained latent editing.

Ownership is refreshed once per diffusion step, then detached for all inner
updates and line-search candidates. No union-outside gate or RGB painting.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F


class SoftAssignmentGuidance(nn.Module):
    def __init__(self,model,slots,source_rgb,*,mode='softmax',focus_slots=(0,),
                 allowance=3.,falloff=3.,leak_floor=.05,preserve_weight=.1,
                 healthy_rms_limit=.025,healthy_local_limit=.12,healthy_mass_drop=.04,
                 baseline_rms_slack=.002,baseline_local_slack=.02,baseline_mass_slack=.01):
        super().__init__()
        if mode not in ('softmax','geometry'):raise ValueError('Unknown assignment guide mode')
        values=(allowance,falloff,preserve_weight,healthy_rms_limit,healthy_local_limit,healthy_mass_drop,
                baseline_rms_slack,baseline_local_slack,baseline_mass_slack)
        if any(not math.isfinite(v) or v<0 for v in values) or falloff<=0 or not 0<=leak_floor<1:
            raise ValueError('Invalid guide configuration')
        if not focus_slots or any(i not in range(len(slots)) for i in focus_slots):raise ValueError('Invalid focus slots')
        if source_rgb.ndim!=4 or source_rgb.shape[:2]!=(1,3):raise ValueError('One RGB image required')
        if not torch.isfinite(source_rgb).all() or (source_rgb<0).any() or (source_rgb>1).any():
            raise ValueError('Source RGB must be finite and in [0,1]')
        self.model=model.eval().requires_grad_(False);self.slots=slots
        self.mode=mode;self.focus_slots=tuple(focus_slots);self.leak_floor=leak_floor
        self.preserve_weight=preserve_weight;self.healthy_rms_limit=healthy_rms_limit
        self.healthy_local_limit=healthy_local_limit;self.healthy_mass_drop=healthy_mass_drop
        self.baseline_rms_slack=baseline_rms_slack;self.baseline_local_slack=baseline_local_slack
        self.baseline_mass_slack=baseline_mass_slack;self.baseline_guards={}
        self.register_buffer('source',source_rgb.detach().float().clone())
        self.register_buffer('source_ink',self.ink(source_rgb).detach())
        healthy=[];bands=[];guard_names=['protected_rms','protected_local_rms']
        target=torch.ones_like(source_rgb)
        for i,slot in enumerate(slots):
            reference=source_rgb.new_tensor(np.asarray(slot['reference']))[None]
            strokes=source_rgb.new_tensor(np.asarray(slot['strokes']))[None]
            distance=np.stack([distance_transform_edt(s<.1) for s in np.asarray(slot['strokes'])])
            d=source_rgb.new_tensor(distance)[None]
            crop=self.crop(self.source_ink,slot,reference.shape[-2:])
            # Calibrate thickness from the observed image, using nearest-stroke
            # pixels; quantiles suppress isolated protrusions. Freeze thereafter.
            nearest=d.argmin(1,keepdim=True)
            radii=[]
            for k in range(strokes.shape[1]):
                samples=d[:,k:k+1][(crop>.5)&(nearest==k)&(d[:,k:k+1]<=6)]
                radius=max(allowance,float(torch.quantile(samples,.85))+.5) if samples.numel() else allowance
                radii.append(min(radius,6.))
            radius=d.new_tensor(radii)[None,:,None,None]
            cost=1-torch.exp(-F.relu(d-radius)/falloff)
            self.register_buffer(f'reference_{i}',reference)
            self.register_buffer(f'strokes_{i}',strokes)
            self.register_buffer(f'cost_{i}',cost)
            bands.append(radii)
            x0,y0,x1,y1=slot['bbox']
            observed=source_rgb[...,y0:y1,x0:x1]
            pixels=observed[0].permute(1,2,0).reshape(-1,3)
            light=pixels.mean(1);bright=pixels[light>=torch.quantile(light,.8)]
            color=bright.median(0).values
            if color.mean()<.7 or (bright-color).abs().median()>.08:
                raise ValueError('This deletion experiment needs a bright flat background')
            target[...,y0:y1,x0:x1]=color[None,:,None,None]
            for k in range(strokes.shape[1]):
                region=((d[:,k:k+1]<=radius[:,k:k+1])&(nearest==k)&(crop>.25)).float()
                sheet=torch.zeros_like(self.source_ink)
                sheet[...,y0:y1,x0:x1]=F.interpolate(region,(y1-y0,x1-x0),mode='nearest')
                healthy.append(sheet)
                guard_names.append(f'slot_{i+1}_stroke_{k+1}_mass_drop')
        self.bands=bands
        self.guard_names=guard_names
        self.register_buffer('target',target.detach())
        self.register_buffer('healthy',torch.cat(healthy,1))
        self.register_buffer('initial_weights',torch.zeros_like(self.source_ink))
        self.register_buffer('weights',torch.zeros_like(self.source_ink))
        self.register_buffer('gradient_support',torch.zeros_like(self.source_ink))
        self.register_buffer('protect',torch.ones_like(self.source_ink))
        initial=self.compute_weights(source_rgb)
        if not (initial>.01).any():raise ValueError('No soft extra-ink signal; inspect reference alignment')
        self.initial_weights.copy_(initial);self.weights.copy_(initial)
        self.refresh(source_rgb)

    @staticmethod
    def ink(rgb):
        weights=rgb.new_tensor([.299,.587,.114])[None,:,None,None]
        return (1-(rgb*weights).sum(1,keepdim=True)).clamp(0,1)

    @staticmethod
    def crop(value,slot,size):
        x0,y0,x1,y1=slot['bbox']
        return F.interpolate(value[...,y0:y1,x0:x1],size,mode='bilinear',align_corners=False)

    @torch.no_grad()
    def compute_weights(self,rgb):
        ink=self.ink(rgb);result=torch.zeros_like(ink)
        for i in self.focus_slots:
            slot=self.slots[i];reference=getattr(self,f'reference_{i}')
            strokes=getattr(self,f'strokes_{i}');cost=getattr(self,f'cost_{i}')
            crop=self.crop(ink,slot,reference.shape[-2:])
            if self.mode=='softmax':
                p=self.model(crop,reference,strokes)['probabilities'].float()
                if p.shape!=cost.shape or not torch.isfinite(p).all():raise ValueError('Invalid extractor probabilities')
                if (p<0).any() or not torch.allclose(p.sum(1),torch.ones_like(p[:,0]),atol=1e-3):
                    raise ValueError('A normalized stroke softmax is required')
                pressure=(p.detach()*cost).sum(1,keepdim=True)
            else:pressure=cost.amin(1,keepdim=True)
            pressure=F.relu(pressure-self.leak_floor)/(1-self.leak_floor)
            source=self.crop(self.source_ink,slot,reference.shape[-2:])
            selection=(.5*source+.5*crop-.05).clamp_min(0)
            x0,y0,x1,y1=slot['bbox']
            result[...,y0:y1,x0:x1]=F.interpolate(pressure*selection,(y1-y0,x1-x0),mode='bilinear',align_corners=False)
        return result.detach()

    @torch.no_grad()
    def refresh(self,rgb):
        # Retain pressure on initial defects so that fading residuals do not
        # disappear from the objective as the current ink or ownership changes.
        self.weights.copy_(torch.maximum(self.compute_weights(rgb),.25*self.initial_weights))
        self.gradient_support.copy_(F.max_pool2d(self.weights,25,1,12).clamp(0,1).sqrt())
        # Binary decisions are for the guard only; the editing loss stays soft.
        edit_halo=F.max_pool2d((self.initial_weights>.03).float(),17,1,8)
        normal=self.healthy.amax(1,keepdim=True)
        self.protect.copy_(torch.maximum(1-edit_halo,normal))

    def terms(self,rgb,*,weights=None):
        weights=self.weights if weights is None else weights
        difference=rgb-self.target
        error=(difference.square()+.5*difference.abs()).mean(1,keepdim=True)
        edit=(weights*error).sum()/weights.sum().clamp_min(1e-6)
        change=(rgb-self.source).square().mean(1,keepdim=True)
        keep=(change*self.protect).sum()/self.protect.sum().clamp_min(1)
        return edit,keep

    def forward(self,rgb):
        edit,keep=self.terms(rgb)
        rows=[]
        for i,slot in enumerate(self.slots):
            x0,y0,x1,y1=slot['bbox'];w=self.weights[...,y0:y1,x0:x1]
            error=(rgb-self.target).abs().mean(1,keepdim=True)[...,y0:y1,x0:x1]
            rows.append(dict(slot=i,edit_loss=(w*error).sum()/w.sum().clamp_min(1e-6),per_stroke=[]))
        return edit+self.preserve_weight*keep,dict(per_glyph=rows,edit_loss=edit,
            structure_loss=edit,preservation_loss=keep)

    @torch.no_grad()
    def guard_vector(self,rgb):
        change=(rgb-self.source).square().mean(1,keepdim=True)*self.protect
        rms=(change.sum()/self.protect.sum().clamp_min(1)).sqrt()
        local=F.avg_pool2d(change,5,1,2).amax().sqrt()
        ink=self.ink(rgb)
        original=(self.source_ink*self.healthy).sum((-2,-1))
        current=(ink*self.healthy).sum((-2,-1))
        drops=((original-current)/original.clamp_min(1)).clamp_min(0).flatten()
        return torch.cat((torch.stack((rms,local)),drops))

    @torch.no_grad()
    def guard_values(self,rgb):
        vector=self.guard_vector(rgb)
        return torch.stack((vector[0],vector[1],vector[2:].amax()))

    def begin_baseline(self):
        self.baseline_guards.clear()

    @torch.no_grad()
    def record_baseline(self,index,rgb):
        self.baseline_guards[int(index)]=self.guard_vector(rgb).detach().clone()

    @torch.no_grad()
    def guard_limits(self,anchor_rgb,step_index=None):
        anchor=self.guard_vector(anchor_rgb)
        baseline=self.baseline_guards.get(step_index,anchor)
        floors=baseline.new_full(baseline.shape,self.healthy_mass_drop)
        floors[0]=self.healthy_rms_limit;floors[1]=self.healthy_local_limit
        slack=baseline.new_full(baseline.shape,self.baseline_mass_slack)
        slack[0]=self.baseline_rms_slack;slack[1]=self.baseline_local_slack
        # Anchor to the SAME un-guided trajectory, not to the previous accepted
        # edit. This budget cannot accumulate anew with each inner iteration.
        return torch.maximum(floors,baseline)+slack

    def optimize_clean(self,clean,decode,*,iterations,step_ratio,backtracks,budget_ratio,step_index=None):
        with torch.no_grad():
            anchor=clean.detach().float();anchor_rgb=decode(anchor)
            self.refresh(anchor_rgb)
            limits=self.guard_limits(anchor_rgb,step_index)
            initial_edit,initial_keep=self.terms(anchor_rgb)
            initial_fixed_edit,_=self.terms(anchor_rgb,weights=self.initial_weights)
            initial_guard=self.guard_vector(anchor_rgb)
            initial_total=initial_edit+self.preserve_weight*initial_keep
        value=anchor.clone();attempts=[];accepted_count=0
        scale=anchor.square().mean().sqrt().clamp_min(1e-3)
        support=F.adaptive_max_pool2d(self.gradient_support,anchor.shape[-2:])
        for inner in range(iterations):
            with torch.enable_grad():
                leaf=value.detach().requires_grad_(True)
                rgb=decode(leaf);edit,keep=self.terms(rgb)
                loss=edit+self.preserve_weight*keep
                gradient,=torch.autograd.grad(loss,leaf)
            gradient=gradient*support
            if not torch.isfinite(gradient).all() or not torch.isfinite(loss):raise FloatingPointError('Nonfinite soft-assignment gradient')
            norm=gradient.square().mean().sqrt()
            if norm<1e-12:break
            direction=gradient/norm
            accepted=False
            for retry in range(backtracks+1):
                candidate=value-step_ratio*(.5**retry)*scale*direction
                displacement=candidate-anchor
                move=displacement.square().mean().sqrt()
                candidate=anchor+displacement*torch.clamp(budget_ratio*scale/move.clamp_min(1e-12),max=1)
                with torch.no_grad():
                    image=decode(candidate);after_edit,after_keep=self.terms(image)
                    guard=self.guard_vector(image)
                if not torch.isfinite(after_edit) or not torch.isfinite(after_keep) or not torch.isfinite(guard).all():
                    attempts.append(dict(inner=inner,backtrack=retry,reason='nonfinite_candidate'))
                    continue
                reason=('edit_not_lower' if after_edit>=edit.detach()-1e-8 else
                        'healthy_guard' if (guard>limits).any() else 'accepted')
                attempts.append(dict(inner=inner,backtrack=retry,edit_before=float(edit.detach()),
                    edit_after=float(after_edit),protected_rms=float(guard[0]),
                    protected_local_rms=float(guard[1]),healthy_mass_drop=float(guard[2:].amax()),
                    violated_guards=(guard>limits).nonzero().flatten().cpu().tolist(),
                    guard_failures=[dict(name=self.guard_names[j],value=float(guard[j]),limit=float(limits[j]),
                        excess=float(guard[j]-limits[j])) for j in (guard>limits).nonzero().flatten().cpu().tolist()],reason=reason))
                if reason=='accepted':
                    value=candidate.detach();accepted_count+=1;accepted=True;break
            if not accepted:break
        with torch.no_grad():
            final_rgb=decode(value);final_edit,final_keep=self.terms(final_rgb)
            final_guard=self.guard_values(final_rgb)
            final_fixed_edit,_=self.terms(final_rgb,weights=self.initial_weights)
        return value,dict(accepted=accepted_count>0,accepted_inner_steps=accepted_count,
            loss_before=float(initial_total),loss_after=float(final_edit+self.preserve_weight*final_keep),
            edit_loss_before=float(initial_edit),edit_loss_after=float(final_edit),
            fixed_map_edit_before=float(initial_fixed_edit),fixed_map_edit_after=float(final_fixed_edit),
            anchor_guard_failures=[dict(name=self.guard_names[j],value=float(initial_guard[j]),limit=float(limits[j]))
                for j in (initial_guard>limits).nonzero().flatten().cpu().tolist()],
            preservation_loss_before=float(initial_keep),preservation_loss_after=float(final_keep),
            protected_rms=float(final_guard[0]),protected_local_rms=float(final_guard[1]),
            healthy_mass_drop=float(final_guard[2]),
            guard_limits=[float(limits[0]),float(limits[1]),float(limits[2:].amax())],
            per_stroke_guard_limits=limits[2:].cpu().tolist(),
            guard_reference='matched_unguided_step' if step_index in self.baseline_guards else 'anchor_fallback',
            soft_weight_sum=float(self.weights.sum()),attempts=attempts)

    @torch.no_grad()
    def diagnostics(self,rgb):
        edit,keep=self.terms(rgb,weights=self.initial_weights)
        guard=self.guard_values(rgb)
        return dict(initial_map_edit_loss=float(edit),protected_mse=float(keep),
            protected_rms=float(guard[0]),protected_local_rms=float(guard[1]),healthy_mass_drop=float(guard[2]))
