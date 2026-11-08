"""Fixed reference erase/fill maps; no extractor ownership gradients."""
import torch
from torch.nn import functional as F


class LocalReferenceEdit:
    def __init__(self, reference, anchor, threshold=.2, radius=1, halo_radius=0):
        if reference.shape != anchor.shape or reference.ndim != 4 or reference.shape[1] != 3:
            raise ValueError('Matching BCHW RGB sheets required')
        self.anchor=anchor.detach()
        self.reference=reference.detach()
        ref=1-reference.mean(1,keepdim=True).clamp(0,1)
        ink=1-anchor.mean(1,keepdim=True).clamp(0,1)
        support=F.max_pool2d(ref,2*radius+1,1,radius)
        self.erase=((ink-support)>threshold).float().detach()
        if type(halo_radius) is not int or halo_radius<0:raise ValueError('Nonnegative halo radius required')
        if halo_radius:
            # Expand only near detected defects and only into clear reference background.
            halo=F.max_pool2d(self.erase,2*halo_radius+1,1,halo_radius)*(support<.05)
            self.erase=torch.maximum(self.erase,halo).detach()
        self.fill=((ref-ink)>threshold).float().detach()*(1-self.erase)
        self.protect=1-(self.erase+self.fill).clamp(0,1)
        self.rebase(anchor)

    def rebase(self,anchor):
        self.anchor=anchor.detach()
        self.target=torch.where(self.erase.bool(),torch.ones_like(anchor),anchor)
        self.target=torch.where(self.fill.bool(),self.reference,self.target).detach()

    def measure(self,rgb):
        # Unclamped decoder RGB preserves gradients even for saturated black.
        error=(rgb-self.target).square().mean(1,keepdim=True)
        erase=(error*self.erase).sum()/self.erase.sum().clamp_min(1)
        fill=(error*self.fill).sum()/self.fill.sum().clamp_min(1)
        change=(rgb-self.anchor).square().mean(1,keepdim=True)*self.protect
        protected=change.sum()/self.protect.sum().clamp_min(1)
        peak=F.avg_pool2d(change,5,1,2).amax().clamp_min(1e-12).sqrt()
        return dict(edit=erase+fill,erase=erase,fill=fill,protected=protected,peak=peak)


def optimize_local(clean,decode,editor,iterations=24,step_ratio=.01,budget_ratio=.08,
                   protect_weight=20.,max_protected_rms=.03,max_protected_peak=.12):
    if iterations<1 or step_ratio<=0 or budget_ratio<=0:
        raise ValueError('Positive iteration count and step budgets required')
    origin=clean.detach().float();current=origin.clone();best=origin.clone()
    scale=origin.square().mean().sqrt().clamp_min(1e-3)
    with torch.no_grad():before=editor.measure(decode(origin))
    best_error=float(before['edit']);history=[]
    for index in range(iterations):
        with torch.enable_grad():
            leaf=current.detach().requires_grad_(True)
            m=editor.measure(decode(leaf))
            loss=m['edit']+protect_weight*(m['protected']+m['peak'].square())
            gradient,=torch.autograd.grad(loss,leaf)
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite local edit gradient')
        norm=gradient.square().mean().sqrt()
        if norm<1e-12:break
        # Search from the last feasible state. A large normalized step can
        # cross the protection boundary; shrinking it retains useful progress.
        moved=False
        with torch.no_grad():
            for attempt in range(7):
                candidate=leaf-step_ratio*(.5**attempt)*scale*gradient/norm
                delta=candidate-origin
                candidate=origin+delta*(budget_ratio*scale/delta.square().mean().sqrt().clamp_min(1e-12)).clamp(max=1)
                values=editor.measure(decode(candidate))
                row={k:float(v) for k,v in values.items()}
                ok=(row['protected']**.5<=max_protected_rms and row['peak']<=max_protected_peak)
                energy=row['edit']+protect_weight*(row['protected']+row['peak']**2)
                accept=ok and row['edit']<best_error and energy<float(loss.detach())
                row.update(iteration=index,backtrack=attempt,feasible=ok,accepted=accept,
                           step_ratio=step_ratio*(.5**attempt))
                history.append(row)
                if accept:
                    current=candidate.clone();best=current.clone();best_error=row['edit']
                    moved=True
                    break
        if not moved:break
    return best,dict(accepted=best_error<float(before['edit']),loss_before=float(before['edit']),
                     loss_after=best_error,local_iterations=history)
