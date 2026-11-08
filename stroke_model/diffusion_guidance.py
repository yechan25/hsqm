"""Frozen Topo soft masks -> differentiable, reference-aware stroke objective."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

from .stroke_structure import StrokeStructureLoss, StructureConfig
from .structural_indicator import StructuralIndicator, IndicatorConfig
from .source_quality_v5 import foreground_evidence


class TopoGuidance(nn.Module):
    def __init__(self, model, slots, structure_config=None, extra_weight=.05, range_weight=.01,
                 focus_strength=0., max_stroke_weight=3.):
        super().__init__()
        if len(slots)!=4:raise ValueError('This experiment requires exactly four glyph slots')
        self.model=model.eval().requires_grad_(False)
        self.slots=slots
        self.structure_config=structure_config or StructureConfig()
        self.metrics=nn.ModuleList()
        self.extra_weight=float(extra_weight)
        self.range_weight=float(range_weight)
        self.focus_strength=float(focus_strength)
        self.max_stroke_weight=float(max_stroke_weight)
        if (not math.isfinite(self.focus_strength) or self.focus_strength<0
                or not math.isfinite(self.max_stroke_weight) or self.max_stroke_weight<1):
            raise ValueError('Invalid stroke focus settings')
        if self.extra_weight<0 or self.range_weight<0:raise ValueError('Nonnegative auxiliary weights required')
        self.scorer=StructuralIndicator(IndicatorConfig(extra_weight=0))
        for i,slot in enumerate(slots):
            ref=np.asarray(slot['reference'],np.float32)
            strokes=np.asarray(slot['strokes'],np.float32)
            self.register_buffer(f'reference_{i}',torch.from_numpy(ref)[None])
            self.register_buffer(f'strokes_{i}',torch.from_numpy(strokes)[None])
            distance=distance_transform_edt(ref[0]<.3)
            support=1/(1+np.exp(np.clip((distance-3.)/.5,-80,80)))
            support=np.minimum(support/(1/(1+np.exp(-6.))),1).astype(np.float32)
            self.register_buffer(f'support_{i}',torch.from_numpy(support)[None,None])
            self.metrics.append(StrokeStructureLoss(strokes,self.structure_config))

    @staticmethod
    def freeze_priority(details):
        """Freeze priorities for ALL backtracking candidates at this timestep."""
        return [[float(s['weight']) for s in row['per_stroke']] for row in details['per_glyph']]

    def forward(self, rgb, return_masks=False, stroke_weights=None):
        if rgb.ndim!=4 or rgb.shape[0]!=1 or rgb.shape[1]!=3:
            raise ValueError('Expected one jointly generated RGB sheet: 1,3,H,W')
        range_loss=(F.relu(-rgb).square()+F.relu(rgb-1).square()).mean()
        bounded=rgb.clamp(0,1)
        luminance=(bounded*bounded.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        rows,losses=[],[]
        if stroke_weights is not None and len(stroke_weights)!=len(self.slots):
            raise ValueError('Need stroke weights for each of the four slots')
        for i,(slot,metric) in enumerate(zip(self.slots,self.metrics)):
            ref=getattr(self,f'reference_{i}')
            strokes=getattr(self,f'strokes_{i}')
            x0,y0,x1,y1=slot['bbox']
            if not (0<=x0<x1<=rgb.shape[-1] and 0<=y0<y1<=rgb.shape[-2]):
                raise ValueError('Crop outside generated sheet')
            ink=F.interpolate(1-luminance[...,y0:y1,x0:x1],ref.shape[-2:],mode='bilinear',align_corners=False)
            prediction=self.model(ink.float(),ref,strokes)
            measured=metric(prediction['masks'].float(),return_maps=return_masks)
            if not all(c['pairs'] and c['attachment'] for c in measured['convergence']):
                raise RuntimeError('Stroke reconstruction did not converge; do not guide on truncated measurements')
            values=measured['per_stroke']
            missing=values['missing'].mean()
            cfg=self.structure_config
            per_loss=(cfg.missing_weight*values['missing']+cfg.split_weight*torch.log1p(values['split']/.2)
                      +cfg.junction_weight*torch.log1p(values['junction']/.01)+cfg.width_weight*values['width'])
            scoring=self.scorer(values['missing'],values['split'],values['junction'],extra=0.)
            scores=scoring['stroke_scores']
            weights=(1+self.focus_strength*(1-scores.detach()/100)).clamp(1,self.max_stroke_weight)
            if stroke_weights is not None:
                weights=torch.as_tensor(stroke_weights[i],device=per_loss.device,dtype=per_loss.dtype)[None].detach()
                if weights.shape!=per_loss.shape or not torch.isfinite(weights).all() or (weights<=0).any():
                    raise ValueError('Frozen stroke weights must be positive and match the stroke count')
            contributions=weights*per_loss/weights.sum(1,keepdim=True)
            stroke_loss=contributions.sum(1).mean()
            # Separate weak source-ink auxiliary, never substituted for Topo loss.
            evidence=foreground_evidence(ink)
            extra=(evidence*(1-getattr(self,f'support_{i}'))).sum()/ref.sum().clamp_min(1)
            losses.append(stroke_loss+self.extra_weight*extra)
            score=scoring['score'][0]
            per_stroke=[dict(stroke=k+1,score=scores[0,k],weight=weights[0,k],
                weight_share=weights[0,k]/weights.sum(),loss=per_loss[0,k],
                contribution=contributions[0,k]/len(self.slots),
                **{name:values[name][0,k] for name in ('missing','split','junction','width')})
                for k in range(per_loss.shape[1])]
            row=dict(slot=i,stroke_loss=stroke_loss,missing=missing,
                split=values['split'].mean(),junction=values['junction'].mean(),
                uniform_stroke_loss=per_loss.mean(),per_stroke=per_stroke,
                extra_auxiliary=extra,extracted_stroke_score_no_extra=score)
            if return_masks:
                row.update(image=ink,soft_masks=prediction['masks'],probabilities=prediction['probabilities'],
                           defect_maps=measured['maps'])
            rows.append(row)
        loss=torch.stack(losses).mean()+self.range_weight*range_loss
        return loss,dict(per_glyph=rows,range_loss=range_loss)
