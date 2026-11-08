"""Frozen Topo soft masks -> differentiable, reference-aware stroke objective."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

from .stroke_structure import StrokeStructureLoss, StructureConfig
from .structural_indicator import StructuralIndicator, IndicatorConfig
from .source_quality_v5 import foreground_evidence


class TopoGuidance(nn.Module):
    def __init__(self, model, slots, structure_config=None, extra_weight=.05, range_weight=.01):
        super().__init__()
        if len(slots)!=4:raise ValueError('This experiment requires exactly four glyph slots')
        self.model=model.eval().requires_grad_(False)
        self.slots=slots
        self.structure_config=structure_config or StructureConfig()
        self.metrics=nn.ModuleList()
        self.extra_weight=float(extra_weight)
        self.range_weight=float(range_weight)
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

    def forward(self, rgb, return_masks=False):
        if rgb.ndim!=4 or rgb.shape[0]!=1 or rgb.shape[1]!=3:
            raise ValueError('Expected one jointly generated RGB sheet: 1,3,H,W')
        range_loss=(F.relu(-rgb).square()+F.relu(rgb-1).square()).mean()
        bounded=rgb.clamp(0,1)
        luminance=(bounded*bounded.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        rows,losses=[],[]
        for i,(slot,metric) in enumerate(zip(self.slots,self.metrics)):
            ref=getattr(self,f'reference_{i}')
            strokes=getattr(self,f'strokes_{i}')
            x0,y0,x1,y1=slot['bbox']
            if not (0<=x0<x1<=rgb.shape[-1] and 0<=y0<y1<=rgb.shape[-2]):
                raise ValueError('Crop outside generated sheet')
            ink=F.interpolate(1-luminance[...,y0:y1,x0:x1],ref.shape[-2:],mode='bilinear',align_corners=False)
            prediction=self.model(ink.float(),ref,strokes)
            measured=metric(prediction['masks'].float())
            if not all(c['pairs'] and c['attachment'] for c in measured['convergence']):
                raise RuntimeError('Stroke reconstruction did not converge; do not guide on truncated measurements')
            values=measured['per_stroke']
            missing=values['missing'].mean()
            split=torch.log1p(values['split']/.2).mean()
            junction=torch.log1p(values['junction']/.01).mean()
            cfg=self.structure_config
            stroke_loss=(cfg.missing_weight*missing+cfg.split_weight*split+cfg.junction_weight*junction
                         +cfg.width_weight*values['width'].mean())
            # Separate weak source-ink auxiliary, never substituted for Topo loss.
            evidence=foreground_evidence(ink)
            extra=(evidence*(1-getattr(self,f'support_{i}'))).sum()/ref.sum().clamp_min(1)
            losses.append(stroke_loss+self.extra_weight*extra)
            score=self.scorer(values['missing'],values['split'],values['junction'],extra=0.)['score'][0]
            row=dict(slot=i,stroke_loss=stroke_loss,missing=missing,
                split=values['split'].mean(),junction=values['junction'].mean(),
                extra_auxiliary=extra,extracted_stroke_score_no_extra=score)
            if return_masks:
                row.update(image=ink,soft_masks=prediction['masks'],probabilities=prediction['probabilities'])
            rows.append(row)
        loss=torch.stack(losses).mean()+self.range_weight*range_loss
        return loss,dict(per_glyph=rows,range_loss=range_loss)
