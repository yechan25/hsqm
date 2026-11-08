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
from .stroke_appearance import StrokeAppearanceLoss,AppearanceConfig
from .source_ink_guard import SourceInkGuard


class TopoGuidance(nn.Module):
    def __init__(self, model, slots, structure_config=None, extra_weight=.05, range_weight=.01,
                 focus_strength=0., max_stroke_weight=3.,appearance_weight=0.,appearance_config=None,
                 mask_gradient='full',source_spill_weight=0.):
        super().__init__()
        if len(slots)!=4:raise ValueError('This experiment requires exactly four glyph slots')
        self.model=model.eval().requires_grad_(False)
        self.slots=slots
        self.structure_config=structure_config or StructureConfig()
        self.metrics=nn.ModuleList()
        self.appearance_metrics=nn.ModuleList()
        self.appearance_config=appearance_config or AppearanceConfig()
        self.appearance_weight=float(appearance_weight)
        if mask_gradient not in ('full','ink_only'):raise ValueError('Unknown Topo gradient mode')
        self.mask_gradient=mask_gradient
        self.source_spill_weight=float(source_spill_weight)
        if not math.isfinite(self.source_spill_weight) or self.source_spill_weight<0:
            raise ValueError('Source spill weight must be finite and nonnegative')
        self.source_guard=None
        if not math.isfinite(self.appearance_weight) or self.appearance_weight<0:
            raise ValueError('Appearance weight must be finite and nonnegative')
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
            # Topo partitions ink among strokes. Overlap must not demand two
            # fully opaque strokes at the same pixel, which is unattainable.
            target=strokes/np.maximum(strokes.sum(0,keepdims=True),1e-6)*ref
            self.appearance_metrics.append(StrokeAppearanceLoss(target,self.appearance_config))

    @staticmethod
    def freeze_priority(details):
        """Freeze priorities for ALL backtracking candidates at this timestep."""
        return [[float(s['weight']) for s in row['per_stroke']] for row in details['per_glyph']]

    @staticmethod
    def _luminance(rgb):
        bounded=rgb.clamp(0,1)
        return (bounded*bounded.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)

    def _crop_ink(self,luminance,index):
        x0,y0,x1,y1=self.slots[index]['bbox']
        if not (0<=x0<x1<=luminance.shape[-1] and 0<=y0<y1<=luminance.shape[-2]):
            raise ValueError('Crop outside generated sheet')
        ref=getattr(self,f'reference_{index}')
        return F.interpolate(1-luminance[...,y0:y1,x0:x1],ref.shape[-2:],mode='bilinear',align_corners=False)

    @torch.no_grad()
    def calibrate_appearance(self,reference_rgb):
        """Use this frozen extractor's NORMAL reference response as appearance target.

        The same displayed reference sheet and torch crop/resize are required.
        This is not a subtraction of a scalar bias: the entire target soft mask
        is measured once, frozen, and shared by all trajectories/candidates.
        """
        self.set_reference_sheet(reference_rgb)
        luminance=self._luminance(reference_rgb)
        summary=[]
        for i in range(len(self.slots)):
            ref=getattr(self,f'reference_{i}');strokes=getattr(self,f'strokes_{i}')
            ink=self._crop_ink(luminance,i)
            target=self.model(ink.float(),ref,strokes)['masks'][0].detach().float()
            legacy=self.appearance_metrics[i](target[None])['loss'].mean()
            metric=StrokeAppearanceLoss(target,self.appearance_config).to(target.device)
            self.appearance_metrics[i]=metric
            summary.append(dict(slot=i,legacy_reference_error=float(legacy),
                                calibrated_reference_error=float(metric(target[None])['loss'].mean())))
        return summary

    def set_reference_sheet(self,reference_rgb):
        self.source_guard=SourceInkGuard(reference_rgb).to(reference_rgb.device)

    def forward(self, rgb, return_masks=False, stroke_weights=None):
        if rgb.ndim!=4 or rgb.shape[0]!=1 or rgb.shape[1]!=3:
            raise ValueError('Expected one jointly generated RGB sheet: 1,3,H,W')
        range_loss=(F.relu(-rgb).square()+F.relu(rgb-1).square()).mean()
        luminance=self._luminance(rgb)
        rows,losses=[],[]
        if stroke_weights is not None and len(stroke_weights)!=len(self.slots):
            raise ValueError('Need stroke weights for each of the four slots')
        for i,(slot,metric) in enumerate(zip(self.slots,self.metrics)):
            ref=getattr(self,f'reference_{i}')
            strokes=getattr(self,f'strokes_{i}')
            ink=self._crop_ink(luminance,i)
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.mask_gradient=='full'):
                prediction=self.model(ink.float(),ref,strokes)
            masks=prediction['masks'].float()
            if self.mask_gradient=='ink_only':
                # Preserve the measured masks EXACTLY in forward, but do not
                # optimize RGB to change the extractor's channel classification.
                masks=masks.detach()+(ink-ink.detach())*prediction['probabilities'].detach()
            measured=metric(masks,return_maps=return_masks)
            if not all(c['pairs'] and c['attachment'] for c in measured['convergence']):
                raise RuntimeError('Stroke reconstruction did not converge; do not guide on truncated measurements')
            values=measured['per_stroke']
            missing=values['missing'].mean()
            cfg=self.structure_config
            structure_loss=(cfg.missing_weight*values['missing']+cfg.split_weight*torch.log1p(values['split']/.2)
                      +cfg.junction_weight*torch.log1p(values['junction']/.01)+cfg.width_weight*values['width'])
            appearance=self.appearance_metrics[i](masks,return_maps=return_masks)
            per_loss=structure_loss+self.appearance_weight*appearance['loss']
            scoring=self.scorer(values['missing'],values['split'],values['junction'],extra=0.)
            scores=scoring['stroke_scores']
            severity=1-scores.detach()/100
            if self.appearance_weight>0:
                severity=torch.maximum(severity,1-torch.exp(-appearance['loss'].detach()/self.appearance_config.priority_scale))
            weights=(1+self.focus_strength*severity).clamp(1,self.max_stroke_weight)
            if stroke_weights is not None:
                weights=torch.as_tensor(stroke_weights[i],device=per_loss.device,dtype=per_loss.dtype)[None].detach()
                if weights.shape!=per_loss.shape or not torch.isfinite(weights).all() or (weights<=0).any():
                    raise ValueError('Frozen stroke weights must be positive and match the stroke count')
            contributions=weights*per_loss/weights.sum(1,keepdim=True)
            guidance_stroke_loss=contributions.sum(1).mean()
            stroke_loss=(weights*structure_loss).sum()/weights.sum()
            # Separate weak source-ink auxiliary, never substituted for Topo loss.
            evidence=foreground_evidence(ink)
            extra=(evidence*(1-getattr(self,f'support_{i}'))).sum()/ref.sum().clamp_min(1)
            losses.append(guidance_stroke_loss+self.extra_weight*extra)
            score=scoring['score'][0]
            per_stroke=[dict(stroke=k+1,score=scores[0,k],weight=weights[0,k],
                weight_share=weights[0,k]/weights.sum(),loss=per_loss[0,k],
                structure_loss=structure_loss[0,k],appearance_loss=appearance['loss'][0,k],
                contour=appearance['contour'][0,k],local_mass=appearance['local_mass'][0,k],
                contribution=contributions[0,k]/len(self.slots),
                **{name:values[name][0,k] for name in ('missing','split','junction','width')})
                for k in range(per_loss.shape[1])]
            row=dict(slot=i,stroke_loss=stroke_loss,missing=missing,
                split=values['split'].mean(),junction=values['junction'].mean(),
                uniform_stroke_loss=structure_loss.mean(),guidance_stroke_loss=guidance_stroke_loss,
                appearance_loss=appearance['loss'].mean(),contour=appearance['contour'].mean(),
                local_mass=appearance['local_mass'].mean(),per_stroke=per_stroke,
                extra_auxiliary=extra,extracted_stroke_score_no_extra=score)
            if return_masks:
                row.update(image=ink,soft_masks=prediction['masks'],probabilities=prediction['probabilities'],
                           defect_maps=measured['maps'])
                row['appearance_maps']={name:appearance[name] for name in ('contour_map','local_mass_map')}
            rows.append(row)
        loss=torch.stack(losses).mean()+self.range_weight*range_loss
        spill=self.source_guard(rgb) if self.source_guard is not None else None
        if spill is not None:loss=loss+self.source_spill_weight*spill['loss']
        return loss,dict(per_glyph=rows,range_loss=range_loss,
                         source_spill_loss=spill['loss'] if spill is not None else None,
                         source_spill_map=spill['local_excess'].detach() if spill is not None else None)
