"""Isolate the existing stroke objective's sensitivity, without SD/Topo weights.

Usage: python -m tests.diagnose_guidance_sensitivity  (or run this file directly).
The masks are idealized extracted strokes, NOT actual diffusion outputs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stroke_model.stroke_structure import StrokeStructureLoss
from stroke_model.structural_indicator import StructuralIndicator,IndicatorConfig
from stroke_model.stroke_appearance import StrokeAppearanceLoss


def diagnose(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    reference=np.zeros((1,24,24),np.float32)
    reference[0,10:13,4:20]=1
    variants={'reference':reference.copy()}
    thick=reference.copy();thick[0,7:16,4:20]=1
    variants['three_times_thicker']=thick
    bump=reference.copy();bump[0,9,10:15]=1
    variants['edge_bump']=bump
    notch=reference.copy();notch[0,10,10:15]=0
    variants['edge_notch']=notch
    broken=reference.copy();broken[:,:,11:13]=0
    variants['broken_stroke_control']=broken
    metric=StrokeStructureLoss(reference)  # Actual default settings; no truncated cap.
    scorer=StructuralIndicator(IndicatorConfig(extra_weight=0))
    appearance=StrokeAppearanceLoss(reference)
    rows=[]
    fig,axes=plt.subplots(2,len(variants),figsize=(14,6),layout='constrained')
    for column,(name,ink) in enumerate(variants.items()):
        axis=axes[0,column]
        masks=torch.from_numpy(ink)[None].requires_grad_(True)
        measured=metric(masks);values=measured['per_stroke']
        loss=(values['missing']+torch.log1p(values['split']/.2)+torch.log1p(values['junction']/.01)).mean()
        gradient,=torch.autograd.grad(loss,masks)
        score=scorer(values['missing'],values['split'],values['junction'],extra=0)['score'][0]
        row=dict(case=name,loss=float(loss.detach()),score=float(score.detach()),
            gradient_rms=float(gradient.square().mean().sqrt()),
            converged=all(c['pairs'] and c['attachment'] for c in measured['convergence']))
        repaired=torch.from_numpy(ink)[None].clone().requires_grad_(True)
        shape=appearance(repaired)['loss'].sum()
        shape_gradient,=torch.autograd.grad(shape,repaired)
        row.update(appearance_before=float(shape.detach()),
                   appearance_gradient_rms=float(shape_gradient.square().mean().sqrt()))
        # Only a synthetic soft-mask descent demonstration, not diffusion.
        for _ in range(25):
            repaired=repaired.detach().requires_grad_(True)
            shape=appearance(repaired)['loss'].sum()
            g,=torch.autograd.grad(shape,repaired)
            repaired=(repaired-.03*g/g.abs().max().clamp_min(1e-12)).clamp(0,1)
        row['appearance_after']=float(appearance(repaired)['loss'].sum().detach())
        if not row['converged']:raise RuntimeError('Default reconstruction did not converge')
        rows.append(row)
        axis.imshow(ink[0],cmap='gray_r',vmin=0,vmax=1);axis.axis('off')
        axis.set_title(f"{name.replace('_',' ')}\nscore={row['score']:.2f}, loss={row['loss']:.3g}\ngradient RMS={row['gradient_rms']:.3g}",fontsize=10)
        axes[1,column].imshow(repaired[0,0].detach().numpy(),cmap='gray_r',vmin=0,vmax=1)
        axes[1,column].axis('off')
        axes[1,column].set_title(f"Synthetic mask descent, not diffusion\nappearance: {row['appearance_before']:.4g} -> {row['appearance_after']:.4g}\nnew gradient RMS={row['appearance_gradient_rms']:.3g}",fontsize=9)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(purpose='Isolate structural objective sensitivity; not pretrained-model validation',
        ideal_extracted_masks=True,extra_auxiliary_included=False,
        config='StrokeStructureLoss defaults; primary score extra_weight=0',results=rows)
    (output/'sensitivity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig.savefig(output/'sensitivity.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/guidance_sensitivity')
    diagnose(parser.parse_args().output)
