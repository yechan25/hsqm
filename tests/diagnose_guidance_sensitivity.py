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
    rows=[]
    fig,axes=plt.subplots(1,len(variants),figsize=(14,3),layout='constrained')
    for axis,(name,ink) in zip(axes,variants.items()):
        masks=torch.from_numpy(ink)[None].requires_grad_(True)
        measured=metric(masks);values=measured['per_stroke']
        loss=(values['missing']+torch.log1p(values['split']/.2)+torch.log1p(values['junction']/.01)).mean()
        gradient,=torch.autograd.grad(loss,masks)
        score=scorer(values['missing'],values['split'],values['junction'],extra=0)['score'][0]
        row=dict(case=name,loss=float(loss.detach()),score=float(score.detach()),
            gradient_rms=float(gradient.square().mean().sqrt()),
            converged=all(c['pairs'] and c['attachment'] for c in measured['convergence']))
        if not row['converged']:raise RuntimeError('Default reconstruction did not converge')
        rows.append(row)
        axis.imshow(ink[0],cmap='gray_r',vmin=0,vmax=1);axis.axis('off')
        axis.set_title(f"{name.replace('_',' ')}\nscore={row['score']:.2f}, loss={row['loss']:.3g}\ngradient RMS={row['gradient_rms']:.3g}",fontsize=10)
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
