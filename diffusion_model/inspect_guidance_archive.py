"""Audit a saved guidance ZIP without checkpoints, extraction, or resampling."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from stroke_model.stroke_appearance import StrokeAppearanceLoss,AppearanceConfig
from stroke_model.source_ink_guard import SourceInkGuard


def inspect_archive(archive_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with zipfile.ZipFile(archive_path) as archive:
        report=json.loads(archive.read('generation/generation_report.json'))
        scores=pd.read_csv(archive.open('topo_evaluation/topo_scores.csv'))
        old_strokes=pd.read_csv(archive.open('topo_evaluation/stroke_scores.csv'))
        trace=[json.loads(line) for line in archive.read('generation/trace_guided.jsonl').splitlines()]
        for slot in sorted(scores.slot.unique()):
            with np.load(io.BytesIO(archive.read(f'topo_evaluation/reference_slot_{slot}.npz'))) as data:
                target=data['soft_masks'].copy()
            metric=StrokeAppearanceLoss(target,AppearanceConfig(**report.get('appearance_config',{})))
            for method in scores.method.unique():
                with np.load(io.BytesIO(archive.read(f'topo_evaluation/{method}_slot_{slot}.npz'))) as data:
                    masks=torch.from_numpy(data['soft_masks'].copy())[None]
                measured=metric(masks)['loss'][0]
                old=old_strokes[(old_strokes.method==method)&(old_strokes.slot==slot)].sort_values('stroke')
                for k,(value,(_,before)) in enumerate(zip(measured,old.iterrows())):
                    rows.append(dict(method=method,slot=int(slot),stroke=k+1,
                        saved_appearance_loss=float(before.appearance_loss),
                        reference_response_appearance_loss=float(value)))
        pages={p['method']:p for p in report['pages']}
        images={}
        for method in ('reference','baseline','guided'):
            page=pages[method]
            name=f"generation/page_{page['page_id']:03d}_{method}.png"
            with Image.open(io.BytesIO(archive.read(name))) as image:
                images[method]=np.array(image.convert('RGB'),dtype=np.float32)
        diff=abs(images['guided']-images['baseline'])
    guard=SourceInkGuard(torch.from_numpy(images['reference']/255).permute(2,0,1)[None])
    raw={method:guard(torch.from_numpy(image/255).permute(2,0,1)[None]) for method,image in images.items()}
    increase=(raw['guided']['local_excess']-raw['baseline']['local_excess']).clamp_min(0)[0,0]
    peak=np.unravel_index(int(increase.argmax()),tuple(increase.shape))
    allowance=report.get('guidance_config',{}).get('max_new_local_ink')
    allowance=.002 if allowance is None else allowance
    accepted=[r for r in trace if r['accepted']]
    transfer=[r['state_delta_rms']/max(r['clean_delta_rms'],1e-12) for r in accepted
              if 'state_delta_rms' in r and 'clean_delta_rms' in r]
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'reference_response_reanalysis.csv',index=False,encoding='utf-8-sig')
    appearance=frame.groupby(['method','slot'])[['saved_appearance_loss','reference_response_appearance_loss']].mean().reset_index()
    appearance.to_csv(output/'reference_response_summary.csv',index=False,encoding='utf-8-sig')
    summary=dict(source_archive_sha256=hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
        resampled=False,uses_saved_extracted_masks=True,
        note='Recomputed reference-relative measurements on OLD images; not a new generation result.',
        accepted_steps=len(accepted),active_steps=sum(r['active'] for r in trace),
        mean_state_to_clean_transfer_ratio=float(np.mean(transfer)) if transfer else None,
        min_state_to_clean_transfer_ratio=min(transfer) if transfer else None,
        pixel_mae_255=float(diff.mean()),
        changed_pixel_over_8_percent=float((diff.max(-1)>8).mean()*100),
        max_new_local_ink_vs_baseline=float(increase.max()),
        new_local_ink_peak_xy=[int(peak[1]),int(peak[0])],
        passes_final_ink_check=float(increase.max())<=allowance+1e-6,
        final_active_attempts=next((r.get('attempts',[]) for r in reversed(trace) if r['active']),[]),
        primary_scores=scores[['method','slot','extracted_stroke_score_no_extra']].to_dict('records'),
        appearance=appearance.to_dict('records'))
    (output/'archive_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--output',default='outputs/guidance_archive_audit')
    args=parser.parse_args()
    summary=inspect_archive(args.archive,args.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('primary_scores','appearance')},indent=2))
