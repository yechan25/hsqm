"""Inspect paired outputs and traces, including runs from older samplers."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def diagnose_pair(output,show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    output=Path(output)
    report=json.loads((output/'generation_report.json').read_text(encoding='utf-8'))
    pages={p['method']:p for p in report['pages']}
    arrays={}
    for method in ('baseline','guided'):
        with Image.open(pages[method]['input']) as image:arrays[method]=np.array(image.convert('RGB'),dtype=np.float32)
    if arrays['baseline'].shape!=arrays['guided'].shape:raise ValueError('Paired image sizes differ')
    difference=abs(arrays['guided']-arrays['baseline'])
    trace_path=output/'trace_guided.jsonl'
    trace=[json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    accepted=[r for r in trace if r['accepted']]
    stroke_trace=[]
    for record in trace:
        after={row['slot']:row for row in record.get('per_glyph_after',[])}
        for row in record.get('per_glyph_before',[]):
            new={s['stroke']:s for s in after.get(row['slot'],{}).get('per_stroke',[])}
            for stroke in row.get('per_stroke',[]):
                changed=new.get(stroke['stroke'],{})
                entry=dict(step=record['step']+1,timestep=record['timestep'],
                    accepted=record['accepted'],slot=row['slot']+1,stroke=stroke['stroke'],
                    score_before=stroke['score'],score_after=changed.get('score'),
                    loss_before=stroke['loss'],loss_after=changed.get('loss'),
                    weight=stroke['weight'],weight_share=stroke['weight_share'])
                for key in ('structure_loss','appearance_loss','contour','local_mass'):
                    entry[key+'_before']=stroke.get(key)
                    entry[key+'_after']=changed.get(key)
                stroke_trace.append(entry)
    if stroke_trace:
        pd.DataFrame(stroke_trace).to_csv(output/'stroke_guidance_trace.csv',index=False,encoding='utf-8-sig')
    mean=lambda values:float(np.mean(values)) if values else None
    gains=[]
    extra_gains=[]
    appearance_gains=[]
    for row in accepted:
        before,after=row.get('per_glyph_before',[]),row.get('per_glyph_after',[])
        if before and after:
            gains.append(mean([a['stroke_loss']-b['stroke_loss'] for a,b in zip(before,after) if 'stroke_loss' in a]))
            extra_gains.append(mean([a['extra_auxiliary']-b['extra_auxiliary'] for a,b in zip(before,after) if 'extra_auxiliary' in a]))
            appearance_gains.append(mean([a['appearance_loss']-b['appearance_loss'] for a,b in zip(before,after) if 'appearance_loss' in a]))
    summary=dict(accepted_steps=len(accepted),active_steps=sum(r['active'] for r in trace),
        identical_pngs=bool(not difference.any()),png_mae_255=float(difference.mean()),
        maximum_channel_difference_255=float(difference.max()),
        changed_pixel_percent=float((difference.max(-1)>0).mean()*100),
        mean_accepted_proxy_loss_drop=mean([r['loss_before']-r['loss_after'] for r in accepted]),
        mean_accepted_stroke_loss_drop=mean([v for v in gains if v is not None]),
        mean_accepted_extra_auxiliary_drop=mean([v for v in extra_gains if v is not None]),
        mean_accepted_appearance_loss_drop=mean([v for v in appearance_gains if v is not None]),
        mean_state_delta_rms=mean([r['state_delta_rms'] for r in accepted if 'state_delta_rms' in r]),
        mean_state_change_to_base_step_ratio=mean([r['state_change_to_base_step_ratio'] for r in accepted if 'state_change_to_base_step_ratio' in r]),
        mean_state_to_clean_transfer_ratio=mean([r['state_delta_rms']/max(r['clean_delta_rms'],1e-12) for r in accepted if 'state_delta_rms' in r and 'clean_delta_rms' in r]),
        meaningful_changed_pixel_percent_over_8=float((difference.max(-1)>8).mean()*100),
        ddim_state_dtype=report.get('ddim_state_dtype','not recorded in this older run'),
        final_latent_rms_difference=report.get('final_latent_rms_difference'))
    # Check native RGB, not Topo channels or the globally averaged score.
    import torch
    from stroke_model.source_ink_guard import SourceInkGuard
    with Image.open(pages['reference']['input']) as reference:
        ref_rgb=torch.from_numpy(np.asarray(reference.convert('RGB'),np.float32).copy()/255).permute(2,0,1)[None]
    guard=SourceInkGuard(ref_rgb)
    raw={method:guard(torch.from_numpy(array.copy()/255).permute(2,0,1)[None])
         for method,array in arrays.items()}
    added=(raw['guided']['local_excess']-raw['baseline']['local_excess']).clamp_min(0)[0,0].numpy()
    peak=np.unravel_index(int(added.argmax()),added.shape)
    allowance=report.get('guidance_config',{}).get('max_new_local_ink')
    allowance=.002 if allowance is None else allowance
    summary.update(max_new_local_ink_vs_baseline=float(added.max()),
                   new_local_ink_peak_xy=[int(peak[1]),int(peak[0])],
                   passes_final_ink_check=bool(added.max()<=allowance+1e-6))
    fig2,axes2=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    axes2[0].imshow(arrays['guided'].astype(np.uint8));axes2[0].set_title('Guided: original RGB')
    heat2=axes2[1].imshow(added,cmap='magma',vmin=0,vmax=max(float(added.max()),1e-6))
    axes2[1].set_title('New local reference-excess ink\nvs baseline; native 5x5 patches')
    fig2.colorbar(heat2,ax=axes2[1],fraction=.045)
    for axis in axes2:axis.axis('off')
    fig2.savefig(output/'new_ink_local_difference.png',dpi=140)
    if show:plt.show()
    plt.close(fig2)
    (output/'guidance_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    per_slot=[]
    for sample in report['results']:
        if sample.get('method')!='baseline':continue
        x0,y0,x1,y1=sample['crop_bbox']
        d=difference[y0:y1,x0:x1]
        per_slot.append(dict(slot=sample['slot']+1,mae_255=float(d.mean()),
            changed_pixel_percent=float((d.max(-1)>0).mean()*100),maximum_difference_255=float(d.max())))
    frame=pd.DataFrame(per_slot)
    frame.to_csv(output/'pixel_differences.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,method in zip(axes,('baseline','guided')):
        ax.imshow(arrays[method].astype(np.uint8));ax.set_title(method)
    heat=axes[2].imshow(difference.mean(-1),cmap='magma',vmin=0,vmax=max(float(difference.mean(-1).max()),1))
    axes[2].set_title('Absolute pixel difference\nAmplified display; not an improvement map')
    fig.colorbar(heat,ax=axes[2],fraction=.045,label='Mean channel difference (0-255)')
    for ax in axes:ax.axis('off')
    fig.savefig(output/'paired_pixel_difference.png',dpi=140)
    if show:plt.show()
    plt.close(fig)
    print('채택/적용 시도:',summary['accepted_steps'],'/',summary['active_steps'])
    print('PNG 완전 동일:',summary['identical_pngs'],'| 평균 밝기 차이(0~255):',summary['png_mae_255'])
    print('값이 달라진 픽셀 비율(%):',summary['changed_pixel_percent'])
    print('평균 추출 획 손실 감소:',summary['mean_accepted_stroke_loss_drop'])
    print('평균 주변 잉크 보조항 감소:',summary['mean_accepted_extra_auxiliary_drop'])
    print('평균 윤곽·국소 잉크량 오차 감소:',summary['mean_accepted_appearance_loss_drop'])
    print('평균 실제 다음 상태 변화 RMS:',summary['mean_state_delta_rms'])
    print('예상 완성 보정 대비 다음 상태 전달 비율:',summary['mean_state_to_clean_transfer_ratio'])
    print('채널값이 8/255 넘게 변한 픽셀 비율(%):',summary['meaningful_changed_pixel_percent_over_8'])
    print('최종 새 국소 잉크 증가 / 허용량:',summary['max_new_local_ink_vs_baseline'],'/',allowance)
    if not summary['passes_final_ink_check']:
        print('최종 원본에 새 국소 잉크 증가가 있습니다. 점수 상승만으로 개선으로 판정하지 마세요.')
    if summary['mean_state_delta_rms'] is None:
        print('이전 실행 로그에는 실제 상태 변화량이 없습니다. 기존 PNG 차이는 위에서 확인할 수 있습니다.')
    return summary,frame
