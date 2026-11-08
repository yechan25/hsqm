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
    mean=lambda values:float(np.mean(values)) if values else None
    gains=[]
    extra_gains=[]
    for row in accepted:
        before,after=row.get('per_glyph_before',[]),row.get('per_glyph_after',[])
        if before and after:
            gains.append(mean([a['stroke_loss']-b['stroke_loss'] for a,b in zip(before,after) if 'stroke_loss' in a]))
            extra_gains.append(mean([a['extra_auxiliary']-b['extra_auxiliary'] for a,b in zip(before,after) if 'extra_auxiliary' in a]))
    summary=dict(accepted_steps=len(accepted),active_steps=sum(r['active'] for r in trace),
        identical_pngs=bool(not difference.any()),png_mae_255=float(difference.mean()),
        maximum_channel_difference_255=float(difference.max()),
        changed_pixel_percent=float((difference.max(-1)>0).mean()*100),
        mean_accepted_proxy_loss_drop=mean([r['loss_before']-r['loss_after'] for r in accepted]),
        mean_accepted_stroke_loss_drop=mean([v for v in gains if v is not None]),
        mean_accepted_extra_auxiliary_drop=mean([v for v in extra_gains if v is not None]),
        mean_state_delta_rms=mean([r['state_delta_rms'] for r in accepted if 'state_delta_rms' in r]),
        mean_state_change_to_base_step_ratio=mean([r['state_change_to_base_step_ratio'] for r in accepted if 'state_change_to_base_step_ratio' in r]),
        ddim_state_dtype=report.get('ddim_state_dtype','not recorded in this older run'),
        final_latent_rms_difference=report.get('final_latent_rms_difference'))
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
    print('평균 실제 다음 상태 변화 RMS:',summary['mean_state_delta_rms'])
    if summary['mean_state_delta_rms'] is None:
        print('이전 실행 로그에는 실제 상태 변화량이 없습니다. 기존 PNG 차이는 위에서 확인할 수 있습니다.')
    return summary,frame
