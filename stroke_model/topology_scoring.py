"""Topo extraction display + existing source-ink scoring, explicitly independent.

No training, score recalibration, or use of extractor channel defects as handwriting defects.
"""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .infer_topology import run_topology_inference, export_hard_masks
from .canvas_topology import run_topology_canvas
from .structural_indicator import IndicatorConfig, score_structure_report, export_indicator_debug
from .source_structure import SourceConfig


def _config(value,kind):
    return kind(**value) if isinstance(value,dict) else value or kind()


def extraction_samples(report):
    """Make source-scoring records; stroke IDs are metadata, NOT fake zero errors."""
    entries=report.get('results')
    if entries is None and report.get('npz'):
        entries=[dict(npz=report['npz'],input=str(Path(report['npz']).parent/'input.png'))]
    if not entries:raise ValueError('No successful Topo extraction snapshots to score')
    samples=[]
    for entry in entries:
        path=Path(entry['npz'])
        with np.load(path,allow_pickle=False) as data:
            required=('image','reference_strokes','soft_masks')
            if any(key not in data for key in required) or not any(key in data for key in ('reference_image','reference')):
                raise ValueError(f'Need original image, full reference, reference_strokes and soft_masks: {path}')
            strokes=data['reference_strokes'];masks=data['soft_masks']
            if strokes.ndim!=3 or masks.shape!=strokes.shape:
                raise ValueError('Topo masks and reference stroke shapes differ')
            count=len(strokes)
        samples.append(dict(input=entry.get('input',str(path)),npz=str(path),csv_row=entry.get('csv_row'),
            strokes=[dict(stroke=k+1,measurement_status='extractor_errors_not_scored') for k in range(count)]))
    return samples


def plot_extraction_with_score(sample,score,prefix,output_mode='partition',show=True):
    import matplotlib.pyplot as plt
    with np.load(sample['npz'],allow_pickle=False) as data:
        image=data['image'];masks=data['soft_masks'];reference=data['reference_image'] if 'reference_image' in data else data['reference']
    hard=export_hard_masks(image,masks,output_mode=output_mode)
    k=len(masks);ink=image[0]>.5
    palette=plt.get_cmap('tab20')(np.arange(k)%20)[:,:3]
    colored=np.zeros((*ink.shape,3));colored[ink]=palette[masks.argmax(0)[ink]]
    figure,axes=plt.subplots(2,k+2,figsize=(2*(k+2),5),squeeze=False)
    for ax in axes.flat:ax.axis('off')
    axes[0,0].imshow(image[0],cmap='gray',vmin=0,vmax=1);axes[0,0].set_title('Scored input')
    axes[1,0].imshow(reference[0],cmap='gray',vmin=0,vmax=1);axes[1,0].set_title('Reference')
    axes[0,1].imshow(colored);axes[0,1].set_title('Topo assignment')
    lines=[f"SOURCE SCORE\n{score['structural_score']:.1f} / 100",'Score caps:']
    for key,label in (('missing','Missing'),('split','Split'),('junction','Junction'),('extra','Extra')):
        lines.append(f"{label}: {score[key+'_score_cap']:.1f}")
    axes[1,1].text(.03,.95,'\n'.join(lines),va='top',fontsize=10)
    for index in range(k):
        axes[0,index+2].imshow(masks[index],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[0,index+2].set_title(f'Soft {index+1}')
        axes[1,index+2].imshow(hard[index],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[1,index+2].set_title(f'Exclusive {index+1}' if output_mode=='partition' else f'Hard {index+1}')
    figure.suptitle(f"Source ink score = {score['structural_score']:.1f}; Topo masks are displayed, not scored")
    figure.tight_layout();path=str(prefix)+'.png';figure.savefig(path,dpi=140)
    if show:plt.show()
    plt.close(figure)
    return path


def score_topology_results(extraction_report,score_config=None,source_config=None,
                           save_dir=None,show=True,export_debug=True):
    cfg=_config(score_config,IndicatorConfig);source_cfg=_config(source_config,SourceConfig)
    samples=extraction_samples(extraction_report)
    base=Path(extraction_report.get('checkpoint',samples[0]['npz'])).parent
    destination=Path(save_dir) if save_dir else base/('topo_scores_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True,exist_ok=False)
    source_report=dict(mode='topo_display_and_source_score',checkpoint=extraction_report.get('checkpoint'),
        uses_input_stroke_annotations=False,extraction_used_for_numeric_score=False,results=samples)
    (destination/'source_report.json').write_text(json.dumps(source_report,ensure_ascii=False,indent=2),encoding='utf-8')
    results=[];combined_figures=[];stroke_rows=[]
    for number,sample in enumerate(samples,1):
        print(f'원본 점수 계산 {number}/{len(samples)}: {sample["input"]}',flush=True)
        measured=score_structure_report({'results':[sample]},config=cfg,source_config=source_cfg,
            measurement_mode='source',save_dir=destination/f'sample_{number:04d}',show_images=False,
            show_stroke_table=False)
        score=dict(measured['results'][0]);score['sample']=number
        score['strokes']=[{**s,'sample':number} for s in score['strokes']]
        results.append(score);stroke_rows.extend(score['strokes'])
        print(f"원본 점수: {score['structural_score']:.1f}/100 / 추출 마스크의 파임, 혼입은 점수 입력이 아닙니다.",flush=True)
        figure=plot_extraction_with_score(sample,score,destination/f'combined_{number:04d}',
            extraction_report.get('output_mode','partition'),show)
        combined_figures.append(figure)
        if show:
            from IPython.display import display
            display(pd.DataFrame(score['strokes'])[['stroke','stroke_score','missing','split','junction']])
    indicator=dict(indicator='structural_indicator_v4_registered_source',measurement_mode='source',
        config=asdict(cfg),source_config=asdict(source_cfg),trained=False,calibrated_to_human_quality=False,
        save_dir=str(destination),results=results,extraction_used_for_numeric_score=False)
    (destination/'indicator_report.json').write_text(json.dumps(indicator,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('strokes','extractor_strokes','registration')}
                  for r in results]).to_csv(destination/'glyph_scores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stroke_rows).to_csv(destination/'stroke_scores.csv',index=False,encoding='utf-8-sig')
    report=dict(checkpoint=extraction_report.get('checkpoint'),checkpoint_epoch=extraction_report.get('checkpoint_epoch'),
        scoring_input='original_ink',topology_masks_used_for_score=False,extraction=extraction_report,
        indicator=indicator,combined_figures=combined_figures,save_dir=str(destination))
    if export_debug:
        report['debug_zip']=export_indicator_debug(source_report,indicator,destination/'indicator_debug.zip')
    (destination/'combined_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('추출·점수 결과 저장:',destination,flush=True)
    return report


def run_scored_topology_csv(checkpoint,csv_path,dataset_root=None,input_files=None,sample_rows=None,
                            input_pattern='*modified*.png',max_samples=8,score_config=None,source_config=None,
                            show=True,export_debug=True):
    extraction=run_topology_inference(checkpoint,csv_path,dataset_root=dataset_root,input_files=input_files,
        sample_rows=sample_rows,input_pattern=input_pattern,max_samples=max_samples,show=False)
    return score_topology_results(extraction,score_config,source_config,show=show,export_debug=export_debug)


def run_scored_topology_canvas(checkpoint,csv_path,dataset_root=None,canvas_size=384,pen_width=18,
                               score_config=None,source_config=None,show=True,export_debug=True):
    extraction=run_topology_canvas(checkpoint,csv_path,dataset_root=dataset_root,canvas_size=canvas_size,
        pen_width=pen_width,show=False)
    return score_topology_results(extraction,score_config,source_config,show=show,export_debug=export_debug)
