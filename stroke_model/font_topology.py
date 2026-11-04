"""Cross-typeface Topo extraction; no input-stroke annotations or retraining."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image
import torch

from font_test.render import render_glyph
from .convert_reference_csv import Resolver,existing_path,load_ink,path_list
from .infer_reference import inference_columns
from .infer_topology import load_topology,export_hard_masks,ink_diagnostics,preview_topology


def reference_catalog(csv_path):
    """One entry per reference. Never interpret a numeric ID as a Unicode point."""
    frame=pd.read_csv(csv_path);columns=inference_columns(frame);entries=[]
    char_columns=[c for c in frame.columns if str(c).lower() in ('char','character','glyph','text','letter')]
    for index in frame.drop_duplicates(columns['reference']).index:
        row=frame.iloc[index];glyph=None
        for col in char_columns:
            value=unicodedata.normalize('NFC',str(row[col]).strip())
            if len(value)==1 and not value.isspace():glyph=value;break
        filename=Path(str(row[columns['reference']]).replace('\\','/')).name
        if glyph is None:
            found=re.findall('[가-힣]',unicodedata.normalize('NFC',filename))
            if len(found)==1:glyph=found[0]
        entries.append(dict(csv_row=int(index)+1,reference=str(row[columns['reference']]),glyph=glyph,
                            label=f'{glyph or "?"} | row {int(index)+1} | {filename}'))
    return entries


def run_topology_fonts(checkpoint,csv_path,csv_row,glyph,font_paths,dataset_root=None,
                       placement='reference_bbox',include_reference_control=True,save_dir=None,show=True):
    """Use exactly the selected CSV reference/strokes for all rendered fonts.

    Caller must confirm glyph matches reference preview; filenames may be IDs.
    Raw fonts are NOT warped by the model flow before inference. Whole-glyph
    placement is explicit in the report. No scores imply segmentation accuracy.
    """
    checkpoint,csv_path=existing_path(checkpoint),existing_path(csv_path)
    frame=pd.read_csv(csv_path);columns=inference_columns(frame)
    if type(csv_row) is not int or not 1<=csv_row<=len(frame):raise ValueError('Invalid one-based CSV row')
    row=frame.iloc[csv_row-1]
    resolve=Resolver(csv_path,existing_path(dataset_root) if dataset_root else csv_path.parent)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,size,epoch=load_topology(checkpoint,device);mode=getattr(model,'output_mode','independent')
    reference=load_ink(resolve(row[columns['reference']]),size)
    paths=path_list(row[columns['strokes']])
    if not paths:raise ValueError('Reference stroke list is empty')
    strokes=np.concatenate([load_ink(resolve(p),size) for p in paths])
    if not (strokes.reshape(len(strokes),-1).max(1)>.5).all():raise ValueError('Empty reference stroke image; choose another row')
    if isinstance(font_paths,(str,Path)):font_paths=[font_paths]
    fonts=list(dict.fromkeys(str(existing_path(p)) for p in font_paths))
    if not fonts:raise ValueError('Select at least one font')
    # Validate all selected fonts before creating partial inference outputs.
    rendered=[(path,render_glyph(glyph,path,reference,placement)) for path in fonts]
    destination=Path(save_dir) if save_dir else checkpoint.parent/('font_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True,exist_ok=False)
    report=dict(checkpoint=str(checkpoint),checkpoint_epoch=epoch,output_mode=mode,csv=str(csv_path),
        csv_row=csv_row,glyph=glyph,reference=str(row[columns['reference']]),placement=placement,
        uses_input_stroke_annotations=False,segmentation_accuracy_assessed=False,results=[])
    inputs=([(None,reference)] if include_reference_control else [])+rendered
    for number,(font_path,image) in enumerate(inputs):
        label='Reference control (same image)' if font_path is None else Path(font_path).name
        print(f'{number+1}/{len(inputs)}: {label}',flush=True)
        with torch.inference_mode():
            output=model(*[torch.from_numpy(x)[None].to(device) for x in (image,reference,strokes)])
        probabilities,masks=[output[key][0].float().cpu().numpy() for key in ('probabilities','masks')]
        prefix=destination/f'font_{number:02d}'
        input_path=str(prefix)+'_input.png'
        Image.fromarray(np.uint8(np.clip(1-image[0],0,1)*255)).save(input_path)
        np.savez_compressed(str(prefix)+'.npz',image=image,reference=reference,reference_strokes=strokes,
            probabilities=probabilities,soft_masks=masks,hard_masks=export_hard_masks(image,masks,.5,mode))
        preview_topology(image,reference,strokes,probabilities,masks,prefix,label,.5,show,mode)
        report['results'].append(dict(font=font_path,font_sha256=hashlib.sha256(Path(font_path).read_bytes()).hexdigest() if font_path else None,
            label=label,control=font_path is None,input=input_path,npz=str(prefix)+'.npz',
            **ink_diagnostics(image,masks,output_mode=mode)))
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(len(inputs),len(strokes)+1,
        figsize=(2*(len(strokes)+1),2*len(inputs)),squeeze=False)
    for index,result in enumerate(report['results']):
        with np.load(result['npz'],allow_pickle=False) as data:
            panels=[data['image'][0],*data['hard_masks']]
        for channel,(ax,content) in enumerate(zip(axes[index],panels)):
            ax.imshow(content,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
            if index==0:ax.set_title('Input' if channel==0 else f'Stroke {channel}')
        axes[index,0].text(-.04,.5,result['label'],rotation=90,va='center',ha='right',
                          transform=axes[index,0].transAxes,fontsize=8)
    figure.suptitle('Cross-font extraction (no stroke-level ground truth)')
    figure.tight_layout();report['comparison']=str(destination/'font_comparison.png')
    figure.savefig(report['comparison'],dpi=140)
    if show:plt.show()
    plt.close(figure)
    (destination/'font_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(report['results']).to_csv(destination/'font_results.csv',index=False,encoding='utf-8-sig')
    print('Saved:',destination,flush=True)
    return report
