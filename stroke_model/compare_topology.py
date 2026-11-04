"""Read-only checkpoint comparison on identical input; no automatic quality verdict."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import torch

from .convert_reference_csv import Resolver, existing_path, load_ink
from .infer_reference import inference_columns, load_inference_row
from .infer_topology import select_pattern_rows, export_hard_masks
from .reference_model import ReferenceStrokeModel
from .reference_v2_topology import ARCHITECTURE


def load_comparison_model(path,device):
    state=torch.load(existing_path(path),map_location='cpu',weights_only=True)
    if state.get('architecture') not in ('reference_v2',ARCHITECTURE):
        raise ValueError('Compare V2 or V2-based Topo checkpoints, not legacy independent Topo')
    model=ReferenceStrokeModel(image_size=state['image_size'],width=state['width'],pretrained=False)
    model.load_state_dict(state['model'],strict=True)
    metadata={key:state.get(key) for key in ('architecture','image_size','epoch','ownership_weight',
        'affinity_weight','ownership_config','junction_config')}
    return model.to(device).eval(),metadata


def find_before_checkpoint(after):
    source=Path(after).parent/'initial_checkpoint_source.txt'
    if source.is_file():
        value=source.read_text(encoding='utf-8').strip()
        if value and value!='None':
            return existing_path(value)
    raise FileNotFoundError('튜닝 전 모델을 자동 확인할 수 없습니다. BEFORE_CHECKPOINT에 정확한 경로를 지정하세요.')


def assignment_changes(before_p,after_p,image):
    if before_p.shape!=after_p.shape:
        raise ValueError('Prediction shapes differ')
    ink=image[0]>.5
    before,after=before_p.argmax(0),after_p.argmax(0)
    k=len(before_p)
    transfers=np.bincount((before[ink]*k+after[ink]).ravel(),minlength=k*k).reshape(k,k)
    rows=[]
    for index in range(k):
        b=(before==index)&ink;a=(after==index)&ink
        rows.append(dict(stroke=index+1,before_pixels=int(b.sum()),after_pixels=int(a.sum()),
            gained_pixels=int((a&~b).sum()),lost_pixels=int((b&~a).sum()),
            before_components=int(ndi.label(b,np.ones((3,3)))[1]),
            after_components=int(ndi.label(a,np.ones((3,3)))[1])))
    top=np.sort(after_p,axis=0)
    margin=top[-1]-top[-2] if k>1 else np.ones_like(top[-1])
    diagnostics=dict(changed_pixels=int(((before!=after)&ink).sum()),input_pixels=int(ink.sum()),
        low_winner_probability_pixels=int(((top[-1]<.5)&ink).sum()),
        low_winner_margin_pixels=int(((margin<.15)&ink).sum()))
    return before,after,transfers,rows,diagnostics,margin


def plot_comparison(image,reference,strokes,before_p,after_p,prefix,focus_strokes=None,show=True):
    import matplotlib.pyplot as plt
    b,a,transfers,rows,diagnostics,margin=assignment_changes(before_p,after_p,image)
    ink=image[0]>.5;k=len(strokes)
    palette=plt.get_cmap('tab20')(np.arange(k)%20)[:,:3]
    def colored(labels):
        result=np.zeros((*ink.shape,3));result[ink]=palette[labels[ink]];return result
    changed=np.zeros((*ink.shape,3));changed[ink]=.25;changed[ink&(b!=a)]=(1,0,0)
    figure,axes=plt.subplots(1,7,figsize=(18,3))
    panels=((image[0],'Same input'),(reference[0],'Reference'),(colored(b),'Before'),
        (colored(a),'After'),(changed,'Changed ink = red'),
        (np.where(ink,after_p.max(0),np.nan),'After max probability'),
        (np.where(ink,margin,np.nan),'After winner margin'))
    for ax,(content,title) in zip(axes,panels):
        ax.imshow(content,cmap='gray' if content.ndim==3 or title in ('Same input','Reference') else 'viridis',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(title);ax.axis('off')
    figure.tight_layout();figure.savefig(str(prefix)+'_comparison.png',dpi=140)
    if show:plt.show()
    plt.close(figure)
    selected=list(range(k)) if focus_strokes is None else [n-1 for n in focus_strokes if 1<=n<=k]
    if not selected:selected=list(range(k))
    before_masks=image*before_p;after_masks=image*after_p
    bh=export_hard_masks(image,before_masks,output_mode='partition')
    ah=export_hard_masks(image,after_masks,output_mode='partition')
    figure,axes=plt.subplots(5,len(selected),figsize=(2.5*len(selected),10),squeeze=False)
    for col,index in enumerate(selected):
        for row,(content,title) in enumerate(((strokes[index],'Ref stroke'),(before_masks[index],'Before soft'),
            (after_masks[index],'After soft'),(bh[index],'Before exclusive'),(ah[index],'After exclusive'))):
            axes[row,col].imshow(content,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            axes[row,col].set_title(f'{title} {index+1}');axes[row,col].axis('off')
    figure.tight_layout();figure.savefig(str(prefix)+'_strokes.png',dpi=140)
    if show:plt.show()
    plt.close(figure)
    pd.DataFrame(rows).to_csv(str(prefix)+'_stroke_changes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(transfers,index=[f'before_{i+1}' for i in range(k)],
        columns=[f'after_{i+1}' for i in range(k)]).to_csv(str(prefix)+'_transfers.csv',encoding='utf-8-sig')
    return dict(diagnostics=diagnostics,strokes=rows,transfers=transfers.tolist())


def compare_checkpoints(before,after,csv_path,dataset_root=None,sample_rows=None,input_pattern='*modified*.png',
                        input_image=None,reference_row=1,focus_strokes=None,save_dir=None,show=True):
    before,after,csv_path=map(existing_path,(before,after,csv_path))
    if before.resolve()==after.resolve():raise ValueError('Before and after are the same checkpoint')
    frame=pd.read_csv(csv_path);columns=inference_columns(frame)
    resolve=Resolver(csv_path,existing_path(dataset_root) if dataset_root else csv_path.parent)
    selected=([reference_row] if input_image else list(sample_rows) if sample_rows is not None else
        select_pattern_rows(frame,columns['image'],input_pattern) if input_pattern else list(range(1,len(frame)+1)))
    if not selected or any(not isinstance(n,int) or not 1<=n<=len(frame) for n in selected):
        raise ValueError('선택된 CSV 행이 없습니다. SAMPLE_ROWS 또는 INPUT_PATTERN을 확인하세요.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bm,bmeta=load_comparison_model(before,device);am,ameta=load_comparison_model(after,device)
    if bmeta['image_size']!=ameta['image_size']:raise ValueError('Checkpoint image sizes differ')
    size=ameta['image_size']
    destination=Path(save_dir) if save_dir else after.parent/('comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True,exist_ok=False)
    report=dict(before_checkpoint=str(before),after_checkpoint=str(after),before=bmeta,after=ameta,
        comparison_is_not_accuracy_score=True,results=[],skipped=[])
    print('Before:',before,bmeta,flush=True);print('After:',after,ameta,flush=True)
    for number in selected:
        try:
            row=frame.iloc[number-1].copy()
            # A saved canvas image replaces only the input; reference correspondence stays explicit.
            if input_image:row[columns['image']]=str(existing_path(input_image).resolve())
            image,reference,strokes,input_path,_=load_inference_row(row,columns,resolve,size)
        except ValueError as exc:
            if str(exc)!='Empty reference stroke image':raise
            report['skipped'].append(dict(csv_row=number,reason=str(exc)));continue
        tensors=[x[None].to(device) for x in (image,reference,strokes)]
        with torch.inference_mode():
            bo=bm(*tensors);ao=am(*tensors)
        bp,ap=[out['probabilities'][0].float().cpu().numpy() for out in (bo,ao)]
        prefix=destination/f'row_{number:05d}'
        payload=dict(image=image.numpy(),reference=reference.numpy(),reference_strokes=strokes.numpy(),
                     before_probabilities=bp,after_probabilities=ap)
        for name,out in (('before',bo),('after',ao)):
            for key in ('hints','flow'):
                payload[f'{name}_{key}']=out[key][0].float().cpu().numpy()
        np.savez_compressed(str(prefix)+'.npz',**payload)
        entry=plot_comparison(image.numpy(),reference.numpy(),strokes.numpy(),bp,ap,prefix,focus_strokes,show)
        entry.update(csv_row=number,input=str(input_path),input_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest())
        report['results'].append(entry)
        print(f'CSV {number}: {input_path.name}',entry['diagnostics'],flush=True)
        print(pd.DataFrame(entry['strokes']).to_string(index=False),flush=True)
        moved=[dict(from_stroke=i+1,to_stroke=j+1,pixels=value)
               for i,line in enumerate(entry['transfers']) for j,value in enumerate(line) if i!=j and value]
        if moved:
            print('Changed assignments (not an accuracy score):',flush=True)
            print(pd.DataFrame(moved).sort_values('pixels',ascending=False).to_string(index=False),flush=True)
    (destination/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('비교 저장:',destination,flush=True)
    return report


def inspect_training_supervision(manifest,sample_rows=(1,),save_dir=None,show=True):
    """Inspect ACTUALLY SAVED training maps, never inferred labels for a test canvas."""
    import matplotlib.pyplot as plt
    from .train_reference import read_manifest
    rows=read_manifest(existing_path(manifest));results=[]
    for number in sample_rows:
        if not 1<=number<=len(rows):raise ValueError('Training manifest row out of range')
        row=rows[number-1]
        if row['split']!='train' or not row.get('ownership_path'):
            results.append(dict(manifest_row=number,status='no_saved_training_ownership'));continue
        with np.load(row['path'],allow_pickle=False) as arrays:
            image=arrays['image'];labels=arrays['labels'];strokes=arrays['strokes'];target=arrays['target_masks']
        with np.load(existing_path(row['ownership_path']),allow_pickle=False) as arrays:
            owners=arrays['owners'];weight=arrays['reliability']
        k=len(strokes);palette=plt.get_cmap('tab20')(np.arange(k)%20)[:,:3]
        def colored(ids):
            out=np.zeros((*ids.shape,3));valid=ids>=0;out[valid]=palette[ids[valid]];return out
        figure,axes=plt.subplots(1,4,figsize=(11,3))
        for ax,(content,title) in zip(axes,((image[0],'Training input'),(colored(labels),'Original labels'),
            (colored(np.where(weight>0,owners,-1)),'SAVED weak owners'),(weight/255,'Reliability (heuristic)'))):
            ax.imshow(content,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title);ax.axis('off')
        figure.tight_layout()
        if save_dir:
            folder=Path(save_dir);folder.mkdir(parents=True,exist_ok=True)
            figure.savefig(folder/f'train_{number:05d}_ownership.png',dpi=140)
            np.savez_compressed(folder/f'train_{number:05d}.npz',image=image,labels=labels,target_masks=target,owners=owners,reliability=weight)
        if show:plt.show()
        plt.close(figure)
        results.append(dict(manifest_row=number,status='saved_training_map',image_npz=row['path'],ownership_npz=row['ownership_path']))
    print(results,flush=True)
    return results


def compare_canvas(before,after,csv_path,dataset_root=None,focus_strokes=None,
                   canvas_size=384,pen_width=18,save_dir=None,show=True):
    """Draw once, save the raw canvas, then compare both models on that image."""
    import base64
    import io
    from PIL import Image
    from .canvas_topology import canvas_prompt, reference_canvas_options
    before,after,csv_path=map(existing_path,(before,after,csv_path))
    state=torch.load(after,map_location='cpu',weights_only=True)
    size=state['image_size'];del state
    frame=pd.read_csv(csv_path);columns=inference_columns(frame)
    resolve=Resolver(csv_path,existing_path(dataset_root) if dataset_root else csv_path.parent)
    options=reference_canvas_options(frame,columns,resolve,size)
    print('한 번 그린 동일한 글씨를 두 모델에 적용합니다. 기준을 고르고 획 추출을 누르세요.',flush=True)
    answer=canvas_prompt(options,canvas_size,pen_width)
    destination=Path(save_dir) if save_dir else after.parent/('canvas_comparison_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True,exist_ok=False)
    raw=Image.open(io.BytesIO(base64.b64decode(answer['png'].split(',',1)[1]))).convert('RGB')
    image_path=destination/'input.png';raw.save(image_path)
    if load_ink(image_path,size).sum()<1:raise ValueError('Canvas is empty')
    report=compare_checkpoints(before,after,csv_path,dataset_root=dataset_root,
        input_image=image_path,reference_row=int(answer['row'])+1,focus_strokes=focus_strokes,
        save_dir=destination/'results',show=show)
    report['canvas_input']=str(image_path)
    report['reference_csv_row']=int(answer['row'])+1
    (destination/'canvas_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('같은 그림으로 다시 비교할 때 사용할 INPUT_IMAGE:',image_path,flush=True)
    return report
