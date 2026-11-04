"""Colab canvas inference, including V2 softmax and legacy independent outputs."""
import base64
from datetime import datetime
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .convert_reference_csv import Resolver, existing_path, load_ink, path_list
from .infer_reference import inference_columns
from .infer_topology import load_topology, export_hard_masks, ink_diagnostics, preview_topology


def canvas_prompt(options, size, pen):
    from google.colab.output import eval_js
    script = r'''
    (async () => {
      const cfg = __CONFIG__, wrap = document.createElement('div');
      const select = document.createElement('select');
      for (const item of cfg.options) {
        const option = document.createElement('option');
        option.value=item.row; option.textContent=item.label; select.append(option);
      }
      const preview=document.createElement('img');
      preview.style.cssText='display:block;width:160px;border:1px solid #aaa;margin:8px 0';
      const update=()=>{preview.src=cfg.options.find(x=>x.row===Number(select.value)).preview;};
      select.onchange=update; update();
      const canvas=document.createElement('canvas'); canvas.width=canvas.height=cfg.size;
      canvas.style.cssText=`width:${cfg.size}px;max-width:100%;border:1px solid #888;touch-action:none`;
      const ctx=canvas.getContext('2d',{willReadFrequently:true});
      ctx.fillStyle='white';ctx.fillRect(0,0,cfg.size,cfg.size);
      ctx.fillStyle=ctx.strokeStyle='black';ctx.lineCap=ctx.lineJoin='round';
      const undo=document.createElement('button');undo.textContent='한 획 취소';
      const clear=document.createElement('button');clear.textContent='전체 지우기';
      const submit=document.createElement('button');submit.textContent='획 추출';
      const slider=document.createElement('input');slider.type='range';slider.min=2;slider.max=48;slider.value=cfg.pen;
      const controls=document.createElement('div');controls.style.marginTop='8px';
      controls.append(undo,clear,document.createTextNode(' 펜 굵기 '),slider,submit);
      wrap.append(select,preview,canvas,controls);document.body.append(wrap);
      let drawing=false,done=false,hasInk=false;const history=[];
      const save=()=>history.push({pixels:ctx.getImageData(0,0,cfg.size,cfg.size),hasInk});
      const point=e=>{const r=canvas.getBoundingClientRect();return [(e.clientX-r.left)*cfg.size/r.width,(e.clientY-r.top)*cfg.size/r.height];};
      canvas.onpointerdown=e=>{if(done)return;e.preventDefault();save();drawing=true;hasInk=true;
        canvas.setPointerCapture(e.pointerId);const [x,y]=point(e);ctx.lineWidth=Number(slider.value);
        ctx.beginPath();ctx.arc(x,y,ctx.lineWidth/2,0,2*Math.PI);ctx.fill();ctx.beginPath();ctx.moveTo(x,y);};
      canvas.onpointermove=e=>{if(!drawing||done)return;e.preventDefault();const [x,y]=point(e);ctx.lineTo(x,y);ctx.stroke();};
      canvas.onpointerup=canvas.onpointercancel=()=>{drawing=false;};
      undo.onclick=()=>{if(done||!history.length)return;drawing=false;const old=history.pop();ctx.putImageData(old.pixels,0,0);hasInk=old.hasInk;};
      clear.onclick=()=>{if(done)return;save();drawing=false;ctx.save();ctx.fillStyle='white';ctx.fillRect(0,0,cfg.size,cfg.size);ctx.restore();hasInk=false;};
      return await new Promise(resolve=>{submit.onclick=()=>{
        if(done)return;if(!hasInk){alert('먼저 글자를 써 주세요.');return;}
        done=true;drawing=false;for(const element of [select,undo,clear,slider,submit])element.disabled=true;
        resolve({row:Number(select.value),png:canvas.toDataURL('image/png')});
      };});
    })()
    '''
    return eval_js(script.replace('__CONFIG__',json.dumps(dict(options=options,size=size,pen=pen))))


def reference_canvas_options(frame,columns,resolve,size):
    options=[]
    print('기준 글자 선택 목록 준비 중...',flush=True)
    for index in frame.drop_duplicates(columns['reference']).index:
        value=frame.iloc[index][columns['reference']]
        reference=load_ink(resolve(value),size)
        stream=io.BytesIO()
        Image.fromarray(np.uint8(np.clip(1-reference[0],0,1)*255)).save(stream,format='PNG')
        options.append(dict(row=int(index),label=Path(str(value).replace('\\','/')).name,
            preview='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')))
    if not options:
        raise ValueError('CSV has no reference glyphs')
    return options


def run_topology_canvas(checkpoint,csv_path,dataset_root=None,canvas_size=384,pen_width=18,
                        save_dir=None,show=True):
    checkpoint,csv_path=existing_path(checkpoint),existing_path(csv_path)
    frame=pd.read_csv(csv_path);columns=inference_columns(frame)
    resolve=Resolver(csv_path,existing_path(dataset_root) if dataset_root else csv_path.parent)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,size,epoch=load_topology(checkpoint,device)
    mode=getattr(model,'output_mode','independent')
    options=reference_canvas_options(frame,columns,resolve,size)
    print('기준 글자를 선택하고 비슷한 위치·크기로 쓴 뒤 획 추출을 누르세요.',flush=True)
    answer=canvas_prompt(options,canvas_size,pen_width)
    row=frame.iloc[int(answer['row'])]
    reference=load_ink(resolve(row[columns['reference']]),size)
    paths=path_list(row[columns['strokes']])
    if not paths:
        raise ValueError('Empty reference stroke list')
    strokes=np.concatenate([load_ink(resolve(p),size) for p in paths])
    if not (strokes.reshape(len(strokes),-1).max(1)>.5).all():
        raise ValueError('빈 기준 획입니다. 캔버스 셀을 재실행하고 다른 기준 글자를 선택하세요.')
    raw=Image.open(io.BytesIO(base64.b64decode(answer['png'].split(',',1)[1]))).convert('RGB')
    destination=Path(save_dir) if save_dir else checkpoint.parent/('canvas_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True,exist_ok=False)
    raw.save(destination/'input.png')
    image=load_ink(destination/'input.png',size)
    if image.max()<=0:
        raise ValueError('Canvas is empty')
    with torch.inference_mode():
        output=model(*[torch.from_numpy(x)[None].to(device) for x in (image,reference,strokes)])
    probabilities,masks=[output[key][0].float().cpu().numpy() for key in ('probabilities','masks')]
    np.savez_compressed(destination/'prediction.npz',image=image,reference=reference,reference_strokes=strokes,
        probabilities=probabilities,soft_masks=masks,hard_masks=export_hard_masks(image,masks,.5,mode))
    stats=ink_diagnostics(image,masks,output_mode=mode)
    preview_topology(image,reference,strokes,probabilities,masks,destination/'canvas',
        f'Canvas: {row[columns["reference"]]} | {mode}',.5,show,mode)
    report=dict(checkpoint=str(checkpoint),checkpoint_epoch=epoch,output_mode=mode,
        reference=str(row[columns['reference']]),diagnostics=stats,
        uses_input_stroke_annotations=False,npz=str(destination/'prediction.npz'))
    (destination/'canvas.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('저장:',destination,flush=True)
    return report
