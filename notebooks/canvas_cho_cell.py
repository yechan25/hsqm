# 기존 점수 Colab의 준비/모듈 로드 셀 실행 후, 이 셀을 추가하세요.
REFERENCE_INPUT_FILE = "I_0_11.png"  # 이 CSV 행의 '초' 정답 이미지·정답 획 사용
CANVAS_SIZE = 384
PEN_WIDTH = 18  # 평가 해상도 128에서 약 6px
EVAL_SIZE = 128
CANVAS_SAVE_ROOT = globals().get("CANVAS_SAVE_ROOT", "/content/canvas_scores")

import base64, io, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from google.colab.output import eval_js
from stroke_model.convert_reference_csv import Resolver, existing_path, load_ink, path_list
from stroke_model.infer_reference import inference_columns, select_input_files
from _structural_indicator_runtime import IndicatorConfig, SourceConfig, score_structure_report

# 입력 사진이 아니라 이 행의 정답 이미지와 정답 획 경로만 읽습니다.
_csv = existing_path(CSV_PATH)
_frame = pd.read_csv(_csv)
_cols = inference_columns(_frame)
_row_number = select_input_files(_frame, _cols['image'], [REFERENCE_INPUT_FILE])[0]
_row = _frame.iloc[_row_number - 1]
_root = globals().get('DATASET_ROOT')
_resolve = Resolver(_csv, existing_path(_root) if _root else _csv.parent)
_target_path = _resolve(_row[_cols['reference']])
_target = load_ink(_target_path, EVAL_SIZE)
_paths = path_list(_row[_cols['strokes']])
if not _paths:
    raise ValueError('정답 획 경로가 비어 있습니다.')
_strokes = np.concatenate([load_ink(_resolve(p), EVAL_SIZE) for p in _paths])
if not (_strokes.reshape(len(_strokes), -1).max(1) > .5).all():
    raise ValueError('빈 정답 획이 있습니다.')
_y, _x = np.where(_target[0] > .1)
if not len(_x):
    raise ValueError('정답 이미지가 비어 있습니다.')

_fig, _axes = plt.subplots(1, len(_strokes)+1, figsize=(2*(len(_strokes)+1), 2.3))
for _ax, _ink, _label in zip(_axes, [_target[0], *_strokes], ['Target', *[f'Stroke {k+1}' for k in range(len(_strokes))]]):
    _ax.imshow(_ink, cmap='gray_r', vmin=0, vmax=1)
    _ax.set_title(_label); _ax.axis('off')
plt.show()
print('점선 영역에 위 정답과 비슷한 위치·크기로 초를 쓴 뒤 평가를 누르세요.')
print('다시 쓰려면 셀을 다시 실행하세요. 저장 위치:', CANVAS_SAVE_ROOT)
_canvas_config = dict(size=CANVAS_SIZE, pen=PEN_WIDTH,
    box=[float(_x.min()/EVAL_SIZE), float(_y.min()/EVAL_SIZE),
         float((_x.max()-_x.min()+1)/EVAL_SIZE), float((_y.max()-_y.min()+1)/EVAL_SIZE)])
_js = r'''
(async () => {
  const cfg = __CONFIG__;
  const wrap = document.createElement('div');
  const board = document.createElement('div');
  board.style.cssText = `position:relative;width:${cfg.size}px;max-width:100%`;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = cfg.size;
  canvas.style.cssText = 'width:100%;display:block;border:1px solid #888;touch-action:none;background:white';
  const guide = document.createElement('div');
  guide.style.cssText = `position:absolute;pointer-events:none;border:1px dashed #aaa;box-sizing:border-box;left:${100*cfg.box[0]}%;top:${100*cfg.box[1]}%;width:${100*cfg.box[2]}%;height:${100*cfg.box[3]}%`;
  board.append(canvas, guide); wrap.append(board);
  const ctx = canvas.getContext('2d', {willReadFrequently:true});
  ctx.fillStyle = 'white'; ctx.fillRect(0,0,cfg.size,cfg.size);
  ctx.strokeStyle = ctx.fillStyle = 'black'; ctx.lineCap = ctx.lineJoin = 'round';
  const controls = document.createElement('div'); controls.style.marginTop = '8px';
  const undo = document.createElement('button'); undo.textContent = '한 획 취소';
  const clear = document.createElement('button'); clear.textContent = '지우기';
  const submit = document.createElement('button'); submit.textContent = '평가';
  const label = document.createElement('span'); label.textContent = '  펜 굵기 ';
  const slider = document.createElement('input'); slider.type='range'; slider.min=4; slider.max=36; slider.value=cfg.pen;
  controls.append(undo,clear,label,slider,submit); wrap.append(controls); document.body.append(wrap);
  let drawing=false, done=false; const history=[];
  const point = e => {const r=canvas.getBoundingClientRect(); return [(e.clientX-r.left)*cfg.size/r.width,(e.clientY-r.top)*cfg.size/r.height];};
  canvas.onpointerdown = e => {
    if(done)return; e.preventDefault(); history.push(ctx.getImageData(0,0,cfg.size,cfg.size));
    drawing=true; canvas.setPointerCapture(e.pointerId);
    const [x,y]=point(e); ctx.lineWidth=Number(slider.value);
    ctx.beginPath();ctx.arc(x,y,ctx.lineWidth/2,0,2*Math.PI);ctx.fill();
    ctx.beginPath();ctx.moveTo(x,y);
  };
  canvas.onpointermove = e => {if(!drawing||done)return;e.preventDefault();const [x,y]=point(e);ctx.lineTo(x,y);ctx.stroke();};
  canvas.onpointerup = canvas.onpointercancel = () => {drawing=false;};
  undo.onclick = () => {if(!done && history.length)ctx.putImageData(history.pop(),0,0);};
  clear.onclick = () => {if(done)return;history.push(ctx.getImageData(0,0,cfg.size,cfg.size));ctx.save();ctx.fillStyle='white';ctx.fillRect(0,0,cfg.size,cfg.size);ctx.restore();};
  return await new Promise(resolve => {submit.onclick = () => {
    if(done)return;done=true;drawing=false;
    for(const element of [submit,undo,clear,slider])element.disabled=true;
    resolve(canvas.toDataURL('image/png'));
  };});
})()
'''
_data_url = eval_js(_js.replace('__CONFIG__', json.dumps(_canvas_config)))
_raw = Image.open(io.BytesIO(base64.b64decode(_data_url.split(',', 1)[1]))).convert('RGB')
_small = _raw.convert('L').resize((EVAL_SIZE, EVAL_SIZE), Image.Resampling.LANCZOS)
_ink = (1 - np.asarray(_small, dtype=np.float32)/255)[None]
if _ink.sum() < 1:
    raise ValueError('캔버스가 비어 있습니다. 셀을 다시 실행하고 글자를 써 주세요.')
_folder = Path(CANVAS_SAVE_ROOT)/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
_folder.mkdir(parents=True)
_raw.save(_folder/'canvas_cho.png')
_npz = _folder/'canvas_cho.npz'
np.savez_compressed(_npz, image=_ink, reference_image=_target, reference_strokes=_strokes)
# 추출 모델을 실행하지 않으므로 가짜 추출 오류값을 넣지 않습니다.
canvas_structure_results = dict(mode='canvas_source', target=str(_target_path), results=[
    dict(input=str(_folder/'canvas_cho.png'), npz=str(_npz),
         strokes=[dict(stroke=k+1) for k in range(len(_strokes))])])
canvas_indicator_results = score_structure_report(
    canvas_structure_results,
    config=IndicatorConfig(**globals().get('SCORE_CONFIG', {})),
    measurement_mode='source', source_config=SourceConfig(**globals().get('SOURCE_CONFIG', {})),
    save_dir=_folder/'scores', show_images=True, images_per_row=1, show_stroke_table=True,
)
