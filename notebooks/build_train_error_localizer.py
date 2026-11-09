"""Gate A training notebook using the established Drive/data preparation."""
import ast,json
from pathlib import Path
from build_structure_generalization import build as base

def build():
    nb=json.loads(base().read_text(encoding='utf-8'))
    def put(i,text):
        if nb['cells'][i]['cell_type']=='code':ast.parse(text)
        nb['cells'][i]['source']=text.strip().splitlines(keepends=True)
    put(0,"""# 새 오류 위치 탐지기 학습 — Gate A
기존 best.pt 획 추출기는 동결하고 새 탐지기를 학습합니다. 아직 자동 수정/확산 연결 단계가 아닙니다.
같은 폰트의 정상 글자에 부착 돌출/분리 잡획을 합성해 정확한 변경 위치를 정답으로 사용합니다.
정상 이미지도 오류 없음으로 학습합니다. 깨끗한 RGB는 새 모델의 입력이 아닙니다.
문자와 입력 폰트를 train/validation/test에서 분리합니다. 기존 추출기의 사전학습 중복은 배제하지 못합니다.
실제 확산 출력과 사진의 성능은 별도 검수해야 합니다.

GPU 런타임에서 1 준비 → 2 기존 체크포인트·자료 → 3 데이터/주석 확인 → 4 학습 → 5 held-out 결과·ZIP → 6 실제 단일 글자 crop 진단(선택).
기본 96문자, 10에폭, batch4는 실험 시작값이며 학습 시간/최적 학습량은 미검증입니다.
""")
    settings=''.join(nb['cells'][2]['source']).split('# Separate experimental path')[0]
    settings=settings.replace('3번 셀에서 기준 글자를 선택하세요.','3번 셀에서 합성 오류 주석을 확인하세요.')
    put(2,settings+"""
EPOCHS=10
BATCH_SIZE=4
WIDTH=32
LEARNING_RATE=2e-4
MAX_CHARACTERS=96
VARIANTS=6
TRAIN_SEED=42
# CHECKPOINT is the EXISTING stroke extractor, not the new error detector.
print('오류 탐지기만 새로 학습하며 기존 획 모델은 동결합니다.')
""")
    put(3,"""
from stroke_model.error_synthesis import build_error_sets,glyph_from_entry
from stroke_model.infer_topology import load_topology
catalog=training_catalog(TRAIN_MANIFEST)
font_root=Path('/usr/share/fonts/truetype/nanum')
def find_font(*names):return next((str(font_root/n) for n in names if (font_root/n).is_file()),None)
fonts=dict(train=[find_font('NanumGothic.ttf'),find_font('NanumGothicBold.ttf')],
           val=[find_font('NanumMyeongjo.ttf')],test=[find_font('NanumSquareR.ttf','NanumBarunGothic.ttf')])
fonts={k:[p for p in v if p] for k,v in fonts.items()}
sets,split_metadata=build_error_sets(catalog,fonts,MAX_CHARACTERS,VARIANTS,TRAIN_SEED)
extractor,size,extractor_epoch=load_topology(CHECKPOINT,torch.device('cuda'))
from datetime import datetime
RUN_DIR=HSQM/'error_localizer_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
for split,data in sets.items():print(split,'characters',len(split_metadata['characters'][split]),'samples',len(data),'fonts',fonts[split])
print('문자 정보 확인:',[(glyph_from_entry(e),e.get('reference',e.get('path'))) for e in sets['train'].entries[:8]])
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,3,figsize=(9,9))
for i in range(3):
    s=sets['train'][i]
    for ax,image,title in zip(axes[i],[s['clean'].permute(1,2,0),s['rgb'].permute(1,2,0),s['extra'][0]],['Clean (label only)','Synthetic input','Extra ink ground truth']):
        ax.imshow(image,cmap='magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
plt.tight_layout();plt.show()
print('4번 실행 전, 원래 획이 extra 정답으로 잡히지 않는지 확인하세요.')
""")
    put(4,"""
from stroke_model.train_error_localizer import fit_error_localizer
import hashlib
metadata=dict(split_metadata,extractor_checkpoint=str(CHECKPOINT),
    extractor_sha256=hashlib.sha256(Path(CHECKPOINT).read_bytes()).hexdigest(),git_commit=commit,
    data_manifest=str(TRAIN_MANIFEST))
model,history=fit_error_localizer(extractor,sets['train'],sets['val'],RUN_DIR,device='cuda',
    epochs=EPOCHS,batch_size=BATCH_SIZE,learning_rate=LEARNING_RATE,width=WIDTH,seed=TRAIN_SEED,metadata=metadata)
(RUN_DIR/'git_commit.txt').write_text(commit)
print('새 오류 탐지기:',RUN_DIR/'best.pt')
print('threshold 미보정: 아직 자동 삭제에 연결하지 마세요.')
""")
    put(5,"""
from stroke_model.train_error_localizer import evaluate_errors
metrics=evaluate_errors(model,sets['test'],'cuda',BATCH_SIZE)
(RUN_DIR/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
print('held-out test:',metrics)
fig,axes=plt.subplots(3,4,figsize=(12,9))
for i in range(3):
    s=sets['test'][i]
    with torch.no_grad():
        probability=model(s['rgb'][None].cuda(),s['reference'][None].cuda(),s['strokes'][None].cuda(),s['codepoint'][None].cuda())['extra_probability'][0,0].cpu()
    for ax,image,title in zip(axes[i],[s['clean'].permute(1,2,0),s['rgb'].permute(1,2,0),s['extra'][0],probability],['Clean (evaluation only)','Input','Ground truth','Predicted extra']):
        ax.imshow(image,cmap='magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
plt.tight_layout();fig.savefig(RUN_DIR/'heldout_examples.png',dpi=140);plt.show()
fig,ax=plt.subplots()
ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='train')
ax.plot([r['epoch'] for r in history],[r['validation']['loss'] for r in history],label='validation')
ax.legend();fig.savefig(RUN_DIR/'loss.png');plt.show()
import zipfile
from google.colab import files
archive=RUN_DIR.parent/(RUN_DIR.name+'_error_training.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in RUN_DIR.iterdir():
        if path.is_file():z.write(path,path.name)
files.download(str(archive))
""")
    nb['cells'].append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=[]))
    put(6,"""
# Optional: ONE generated character crop, not the entire 2x2 sheet.
REAL_IMAGE=None
REAL_CHAR='가'
if REAL_IMAGE is not None:
    import numpy as np
    from PIL import Image
    from diffusion_model.score_pilot import load_catalog_reference
    entry=next((e for e in catalog if glyph_from_entry(e)==REAL_CHAR),None)
    if entry is None:raise ValueError('해당 문자의 표준 획 조건이 없습니다.')
    ref,strokes=load_catalog_reference(entry)
    with Image.open(REAL_IMAGE) as im:rgb=np.asarray(im.convert('RGB').resize((size,size)),np.float32)/255
    with torch.no_grad():
        out=model(torch.from_numpy(rgb).permute(2,0,1)[None].cuda(),torch.from_numpy(ref)[None].cuda(),torch.from_numpy(strokes)[None].cuda(),torch.tensor([ord(REAL_CHAR)],device='cuda'))
    fig,axes=plt.subplots(1,2,figsize=(8,4))
    axes[0].imshow(rgb);axes[0].set_title('Generated crop')
    axes[1].imshow(out['extra_probability'][0,0].cpu(),cmap='magma',vmin=0,vmax=1);axes[1].set_title('Extra ink probability (uncalibrated)')
    for ax in axes:ax.axis('off')
    plt.show()
""")
    path=Path(__file__).with_name('train_error_localizer.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path
if __name__=='__main__':print(build())
