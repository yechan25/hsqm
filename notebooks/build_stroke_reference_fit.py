"""Inspect reference deformation before attempting any reverse diffusion."""
import ast
import json
from pathlib import Path
from build_soft_assignment_repair import build_drive

ROOT=Path(__file__).resolve().parents[1]


def build():
    nb=json.loads(build_drive().read_text(encoding='utf-8'))
    def put(i,source):
        if nb['cells'][i]['cell_type']=='code':ast.parse(source)
        nb['cells'][i]['source']=source.strip().splitlines(keepends=True)
    put(0,'''# 획 추출 → 원본 획 변형 확인 (역확산 없음)

1 Drive·기존 오류 이미지·정확히 대응하는 best.pt → 2 네 칸의 획/모델 flow 추출 → 3 기준 획 변형·정상 후보 확인.
기존 모델의 softmax 마스크와 learned flow hints를 사용합니다. 새 모델 학습과 확산 모델 다운로드는 없습니다.
직선은 직선으로, 곡선은 기준 구조를 유지하는 부드러운 변형으로 맞춥니다. 짧은 가지·작은 분리 문양을 따라가지 않도록 주 경로를 사용합니다.
폐곡선/복잡한 분기는 모양을 잘라버리는 대신 제한된 affine으로 대체하고 경고합니다. 모든 글자를 정상 복원한다고 보장하지 않습니다.
후보는 후속 역확산의 기준 후보일 뿐 실제 역확산 출력이나 정답 이미지가 아닙니다. 이 노트북은 변형의 타당성을 먼저 확인하는 단계입니다.
Drive에는 원본 ZIP 또는 원래 실험 폴더와 기존 획 추출기가 필요합니다. 결과 ZIP을 Drive에도 저장합니다.
''')
    source=''.join(nb['cells'][1]['source'])
    a=source.index('runtime_helpers=');b=source.index('for name,code in runtime_helpers.items():',a)
    helpers={p:(ROOT/p).read_text(encoding='utf-8') for p in (
        'diffusion_model/saved_repair_demo.py','stroke_model/stroke_reference_fit.py')}
    source=source[:a]+'runtime_helpers='+repr(helpers)+'\n'+source[b:]
    source=source.replace("'diffusers==0.37.0',",'')
    source=source.replace("Path('/content/soft_assignment_runs')","Path('/content/stroke_reference_fit_runs')")
    put(1,source)
    put(2,'''from diffusion_model.score_pilot import training_catalog
from diffusion_model.multiglyph_pilot import compose_sheet
from stroke_model.infer_topology import load_topology
from torch.nn import functional as F

extractor,size,epoch=load_topology(CHECKPOINT,torch.device('cuda'))
extractor.requires_grad_(False)
_,slots=compose_sheet(training_catalog(demo['manifest']),demo['size'])
for slot,box in zip(slots,demo['boxes']):slot['bbox']=box
with Image.open(demo['source']) as image:source_array=np.asarray(image.convert('RGB'),np.float32)/255
source_tensor=torch.from_numpy(source_array.copy()).permute(2,0,1)[None].cuda()
extractions=[]
for i,slot in enumerate(slots):
    x0,y0,x1,y1=slot['bbox']
    reference=torch.as_tensor(slot['reference'],device='cuda',dtype=torch.float32)[None]
    strokes=torch.as_tensor(slot['strokes'],device='cuda',dtype=torch.float32)[None]
    if reference.shape[-2:]!=(size,size):raise ValueError('Checkpoint/reference resolution mismatch')
    crop=F.interpolate(source_tensor[...,y0:y1,x0:x1],(size,size),mode='bilinear',align_corners=False)
    ink=1-(crop*crop.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    with torch.no_grad():output=extractor(ink,reference,strokes)
    data=dict(reference=reference[0,0].cpu().numpy(),strokes=strokes[0].cpu().numpy(),
        ink=ink[0,0].cpu().numpy(),probabilities=output['probabilities'][0].cpu().numpy(),
        masks=output['masks'][0].cpu().numpy(),
        hints=output.get('hints',strokes)[0].cpu().numpy())
    if 'flow' in output:data['flow']=output['flow'][0].cpu().numpy()
    data['hint_from_model']=np.array('hints' in output)
    extractions.append(data)
    np.savez_compressed(RUN_DIR/f'extraction_slot_{i+1}.npz',**data)
    fig,axes=plt.subplots(3,len(strokes[0]),figsize=(3*len(strokes[0]),7),squeeze=False)
    for k in range(len(strokes[0])):
        for row,(key,title) in enumerate([('strokes','Original reference'),('hints','Model flow warp'),('masks','Extracted ink')]):
            axes[row,k].imshow(data[key][k],cmap='gray',vmin=0,vmax=1);axes[row,k].set_title(f'{title} {k+1}');axes[row,k].axis('off')
    fig.suptitle(f'Slot {i+1}');fig.tight_layout();fig.savefig(RUN_DIR/f'extraction_slot_{i+1}.png',dpi=120);plt.show()
extractor.cpu();gc.collect();torch.cuda.empty_cache()
print('네 칸의 추출을 저장했습니다. 3번은 CPU에서 기준 획을 변형합니다.')
''')
    put(3,'''from stroke_model.stroke_reference_fit import ReferenceFitConfig,fit_reference_glyph
from scipy.ndimage import distance_transform_edt
import pandas as pd
import zipfile

CONFIG=ReferenceFitConfig(max_displacement=16.,smoothness=12.,shape_prior=.15,max_junction_shift=2.)
RUN_GEOMETRY_CONTROL=True
results=[];controls=[];summary=[]
for i,data in enumerate(extractions):
    result=fit_reference_glyph(data['strokes'],data['masks'],hints=data['hints'],glyph_reference=data['reference'],config=CONFIG)
    results.append(result)
    # Strong reference-only correspondence control, using exactly the same ink.
    distance=np.stack([distance_transform_edt(s<.2) for s in data['strokes']])
    ownership=distance.argmin(0)
    nearest=np.stack([data['ink']*(ownership==k) for k in range(len(distance))])
    control=fit_reference_glyph(data['strokes'],nearest,hints=data['strokes'],glyph_reference=data['reference'],config=CONFIG) if RUN_GEOMETRY_CONTROL else None
    controls.append(control)
    arrays=dict(warped_strokes=result['strokes'],candidate_ink=result['candidate'],
        reference_strokes=data['strokes'],extracted_masks=data['masks'],model_hints=data['hints'])
    if control is not None:arrays['geometry_candidate_ink']=control['candidate']
    np.savez_compressed(RUN_DIR/f'warped_slot_{i+1}.npz',**arrays)
    fig,axes=plt.subplots(len(result['fits']),5,figsize=(15,2.7*len(result['fits'])),squeeze=False)
    for k,item in enumerate(result['fits']):
        observed=data['masks'][k];fitted=item['warped']
        overlap=np.stack((observed,np.minimum(observed,fitted),fitted),-1).clip(0,1)
        for ax,(image,title) in zip(axes[k],[(data['strokes'][k],'Original'),(data['hints'][k],'Learned warp'),
            (observed,'Extracted'),(fitted,'Fitted reference'),(overlap,'Red=input / blue=candidate')]):
            ax.imshow(image,cmap='gray',vmin=0,vmax=1);ax.set_title(f'{k+1}: {title}');ax.axis('off')
        if item['reference_path'] is not None:
            rp=item['reference_path'];tp=item['target_path']
            axes[k,3].plot(tp[:,0],tp[:,1],color='cyan',linewidth=.6)
        info=dict(slot=i+1,stroke=k+1,**item['metadata'])
        summary.append(info)
    fig.suptitle(f'Slot {i+1}: inspect every fitted stroke');fig.tight_layout();fig.savefig(RUN_DIR/f'fit_slot_{i+1}.png',dpi=120);plt.show()
    (RUN_DIR/f'fit_slot_{i+1}.json').write_text(json.dumps(dict(strokes=[x['metadata'] for x in result['fits']],contacts=result['contacts']),ensure_ascii=False,indent=2))

def compose_candidate(masks_by_slot):
    canvas=Image.new('RGB',(demo['size'],demo['size']),'white')
    for slot,mask in zip(slots,masks_by_slot):
        x0,y0,x1,y1=slot['bbox'];crop=source_array[y0:y1,x0:x1]
        brightness=crop.mean(-1)
        background=np.median(crop[brightness>=np.quantile(brightness,.8)],axis=0)
        foreground=np.median(crop[brightness<=np.quantile(brightness,.05)],axis=0)
        alpha=np.asarray(Image.fromarray(mask.astype(np.float32)).resize((x1-x0,y1-y0),Image.Resampling.BICUBIC)).clip(0,1)
        rgb=background[None,None]*(1-alpha[...,None])+foreground[None,None]*alpha[...,None]
        canvas.paste(Image.fromarray(np.rint(rgb*255).clip(0,255).astype('uint8')),(x0,y0))
    return canvas
candidate=compose_candidate([r['candidate'] for r in results]);candidate.save(RUN_DIR/'normal_candidate.png')
original=compose_candidate([d['reference'] for d in extractions]);original.save(RUN_DIR/'unfitted_reference.png')
hint_candidate=compose_candidate([d['hints'].max(0) for d in extractions]);hint_candidate.save(RUN_DIR/'learned_hint_candidate.png')
panels=[('Erroneous input',Image.open(demo['source']).convert('RGB')),('Unfitted full reference',original),
    ('Model hints (thin)',hint_candidate),('Fitted normal candidate',candidate)]
if RUN_GEOMETRY_CONTROL:
    geometry_candidate=compose_candidate([r['candidate'] for r in controls]);geometry_candidate.save(RUN_DIR/'geometry_candidate.png')
    panels.append(('Without learned extraction',geometry_candidate))
fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),5))
for ax,(title,image) in zip(axes,panels):ax.imshow(image);ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'candidate_comparison.png',dpi=140);plt.show()
source=np.asarray(panels[0][1],np.float32)/255;target=np.asarray(candidate,np.float32)/255
source_ink=1-source.mean(-1);target_ink=1-target.mean(-1)
erase=np.maximum(source_ink-target_ink,0);restore=np.maximum(target_ink-source_ink,0)
np.savez_compressed(RUN_DIR/'candidate_difference.npz',erase=erase,restore=restore,source_ink=source_ink,candidate_ink=target_ink)
fig,axes=plt.subplots(1,3,figsize=(15,5))
axes[0].imshow(candidate);axes[0].set_title('Candidate ONLY — not diffusion output')
axes[1].imshow(erase,cmap='magma',vmin=0,vmax=1);axes[1].set_title('Proposed erase (includes alignment error)')
axes[2].imshow(restore,cmap='magma',vmin=0,vmax=1);axes[2].set_title('Proposed restore (includes alignment error)')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(RUN_DIR/'candidate_guides.png',dpi=140);plt.show()
pd.DataFrame(summary).to_csv(RUN_DIR/'stroke_fit_diagnostics.csv',index=False)
warnings=[dict(slot=r['slot'],stroke=r['stroke'],warnings=r['warnings']) for r in summary if r.get('warnings')]
display(pd.DataFrame(summary)[['slot','stroke','method','warnings']])
print('경고:',len(warnings),'개. 후보의 획 모양/접점/굵기를 확인한 뒤에만 다음 역확산 실험에 사용하세요.')
print('이 단계는 실제 역확산을 실행하지 않습니다.')
(RUN_DIR/'runtime_helpers.json').write_text(json.dumps(runtime_helpers,ensure_ascii=False))
(RUN_DIR/'git_commit.txt').write_text(commit)
archive=RUN_DIR.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for path in RUN_DIR.rglob('*'):
        if path.is_file():z.write(path,path.relative_to(RUN_DIR))
drive_folder=HSQM/'stroke_reference_fit_results';drive_folder.mkdir(exist_ok=True)
shutil.copy2(archive,drive_folder/archive.name)
print('Drive 결과:',drive_folder/archive.name)
files.download(str(archive))
''')
    path=Path(__file__).with_name('stroke_reference_fit.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    return path


if __name__=='__main__':print(build())
