"""Connect the existing 2x2 generator/extractor to an aligned target-image guide.

The CLI prepares and inspects targets on CPU. The Python repair_generated_sheet
API uses an already loaded pipeline/extractor for actual img2img re-denoising.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from stroke_model.target_stroke_plan import build_target_plan
from .stage1 import read_archive, load_conditions, json_write


def extract_sheet_masks(source,conditions,extractor):
    device=next(extractor.parameters(),torch.empty(0)).device
    rgb=torch.as_tensor(np.asarray(source.convert('RGB')).copy(),device=device).permute(2,0,1)[None].float()/255
    ink=1-(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    side=source.width//2;result=[]
    extractor.eval().requires_grad_(False)
    with torch.no_grad():
        for gi,condition in enumerate(conditions):
            x0=(gi%2)*side;y0=(gi//2)*side
            ref=torch.as_tensor(condition['reference'],device=device)[None].float()
            strokes=torch.as_tensor(condition['strokes'],device=device)[None].float()
            crop=F.interpolate(ink[...,y0:y0+side,x0:x0+side],ref.shape[-2:],mode='bilinear',align_corners=False)
            result.append(extractor(crop,ref,strokes)['masks'][0].float().cpu().numpy())
    return result


def save_plan(guide,rendered,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    def image(array):return Image.fromarray(np.rint(array*255).clip(0,255).astype('uint8'))
    image(guide.source).save(out/'input.png');image(guide.target).save(out/'aligned_target.png')
    image(guide.weights).save(out/'edit_strength.png')
    preview=guide.project(torch.tensor(guide.source.copy()).permute(2,0,1)[None])
    image(preview[0].permute(1,2,0).numpy()).save(out/'cpu_blend_diagnostic_only.png')
    for gi,item in enumerate(rendered,1):np.savez_compressed(out/f'glyph_{gi}_target_strokes.npz',**item)
    np.savez_compressed(out/'guide.npz',target=guide.target,weights=guide.weights,source=guide.source)
    json_write(out/'plan.json',dict(**guide.report,guidance=guide.metadata,diffusion_run=False))
    names=[('generated input','input.png'),('width-rendered + aligned target','aligned_target.png'),
           ('score-controlled edit strength','edit_strength.png'),('CPU blend: NOT diffusion','cpu_blend_diagnostic_only.png')]
    cell=288;canvas=Image.new('RGB',(cell*4,cell+35),'white');draw=ImageDraw.Draw(canvas)
    for i,(title,name) in enumerate(names):
        draw.text((i*cell+5,9),title,fill='black')
        with Image.open(out/name) as im:canvas.paste(im.convert('RGB').resize((cell,cell)),(i*cell,35))
    canvas.save(out/'comparison.png')


def repair_generated_sheet(pipe,extractor,source,conditions,output,*,cached_masks=None,scores=None,
                           config=None,registration_config=None,steps=60,strength=.25,seed=2026,
                           prompt='',negative_prompt='',progress=None):
    """Call directly after existing generate_cases / generation of a 2x2 sheet."""
    from diffusion_model.stroke_projected_diffusion import run_projected_pair
    masks=cached_masks if cached_masks is not None else extract_sheet_masks(source,conditions,extractor)
    guide,rendered=build_target_plan(source,conditions,masks,scores=scores,config=config,
                                    registration_config=registration_config)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    save_plan(guide,rendered,root/'target_plan')
    report=run_projected_pair(pipe,source,guide,root/'reverse',steps=steps,strength=strength,seed=seed,
        prompt=prompt,negative_prompt=negative_prompt,start_from_noise=False,save_final_projection=False,progress=progress)
    json_write(root/'pipeline.json',dict(stages=['existing_2x2_generation','stroke_extraction',
        'observed_width_rendering','shared_glyph_affine','score_to_edit_plan','source_initialized_rediffusion'],
        score_source='provided' if scores is not None else 'uncalibrated_geometry_proxy_v1',
        new_training=False,source_content_used=report['source_content_used'],final_pixel_replacement=False,
        plan='target_plan/plan.json',diffusion='reverse/report.json'))
    return guide,report


def _cached_masks(z,case,source,conditions):
    prefix=case['case_id'];result=[]
    for slot,condition in enumerate(conditions,1):
        name=f'{prefix}/audit/extraction_input/slot_{slot}.npz'
        if name not in z.namelist():return None
        with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as a:
            for key in ['reference','strokes']:
                if key in a and not np.array_equal(a[key],condition[key]):raise ValueError('Cached conditions differ')
            result.append(a['extracted_masks'].copy())
    with Image.open(io.BytesIO(z.read(prefix+'/input.png'))) as original:
        if not np.array_equal(np.asarray(original.convert('RGB')),np.asarray(source)):
            raise ValueError('Cached extraction uses different source pixels')
    return result


def prepare_archive(archive,output,*,case_ids=None,observations_archive=None,scores=None,config=None):
    root=Path(output)
    cases,_=read_archive(archive,root,case_ids,include_controls=True)
    results=[]
    with zipfile.ZipFile(archive) as z:
        for case in cases:
            directory=Path(case['directory'])
            with Image.open(directory/'input.png') as image:source=image.convert('RGB').copy()
            conditions=load_conditions(directory);masks=_cached_masks(z,case,source,conditions)
            if masks is None and observations_archive:
                with zipfile.ZipFile(observations_archive) as previous:masks=_cached_masks(previous,case,source,conditions)
            if masks is None:
                results.append(dict(case_id=case['case_id'],status='deferred_no_saved_extraction',diffusion_run=False))
                continue
            try:
                given=None if scores is None else scores.get(case['case_id'])
                guide,rendered=build_target_plan(source,conditions,masks,scores=given,config=config)
                save_plan(guide,rendered,directory/'target_plan')
                results.append(dict(case_id=case['case_id'],status='target_ready_cpu_only',
                    strokes=len(guide.report['strokes']),selected_strokes=sum(s['edit_strength']>0 for s in guide.report['strokes']),
                    score_source='provided' if given is not None else 'geometry_proxy_v1',diffusion_run=False))
            except ValueError as exc:
                results.append(dict(case_id=case['case_id'],status='deferred_invalid_plan',error=str(exc),diffusion_run=False))
            json_write(root/'target_plan_summary.json',results)
    json_write(root/'target_plan_summary.json',results)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--observations-archive');parser.add_argument('--case',action='append',dest='cases')
    parser.add_argument('--scores',help='JSON: case_id -> four lists of consistency scores in [0,100], in stroke-ID order')
    args=parser.parse_args()
    given=json.loads(Path(args.scores).read_text(encoding='utf-8')) if args.scores else None
    print(json.dumps(prepare_archive(args.archive,args.output,case_ids=args.cases,
        observations_archive=args.observations_archive,scores=given),indent=2))
