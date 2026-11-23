"""CPU-only target-stroke projection inspection from an existing audit ZIP.

No models are loaded and no diffusion is run. This is a geometry/constraint
experiment, not evidence that generated characters are correct.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch

from stroke_model.stroke_projection import StrokeProjection, StrokeProjectionConfig
from .stage1 import json_write


def _image(z,name):
    with Image.open(io.BytesIO(z.read(name))) as image:return image.convert('RGB').copy()


def _tensor(image):
    return torch.tensor(np.asarray(image).copy()).permute(2,0,1)[None].float()/255


def _save(rgb,path):
    Image.fromarray(np.rint(rgb[0].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8')).save(path)


def observed_widths(strokes, observed):
    result=[];notes=[]
    for k,stroke in enumerate(strokes):
        path=skeletonize(stroke>.35)
        if not path.any():raise ValueError('Empty target stroke')
        width=float(np.median(distance_transform_edt(stroke>.35)[path]))
        reason='target_template_width'
        if observed is not None:
            mask=observed[k]>.35
            coverage=float(mask[path].mean())
            samples=skeletonize(mask)&(distance_transform_edt(~path)<=8)
            if coverage>=.6 and samples.sum()>=12:
                width=float(np.median(distance_transform_edt(mask)[samples]))
                reason='observed_center_width'
        result.append(width);notes.append(dict(stroke=k+1,half_width=width,source=reason))
    return result,notes


def run(archive,output,case_ids=None,observations_archive=None,allow_template_widths=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=StrokeProjectionConfig();summaries=[]
    with zipfile.ZipFile(archive) as z, ExitStack() as stack:
        extra=stack.enter_context(zipfile.ZipFile(observations_archive)) if observations_archive else None
        metadata=json.loads(z.read('archive_metadata.json'))
        cases=metadata['selected_cases']
        if case_ids:
            available={c['case_id'] for c in cases}
            if set(case_ids)-available:raise ValueError('Unknown case IDs')
            cases=[c for c in cases if c['case_id'] in case_ids]
        names=set(z.namelist())
        for case in cases:
            identifier=case['case_id']
            if not re.fullmatch(r'group_[0-9]+_(page_[0-9]+|normal_control)',identifier):
                raise ValueError('Invalid case identifier')
            source=_image(z,identifier+'/input.png');width,height=source.size
            if width!=height or width%2:raise ValueError('Expected a four-cell square sheet')
            side=width//2;glyphs=[];width_notes=[];template=Image.new('RGB',source.size,'white');missing_observations=[]
            for slot in range(1,5):
                with np.load(io.BytesIO(z.read(f'{identifier}/condition_{slot}.npz')),allow_pickle=False) as a:
                    strokes=a['strokes'].copy()
                observed=None
                member=f'{identifier}/audit/extraction_input/slot_{slot}.npz'
                observation_zip=z if member in names else extra
                if observation_zip is not None and member in observation_zip.namelist():
                    saved_source=_image(observation_zip,identifier+'/input.png')
                    if not np.array_equal(np.asarray(saved_source),np.asarray(source)):
                        raise ValueError('Observation archive uses different source pixels: '+identifier)
                    with np.load(io.BytesIO(observation_zip.read(member)),allow_pickle=False) as a:
                        observed=a['extracted_masks'].copy()
                    if observed.shape!=strokes.shape:raise ValueError('Observed stroke count/shape differs')
                else:
                    missing_observations.append(slot)
                widths,notes=observed_widths(strokes,observed)
                x0=((slot-1)%2)*side;y0=((slot-1)//2)*side
                glyphs.append(dict(strokes=strokes,bbox=(x0,y0,x0+side,y0+side),half_widths=widths))
                width_notes.append(dict(slot=slot,strokes=notes))
                patch=Image.fromarray(np.rint((1-strokes.max(0))*255).clip(0,255).astype('uint8')).convert('RGB')
                template.paste(patch.resize((side,side)),(x0,y0))
            target=output/identifier;target.mkdir()
            source.save(target/'input.png');template.save(target/'target_stroke_template.png')
            if missing_observations and not allow_template_widths:
                report=dict(case_id=identifier,kind=case['kind'],stroke_count=sum(len(g['strokes']) for g in glyphs),
                    status='deferred_missing_observed_widths',slots=missing_observations,diffusion_run=False,
                    reason='Provide same-input saved extraction data, or explicitly allow template-width stress test.')
                json_write(target/'report.json',report);summaries.append(report)
                continue
            constraints=StrokeProjection.from_glyphs(glyphs,source.size,cfg)
            overlay=np.asarray(source).copy()
            overlay[constraints.free]=[210,220,255]
            overlay[constraints.foreground]=[0,140,80]
            overlay[constraints.background]=[245,245,245]
            Image.fromarray(overlay).save(target/'constraint_regions.png')
            # Retain individual stroke identities in the visual evidence too.
            columns=max(sum(i['glyph']==g for i in constraints.identities) for g in range(1,5))
            tile=128;grid=Image.new('RGB',(columns*tile,4*(tile+24)),'white');pen=ImageDraw.Draw(grid)
            for k,identity in enumerate(constraints.identities):
                gi=identity['glyph']-1;si=identity['stroke']-1
                x0=(gi%2)*side;y0=(gi//2)*side
                base=np.full((side,side,3),255,np.uint8)
                permit=constraints.allowed[k,y0:y0+side,x0:x0+side]
                core=constraints.cores[k,y0:y0+side,x0:x0+side]
                base[permit]=[210,220,255];base[core]=[0,140,80]
                gx=si*tile;gy=gi*(tile+24)
                pen.text((gx+3,gy+5),f'g{gi+1} stroke {si+1}',fill='black')
                grid.paste(Image.fromarray(base).resize((tile,tile)),(gx,gy+24))
            grid.save(target/'per_stroke_constraints.png')
            np.savez_compressed(target/'constraints.npz',cores=constraints.cores,allowed=constraints.allowed,roi=constraints.roi)
            candidates=[('input',source)]
            for arm in ['redenoise_only','boundary_guarded','observed_width']:
                name=f'{identifier}/audit/{arm}.png'
                if name in names:candidates.append((arm,_image(z,name)))
            results=[]
            for arm,image in candidates:
                original=_tensor(image);projected=constraints.project(original)
                _save(projected,target/(arm+'_pixel_projected.png'))
                # Same masks collapsed to one channel must give identical RGB.
                union=StrokeProjection(constraints.foreground[None],constraints.allowed.any(0)[None],
                    constraints.roi,[dict(glyph=0,stroke=0)],cfg)
                results.append(dict(arm=arm,before=constraints.diagnostics(original),after=constraints.diagnostics(projected),
                    changed_pixels=int((projected-original).abs().amax(1).gt(1e-6).sum()),
                    union_equivalent=bool(torch.equal(projected,union.project(original)))))
            panels=[('input','input.png'),('target strokes (not GT style)','target_stroke_template.png'),
                    ('core / free boundary / outside','constraint_regions.png'),('CPU pixel projection only','input_pixel_projected.png')]
            if any(a=='observed_width' for a,_ in candidates):
                saved=_image(z,f'{identifier}/audit/observed_width.png');saved.save(target/'saved_diffusion.png')
                panels += [('saved diffusion output','saved_diffusion.png'),('saved output + CPU projection','observed_width_pixel_projected.png')]
            cell=240;canvas=Image.new('RGB',(cell*len(panels),cell+36),'white');draw=ImageDraw.Draw(canvas)
            for i,(label,path) in enumerate(panels):
                draw.text((i*cell+4,10),label,fill='black')
                with Image.open(target/path) as image:canvas.paste(image.resize((cell,cell)),(i*cell,36))
            canvas.save(target/'comparison.png')
            report=dict(case_id=identifier,kind=case['kind'],config=asdict(cfg),width_estimation=width_notes,
                status='cpu_projection_only',template_width_stress_test=bool(missing_observations),
                stroke_count=len(constraints.cores),identities=constraints.identities,results=results,
                target_geometry='Canonical target strokes placed by bbox; not fitted to erroneous input geometry',
                diffusion_run=False,additional_training=False,
                limitations=['Target strokes are assumed correct; extractor does not verify correctness.',
                    'CPU projection may overwrite legitimate input geometry/style; constraints passing is not text accuracy.',
                    'Identical aggregated core/allowed masks give identical RGB projection; per-stroke benefit is not established.',
                    '8-bit PNG saving can shift ink thresholds by up to 1/255.'])
            json_write(target/'report.json',report);summaries.append(report)
    json_write(output/'summary.json',summaries)
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--case',action='append',dest='cases')
    parser.add_argument('--observations-archive',help='Optional older ZIP with extraction masks for identical source pixels')
    parser.add_argument('--allow-template-widths',action='store_true',help='Explicit stress test; may thin healthy strokes')
    args=parser.parse_args()
    result=run(args.archive,args.output,args.cases,args.observations_archive,args.allow_template_widths)
    print(json.dumps([dict(case=r['case_id'],strokes=r['stroke_count'],status=r['status'],diffusion_run=False) for r in result],indent=2))
