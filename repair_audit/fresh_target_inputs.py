"""Create repeatable whole-sheet img2img inputs from saved canonical strokes."""
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from .stage1 import load_conditions, json_write


def canonical_sheet(conditions, size):
    if len(conditions) != 4 or size % 16:
        raise ValueError('Expected four conditions and a sheet size divisible by 16')
    sheet = Image.new('RGB', (size, size), 'white')
    for i, condition in enumerate(conditions):
        reference = np.asarray(condition['reference'])[0]
        glyph = Image.fromarray(np.rint((1-reference)*255).clip(0,255).astype('uint8'))
        sheet.paste(glyph.convert('RGB').resize((size//2,size//2),Image.Resampling.BICUBIC),
                    ((i%2)*(size//2),(i//2)*(size//2)))
    return sheet


def generate_fresh_cases(pipe, seed_cases, output, *, strengths=(.65,), seeds=(100,101),
                         steps=60, cfg_scale=7.5, on_saved=None):
    """One diffusion call sees all four glyphs. Keep every output, even failures.

    Saved ZIPs supply reference conditions, not a pool of pre-repaired images.
    The original canonical sheet is also returned as a normal preservation control.
    """
    strengths, seeds = list(strengths), list(seeds)
    if (not strengths or not seeds or type(steps) is not int or steps<2
        or any(not math.isfinite(s) or not 0<s<1 or int(steps*s)<1 for s in strengths)
        or any(type(s) is not int or s<0 for s in seeds)
        or len(set(strengths))!=len(strengths) or len(set(seeds))!=len(seeds)):
        raise ValueError('Invalid or duplicate generation settings')
    controls = [c for c in seed_cases if c['kind']=='normal_control']
    if not controls: raise ValueError('No canonical control groups in the source ZIP')
    root = Path(output); root.mkdir(parents=True,exist_ok=False)
    cases=[]
    prompt = ('a typography sheet with 4 Korean Hangul glyphs arranged in 2 rows and 2 columns, '
              'black ink on plain white background, flat front view, no grid lines, '
              'preserve the glyphs and layout of the input image')
    negative = 'photograph, scenery, paper texture, shadows, colored background, grid lines, decorations'
    report = dict(kind='canonical_joint_sheet_img2img',strengths=strengths,seeds=seeds,
        steps=steps,cfg_scale=cfg_scale,prompt=prompt,negative_prompt=negative,
        filtering='none; wrong, displaced and empty glyphs are retained',cases=cases)
    def save(image,reference_case,case_id,**generation):
        directory=root/case_id;directory.mkdir()
        image.save(directory/'input.png')
        for slot in range(1,5):
            shutil.copy2(Path(reference_case['directory'])/f'condition_{slot}.npz',directory/f'condition_{slot}.npz')
        item=dict(reference_case,case_id=case_id,directory=str(directory),
                  kind='normal_control' if generation['control'] else 'generated',
                  prompt=prompt,negative_prompt=negative,generation=generation,
                  source_reference_case=reference_case['case_id'],reused_stage1_original=False)
        json_write(directory/'input_metadata.json',item)
        cases.append(item);json_write(root/'generation_report.json',report)
        if on_saved: on_saved(item,cases)
    for control in controls:
        with Image.open(Path(control['directory'])/'input.png') as old: size=old.width
        initial=canonical_sheet(load_conditions(control['directory']),size)
        group=control['case_id'].removesuffix('_normal_control')
        save(initial,control,group+'_normal_control',control=True,strength=None,seed=None)
        for si,strength in enumerate(strengths):
            for ri,seed in enumerate(seeds):
                case_id=f'{group}_page_{100+si*len(seeds)+ri:03d}'
                print('Joint 2x2 generation:',case_id,'strength=',strength,'seed=',seed,flush=True)
                generated=pipe(prompt=prompt,negative_prompt=negative,image=initial,
                    strength=float(strength),num_inference_steps=steps,guidance_scale=cfg_scale,
                    generator=torch.Generator('cpu').manual_seed(seed))
                image=generated.images[0]
                if image.size!=initial.size:raise ValueError('Generation changed fixed sheet dimensions')
                flags=getattr(generated,'nsfw_content_detected',None)
                save(image,control,case_id,control=False,strength=strength,seed=seed,
                     safety_filtered=bool(flags[0]) if flags is not None else None)
    return cases
