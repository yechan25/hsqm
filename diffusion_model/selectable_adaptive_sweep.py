"""Select four references, generate independent cases, fit and repair all cases.

The notebook supplies the historical adaptive runtime. This module orchestrates
experiments and records failures; it does not alter its losses or accept updates.
"""
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .score_pilot import training_catalog, load_catalog_reference
from .multiglyph_pilot import compose_sheet, generate_multiglyph_pilot
from .fitted_candidate_diffusion import image_tensor, run_pair
from stroke_model.stroke_reference_fit import fit_reference_glyph, ReferenceFitConfig
from stroke_model.adaptive_ink_constraints import AdaptiveInkConstraints


ADAPTIVE_COMMIT = 'd74e26e19d1e13360fdecde5d9d80d889344d990'


def select_entries(manifest, rows):
    rows = list(rows)
    catalog = {r['manifest_row']: r for r in training_catalog(manifest)}
    if len(rows) != 4 or len(set(rows)) != 4 or any(type(i) is not int or i not in catalog for i in rows):
        raise ValueError('Choose exactly four distinct row IDs shown in the reference catalog')
    return [catalog[i] for i in rows]


def prepare_groups(manifest, selections, output):
    """Export only selected canonical references; retain original row identities."""
    choices = [list(rows) for rows in selections]
    if not choices or len({tuple(r) for r in choices}) != len(choices):
        raise ValueError('Choose at least one group, without duplicate ordered groups')
    entries = [select_entries(manifest, rows) for rows in choices]
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    groups = []
    for i, (rows, selected) in enumerate(zip(choices, entries)):
        root = output/f'group_{i:03d}';conditions = root/'source_conditions'
        conditions.mkdir(parents=True)
        local = []
        for slot, entry in enumerate(selected):
            reference, strokes = load_catalog_reference(entry)
            name = f'condition_{slot+1}.npz'
            np.savez_compressed(conditions/name, reference=reference, strokes=strokes)
            local.append(dict(path=name, split='train', char_id=entry.get('char_id', ''),
                              original_manifest_row=rows[slot]))
        path = conditions/'conditions.json'
        path.write_text(json.dumps(local, ensure_ascii=False, indent=2), encoding='utf-8')
        if len(training_catalog(path)) != 4:
            raise ValueError('Selected rows contain identical reference/stroke contents; choose four different glyphs')
        group = dict(group_id=f'group_{i:03d}', original_rows=rows,
                     char_ids=[r.get('char_id', '') for r in selected],
                     manifest=str(path), root=str(root))
        groups.append(group)
    (output/'groups.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding='utf-8')
    return groups


def generate_cases(groups, *, strengths=(.65,), seeds=(100,),
                   steps=100, canvas_size=576, model_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
                   revision=None, pipeline=None, include_control=False):
    cases = []
    for group in groups:
        root = Path(group['root'])
        try:
            report = generate_multiglyph_pilot(group['manifest'], [1, 2, 3, 4], root/'generation',
                glyph_counts=(4,), canvas_size=canvas_size, strengths=strengths, seeds=seeds,
                steps=steps, model_id=model_id, revision=revision, pipeline=pipeline)
        except Exception as exc:
            (root/'generation_failure.json').write_text(json.dumps(dict(type=type(exc).__name__, error=str(exc))), encoding='utf-8')
            partial = root/'generation/generation_report.json'
            if not partial.is_file():continue
            report = json.loads(partial.read_text(encoding='utf-8'))
        for page in report['pages']:
            if page['control'] and not include_control:continue
            case_id = f"{group['group_id']}_page_{page['page_id']:03d}"
            cases.append(dict(case_id=case_id, group_id=group['group_id'], group=group,
                input=page['input'], control=page['control'], strength=page['strength'], seed=page['seed'],
                safety_filtered=page.get('safety_filtered'), label=page['label'],
                prompt=report.get('prompts', {}).get('4', ''), negative_prompt=report.get('negative_prompt', ''),
                output=str(root/case_id), status='generated'))
    return cases


def validate_generation_cases(cases, *, seed, strength):
    """Changing a generation seed/strength requires fresh stage-1 images."""
    if not cases or any(c['control'] or c['seed'] != seed or c['strength'] != strength for c in cases):
        raise ValueError('생성 조건이 다릅니다. 4번 셀에서 해당 seed/strength로 새로 생성한 뒤 5번을 실행하세요.')


def fit_case(case, extractor):
    out = Path(case['output']);out.mkdir(parents=True, exist_ok=False)
    with Image.open(case['input']) as im:source = im.convert('RGB').copy()
    entries = training_catalog(case['group']['manifest'])
    _, slots = compose_sheet(entries, source.width)
    if source.width != source.height:raise ValueError('Expected square joint sheet')
    device = next(extractor.parameters(), torch.tensor(0.)).device
    rgb = image_tensor(source, device)
    ink = 1-(rgb*rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    records = []
    candidate = Image.new('RGB', source.size, 'white')
    source_array = np.asarray(source, np.float32)/255
    for i, slot in enumerate(slots):
        x0, y0, x1, y1 = slot['bbox']
        reference = torch.as_tensor(slot['reference'], device=device, dtype=torch.float32)[None]
        strokes = torch.as_tensor(slot['strokes'], device=device, dtype=torch.float32)[None]
        crop = F.interpolate(ink[..., y0:y1, x0:x1], strokes.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():prediction = extractor(crop, reference, strokes)
        masks = prediction['masks'][0].cpu().numpy()
        hints = prediction.get('hints', strokes)[0].cpu().numpy()
        result = fit_reference_glyph(slot['strokes'], masks, hints=hints, glyph_reference=slot['reference'],
            config=ReferenceFitConfig(max_displacement=16., smoothness=12., shape_prior=.15, max_junction_shift=2.))
        records.append(dict(reference=slot['reference'], strokes=slot['strokes'], fitted=result['strokes'],
            extracted_masks=masks,fit_metadata=[r['metadata'] for r in result['fits']]))
        np.savez_compressed(out/f'warped_slot_{i+1}.npz', warped_strokes=result['strokes'],
                            reference_strokes=slot['strokes'], extracted_masks=masks,
                            probabilities=prediction['probabilities'][0].cpu().numpy(), model_hints=hints)
        (out/f'fit_slot_{i+1}.json').write_text(json.dumps(dict(strokes=[r['metadata'] for r in result['fits']], contacts=result['contacts']), ensure_ascii=False, indent=2), encoding='utf-8')
        pixel_crop = source_array[y0:y1, x0:x1];brightness = pixel_crop.mean(-1)
        background = np.median(pixel_crop[brightness >= np.quantile(brightness, .8)], axis=0)
        foreground = np.median(pixel_crop[brightness <= np.quantile(brightness, .05)], axis=0)
        alpha = np.asarray(Image.fromarray(result['candidate'].astype(np.float32)).resize((x1-x0, y1-y0), Image.Resampling.BICUBIC)).clip(0, 1)
        patch = background[None, None]*(1-alpha[..., None])+foreground[None, None]*alpha[..., None]
        candidate.paste(Image.fromarray(np.rint(patch*255).clip(0, 255).astype('uint8')), (x0, y0))
    source.save(out/'input.png');candidate.save(out/'fitted_candidate.png')
    (out/'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
    return source, candidate, records


def repair_cases(cases, extractor, *, checkpoint_sha256, revision, seed=2026,
                 steps=60, strength=.25, on_case=None, progress=None):
    """Run raw historical adaptive on all cases; failures remain explicit rows."""
    summaries = []
    for case in cases:
        row = {k: case[k] for k in ('case_id', 'group_id', 'control', 'strength', 'seed', 'safety_filtered', 'input', 'output')}
        try:
            source, candidate, records = fit_case(case, extractor)
            source_rgb = image_tensor(source, next(extractor.parameters(), torch.tensor(0.)).device)
            objective = AdaptiveInkConstraints(source_rgb, extractor, records,
                refresh_every=3, enable_restore=False, margin_ratio=.20)
            metadata = dict(case={k: v for k, v in case.items() if k != 'group'},
                checkpoint_sha256=checkpoint_sha256, original_manifest_rows=case['group']['original_rows'],
                prompt=case['prompt'], negative_prompt=case['negative_prompt'],
                candidate_is_ground_truth=False, adaptive_runtime_commit=ADAPTIVE_COMMIT)
            report = run_pair(source, candidate, metadata, Path(case['output'])/'reverse',
                steps=steps, strength=strength, seed=seed, inner_steps=4, step_ratio=.012,
                revision=revision, progress=progress, guidance_mode='adaptive_ink', objective_override=objective)
            if report.get('selection') is not None:raise RuntimeError('Unexpected release selection: historical runtime required')
            row.update(status='completed', guided=str(Path(case['output'])/'reverse/adaptive_guided.png'),
                baseline=str(Path(case['output'])/'reverse/redenoise_only.png'),
                candidate=str(Path(case['output'])/'fitted_candidate.png'), metrics=report['metrics'])
        except Exception as exc:
            row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            Path(case['output']).mkdir(parents=True, exist_ok=True)
            (Path(case['output'])/'failure.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
        summaries.append(row)
        summary_path = Path(case['group']['root']).parent/'all_results.json'
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
        if on_case:on_case(row)
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return summaries
