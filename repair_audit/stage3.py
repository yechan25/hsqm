"""Expand the frozen stage-2 comparison to new reference glyphs.

This evaluates new glyphs from the existing training catalog, not unseen-font
generalization. Shape judgments remain manual; no template straightening is used.
"""
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .stage1 import file_hash, json_write, write_tables


def choose_rows(catalog, count=12, excluded_rows=(1, 2, 3, 5)):
    """Deterministically spread choices across the catalog, without quality filtering."""
    excluded_hashes = {r['reference_sha256'] for r in catalog
                       if r['manifest_row'] in excluded_rows}
    candidates = [r['manifest_row'] for r in catalog
                  if r['manifest_row'] not in excluded_rows
                  and r['reference_sha256'] not in excluded_hashes]
    if type(count) is not int or count < 4 or count % 4 or len(candidates) < count:
        raise ValueError('Need a multiple of four new glyphs (default 12), available: '+str(len(candidates)))
    return [candidates[i] for i in np.linspace(0, len(candidates)-1, count, dtype=int)]


def validate_rows(catalog, rows, excluded_rows=(1, 2, 3, 5)):
    by_row = {r['manifest_row']: r for r in catalog}
    excluded_hashes = {r['reference_sha256'] for r in catalog
                       if r['manifest_row'] in excluded_rows}
    if (not rows or len(rows) % 4 or len(set(rows)) != len(rows)
            or any(type(r) is not int or r not in by_row for r in rows)):
        raise ValueError('Choose distinct catalog row IDs, in groups of four.')
    hashes = [by_row[r]['reference_sha256'] for r in rows]
    if any(r in excluded_rows for r in rows) or excluded_hashes.intersection(hashes):
        raise ValueError('Exclude the four previously inspected glyphs (rows 1,2,3,5 and matching contents).')
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate reference contents; select different glyphs.')
    return [rows[i:i+4] for i in range(0, len(rows), 4)]


def materialize_inputs(groups, output, checkpoint_sha256):
    """Keep every generated page and every normal control; record failed generation."""
    from diffusion_model.multiglyph_pilot import compose_sheet
    from diffusion_model.score_pilot import training_catalog
    output = Path(output)
    cases, failures = [], []
    for group in groups:
        root = Path(group['root'])
        entries = training_catalog(group['manifest'])
        normal, _ = compose_sheet(entries, 576)
        report_path = root/'generation/generation_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
        failure_path = root/'generation_failure.json'
        if failure_path.is_file():
            failures.append(dict(case_id=group['group_id'], stage='generation',
                                 detail=json.loads(failure_path.read_text(encoding='utf-8'))))
        pages = [p for p in report.get('pages', []) if not p.get('control')]
        if not pages and not failure_path.is_file():
            failures.append(dict(case_id=group['group_id'], stage='generation', error='No generated pages'))
        sources = [(group['group_id']+'_normal_control', normal, 'normal_control', {})]
        for page in pages:
            identifier = f"{group['group_id']}_page_{page['page_id']:03d}"
            if page.get('safety_filtered'):
                failures.append(dict(case_id=identifier, stage='generation', error='safety_filtered', page=page))
                continue
            # The generator writes this named member itself; ignore report absolute paths.
            with Image.open(root/'generation'/f"page_{page['page_id']:03d}.png") as image:
                sources.append((identifier, image.convert('RGB').copy(), 'generated', page))
        for identifier, image, kind, page in sources:
            target = output/identifier
            target.mkdir(exist_ok=False)
            image.save(target/'input.png')
            for slot in range(1, 5):
                shutil.copy2(root/'source_conditions'/f'condition_{slot}.npz', target/f'condition_{slot}.npz')
            case = dict(case_id=identifier, directory=str(target), kind=kind,
                        prompt=report.get('prompts', {}).get('4',
                            'a typography sheet with 4 Korean Hangul glyphs arranged in '
                            '2 rows and 2 columns, black ink on plain white background, '
                            'flat front view, no grid lines, preserve the glyphs and layout of the input image'),
                        negative_prompt=report.get('negative_prompt',
                            'photograph, scenery, paper texture, shadows, colored background, grid lines, decorations'),
                        generation_seed=page.get('seed'), generation_strength=page.get('strength'),
                        checkpoint_sha256=checkpoint_sha256, source_sha256=file_hash(target/'input.png'),
                        original_rows=group['original_rows'], char_ids=group['char_ids'])
            json_write(target/'input_metadata.json', case)
            cases.append(case)
    return cases, failures


def shape_comparison(case):
    """Fixed slot crops: no OCR selection, contour smoothing or straightening."""
    out = Path(case['directory'])/'audit'
    arms = ['input', 'vae_roundtrip', 'redenoise_only', 'cyclic_guided', 'boundary_guarded']
    if not all((out/f'{arm}.png').is_file() for arm in arms):
        return
    images = []
    for arm in arms:
        with Image.open(out/f'{arm}.png') as image:
            images.append(image.convert('RGB'))
    cell, margin, heading = 192, 24, 38
    canvas = Image.new('RGB', (cell*len(arms), heading+(cell+margin)*4), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, arm in enumerate(arms):
        draw.text((col*cell+5, 12), arm, fill='black')
    for index in range(4):
        row = case.get('original_rows', ['?']*4)[index]
        y = heading+index*(cell+margin)
        draw.text((5, y+3), f'slot {index+1} / original row {row}', fill='black')
        for col, image in enumerate(images):
            half = image.width//2
            x0, y0 = (index % 2)*half, (index//2)*half
            crop = image.crop((x0, y0, x0+half, y0+half))
            canvas.paste(crop.resize((cell, cell), Image.Resampling.LANCZOS), (col*cell, y+margin))
    canvas.save(out/'shape_comparison.png')


def write_stage3_tables(root, cases, metrics, failures):
    """Extend review questions and preserve any manually filled answers on rerun."""
    root = Path(root)
    review = root/'human_review.csv'
    previous = {}
    if review.is_file():
        with review.open(encoding='utf-8-sig', newline='') as stream:
            previous = {(r['case_id'], r['slot'], r['arm']): r for r in csv.DictReader(stream)}
    write_tables(root, cases, metrics, failures)
    with review.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    fields += ['original_manifest_row', 'catalog_char_id', 'input_has_error',
               'straight_or_curve_worsened', 'endpoint_deformed', 'junction_deformed', 'residual_artifact']
    by_id = {c['case_id']: c for c in cases}
    for row in rows:
        key = (row['case_id'], row['slot'], row['arm'])
        row.update({k: v for k, v in previous.get(key, {}).items() if k in fields and v})
        case, slot = by_id[row['case_id']], int(row['slot'])-1
        row['original_manifest_row'] = case['original_rows'][slot]
        row['catalog_char_id'] = case['char_ids'][slot]
        if case['kind'] == 'normal_control':
            row['input_has_error'] = 'no'
            row['error_reduced'] = 'n/a'
    with review.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
