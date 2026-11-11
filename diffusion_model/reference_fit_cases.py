"""Keep every generated page as a separate reference-fit case, including failures."""
from pathlib import Path


def cases_from_generation(report, *, include_control=True):
    cases=[];seen=set()
    for page in report['pages']:
        if page['glyph_count']!=4:raise ValueError('This fit experiment requires exactly four glyphs per page')
        if page['control'] and not include_control:continue
        page_id=int(page['page_id'])
        if page_id in seen:raise ValueError('Duplicate generation page id')
        seen.add(page_id)
        if page['control']:
            suffix='reference_control'
        else:
            suffix=f"strength_{page['strength']}_seed_{page['seed']}".replace('.','p')
        cases.append(dict(case_id=f'case_{page_id:03d}_{suffix}',page_id=page_id,
            input=str(Path(page['input'])),control=bool(page['control']),
            strength=page.get('strength'),seed=page.get('seed'),
            safety_filtered=page.get('safety_filtered'),label=page['label'],
            source_type='clean_reference_control' if page['control'] else 'reference_initialized_img2img'))
    if not cases:raise ValueError('No cases in generation report')
    return cases
