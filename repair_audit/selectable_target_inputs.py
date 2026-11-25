"""Freeze four user-selected reference glyphs without any prior experiment ZIP."""
from pathlib import Path
import numpy as np

from diffusion_model.score_pilot import load_catalog_reference
from .fresh_target_inputs import canonical_sheet
from .stage1 import json_write


def prepare_selected_sheet(catalog, rows, output, *, size=576):
    rows=list(rows)
    by_row={entry['manifest_row']:entry for entry in catalog}
    if len(rows)!=4 or any(type(row) is not int or row not in by_row for row in rows):
        raise ValueError('Choose four available reference rows, in slot order')
    entries=[by_row[row] for row in rows]
    # Repeated selections are allowed: four copies of one glyph are useful too.
    conditions=[]
    for entry in entries:
        reference,strokes=load_catalog_reference(entry)
        conditions.append(dict(reference=reference,strokes=strokes))
    if len({c['reference'].shape for c in conditions})!=1:
        raise ValueError('Selected references have different resolutions')
    sheet=canonical_sheet(conditions,size)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    directory=root/'group_000_normal_control';directory.mkdir()
    sheet.save(directory/'input.png')
    for i,condition in enumerate(conditions,1):
        np.savez_compressed(directory/f'condition_{i}.npz',**condition)
    case=dict(case_id='group_000_normal_control',kind='normal_control',directory=str(directory),
        original_rows=rows,char_ids=[str(e.get('char_id','')) for e in entries],
        selected_references=entries,source='user_selected_reference_catalog')
    json_write(directory/'input_metadata.json',case)
    json_write(root/'selection.json',dict(slot_order='top-left, top-right, bottom-left, bottom-right',
        selected_rows=rows,references=entries,canvas_size=size))
    return [case]
