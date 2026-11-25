"""Freeze four user-selected reference glyphs without any prior experiment ZIP."""
from pathlib import Path
import hashlib
import json
import numpy as np

from diffusion_model.score_pilot import load_catalog_reference
from .fresh_target_inputs import canonical_sheet
from .stage1 import json_write


def load_reference_image(entry):
    """Read generation appearance only, never that font's stroke annotations."""
    if entry.get('path'):
        with np.load(entry['path'],allow_pickle=False) as data: reference=data['reference'].astype(np.float32)
    else:
        from stroke_model.convert_reference_csv import load_ink
        reference=load_ink(entry['reference'],entry['image_size'])
    if (reference.ndim!=3 or reference.shape[0]!=1 or reference.shape[1]!=reference.shape[2]
        or not np.isfinite(reference).all() or reference.min()<0 or reference.max()>1 or reference.max()<=.5):
        raise ValueError('Invalid generation reference image')
    return reference


def _signature(reference,strokes):
    digest=hashlib.sha256()
    for array in (reference,strokes):
        array=np.ascontiguousarray(array,dtype=np.float32)
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def load_prototype(bank,key):
    root=Path(bank);index=json.loads((root/'index.json').read_text(encoding='utf-8'))
    if key not in index: raise ValueError(f'No locked prototype for character: {key}')
    record=index[key]
    # Filename derives from the key, not from paths in the index.
    name=hashlib.sha256(key.encode()).hexdigest()+'.npz'
    with np.load(root/name,allow_pickle=False) as data:
        reference=data['reference'].copy();strokes=data['strokes'].copy()
    if _signature(reference,strokes)!=record['sha256']:raise ValueError('Locked prototype content changed')
    return dict(reference=reference,strokes=strokes),record


def lock_prototype(catalog,row,key,bank):
    """Pin one immutable template per character; no font-dependent lookup."""
    key=str(key).strip()
    if not key:raise ValueError('An explicit character identity is required')
    entry=next((e for e in catalog if e['manifest_row']==row),None)
    if entry is None:raise ValueError('Unknown prototype row')
    reference,strokes=load_catalog_reference(entry)
    signature=_signature(reference,strokes)
    root=Path(bank);index_path=root/'index.json'
    index=json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {}
    if key in index:
        _,record=load_prototype(root,key)
        if record['sha256']!=signature:
            raise ValueError(f'{key}: prototype already locked; cannot switch it with the input font')
        return record
    root.mkdir(parents=True,exist_ok=True)
    record=dict(character=key,manifest_row=row,sha256=signature,reference=entry)
    np.savez_compressed(root/(hashlib.sha256(key.encode()).hexdigest()+'.npz'),reference=reference,strokes=strokes)
    index[key]=record
    partial=root/'index.partial.json';json_write(partial,index);partial.replace(index_path)
    return record


def prepare_selected_sheet(catalog, rows, output, *, prototype_keys, prototype_bank, size=576):
    rows=list(rows)
    by_row={entry['manifest_row']:entry for entry in catalog}
    if len(rows)!=4 or any(type(row) is not int or row not in by_row for row in rows):
        raise ValueError('Choose four available reference rows, in slot order')
    entries=[by_row[row] for row in rows]
    keys=[str(k).strip() for k in prototype_keys]
    if len(keys)!=4 or any(not k for k in keys):raise ValueError('Need four explicit character identities')
    # Input images and repair references are deliberately independent.
    source_references=[load_reference_image(entry) for entry in entries]
    loaded=[load_prototype(prototype_bank,key) for key in keys]
    conditions=[condition for condition,record in loaded]
    records=[record for condition,record in loaded]
    if len({c['reference'].shape for c in conditions})!=1:
        raise ValueError('Selected references have different resolutions')
    sheet=canonical_sheet([dict(reference=reference) for reference in source_references],size)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    directory=root/'group_000_normal_control';directory.mkdir()
    sheet.save(directory/'input.png')
    sheet.save(directory/'generation_reference_sheet.png')
    for i,condition in enumerate(conditions,1):
        np.savez_compressed(directory/f'condition_{i}.npz',**condition)
    case=dict(case_id='group_000_normal_control',kind='normal_control',directory=str(directory),
        original_rows=rows,char_ids=[str(e.get('char_id','')) for e in entries],
        selected_references=entries,prototype_keys=keys,prototype_records=records,
        source='user_selected_images_with_locked_character_prototypes')
    json_write(directory/'input_metadata.json',case)
    json_write(root/'selection.json',dict(slot_order='top-left, top-right, bottom-left, bottom-right',
        selected_rows=rows,references=entries,canvas_size=size,prototype_keys=keys,prototypes=records,
        identity_matching='user-confirmed character keys; never inferred from font-specific row IDs',
        source_matches_prototype=[bool(np.array_equal(a,b['reference'])) for a,b in zip(source_references,conditions)]))
    return [case]
