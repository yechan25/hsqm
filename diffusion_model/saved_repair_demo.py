"""Load an actual saved baseline and its recorded reference conditions, not a new generation."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
from PIL import Image


def pack_saved_generation(report_path, output):
    """Package an existing Drive experiment without its original dataset/CSV."""
    from stroke_model.convert_reference_csv import existing_path
    report_path=existing_path(report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    pages=[p for p in report['pages'] if p['method']=='baseline']
    rows=[r for r in report['results'] if r['method']=='reference']
    if len(pages)!=1 or len(rows)!=4:raise ValueError('Expected one baseline and four recorded reference slots')
    def locate(recorded):
        basename=PurePosixPath(str(recorded).replace('\\','/')).name
        local=report_path.parent/basename
        return existing_path(local if local.is_file() else recorded)
    source=locate(pages[0]['input'])
    references=[locate(r['npz']) for r in rows]
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(source,'generation/'+source.name)
        for path in references:z.write(path,'generation/'+path.name)
        z.writestr('generation/generation_report.json',json.dumps(report,ensure_ascii=False))
    return output


def prepare_saved_demo(archive, output):
    destination=Path(output)
    if destination.exists():raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        reports=[n for n in names if n.endswith('generation_report.json')]
        if len(reports)!=1:raise ValueError('ZIP must contain exactly one generation_report.json')
        report=json.loads(z.read(reports[0]))
        def member(recorded):
            basename=PurePosixPath(str(recorded).replace('\\','/')).name
            matches=[n for n in names if PurePosixPath(n).name==basename]
            if len(matches)!=1:raise ValueError(f'Missing/ambiguous ZIP file: {basename}')
            return matches[0]
        baseline=[p for p in report['pages'] if p['method']=='baseline']
        if len(baseline)!=1:raise ValueError('A single recorded baseline page is required')
        source_member=member(baseline[0]['input'])
        source_bytes=z.read(source_member)
        with Image.open(io.BytesIO(source_bytes)) as image:
            width,height=image.size
        if width!=height or width%8:raise ValueError('Expected a square saved sheet, size divisible by 8')
        # Reference arrays are metadata conditions AFTER generation, never the repair input.
        templates=[];boxes=[]
        for index in range(4):
            candidates=[r for r in report['results'] if r['method']=='reference' and r['slot']==index]
            if len(candidates)!=1:raise ValueError('Four reference slots required')
            row=candidates[0];boxes.append(row['crop_bbox'])
            try:npz_member=member(row['npz'])
            except ValueError:npz_member=member(f'reference_slot_{index+1}.npz')
            with np.load(io.BytesIO(z.read(npz_member)),allow_pickle=False) as data:
                ref=data['reference'].astype(np.float32)
                strokes=data['reference_strokes'].astype(np.float32)
            if ref.ndim!=3 or strokes.ndim!=3 or ref.shape[0]!=1 or ref.shape[1:]!=strokes.shape[1:]:
                raise ValueError('Malformed saved reference conditions')
            templates.append((ref,strokes))
    destination.mkdir(parents=True)
    source=destination/'saved_error_input.png';source.write_bytes(source_bytes)
    manifest=[]
    for i,(reference,strokes) in enumerate(templates):
        name=f'condition_{i+1}.npz'
        np.savez_compressed(destination/name,reference=reference,strokes=strokes)
        manifest.append(dict(path=name,split='train',char_id=''))
    path=destination/'conditions.json';path.write_text(json.dumps(manifest),encoding='utf-8')
    provenance=dict(source_archive=str(archive),source_member=source_member,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_initialization=report.get('init_mode','not_recorded_in_legacy_report'),
        source_prompt=report.get('prompt'),source_report=report,
        demonstration='reference_conditioned_repair_of_saved_baseline',
        fresh_text_to_image=False,reference_encoded_as_repair_input=False)
    (destination/'source_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return dict(source=str(source),manifest=str(path),rows=[1,2,3,4],boxes=boxes,
        size=width,checkpoint=report['checkpoint'],checkpoint_sha256=report.get('checkpoint_sha256'),
        provenance=provenance)
