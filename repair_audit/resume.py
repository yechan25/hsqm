"""Data-only archive recovery and completed-arm caching for stage 4.

Legacy archives have no mid-arm latent/controller state. Restart partial arms;
never pretend that prediction PNGs are sampler checkpoints.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image

from .stage1 import read_archive, json_write

ARMS = ('redenoise_only', 'cyclic_guided', 'boundary_guarded', 'observed_width')


def rgb_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()


def prepare_resume(archive, output, expected_hashes):
    """Copy only selected case data, never archived Python or absolute paths."""
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('runtime_manifest.json'))
        settings = manifest['settings']
        if not settings.get('boundary_config') or not settings.get('width_config'):
            raise ValueError('Upload a stage-4 result ZIP, not a stage-3 input ZIP')
        if settings.get('reference_width_probe'):
            raise ValueError('Width-probe archives are not supported by this stage-4 resumer')
        # Audit/plot orchestration may evolve. All model and loss sources must match.
        for name, expected in expected_hashes.items():
            if name.startswith(('stroke_model/', 'diffusion_model/')) and manifest['source_hashes'].get(name) != expected:
                raise ValueError('Repair implementation differs: '+name)
    cases, experiment = read_archive(archive, output, include_controls=True)
    destination = Path(output)
    with zipfile.ZipFile(archive) as z:
        for case in cases:
            prefix = case['case_id']+'/audit/'
            for entry in z.infolist():
                name = entry.filename
                if not name.startswith(prefix) or entry.is_dir():
                    continue
                relative = PurePosixPath(name)
                if ('\\' in name or ':' in name or relative.is_absolute()
                        or '..' in relative.parts or relative.suffix not in ('.png','.npz','.json','.csv')):
                    continue
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(entry))
            saved = Path(case['directory'])/'audit/input.png'
            if saved.exists() and rgb_hash(saved) != rgb_hash(Path(case['directory'])/'input.png'):
                raise ValueError('Cached audit input differs: '+case['case_id'])
        for name in ['human_review.csv','automatic_metrics.csv']:
            if name in z.namelist():
                (destination/name).write_bytes(z.read(name))
    json_write(destination/'resume_parent_manifest.json', manifest)
    return cases, experiment, manifest


def complete_arm(out, arm, timesteps):
    out = Path(out)
    png, trace = out/(arm+'.png'), out/(arm+'_trace.json')
    if not png.is_file() or not trace.is_file():
        return False
    try:
        with Image.open(png) as image:
            dimensions = image.size
            image.verify()
        if (out/'input.png').exists():
            with Image.open(out/'input.png') as source:
                if source.size != dimensions:
                    return False
        rows = json.loads(trace.read_text(encoding='utf-8'))
        return (len(rows)==len(timesteps)
                and [r['step'] for r in rows]==list(range(len(timesteps)))
                and [r['timestep'] for r in rows]==list(timesteps))
    except (ValueError, KeyError, OSError, TypeError):
        return False


def archive_partial(out, arm):
    """Keep incomplete artifacts before restarting; paths are fixed arm names."""
    out = Path(out).resolve()
    if arm not in ARMS:
        raise ValueError(arm)
    paths=[out/arm, out/(arm+'.png'), out/(arm+'_trace.json'), out/(arm+'_map_history.json')]
    existing=[p for p in paths if p.exists()]
    if not existing:
        return
    backup=out/'interrupted_attempts'/(arm+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    for path in existing:
        if not path.resolve().is_relative_to(out) or not backup.resolve().is_relative_to(out):
            raise ValueError('Unsafe backup path')
        path.rename(backup/path.name)


def load_records(out, conditions):
    records=[]
    try:
        for slot,condition in enumerate(conditions,1):
            path=Path(out)/'extraction_input'/f'slot_{slot}.npz'
            with np.load(path,allow_pickle=False) as a:
                if any(not np.array_equal(a[k],condition[k]) for k in ('reference','strokes')):
                    raise ValueError('Cached reference conditions differ')
                records.append(dict(reference=a['reference'].copy(),strokes=a['strokes'].copy(),
                    fitted=a['fitted_strokes'].copy(),extracted_masks=a['extracted_masks'].copy(),
                    fit_metadata=json.loads(path.with_name(f'slot_{slot}_fit.json').read_text(encoding='utf-8'))))
    except FileNotFoundError:
        return None
    return records


def completed_case(case, settings):
    """A whole completed case can be reused without loading its sampler state."""
    out=Path(case['directory'])/'audit'
    report=out/'report.json'
    if not report.is_file():
        return None
    data=json.loads(report.read_text(encoding='utf-8'))
    cfg=data['configuration']
    for key in ('steps','strength','seed','boundary_config','width_config'):
        if cfg.get(key)!=settings.get(key):
            raise ValueError('Cached report settings differ: '+key)
    if not all(complete_arm(out,arm,cfg['timesteps']) for arm in ARMS):
        return None
    return data['metrics']
