"""Pretrained img2img outputs on training references, then unchanged source scores.

This is a domain-transfer pilot, not text-to-image accuracy or guidance training.
"""
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image


def training_catalog(manifest):
    """Only explicit train rows; never infer characters from numeric image IDs."""
    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('Expected the stroke-training manifest JSON list')
    groups = {}
    catalog, seen = [], set()
    for index, row in enumerate(rows):
        split = row.get('split')
        if split not in ('train', 'val', 'test'):
            raise ValueError('Every manifest row needs train/val/test split')
        if row.get('group_id'):
            previous = groups.setdefault(row['group_id'], split)
            if previous != split:
                raise ValueError('Training manifest contains group leakage')
        if split != 'train':
            continue
        path = (manifest.parent / row['path']).resolve()
        with np.load(path, allow_pickle=False) as data:
            reference, strokes = [data[key].astype(np.float32) for key in ('reference', 'strokes')]
        if (reference.ndim != 3 or reference.shape[0] != 1 or reference.shape[1] != reference.shape[2]
                or strokes.ndim != 3 or strokes.shape[1:] != reference.shape[1:] or not len(strokes)):
            raise ValueError(f'Invalid reference/strokes shapes: {path}')
        for array in (reference, strokes):
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError(f'Expected finite absolute ink in [0,1]: {path}')
        if reference.max() <= .5 or (strokes.max(axis=(1, 2)) <= .5).any():
            raise ValueError(f'Empty reference or stroke: {path}')
        digest = hashlib.sha256(reference.tobytes() + strokes.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        catalog.append(dict(manifest_row=index + 1, path=str(path),
                            char_id=str(row.get('char_id', '')), reference_sha256=digest))
    if not catalog:
        raise ValueError('No train references in this manifest')
    return catalog


def scored_ink(image, size):
    """Fixed black-on-white luminance conversion; no polarity guesses or cropping."""
    rgba = image.convert('RGBA')
    white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(white, rgba).convert('L')
    gray = gray.resize((size, size), Image.Resampling.BILINEAR)
    return (1 - np.asarray(gray, dtype=np.float32) / 255)[None].copy()


def generate_pilot(manifest, manifest_rows, output, *, strengths=(.25, .45, .65),
                   seeds=(100, 101), steps=30,
                   model_id='stable-diffusion-v1-5/stable-diffusion-v1-5', revision=None,
                   pipeline=None):
    """Save every output, including wrong letters, noise and safety-filter flags.

    pipeline injection is for orchestration tests, never evidence of SD quality.
    The caller must supply the manifest used for the selected stroke checkpoint.
    """
    catalog = {entry['manifest_row']: entry for entry in training_catalog(manifest)}
    selected = list(dict.fromkeys(manifest_rows))
    if not selected or any(type(i) is not int or i not in catalog for i in selected):
        raise ValueError('Select only manifest row numbers shown in the train catalog')
    strengths, seeds = list(strengths), list(seeds)
    if (type(steps) is not int or steps < 2 or not strengths or not seeds
            or any(not np.isfinite(s) or not 0 < s < 1 or int(steps*s) < 1 for s in strengths)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('Need valid strengths, integer seeds and nonempty schedules')
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    import torch
    try:
        diffusers_version = importlib.metadata.version('diffusers')
    except importlib.metadata.PackageNotFoundError:
        diffusers_version = None
    if pipeline is None:
        if not torch.cuda.is_available():
            raise RuntimeError('Run generation on a Colab GPU; local CPU generation is not started')
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
        if revision:
            kwargs['revision'] = revision
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **kwargs)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_attention_slicing()
        pipeline.to('cuda')
    destination.mkdir(parents=True, exist_ok=False)
    prompt = ('a single isolated Korean Hangul glyph, black ink, plain white background, '
              'flat front view, clean printed typography')
    negative = 'photograph, scene, paper texture, shadows, colored background, decorations'
    report = dict(experiment='pretrained_img2img_domain_pilot', free_text_generation=False,
                  guidance_applied=False, model_id=model_id, requested_revision=revision,
                  pipeline_class=type(pipeline).__name__,
                  resolved_commit=getattr(pipeline, 'config', {}).get('_commit_hash'),
                  scheduler=dict(pipeline.scheduler.config),
                  torch_version=torch.__version__, diffusers_version=diffusers_version,
                  training_manifest=str(Path(manifest).resolve()),
                  training_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                  training_manifest_checkpoint_link='caller must use the actual checkpoint training manifest',
                  prompt=prompt, negative_prompt=negative, steps=steps, strengths=strengths,
                  seeds=seeds, preprocessing='fixed 1-luminance; bilinear resize; no crop/threshold/repair',
                  results=[])

    def save(image, entry, reference, strokes, label, **metadata):
        prefix = destination / f'sample_{len(report["results"]):04d}'
        image.save(str(prefix) + '_raw.png')
        ink = scored_ink(image, reference.shape[-1])
        Image.fromarray(np.uint8(np.clip(1-ink[0], 0, 1)*255)).save(str(prefix)+'_scored.png')
        np.savez_compressed(str(prefix)+'.npz', image=ink, reference=reference, reference_strokes=strokes)
        report['results'].append(dict(**entry, label=label, input=str(prefix)+'_raw.png',
                                     npz=str(prefix)+'.npz', **metadata))
        (destination/'generation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    for row in selected:
        entry = catalog[row]
        with np.load(entry['path'], allow_pickle=False) as data:
            reference, strokes = [data[k].astype(np.float32) for k in ('reference', 'strokes')]
        initial = Image.fromarray(np.uint8((1-reference[0])*255)).convert('RGB')
        save(initial, entry, reference, strokes, 'reference_control', control=True)
        initial = initial.resize((512, 512), Image.Resampling.BICUBIC)
        initial.save(destination/f'row_{row}_initial_512.png')
        for strength in strengths:
            for seed in seeds:
                print(f'Generate train row={row}, strength={strength}, seed={seed}', flush=True)
                result = pipeline(prompt=prompt, negative_prompt=negative, image=initial,
                                  strength=float(strength), num_inference_steps=steps,
                                  guidance_scale=7.5, generator=torch.Generator('cpu').manual_seed(seed))
                flags = getattr(result, 'nsfw_content_detected', None)
                save(result.images[0], entry, reference, strokes, f'strength={strength} seed={seed}',
                     control=False, strength=float(strength), seed=seed,
                     safety_filtered=bool(flags[0]) if flags is not None else None)
    return report


def score_pilot(generation_report, checkpoint, *, score_config=None, source_config=None, show=True):
    """Load the frozen existing extractor only AFTER the generator is released."""
    import torch
    from stroke_model.infer_topology import load_topology
    from stroke_model.topology_scoring import score_topology_results
    report = json.loads(Path(generation_report).read_text(encoding='utf-8'))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, size, epoch = load_topology(checkpoint, device)
    extraction = dict(checkpoint=str(checkpoint), checkpoint_epoch=epoch,
                      output_mode=getattr(model, 'output_mode', 'independent'), results=[])
    root = Path(generation_report).parent / ('extraction_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(exist_ok=False)
    for index, sample in enumerate(report['results']):
        with np.load(sample['npz'], allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in ('image', 'reference', 'reference_strokes')}
        if arrays['reference'].shape != (1, size, size):
            raise ValueError('Training reference resolution differs from checkpoint; do not rescale strokes')
        with torch.inference_mode():
            predicted = model(*[torch.from_numpy(arrays[k])[None].to(device)
                                for k in ('image', 'reference', 'reference_strokes')])
        arrays.update({k: predicted[v][0].float().cpu().numpy()
                       for k, v in (('soft_masks', 'masks'), ('probabilities', 'probabilities'))})
        path = root/f'sample_{index:04d}.npz'
        np.savez_compressed(path, **arrays)
        extraction['results'].append({**sample, 'npz': str(path)})
    del model
    result = score_topology_results(extraction, score_config=score_config, source_config=source_config,
                                   save_dir=root/'scores', show=show, export_debug=True)
    import pandas as pd
    rows = []
    for sample, measured in zip(extraction['results'], result['indicator']['results']):
        rows.append({key: sample.get(key) for key in
                     ('manifest_row', 'char_id', 'label', 'control', 'strength', 'seed', 'safety_filtered')}
                    | {key: measured[key] for key in ('structural_score', 'missing_score_cap',
                       'split_score_cap', 'junction_score_cap', 'extra_score_cap')})
    pd.DataFrame(rows).to_csv(root/'pilot_scores.csv', index=False, encoding='utf-8-sig')
    import zipfile
    with zipfile.ZipFile(result['debug_zip'], 'a', zipfile.ZIP_DEFLATED) as archive:
        archive.write(generation_report, 'generation_report.json')
        archive.write(root/'pilot_scores.csv', 'pilot_scores.csv')
        for index, sample in enumerate(report['results']):
            raw = Path(sample['input'])
            archive.write(raw, f'generated_originals/{index:04d}.png')
    return result, pd.DataFrame(rows)
