"""One jointly denoised glyph sheet, evaluated with fixed per-slot crops."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .score_pilot import training_catalog, scored_ink, load_catalog_reference


def compose_sheet(entries, canvas_size=576, cell_pixels=None):
    """Preserve the original full reference canvas inside each square slot."""
    side = math.isqrt(len(entries))
    if side*side != len(entries) or side < 1:
        raise ValueError('Use square glyph counts: 1, 4, 9, 16, ...')
    pixels = canvas_size//side if cell_pixels is None else cell_pixels
    if type(pixels) is not int or pixels < 16 or (pixels*side) % 8:
        raise ValueError('Sheet dimensions must be multiples of 8; cells need at least 16 pixels')
    if cell_pixels is None and canvas_size % side:
        raise ValueError('CANVAS_SIZE must be divisible by every selected grid side')
    sheet = Image.new('RGB', (pixels*side, pixels*side), 'white')
    slots = []
    for index, entry in enumerate(entries):
        reference,strokes=load_catalog_reference(entry)
        glyph = Image.fromarray(np.uint8((1-reference[0])*255)).convert('RGB')
        x, y = (index % side)*pixels, (index//side)*pixels
        bbox = [x, y, x+pixels, y+pixels]
        sheet.paste(glyph.resize((pixels, pixels), Image.Resampling.BICUBIC), (x, y))
        slots.append(dict(entry=entry, reference=reference, strokes=strokes, bbox=bbox, slot=index))
    return sheet, slots


def generate_multiglyph_pilot(manifest, manifest_rows, output, *, glyph_counts=(1, 4, 9),
                             canvas_size=576, cell_pixels=None, strengths=(.65, .85),
                             seeds=(100,), steps=30,
                             model_id='stable-diffusion-v1-5/stable-diffusion-v1-5',
                             revision=None, pipeline=None):
    """A pipeline call sees the ENTIRE sheet. Never generate tiles separately.

    Crops use input slot coordinates, never OCR success or connected components.
    Missing, displaced, merged and incorrect glyphs remain in the output set.
    """
    catalog = {item['manifest_row']: item for item in training_catalog(manifest)}
    selected = list(dict.fromkeys(manifest_rows))
    counts, strengths, seeds = list(glyph_counts), list(strengths), list(seeds)
    if not selected or any(type(i) is not int or i not in catalog for i in selected):
        raise ValueError('Select only reference rows in the training catalog')
    if (not counts or any(type(n) is not int or n < 1 or math.isqrt(n)**2 != n for n in counts)
            or len(set(counts)) != len(counts) or max(counts) > len(selected)):
        raise ValueError('Select enough training references for glyph counts 1/4/9/16/...')
    if (type(steps) is not int or steps < 2 or not strengths or not seeds
            or any(not np.isfinite(s) or not 0 < s < 1 or int(steps*s) < 1 for s in strengths)
            or any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('Invalid strength, seed or step schedule')
    entries = [catalog[row] for row in selected]
    # Validate layout and references before downloading weights/creating outputs.
    layouts = [(count, *compose_sheet(entries[:count], canvas_size, cell_pixels)) for count in counts]
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    import torch
    if pipeline is None:
        if not torch.cuda.is_available():
            raise RuntimeError('Generate on a Colab GPU')
        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
        kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
        if revision:
            kwargs['revision'] = revision
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **kwargs)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.enable_attention_slicing()
        pipeline.to('cuda')
    destination.mkdir(parents=True, exist_ok=False)
    try:
        version = importlib.metadata.version('diffusers')
    except importlib.metadata.PackageNotFoundError:
        version = None
    report = dict(experiment='joint_sheet_img2img_fixed_crop', free_text_generation=False,
        guidance_applied=False, model_id=model_id, requested_revision=revision,
        pipeline_class=type(pipeline).__name__, scheduler=dict(pipeline.scheduler.config),
        torch_version=torch.__version__, diffusers_version=version,
        training_manifest=str(Path(manifest).resolve()),
        training_manifest_sha256=hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        selected_manifest_rows=selected, glyph_counts=counts,
        pixel_budget='fixed_sheet' if cell_pixels is None else 'fixed_cell',
        requested_canvas_size=canvas_size, cell_pixels=cell_pixels,
        strengths=strengths, seeds=seeds, steps=steps,
        crop_policy='fixed input slot coordinates; no OCR filtering, recentering or repair',
        comparison_note='Counts use nested prefixes; only shared reference rows are paired. '
                        'Fixed-sheet mode also changes pixels per glyph. More errors are not guaranteed.',
        results=[], pages=[])

    def save_page(image, slots, count, *, control=False, strength=None, seed=None, flags=None):
        page_id = len(report['pages'])
        path = destination/f'page_{page_id:03d}.png'
        image.save(path)
        label = f'{count} glyphs | ' + ('input control' if control else f'strength={strength} seed={seed}')
        page = dict(page_id=page_id, input=str(path), glyph_count=count, control=control,
                    strength=strength, seed=seed, width=image.width, height=image.height, label=label,
                    safety_filtered=flags, slots=[])
        for slot in slots:
            prefix = destination/f'sample_{len(report["results"]):04d}'
            crop = image.crop(tuple(slot['bbox']))
            crop.save(str(prefix)+'_raw.png')
            ink = scored_ink(crop, slot['reference'].shape[-1])
            Image.fromarray(np.uint8(np.clip(1-ink[0],0,1)*255)).save(str(prefix)+'_scored.png')
            np.savez_compressed(str(prefix)+'.npz', image=ink,
                                reference=slot['reference'], reference_strokes=slot['strokes'])
            sample = dict(**slot['entry'], page_id=page_id, page_input=str(path), glyph_count=count,
                slot=slot['slot'], crop_bbox=slot['bbox'], control=control, strength=strength, seed=seed,
                safety_filtered=flags, label=label+f' | slot={slot["slot"]+1}',
                input=str(prefix)+'_raw.png', npz=str(prefix)+'.npz')
            report['results'].append(sample)
            page['slots'].append(dict(slot=slot['slot'], bbox=slot['bbox'],
                                      manifest_row=slot['entry']['manifest_row']))
        report['pages'].append(page)
        (destination/'generation_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    for count, initial, slots in layouts:
        side = math.isqrt(count)
        save_page(initial, slots, count, control=True)
        prompt = (f'a typography sheet with {count} Korean Hangul glyphs arranged in '
                  f'{side} rows and {side} columns, black ink on plain white background, '
                  'flat front view, no grid lines, preserve the glyphs and layout of the input image')
        report.setdefault('prompts', {})[str(count)] = prompt
        negative = 'photograph, scenery, paper texture, shadows, colored background, grid lines, decorations'
        report['negative_prompt'] = negative
        for strength in strengths:
            for seed in seeds:
                print(f'Joint generation: {count} glyphs in ONE image; strength={strength}; seed={seed}', flush=True)
                generated = pipeline(prompt=prompt, negative_prompt=negative, image=initial,
                    strength=float(strength), num_inference_steps=steps, guidance_scale=7.5,
                    generator=torch.Generator('cpu').manual_seed(seed))
                image = generated.images[0]
                if image.size != initial.size:
                    raise ValueError('Generator changed the sheet dimensions; fixed crops would be invalid')
                filtered = getattr(generated, 'nsfw_content_detected', None)
                save_page(image, slots, count, strength=float(strength), seed=seed,
                          flags=bool(filtered[0]) if filtered is not None else None)
    return report
