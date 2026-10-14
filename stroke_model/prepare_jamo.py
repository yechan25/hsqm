"""Build thick component supervision from FONT_unique_images/N_M_CHAR.png."""
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .convert_reference_csv import existing_path, load_ink, normalized


def scan_jamo(source):
    source = existing_path(source)
    fonts, files = {}, []
    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or not normalized(folder.name).endswith('_unique_images'):
            continue
        font = normalized(folder.name)[:-len('_unique_images')]
        if font in fonts:
            raise ValueError(f'Duplicate normalized font name: {font}')
        glyphs = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() != '.png':
                continue
            match = re.fullmatch(r'(\d+)_([012])_(.+)\.png', normalized(path.name), re.IGNORECASE)
            if not match:
                raise ValueError(f'Expected N_M_CHARACTER.png: {path}')
            class_id, slot, char = int(match[1]), int(match[2]), match[3]
            if class_id not in range(6):
                raise ValueError(f'Class N must be 0..5: {path}')
            if len(char) != 1 or not 0xAC00 <= ord(char) <= 0xD7A3:
                raise ValueError(f'Expected a composed Hangul syllable: {path}')
            entry = glyphs.setdefault((class_id, char), {'class_id': class_id, 'parts': {}})
            if slot in entry['parts']:
                raise ValueError(f'Duplicate character/component: {path}')
            entry['parts'][slot] = path
            stat = path.stat()
            files.append((str(path.relative_to(source)), stat.st_size, stat.st_mtime_ns))
        if not glyphs:
            continue
        for (class_id, char), entry in glyphs.items():
            expected = {0, 1, 2} if (ord(char) - 0xAC00) % 28 else {0, 1}
            if not set(entry['parts']) <= expected:
                raise ValueError(f'{font}/{class_id}/{char}: unexpected slots {sorted(entry["parts"])}')
            entry['missing_slots'] = sorted(expected - set(entry['parts']))
        fonts[font] = glyphs
    if len(fonts) < 2:
        raise ValueError('At least two FONT_unique_images folders are needed for cross-font pairs.')
    return source, fonts, files


def load_parts(entry, size):
    paths = [entry['parts'][slot] for slot in sorted(entry['parts'])]
    shapes = []
    for path in paths:
        with Image.open(path) as image:
            shapes.append(image.size)
    if len(set(shapes)) != 1:
        raise ValueError(f'Components must share the original character canvas: {paths}')
    masks = np.concatenate([load_ink(path, size) for path in paths])
    if not (masks.reshape(len(masks), -1).max(1) > .5).all():
        raise ValueError(f'Empty component mask: {paths}')
    return masks


def prepare_jamo(source, cache_root, reference_font=None, size=128, seed=42, rebuild=False):
    """Use one fixed reference font and split target fonts, or chars for two fonts.

    Input components must already preserve their position on a common canvas.
    Metadata fingerprints detect normal edits; rebuild=True forces regeneration.
    """
    source, fonts, files = scan_jamo(source)
    incomplete = [dict(font=font, class_id=key[0], char=key[1],
                       present_slots=sorted(entry['parts']), missing_slots=entry['missing_slots'])
                  for font, glyphs in fonts.items() for key, entry in glyphs.items()
                  if entry['missing_slots']]
    complete = {font: {key: entry for key, entry in glyphs.items() if not entry['missing_slots']}
                for font, glyphs in fonts.items()}
    inventory = {font: dict(total=len(fonts[font]), complete=len(complete[font]),
                            incomplete=len(fonts[font]) - len(complete[font])) for font in fonts}
    print('Jamo inventory (complete bundles only):', json.dumps(inventory, ensure_ascii=False), flush=True)
    if reference_font:
        reference_font = normalized(reference_font)
        if reference_font not in fonts:
            raise ValueError(f'Unknown reference font {reference_font}; available: {sorted(fonts)}')
    else:
        reference_font = sorted(complete, key=lambda name: (
            -sum(len(set(complete[name]) & set(glyphs)) for font, glyphs in complete.items() if font != name),
            -len(complete[name]), name))[0]
    fonts = complete
    references = fonts[reference_font]
    pairs = [(font, char) for font in sorted(fonts) if font != reference_font
             for char in sorted(fonts[font]) if char in references]
    skipped = [{'font': font, 'class_id': char[0], 'char': char[1]} for font in sorted(fonts) if font != reference_font
               for char in sorted(fonts[font]) if char not in references]
    target_fonts = sorted({font for font, _ in pairs})
    if not pairs:
        raise ValueError('No matching complete class/character bundles between fonts. Provide all required M masks for the same class and character in at least two fonts.')
    by_font = len(target_fonts) >= 2
    groups = target_fonts if by_font else sorted({char for _, char in pairs})
    if len(groups) < 2:
        raise ValueError('Need two target fonts, or at least two characters with two total fonts.')
    random.Random(seed).shuffle(groups)
    validation = set(groups[:max(1, round(len(groups) * .2))])
    split_note = ('Target-font holdout; reference font excluded from target samples.' if by_font else
                  'Character holdout with one target font; NOT unseen-font evaluation.')
    signature = dict(version=2, source=str(source.resolve()), files=files,
                     reference_font=reference_font, size=size, seed=seed)
    key = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    if not rebuild:
        for folder in sorted(cache_root.glob('jamo_*'), reverse=True):
            try:
                if json.loads((folder / 'cache.json').read_text(encoding='utf-8'))['key'] != key:
                    continue
                manifest = folder / 'manifest.json'
                rows = json.loads(manifest.read_text(encoding='utf-8'))
                if rows and all((manifest.parent / row['path']).is_file() for row in rows):
                    print('Reuse Jamo data:', manifest)
                    print('Reference font:', reference_font, '|', split_note)
                    return manifest
            except (OSError, ValueError, KeyError):
                continue
    folder = cache_root / ('jamo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (folder / 'samples').mkdir(parents=True, exist_ok=False)
    rows, reports = [], []
    # Retain only the last reference glyph, so a full Hangul font need not fit in RAM.
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    last_char, ref_masks = None, None
    for index, (font, char) in enumerate(tqdm(pairs, desc='Prepare Jamo', unit='glyph')):
        if char != last_char:
            ref_masks = load_parts(references[char], size)
            last_char = char
        masks = load_parts(fonts[font][char], size)
        image = masks.max(axis=0, keepdims=True)
        reference = ref_masks.max(axis=0, keepdims=True)
        owners = masks > .5
        foreground = image[0] > .5
        counts = owners.sum(0)
        known = foreground & (counts == 1)
        labels = np.full((size, size), -1, dtype=np.int64)
        labels[known] = owners.argmax(0)[known]
        if any(not (labels == k).any() for k in range(len(masks))):
            raise ValueError(f'{font}/{char}: a component has no exclusive foreground pixels; check canvas alignment.')
        path = folder / 'samples' / f'{index:06d}.npz'
        np.savez_compressed(path, image=image, reference=reference, strokes=ref_masks, labels=labels)
        group = font if by_font else f'{char[0]}:{char[1]}'
        rows.append(dict(path=str(path.resolve()), split='val' if (font if by_font else char) in validation else 'train',
                         group_id=('font:' if by_font else 'char:') + group,
                         char_id=char[1], class_id=char[0], font=font, reference_font=reference_font,
                         component_slots=sorted(fonts[font][char]['parts']), source='font_jamo_masks'))
        reports.append(dict(row=index, font=font, class_id=char[0], char=char[1], foreground_pixels=int(foreground.sum()),
                            annotation_coverage=float(known.sum() / max(foreground.sum(), 1)),
                            overlap_pixels=int((foreground & (counts > 1)).sum()), missing_pixels=0))
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(source=str(source), reference_font=reference_font, split_note=split_note,
                  inventory=inventory, skipped_incomplete=incomplete,
                  skipped_missing_reference=skipped, rows=reports,
                  composition='Pixelwise maximum of ink masks on unchanged canvas; overlaps ignored in labels.')
    (folder / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'cache.json').write_text(json.dumps({'key': key}), encoding='utf-8')
    print('Reference font:', reference_font, '|', split_note)
    print(f'Train={sum(r["split"] == "train" for r in rows)}, val={sum(r["split"] == "val" for r in rows)}, skipped={len(skipped)}')
    return manifest
