"""Compose training samples from class-shared thick jamo source images.

The character in N_M_CHARACTER.png identifies only the selected M component.
Generate on local disk; persist one ZIP on Drive instead of thousands of files.
"""
import hashlib
import itertools
import json
import random
import re
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .convert_reference_csv import existing_path, normalized

VOWELS = ((0, 1, 2, 3, 4, 5, 6, 7, 20), (8, 12, 13, 17, 18),
          (9, 10, 11, 14, 15, 16, 19))
VERSION = 4


def decompose(char):
    if len(char) != 1 or not 0xAC00 <= ord(char) <= 0xD7A3:
        raise ValueError(f'Expected a composed Hangul syllable: {char}')
    code = ord(char) - 0xAC00
    return code // 588, (code // 28) % 21, code % 28


def compose(parts):
    initial, vowel, final = parts
    return chr(0xAC00 + (initial * 21 + vowel) * 28 + final)


def scan_jamo(source, minimum_fonts=2):
    """Index by font -> class -> slot -> semantic jamo index -> candidate paths."""
    source = existing_path(source)
    fonts = {}
    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or not normalized(folder.name).endswith('_unique_images'):
            continue
        font = normalized(folder.name)[:-len('_unique_images')]
        if font in fonts:
            raise ValueError(f'Duplicate normalized font name: {font}')
        classes = {}
        for path in sorted(folder.iterdir(), key=lambda p: normalized(p.name)):
            if path.suffix.lower() != '.png':
                continue
            match = re.fullmatch(r'(\d+)_([012])_(.+)\.png', normalized(path.name), re.IGNORECASE)
            if not match:
                raise ValueError(f'Expected N_M_CHARACTER.png: {path}')
            class_id, slot, char = int(match[1]), int(match[2]), match[3]
            if class_id not in range(6):
                raise ValueError(f'Class N must be 0..5: {path}')
            initial, vowel, final = decompose(char)
            expected_class = next(i for i, values in enumerate(VOWELS) if vowel in values)
            expected_class += 3 if final else 0
            if class_id != expected_class or (slot == 2 and final == 0):
                raise ValueError(f'Class/slot disagrees with Hangul decomposition: {path}')
            semantic = (initial, vowel, final)[slot]
            bank = classes.setdefault(class_id, {})
            bank.setdefault(slot, {}).setdefault(semantic, []).append(path)
        if classes:
            fonts[font] = classes
    if len(fonts) < minimum_fonts:
        raise ValueError(f'At least {minimum_fonts} FONT_unique_images folders are needed.')
    return source, fonts


def shared_options(reference, target, class_id):
    count = 2 if class_id < 3 else 3
    options = []
    for slot in range(count):
        available = sorted(set(reference.get(class_id, {}).get(slot, {})) &
                           set(target.get(class_id, {}).get(slot, {})))
        if not available:
            return []
        options.append(available)
    if count == 2:
        options.append([0])
    return options


def transform_mask(mask, rng, scale_range=(.8, 1.), rotation=5., shift_fraction=20/256):
    """Independent target-jamo scale/rotation/shift on the original canvas."""
    size = mask.shape[0]
    scale = rng.uniform(*scale_range)
    angle = rng.uniform(-rotation, rotation)
    shift = (rng.randint(-round(size * shift_fraction), round(size * shift_fraction)),
             rng.randint(-round(size * shift_fraction), round(size * shift_fraction)))
    image = Image.fromarray(mask.astype(np.float32))
    side = max(1, round(size * scale))
    image = image.resize((side, side), Image.Resampling.BILINEAR)
    canvas = Image.new('F', (size, size), 0.)
    canvas.paste(image, ((size - side) // 2, (size - side) // 2))
    canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR,
                           translate=shift, fillcolor=0.)
    return np.clip(np.asarray(canvas), 0., 1.).copy()


def read_components(paths, size=128, workers=8):
    """Read each original PNG once and retain both its mask and canvas size."""
    def read(path):
        with Image.open(path) as image:
            shape = image.size
            rgba = ImageOps.exif_transpose(image).convert('RGBA')
            white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            gray = Image.alpha_composite(white, rgba).convert('L')
            mask = np.asarray(gray.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        if mask.mean() > .5:
            mask = 1 - mask
        if mask.max() <= .5:
            raise ValueError(f'Empty component: {path}')
        return path, (mask, shape)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(tqdm(pool.map(read, paths), total=len(paths), desc='Read source PNGs', unit='file'))


def make_labels(masks, max_overlap):
    owners = masks > .5
    counts = owners.sum(0)
    # Any pairwise overlap, including overlap of only two of three jamo.
    overlap = float((counts > 1).mean())
    labels = np.full(counts.shape, -1, dtype=np.int64)
    known = counts == 1
    labels[known] = owners.argmax(0)[known]
    valid = overlap <= max_overlap and all((labels == k).any() for k in range(len(masks)))
    return labels, overlap, valid


def _local_complete(folder):
    try:
        metadata = json.loads((folder / 'complete.json').read_text(encoding='utf-8'))
        count = sum(1 for p in (folder / 'samples').iterdir() if p.suffix == '.npz')
        return count == metadata['samples'] and (folder / 'manifest.json').is_file()
    except (OSError, ValueError, KeyError):
        return False


def _unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Invalid archive member path')
        bundle.extractall(destination)


def prepare_jamo(source, cache_root, reference_font=None, size=128, seed=42, rebuild=False,
                 samples_per_class=100, workers=8, local_root=None, max_overlap=.001,
                 max_attempts=20):
    """Build unique semantic combinations and independently transform target jamo.

    Cached source data is reused without rescanning Drive. Set rebuild=True when
    source images change. Train/val split is chosen before sample generation.
    """
    if samples_per_class < 1 or workers < 1 or max_attempts < 1 or not 0 <= max_overlap <= 1:
        raise ValueError('Invalid generation settings')
    if reference_font is not None:
        raise ValueError('Use reference_font=None: every sample now uses its own font for reference and input.')
    requested = dict(version=VERSION, source=normalized(source), reference_font=reference_font,
                     size=size, seed=seed, samples_per_class=samples_per_class,
                     max_overlap=max_overlap, max_attempts=max_attempts)
    key = hashlib.sha256(json.dumps(requested, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
    local_root = Path(local_root) if local_root else Path(tempfile.gettempdir()) / 'jamo_prepared'
    local = local_root / key
    marker = local_root / f'{key}.json'
    if not rebuild:
        try:
            cached_local = Path(json.loads(marker.read_text(encoding='utf-8'))['path'])
            if cached_local.is_relative_to(local_root.resolve()):
                local = cached_local
        except (OSError, ValueError, KeyError):
            pass
    cache_root = Path(cache_root)
    archive = cache_root / f'jamo_{key}.zip'
    if not rebuild and _local_complete(local):
        print('Reuse local Jamo dataset:', local, flush=True)
        return local / 'manifest.json'
    if not rebuild and archive.is_file():
        print('Restore Jamo dataset from Drive ZIP:', archive, flush=True)
        # Copy once so extraction never opens individual files on Drive.
        local_root.mkdir(parents=True, exist_ok=True)
        import shutil
        with tempfile.TemporaryDirectory(dir=local_root) as tmp:
            cached = Path(tmp) / 'dataset.zip'
            shutil.copyfile(archive, cached)
            restored = Path(tmp) / 'restored'
            _unpack(cached, restored)
            if not _local_complete(restored):
                raise ValueError('Incomplete Jamo archive; set REBUILD=True')
            local = local_root / (key + '_' + Path(tmp).name)
            shutil.move(str(restored), str(local))
        marker.write_text(json.dumps(dict(path=str(local))), encoding='utf-8')
        return local / 'manifest.json'
    print('Scan source component banks...', flush=True)
    source, fonts = scan_jamo(source)
    target_fonts = sorted(font for font in fonts if
                          any(shared_options(fonts[font], fonts[font], cls) for cls in range(6)))
    if len(target_fonts) < 2:
        raise ValueError('Need at least two usable fonts to hold out validation fonts.')
    shuffled = target_fonts.copy()
    random.Random(seed).shuffle(shuffled)
    val_fonts = set(shuffled[:max(1, round(len(shuffled) * .2))])
    inventory = {font: {str(cls): {str(slot): len(bank) for slot, bank in slots.items()}
                        for cls, slots in classes.items()} for font, classes in fonts.items()}
    paths = sorted({path for classes in fonts.values() for slots in classes.values()
                    for bank in slots.values() for candidates in bank.values() for path in candidates})
    print(f'Load {len(paths)} original PNGs once ({workers} workers); each font uses its own reference', flush=True)
    loaded = read_components(paths, size, workers)
    local_root.mkdir(parents=True, exist_ok=True)
    # Fresh local output also avoids stale samples after an interrupted run.
    local = Path(tempfile.mkdtemp(prefix=key + '_', dir=local_root))
    (local / 'samples').mkdir()
    rng = random.Random(seed)
    rows, reports, skipped = [], [], []
    duplicates = [dict(font=font, class_id=cls, slot=slot, jamo_index=jamo,
                       candidates=[normalized(p.name) for p in candidates])
                  for font, classes in fonts.items() for cls, slots in classes.items()
                  for slot, bank in slots.items() for jamo, candidates in bank.items() if len(candidates) > 1]
    for font in tqdm(target_fonts, desc='Compose locally', unit='font'):
        reference = fonts[font]
        for cls in range(6):
            options = shared_options(reference, fonts[font], cls)
            if not options:
                skipped.append(dict(font=font, class_id=cls, reason='Missing shared semantic jamo bank'))
                continue
            combinations = list(itertools.product(*options))
            rng.shuffle(combinations)
            # Bound work even when a class has many overlapping/rejected combinations.
            for parts in combinations[:samples_per_class]:
                count = 2 if cls < 3 else 3
                # The representative filename can name a different syllable.
                ref_paths = [reference[cls][slot][parts[slot]][0] for slot in range(count)]
                target_paths = [fonts[font][cls][slot][parts[slot]][0] for slot in range(count)]
                if len({loaded[p][1] for p in ref_paths + target_paths}) != 1:
                    raise ValueError(f'Mismatched original canvases: {font}, class {cls}')
                ref_masks = np.stack([loaded[p][0] for p in ref_paths])
                raw_target = np.stack([loaded[p][0] for p in target_paths])
                valid = False
                for attempt in range(max_attempts):
                    masks = np.stack([transform_mask(mask, rng) for mask in raw_target])
                    labels, overlap, valid = make_labels(masks, max_overlap)
                    if valid:
                        break
                if not valid:
                    skipped.append(dict(font=font, class_id=cls, char=compose(parts), reason='Overlap/rejected empty owner'))
                    continue
                index = len(rows)
                path = Path('samples') / f'{index:06d}.npz'
                image = masks.max(0, keepdims=True)
                np.savez_compressed(local / path, image=image, reference=ref_masks.max(0, keepdims=True),
                                    strokes=ref_masks, labels=labels)
                rows.append(dict(path=path.as_posix(), split='val' if font in val_fonts else 'train',
                                 group_id='font:' + font, char_id=compose(parts), class_id=cls,
                                 font=font, reference_font=font, component_slots=list(range(count)),
                                 jamo_indices=list(parts), reference_components=[normalized(p.name) for p in ref_paths],
                                 target_components=[normalized(p.name) for p in target_paths], source='composed_jamo'))
                foreground = image[0] > .5
                known = labels >= 0
                reports.append(dict(row=index, foreground_pixels=int(foreground.sum()),
                                    annotation_coverage=float(known.sum() / max(foreground.sum(), 1)),
                                    overlap_pixels=int((foreground & ~known).sum()), missing_pixels=0,
                                    overlap_canvas_ratio=overlap))
    if {row['split'] for row in rows} != {'train', 'val'}:
        raise ValueError('No usable train/val samples after overlap filtering. Inspect masks/settings.')
    report = dict(source=str(source), reference_font='same_font', reference_mode='same_font', inventory=inventory,
                  split_note='Font holdout before composition; reference and input use the same font per sample.',
                  validation_fonts=sorted(val_fonts), rows=reports, skipped=skipped,
                  duplicate_semantic_candidates=duplicates,
                  composition='Class-shared semantic jamo; transformed target masks merged with pixelwise maximum.',
                  settings=requested)
    (local / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (local / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (local / 'complete.json').write_text(json.dumps(dict(samples=len(rows))), encoding='utf-8')
    print(f'Generated {len(rows)} samples locally; packing one Drive archive...', flush=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    # Build ZIP locally; a single copy publishes the completed cache on Drive.
    import shutil
    packed = local / 'dataset.zip'
    with zipfile.ZipFile(packed, 'w', compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(local.rglob('*')):
            if path.is_file() and path != packed:
                bundle.write(path, path.relative_to(local).as_posix())
    staging = archive.with_suffix('.zip.partial')
    shutil.copyfile(packed, staging)
    staging.replace(archive)
    packed.unlink()
    # Small marker avoids reloading/validating thousands of Drive paths next run.
    marker.write_text(json.dumps(dict(path=str(local))), encoding='utf-8')
    print('Drive cache:', archive, flush=True)
    print('Local manifest:', local / 'manifest.json', flush=True)
    return local / 'manifest.json'
