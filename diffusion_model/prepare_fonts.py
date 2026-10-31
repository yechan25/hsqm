"""Build clean one-character pairs, splitting FONT FAMILIES before rendering."""
import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def inspect_font(path):
    from fontTools.ttLib import TTFont
    with TTFont(path, fontNumber=0) as font:
        cmap = font.getBestCmap() or {}
        # Typography family (16), fallback legacy family (1), not filename/style.
        family = next((record.toUnicode() for name_id in (16, 1)
                       for record in font['name'].names if record.nameID == name_id), path.stem)
    return unicodedata.normalize('NFC', family).casefold(), cmap


def render(path, char, size, scale=1., dx=0, dy=0):
    font = ImageFont.truetype(str(path), max(8, int(size*.76*scale)), index=0)
    left, top, right, bottom = font.getbbox(char)
    if right <= left or bottom <= top:
        raise ValueError('Empty glyph: ' + char)
    canvas = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text(((size-(right-left))/2-left+dx, (size-(bottom-top))/2-top+dy),
              char, font=font, fill=0)
    ink = 1-np.asarray(canvas, dtype=np.float32)/255
    if ink.sum() < 1 or ink[0].sum()+ink[-1].sum()+ink[:, 0].sum()+ink[:, -1].sum() > .1:
        raise ValueError('Empty or clipped glyph; reduce render scale: ' + char)
    return ink[None]


def prepare(font_dir, characters, output, variants=16, size=128, seed=42, val_fraction=.2):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory: ' + str(output))
    if variants < 1 or size < 16 or not 0 < val_fraction < 1:
        raise ValueError('Invalid variants, size or validation fraction')
    characters = list(dict.fromkeys(c for c in unicodedata.normalize('NFC', characters) if not c.isspace()))
    if not characters:
        raise ValueError('No characters')
    fonts = []
    for path in sorted(Path(font_dir).rglob('*')):
        if path.suffix.lower() in ('.ttf', '.otf', '.ttc'):
            family, cmap = inspect_font(path)
            if any(ord(c) in cmap for c in characters):
                fonts.append((path, family, cmap))
    families = sorted({family for _, family, _ in fonts})
    if len(families) < 2:
        raise ValueError('Need at least TWO different font families with glyph coverage; styles alone do not count')
    random.Random(seed).shuffle(families)
    val_groups = set(families[:max(1, min(len(families)-1, round(len(families)*val_fraction)))])
    references = {}
    for char in characters:
        train = [p for p, f, cmap in fonts if f not in val_groups and ord(char) in cmap]
        val = [p for p, f, cmap in fonts if f in val_groups and ord(char) in cmap]
        if not train or not val:
            raise ValueError(f'{char!r} needs at least one training AND validation font. Change character list/font coverage.')
        references[char] = render(train[0], char, size)
    output.mkdir(parents=True, exist_ok=True)
    samples = output/'samples'
    samples.mkdir()
    rows, index = [], 0
    for path, family, cmap in fonts:
        for char in characters:
            if ord(char) not in cmap:
                continue  # coverage already checked; never render missing-glyph boxes
            for variant in range(variants):
                key = f'{seed}:{path.name}:{char}:{variant}'
                rng = random.Random(hashlib.sha256(key.encode()).hexdigest())
                image = render(path, char, size, scale=rng.uniform(.9, 1.02),
                               dx=rng.uniform(-.025, .025)*size, dy=rng.uniform(-.025, .025)*size)
                local = f'samples/{index:07d}.npz'
                np.savez_compressed(output/local, image=image, reference=references[char])
                rows.append({'path': local, 'group_id': 'font:'+family,
                             'split': 'val' if family in val_groups else 'train', 'char_id': char,
                             'source': 'font-rendered-normal', 'font': str(path), 'variant': variant})
                index += 1
    manifest = output/'manifest.json'
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'preparation.json').write_text(json.dumps({'seed': seed, 'characters': characters,
        'font_families': families, 'val_groups': sorted(val_groups), 'variants': variants, 'size': size,
        'note': 'Font pairs have NO individual stroke annotations; stroke loss must stay off.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font-dir', required=True)
    p.add_argument('--characters', default='가나다라마바사아자차카타파하한글서울연구')
    p.add_argument('--output', required=True)
    p.add_argument('--variants', type=int, default=16)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    print(prepare(args.font_dir, args.characters, args.output, args.variants, args.size, args.seed))


if __name__ == '__main__':
    main()
