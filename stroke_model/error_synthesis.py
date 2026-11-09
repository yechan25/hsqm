"""Same-font known-change supervision for Gate A; not real-error ground truth."""
import hashlib
import random
import re
import unicodedata
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.morphology import skeletonize
from diffusion_model.score_pilot import load_catalog_reference
from font_test.render import render_glyph
from fontTools.ttLib import TTFont


def glyph_from_entry(entry):
    """Only explicit Hangul metadata/filename suffix; never guess numeric IDs."""
    char = unicodedata.normalize('NFC', str(entry.get('char_id', '')))
    if len(char) == 1 and '\uac00' <= char <= '\ud7a3':return char
    stem = unicodedata.normalize('NFC', Path(entry.get('reference', '')).stem)
    match = re.search(r'_([가-힣])$', stem)
    return match[1] if match else None


def corrupt_extra(clean_ink, kind, seed):
    """kind=0 clean, 1 attached, 2 detached. Label only newly visible ink."""
    clean = np.asarray(clean_ink, np.float32)
    if clean.ndim != 3 or clean.shape[0] != 1 or not np.isfinite(clean).all() or clean.min()<0 or clean.max()>1:
        raise ValueError('Clean ink must be finite 1,H,W in [0,1]')
    if kind not in (0, 1, 2):raise ValueError('Unknown corruption kind')
    if kind == 0:return clean.copy(), np.zeros_like(clean), np.zeros(clean.shape[1:], np.int64)
    rng = np.random.default_rng(seed)
    foreground = clean[0] > .5
    skeleton = np.argwhere(skeletonize(foreground))
    if not len(skeleton):raise ValueError('Cannot corrupt empty glyph')
    h, w = foreground.shape
    distance = distance_transform_edt(foreground)
    width = max(2, min(8, round(2*np.median(distance[skeleton[:,0], skeleton[:,1]]))))
    near = binary_dilation(foreground, iterations=2)
    for _ in range(100):
        canvas = Image.new('L', (w, h)); draw = ImageDraw.Draw(canvas)
        if kind == 1:
            y, x = skeleton[int(rng.integers(len(skeleton)))]
            angle = rng.uniform(0, 2*np.pi); length = rng.uniform(.09, .23)*min(h,w)
            end = (int(np.clip(x+length*np.cos(angle), 1, w-2)), int(np.clip(y+length*np.sin(angle), 1, h-2)))
            draw.line([(int(x),int(y)),end], fill=255, width=width)
        else:
            x, y = int(rng.integers(2, w-3)), int(rng.integers(2, h-3))
            radius = int(rng.integers(1, max(3,width)))
            draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill=255)
        stroke = np.asarray(canvas, np.float32)/255
        if kind == 2 and ((stroke > 0) & near).any():continue
        bad = np.maximum(clean[0], stroke)
        changed = bad-clean[0] > .05
        if changed.sum() < 4:continue
        labels = np.zeros((h,w), np.int64); labels[changed] = kind
        return bad[None], changed.astype(np.float32)[None], labels
    raise RuntimeError('Unable to make a visible valid extra-ink sample')


class ErrorSynthesisDataset(Dataset):
    def __init__(self, entries, fonts, variants=6, seed=42):
        if not entries or not fonts or type(variants) is not int or variants < 3:raise ValueError('Need entries/fonts and at least 3 integer variants')
        self.entries=list(entries);self.fonts=list(map(str,fonts));self.variants=variants
        self.seed=seed;self.epoch=0;self.cache={}

    def __len__(self):return len(self.entries)*len(self.fonts)*self.variants

    def __getitem__(self,index):
        variant=index%self.variants; pair=index//self.variants
        entry=self.entries[pair//len(self.fonts)];font=self.fonts[pair%len(self.fonts)]
        char=glyph_from_entry(entry)
        if char is None:raise ValueError('Explicit Hangul identity required')
        key=(pair//len(self.fonts),font)
        if key not in self.cache:
            reference,strokes=load_catalog_reference(entry)
            clean=render_glyph(char,font,reference,placement='centered')
            self.cache[key]=(reference,strokes,clean)
        reference,strokes,clean=self.cache[key]
        kind=variant%3
        identity=f'{self.seed}|{self.epoch}|{char}|{font}|{variant}'
        seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little')
        corrupted,extra,kinds=corrupt_extra(clean,kind,seed)
        return dict(rgb=torch.from_numpy(1-corrupted).repeat(3,1,1),
            reference=torch.from_numpy(reference.copy()),strokes=torch.from_numpy(strokes.copy()),
            codepoint=torch.tensor(ord(char)),extra=torch.from_numpy(extra),valid=torch.ones_like(torch.from_numpy(extra)),
            kind=torch.from_numpy(kinds),clean=torch.from_numpy(1-clean.copy()).repeat(3,1,1))


def collate_errors(samples):
    k=max(s['strokes'].shape[0] for s in samples)
    out={name:torch.stack([s[name] for s in samples]) for name in ('rgb','reference','codepoint','extra','valid','kind','clean')}
    out['strokes']=torch.stack([torch.nn.functional.pad(s['strokes'],(0,0,0,0,0,k-len(s['strokes']))) for s in samples])
    out['active']=out['strokes'].sum((-2,-1))>0
    return out


def build_error_sets(catalog, font_groups, max_characters=96, variants=6, seed=42):
    """Character-disjoint AND configured-font-disjoint splits for the NEW head."""
    if type(max_characters) is not int or max_characters<6:raise ValueError('Need at least 6 maximum characters')
    if set(font_groups)!={'train','val','test'} or any(not v for v in font_groups.values()):
        raise ValueError('Supply nonempty train/val/test font groups from different families')
    paths=[str(Path(p).resolve()) for group in font_groups.values() for p in group]
    if len(paths)!=len(set(paths)):raise ValueError('Font file leakage across splits')
    font_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    owners={}
    for split,fonts in font_groups.items():
        for font in fonts:
            digest=font_hashes[str(Path(font).resolve())]
            previous=owners.setdefault(digest,split)
            if previous!=split:raise ValueError('Identical font content appears in different splits')
    by_char={}
    for entry in catalog:
        char=glyph_from_entry(entry)
        if char and char not in by_char:by_char[char]=entry
    candidates=sorted(by_char);random.Random(seed).shuffle(candidates)
    cmaps=[]
    for font in paths:
        with TTFont(font,fontNumber=0,lazy=True) as face:cmaps.append(set((face.getBestCmap() or {}).keys()))
    chars=[c for c in candidates if all(ord(c) in cmap for cmap in cmaps)][:max_characters]
    if len(chars)<6:raise ValueError('Need at least 6 identifiable supported characters; inspect the catalog metadata')
    a=max(2,int(len(chars)*.7));b=max(1,int(len(chars)*.15))
    groups=dict(train=chars[:a],val=chars[a:a+b],test=chars[a+b:])
    sets={split:ErrorSynthesisDataset([by_char[c] for c in group],font_groups[split],variants,seed)
          for split,group in groups.items()}
    metadata=dict(seed=seed,characters=groups,fonts={k:list(map(str,v)) for k,v in font_groups.items()},
        font_hashes=font_hashes,
        scope='new head character/font transfer; extractor pretraining overlap is not excluded',
        template_rows={c:by_char[c].get('manifest_row') for c in chars})
    return sets,metadata
