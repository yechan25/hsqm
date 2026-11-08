"""Independent-font input sheet; no clean target or input stroke annotations."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from font_test.render import render_glyph
from .multiglyph_pilot import compose_sheet


def compose_font_input(entries,glyphs,font_path,canvas_size=576):
    if len(entries)!=4 or len(glyphs)!=4 or any(len(g)!=1 or g.isspace() for g in glyphs):
        raise ValueError('Four characters in template selection order required')
    _,slots=compose_sheet(entries,canvas_size)
    sheet=Image.new('RGB',(canvas_size,canvas_size),'white')
    for glyph,slot in zip(glyphs,slots):
        # Independent placement: do not fit glyphs to template bounding boxes.
        ink=render_glyph(glyph,font_path,slot['reference'],placement='centered')
        x0,y0,x1,y1=slot['bbox']
        image=Image.fromarray(np.rint((1-ink[0])*255).astype(np.uint8)).convert('RGB')
        sheet.paste(image.resize((x1-x0,y1-y0),Image.Resampling.BICUBIC),(x0,y0))
    return sheet,dict(glyphs=list(glyphs),font=str(font_path),
        font_sha256=hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),placement='centered')
