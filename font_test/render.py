"""Deterministic typeface rendering, with no model-dependent image editing."""
from pathlib import Path
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font_supports(font_path,text):
    from fontTools.ttLib import TTFont
    with TTFont(str(font_path),fontNumber=0,lazy=True) as font:
        cmap=font.getBestCmap() or {}
        return all(cmap.get(ord(c)) not in (None,'.notdef') for c in text)


def render_glyph(text,font_path,reference,placement='reference_bbox',supersample=4):
    """Return 1,H,W ink. Fit WITHOUT stretching x/y separately.

    reference_bbox: align whole-glyph center/maximum extent to the reference.
    centered: fit into 75% of the canvas, independent of reference position.
    Neither mode adjusts individual components or creates stroke ground truth.
    """
    text=unicodedata.normalize('NFC',str(text).strip())
    if len(text)!=1 or text.isspace():raise ValueError('Render one character matching the selected reference')
    if not font_supports(font_path,text):raise ValueError(f'Font has no glyph for {text}: {font_path}')
    reference=np.asarray(reference,dtype=np.float32)
    if reference.ndim!=3 or reference.shape[0]!=1:raise ValueError('reference must be 1,H,W')
    if type(supersample) is not int or supersample<1:raise ValueError('Invalid supersampling')
    h,w=reference.shape[-2:];scale=supersample
    if placement=='reference_bbox':
        yy,xx=np.where(reference[0]>.5)
        if not len(xx):raise ValueError('Reference is empty')
        x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    elif placement=='centered':
        x0,y0,x1,y1=w*.125,h*.125,w*.875,h*.875
    else:raise ValueError('placement must be reference_bbox or centered')
    font=ImageFont.truetype(str(Path(font_path)),max(h,w)*scale,index=0)
    left,top,right,bottom=font.getbbox(text)
    if right<=left or bottom<=top:raise ValueError('Font glyph has no visible outline')
    glyph=Image.new('L',(right-left,bottom-top))
    ImageDraw.Draw(glyph).text((-left,-top),text,font=font,fill=255)
    box=glyph.getbbox()
    if box is None:raise ValueError('Font rendered empty ink')
    glyph=glyph.crop(box)
    ratio=min((x1-x0)*scale/glyph.width,(y1-y0)*scale/glyph.height)
    glyph=glyph.resize((max(1,round(glyph.width*ratio)),max(1,round(glyph.height*ratio))),Image.Resampling.LANCZOS)
    canvas=Image.new('L',(w*scale,h*scale))
    cx,cy=(x0+x1)*scale/2,(y0+y1)*scale/2
    canvas.paste(glyph,(round(cx-glyph.width/2),round(cy-glyph.height/2)))
    return (np.asarray(canvas.resize((w,h),Image.Resampling.LANCZOS),dtype=np.float32)/255)[None]
