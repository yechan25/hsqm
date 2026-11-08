"""Inspect fixed maps on a saved failure; does not claim VAE correction."""
import sys,zipfile,io,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stroke_model.local_reference_edit import LocalReferenceEdit

def audit(archive,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        def read(name):
            a=np.asarray(Image.open(io.BytesIO(z.read(name))).convert('RGB'),dtype=np.float32)/255
            return torch.from_numpy(a).permute(2,0,1)[None]
        ref=read('generation/page_000_reference.png');base=read('generation/page_001_baseline.png')
    editor=LocalReferenceEdit(ref,base)
    np.savez_compressed(out/'edit_maps.npz',erase=editor.erase.numpy(),fill=editor.fill.numpy(),protect=editor.protect.numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    panels=[(ref[0,:,:288,:288].permute(1,2,0),'Reference'),(base[0,:,:288,:288].permute(1,2,0),'Baseline'),(editor.erase[0,0,:288,:288],'Erase map'),(editor.fill[0,0,:288,:288],'Fill map')]
    for ax,(a,title) in zip(axes,panels):ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'ga_edit_maps.png',dpi=140);plt.close(fig)
    (out/'audit.json').write_text(json.dumps({'erase_pixels':float(editor.erase.sum()),'fill_pixels':float(editor.fill.sum()),'validation':'map inspection only; not generated correction'},indent=2))

if __name__=='__main__':audit(sys.argv[1],sys.argv[2])
