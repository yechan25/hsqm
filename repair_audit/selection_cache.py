"""One-time local image/stroke snapshots and thumbnails for the picker."""
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from diffusion_model.score_pilot import load_catalog_reference


def thumbnail(reference,size=92):
    image=Image.fromarray(np.uint8(np.clip(1-reference[0],0,1)*255)).resize((size,size))
    buffer=io.BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()


def prepare_selection_cache(catalog,root,*,source_fingerprint,progress=None):
    """Read each source once; interactions later use only local/in-memory data.

    source_fingerprint identifies an explicit data snapshot. Change it when the
    original data change. No model, inference, generation, or normalization change.
    """
    if not catalog:raise ValueError('Empty selection catalog')
    signature=hashlib.sha256(json.dumps(dict(version=1,catalog=catalog,
        source_fingerprint=source_fingerprint),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    destination=root/signature;index=destination/'catalog.json'
    if index.is_file():
        cached=json.loads(index.read_text(encoding='utf-8'))
        if all(Path(e['path']).is_file() and Path(e['thumbnail']).is_file() for e in cached):return cached
        raise ValueError('Incomplete local selection cache; use a new cache directory')
    with tempfile.TemporaryDirectory(prefix='preparing_',dir=root) as tmp:
        stage=Path(tmp);cached=[]
        for i,entry in enumerate(catalog,1):
            reference,strokes=load_catalog_reference(entry)
            name=f"row_{entry['manifest_row']}"
            np.savez_compressed(stage/(name+'.npz'),reference=reference,strokes=strokes)
            (stage/(name+'.png')).write_bytes(thumbnail(reference))
            cached.append(dict(entry,path=str(destination/(name+'.npz')),
                thumbnail=str(destination/(name+'.png')),source_entry=entry,
                cached_array_sha256=hashlib.sha256(reference.tobytes()+strokes.tobytes()).hexdigest()))
            if progress:progress(i,len(catalog))
        (stage/'catalog.json').write_text(json.dumps(cached,ensure_ascii=False,indent=2),encoding='utf-8')
        # Keep the prepared directory, not a partially written cache.
        stage.rename(destination)
    return cached


def selection_thumbnails(catalog):
    return {e['manifest_row']:Path(e['thumbnail']).read_bytes() for e in catalog}
