"""Training-only, conservative expansion of actual sparse stroke annotations.

This is weak supervision, not full ground truth or a reproduction of ScribbleSup.
No network modules or inference rules change. Ambiguous crossings stay unknown.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
import torch
from tqdm.auto import tqdm

from .reference_v2_topology import TopologyDataset, collate_topology


@dataclass(frozen=True)
class OwnershipConfig:
    version: int = 1
    separation_margin: float = 1.5
    width_tolerance: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.separation_margin) or self.separation_margin <= 0:
            raise ValueError('separation_margin must be finite and positive')
        if not np.isfinite(self.width_tolerance) or self.width_tolerance < 0:
            raise ValueError('width_tolerance must be finite and nonnegative')


def expand_ownership(image, target_masks, labels, valid_pixels=None, config=None):
    """Return weak owners/reliability; retain all original single-owner labels.

    Distances use ACTUAL INPUT annotation seeds, never reference geometry or model
    predictions. Restrict propagation to ink components already containing that
    stroke's seeds, and to local ink half-width measured at the nearest seed.
    The heuristic is deliberately conservative and needs a reviewed pilot set.
    """
    cfg = config or OwnershipConfig()
    image=np.asarray(image); target=np.asarray(target_masks); labels=np.asarray(labels)
    if image.ndim!=3 or image.shape[0]!=1 or target.ndim!=3 or target.shape[1:]!=image.shape[1:] or labels.shape!=image.shape[1:]:
        raise ValueError('Expected image 1,H,W; target K,H,W; labels H,W')
    if not len(target) or not np.isfinite(image).all() or not np.isfinite(target).all():
        raise ValueError('Need finite actual masks')
    valid=np.ones_like(image) if valid_pixels is None else np.asarray(valid_pixels)
    if valid.shape!=image.shape:
        raise ValueError('Invalid known-region shape')
    ink=(image[0]>.5)&(valid[0]>.5)
    components,_=ndi.label(ink,np.ones((3,3)))
    # Pad to measure distance to the image boundary as background, too.
    width=ndi.distance_transform_edt(np.pad(ink,1))[1:-1,1:-1]
    distances=np.full(target.shape,np.inf,np.float32)
    allowed=np.zeros(target.shape,bool)
    for k in range(len(target)):
        seeds=(labels==k)&ink
        if not seeds.any():
            continue  # No fabricated seed for absent or unannotated strokes.
        distance,nearest=ndi.distance_transform_edt(~seeds,return_indices=True)
        component_ids=np.unique(components[seeds])
        same_component=np.isin(components,component_ids)&ink
        half_width=width[nearest[0],nearest[1]]
        distances[k,same_component]=distance[same_component]
        allowed[k]=same_component&(distance<=half_width+cfg.width_tolerance)
    order=np.argsort(distances,axis=0,kind='stable')
    first=order[0]
    nearest=np.take_along_axis(distances,first[None],axis=0)[0]
    if len(target)>1:
        second=np.take_along_axis(distances,order[1:2],axis=0)[0]
    else:
        second=np.full_like(nearest,np.inf)
    finite=np.isfinite(nearest)
    gap=np.full_like(nearest,0)
    np.subtract(second,nearest,out=gap,where=finite)
    admissible=np.take_along_axis(allowed,first[None],axis=0)[0]
    # All actual multi-hot crossings remain unknown, including exact seed overlaps.
    overlap=(target>.5).sum(0)>1
    # Suppress expansion near unknown annotation boundaries.
    safe=~ndi.maximum_filter((valid[0]<=.5).astype(np.uint8),size=3,mode='constant',cval=1).astype(bool)
    trusted=ink&finite&admissible&safe&~overlap&(gap>cfg.separation_margin)
    owners=np.full(labels.shape,-1,np.int16)
    reliability=np.zeros(labels.shape,np.uint8)
    owners[trusted]=first[trusted]
    reliability[trusted]=np.uint8(255*np.clip(
        (gap[trusted]-cfg.separation_margin)/(cfg.separation_margin+1),0,1))
    original=(labels>=0)&ink
    owners[original]=labels[original]
    reliability[original]=255
    return owners,reliability


def prepare_ownership_manifest(manifest, config=None):
    """One-time CPU preparation; maps cached locally, never in the batch loop."""
    from .train_reference import read_manifest
    cfg=config or OwnershipConfig(); source=Path(manifest).resolve()
    rows=read_manifest(source)
    signatures=[(row['path'],Path(row['path']).stat().st_size,Path(row['path']).stat().st_mtime_ns) for row in rows]
    key=hashlib.sha256(source.read_bytes()+json.dumps([asdict(cfg),signatures]).encode()).hexdigest()[:16]
    destination=source.parent/('ownership_'+key)
    result=destination/'manifest.json'
    if (destination/'.complete.json').is_file() and result.is_file():
        print('영역 보조 주석 캐시 재사용:',result,flush=True)
        return result
    destination.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); counts=[]
    for index,row in enumerate(tqdm(rows,desc='Prepare weak ownership (once)',mininterval=2)):
        if row['split']!='train':
            continue  # Do not manufacture validation ground truth.
        path=destination/f'{index:06d}.npz'
        with np.load(row['path'],allow_pickle=False) as arrays:
            if 'target_masks' not in arrays:
                raise ValueError('Need actual input target_masks for weak ownership preparation')
            owners,weight=expand_ownership(arrays['image'],arrays['target_masks'],arrays['labels'],
                arrays['valid_pixels'] if 'valid_pixels' in arrays else None,cfg)
            ink=arrays['image'][0]>.5
            original=arrays['labels']>=0
        np.savez_compressed(path,owners=owners,reliability=weight)
        row['ownership_path']=str(path.resolve())
        counts.append(dict(manifest_row=index+1,input_pixels=int(ink.sum()),original_pixels=int((original&ink).sum()),
                           weak_added_pixels=int(((weight>0)&~original).sum())))
    result.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(config=asdict(cfg),preparation_seconds=time.perf_counter()-started,rows=counts,
                policy='Weak labels from actual input seeds; ambiguity remains unknown; not dense GT')
    (destination/'preparation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (destination/'.complete.json').write_text(key)
    print(f'보조 주석 준비: {report["preparation_seconds"]:.1f}s (epoch 시간과 별도)',flush=True)
    return result


def preview_ownership_samples(manifest,count=3):
    """Inspect proposed weak labels before training; never presents them as GT."""
    import matplotlib.pyplot as plt
    from .train_reference import read_manifest
    seen=set();shown=0
    for row in read_manifest(manifest):
        if row['split']!='train' or row['group_id'] in seen:
            continue
        seen.add(row['group_id'])
        with np.load(row['path'],allow_pickle=False) as arrays:
            image=arrays['image'];labels=arrays['labels'];target=arrays['target_masks']
            owners,weight=expand_ownership(image,target,labels,arrays['valid_pixels'] if 'valid_pixels' in arrays else None)
        palette=plt.get_cmap('tab20')(np.arange(len(target))%20)[:,:3]
        def color(ids):
            canvas=np.zeros((*ids.shape,3))
            valid=ids>=0;canvas[valid]=palette[ids[valid]]
            return canvas
        weak=np.where(weight>0,owners,-1)
        unknown=np.zeros((*labels.shape,3));unknown[(image[0]>.5)&(weak<0)]=(1,0,0)
        figure,axes=plt.subplots(1,4,figsize=(11,3))
        for ax,(content,title) in zip(axes,((image[0],'Input'),(color(labels),'Original labels'),
                (color(weak),'Weak owners (NOT GT)'),(unknown,'Ignored ink = red'))):
            ax.imshow(content,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title);ax.axis('off')
        figure.tight_layout();plt.show();plt.close(figure)
        shown+=1
        if shown>=count:break


class OwnershipDataset(TopologyDataset):
    def __getitem__(self,index):
        sample=super().__getitem__(index)
        with np.load(self.rows[index]['ownership_path'],allow_pickle=False) as arrays:
            owners=torch.from_numpy(arrays['owners'].copy()).long()
            weight=torch.from_numpy(arrays['reliability'].copy()).float()/255
        if owners.shape!=sample[3].shape or weight.shape!=owners.shape:
            raise ValueError('Weak ownership map shape does not match labels')
        return (*sample,owners,weight)


def collate_ownership(samples):
    return (*collate_topology([s[:6] for s in samples]),
            torch.stack([s[6] for s in samples]),torch.stack([s[7] for s in samples]))


def ownership_losses(output, owners, reliability, compute_ownership=True, compute_affinity=True):
    """Class-balanced weak CE and same-owner neighbour consistency, O(B*K*H*W).

    No skeletonization, CRF, graph iterations or teacher forward passes here.
    Never smooth across owners, ink gaps or unknown crossing regions.
    """
    p=output['probabilities'].float()
    if owners.shape!=p.shape[:1]+p.shape[2:] or reliability.shape!=owners.shape:
        raise ValueError('Weak ownership maps must be B,H,W')
    valid=(owners>=0)&(reliability>0)
    safe=owners.clamp_min(0)
    selected=p.gather(1,safe[:,None]).squeeze(1)
    w=reliability*valid
    zero=selected.sum()*0
    # Each annotated stroke contributes equally; large regions cannot drown small strokes.
    ce=zero
    if compute_ownership:
        totals=p.new_zeros(p.shape[:2]).scatter_add_(1,safe.flatten(1),(-selected.clamp_min(1e-6).log()*w).flatten(1))
        masses=p.new_zeros(p.shape[:2]).scatter_add_(1,safe.flatten(1),w.flatten(1))
        supported=masses>0
        ce=((totals/masses.clamp_min(1e-6))*supported).sum()/supported.sum().clamp_min(1)
    if not compute_affinity:
        return {'ownership':ce,'affinity':zero}
    numerator=zero;denominator=p.new_zeros(())
    for a,b in (((...,slice(None),slice(None,-1)),(...,slice(None),slice(1,None))),
                ((...,slice(None,-1),slice(None)),(...,slice(1,None),slice(None)))):
        pair=(owners[a]==owners[b])&valid[a]&valid[b]
        pair_weight=torch.minimum(w[a],w[b])*pair
        numerator=numerator+((selected[a]-selected[b]).square()*pair_weight).sum()
        denominator=denominator+pair_weight.sum()
    return {'ownership':ce,'affinity':numerator/denominator.clamp_min(1)}
