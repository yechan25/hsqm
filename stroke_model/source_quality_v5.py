"""Experimental foreground-aware structure + reference-relative contour quality.

No learned weights or human calibration. Contour diagnostics use CPU thresholded
distance fields and are NOT differentiable reverse-diffusion guidance.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter, label
import torch

from .source_structure import SourceConfig, PreparedSourceMetric, register_reference
from .structural_indicator import IndicatorConfig, StructuralIndicator


@dataclass(frozen=True)
class QualityConfig:
    foreground_midpoint: float = .12
    foreground_temperature: float = .03
    contour_sigma: float = 1.5
    contour_position_tolerance: int = 1
    contour_allowance: float = .08
    contour_scale: float = .05
    contour_thresholds: tuple = (.35, .5, .65)
    component_threshold: float = .35
    component_halo: float = 1.5

    def __post_init__(self):
        for name in ('foreground_midpoint','foreground_temperature','contour_sigma','contour_allowance','contour_scale',
                     'component_threshold','component_halo'):
            if not math.isfinite(getattr(self,name)):
                raise ValueError('Quality settings must be finite')
        if (not 0 < self.foreground_midpoint < 1 or self.foreground_temperature <= 0
                or self.contour_sigma <= 0 or self.contour_allowance < 0 or self.contour_scale <= 0
                or type(self.contour_position_tolerance) is not int or self.contour_position_tolerance < 0
                or not self.contour_thresholds or any(not 0 < t < 1 for t in self.contour_thresholds)
                or not 0 < self.component_threshold < 1 or self.component_halo < 0):
            raise ValueError('Invalid quality settings')


def foreground_evidence(ink, config=None):
    """Smooth, fixed contrast gate, NOT binarization or contour smoothing."""
    cfg = config or QualityConfig()
    x = torch.as_tensor(ink)
    if not x.is_floating_point():
        x = x.float()
    if not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError('Ink must be finite in [0,1]')
    gate = torch.sigmoid((x-cfg.foreground_midpoint)/cfg.foreground_temperature)
    normalizer = torch.sigmoid(x.new_tensor((1-cfg.foreground_midpoint)/cfg.foreground_temperature))
    return x*gate/normalizer


def _response(ink, threshold, sigma):
    binary = ink > threshold
    if not binary.any() or binary.all():
        return np.zeros_like(ink), np.zeros_like(ink, dtype=bool)
    field = distance_transform_edt(binary)-distance_transform_edt(~binary)
    boundary = np.abs(field) <= 1.01
    return (field-gaussian_filter(field,sigma))*boundary, boundary


def separate_unassigned_ink(image, support, config=None):
    """Separate whole components with NO overlap with the expected support.

    Does not select the largest component: legitimate disconnected Hangul strokes
    remain when they overlap ANY expected stroke. Unassigned is a geometric label,
    never proof of harmless dirt. The strict whole-crop score is retained.
    """
    cfg=config or QualityConfig()
    image,support=[np.asarray(x,dtype=np.float32).squeeze() for x in (image,support)]
    if image.ndim!=2 or support.shape!=image.shape:
        raise ValueError('Need matching H,W image/support')
    labels,count=label(image>cfg.component_threshold,np.ones((3,3),bool))
    unmatched=[]
    for index in range(1,count+1):
        if not ((labels==index)&(support>=.5)).any():
            unmatched.append(index)
    mask=np.zeros_like(image,dtype=bool)
    if unmatched:
        # Include nearby anti-alias pixels, without masking another component.
        distance,nearest=distance_transform_edt(labels==0,return_indices=True)
        nearest_label=labels[tuple(nearest)]
        mask=np.isin(nearest_label,unmatched)&(distance<=cfg.component_halo)
    unassigned=image*mask
    return dict(body_image=image*(~mask),unassigned_ink_map=unassigned,
        unassigned_component_count=len(unmatched),
        unassigned_ink_fraction=float(unassigned.sum()/max(float(image.sum()),1.)),
        unassigned_ink_review_required=bool(unmatched))


def contour_diagnosis(image, reference, config=None):
    """Excess small-scale boundary response beyond nearby reference response.

    Positive/negative response marks convex/concave candidates, not proven
    semantic defects. A matching serif/corner has zero excess by construction.
    All maps are in canonical registered coordinates.
    """
    cfg = config or QualityConfig()
    image, reference = [np.asarray(x,dtype=np.float32).squeeze() for x in (image,reference)]
    if image.ndim != 2 or reference.shape != image.shape:
        raise ValueError('Need matching H,W ink arrays')
    for array in (image,reference):
        if not np.isfinite(array).all() or array.min()<0 or array.max()>1:
            raise ValueError('Expected finite ink in [0,1]')
    positive, negative, boundary_maps, values = [], [], [], []
    for threshold in cfg.contour_thresholds:
        response,boundary = _response(image,threshold,cfg.contour_sigma)
        expected,_ = _response(reference,threshold,cfg.contour_sigma)
        allowance = maximum_filter(abs(expected),2*cfg.contour_position_tolerance+1)
        excess = np.maximum(abs(response)-allowance-cfg.contour_allowance,0)*boundary
        positive.append(excess*(response>0))
        negative.append(excess*(response<0))
        boundary_maps.append(boundary)
        values.append(float(excess.sum()/max(int(boundary.sum()),1)))
    value = float(np.mean(values))
    return dict(contour_excess=value, contour_score_cap=100*math.exp(-value/cfg.contour_scale),
                protrusion_map=np.mean(positive,axis=0), indentation_map=np.mean(negative,axis=0),
                boundary_map=np.mean(boundary_maps,axis=0), per_threshold_excess=values)


def assess_source_quality(image, reference, strokes, *, registration=None,
                          config=None, score_config=None, source_config=None):
    cfg = config or QualityConfig()
    score_cfg = score_config or IndicatorConfig()
    source_cfg = source_config or SourceConfig()
    source = torch.as_tensor(np.asarray(image),dtype=torch.float32)
    target = torch.as_tensor(np.asarray(reference),dtype=torch.float32)
    cleaned = foreground_evidence(source,cfg)
    target_cleaned = foreground_evidence(target,cfg).squeeze().numpy()
    if registration is None:
        _,_,registration = register_reference(source.numpy(),target.numpy(),strokes,source_cfg)
    metric = PreparedSourceMetric(target_cleaned,strokes,source_cfg,parameters=registration['parameters'])
    with torch.no_grad():
        measured = metric(cleaned)
        structural = StructuralIndicator(score_cfg)(measured['missing'],measured['split'],
                                                    measured['junction'],extra=measured['extra'])
    registered = measured['registered_image'][0,0].numpy()
    contour = contour_diagnosis(registered,target_cleaned,cfg)
    clean_score = float(structural['score'][0])
    separated=separate_unassigned_ink(cleaned.squeeze().numpy(),measured['support'][0,0].numpy(),cfg)
    if separated['unassigned_component_count']:
        with torch.no_grad():
            body_measured=metric(torch.from_numpy(separated['body_image']))
            body_structure=StructuralIndicator(score_cfg)(body_measured['missing'],body_measured['split'],
                body_measured['junction'],extra=body_measured['extra'])
        body_contour=contour_diagnosis(body_measured['registered_image'][0,0].numpy(),target_cleaned,cfg)
        body_structural_score=float(body_structure['score'][0])
    else:
        body_measured,body_contour=measured,contour
        body_structural_score=clean_score
    return dict(foreground_structural_score=clean_score, foreground_extra=float(measured['extra'][0]),
                experimental_score=min(clean_score,contour['contour_score_cap']),
                body_structural_score=body_structural_score,
                body_contour_score_cap=body_contour['contour_score_cap'],
                body_diagnostic_score=min(body_structural_score,body_contour['contour_score_cap']),
                body_registered_image=body_measured['registered_image'][0,0].numpy(),
                body_protrusion_map=body_contour['protrusion_map'],
                body_indentation_map=body_contour['indentation_map'],
                **separated,
                registration=registration, source_image=source.squeeze().numpy(),
                foreground_image=cleaned.squeeze().numpy(),registered_image=registered,
                reference_image=target_cleaned,foreground_extra_map=measured['extra_map'][0,0].numpy(),
                **contour)


def compare_quality_v5(result, *, config=None, show=True):
    """Use saved source/reference arrays; never rerun or score extractor masks."""
    from datetime import datetime
    import pandas as pd
    import matplotlib.pyplot as plt
    cfg = QualityConfig(**config) if isinstance(config,dict) else config or QualityConfig()
    score_cfg = IndicatorConfig(**result['indicator']['config'])
    source_cfg = SourceConfig(**result['indicator']['source_config'])
    destination = Path(result['save_dir'])/('quality_v5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(exist_ok=False)
    samples = result['extraction']['results']
    old = result['indicator']['results']
    if len(samples)!=len(old):
        raise ValueError('Sample/score count mismatch')
    rows = []
    for index,(sample,previous) in enumerate(zip(samples,old),1):
        print(f'Foreground/contour diagnosis {index}/{len(samples)}',flush=True)
        with np.load(sample['npz'],allow_pickle=False) as data:
            reference = data['reference_image'] if 'reference_image' in data else data['reference']
            assessed = assess_source_quality(data['image'],reference,data['reference_strokes'],
                registration=previous.get('registration'),config=cfg,score_config=score_cfg,source_config=source_cfg)
        maps = {key:value for key,value in assessed.items() if isinstance(value,np.ndarray)}
        path = destination/f'sample_{index:04d}.npz'
        np.savez_compressed(path,**maps)
        row = {key:sample.get(key) for key in ('label','manifest_row','control','glyph_count','page_id','slot')}
        row.update(sample=index,raw_source_score=previous['structural_score'],
                   **{key:assessed[key] for key in ('foreground_structural_score','foreground_extra',
                       'contour_excess','contour_score_cap','experimental_score','body_structural_score',
                       'body_contour_score_cap','body_diagnostic_score','unassigned_component_count',
                       'unassigned_ink_fraction','unassigned_ink_review_required')},maps=str(path))
        rows.append(row)
        fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
        axes[0].imshow(assessed['source_image'],cmap='gray_r',vmin=0,vmax=1)
        axes[0].set_title('Original scored input')
        axes[1].imshow(assessed['foreground_image'],cmap='gray_r',vmin=0,vmax=1)
        shape=(*assessed['foreground_image'].shape,3)
        axes[1].imshow(np.ones(shape)*[1,0,1],alpha=np.clip(assessed['unassigned_ink_map']/.2,0,.9))
        axes[1].set_title('Magenta: unassigned components\nRole must be reviewed')
        axes[2].imshow(registered:=assessed['body_registered_image'],cmap='gray_r',vmin=0,vmax=1)
        # Alpha amplification is display-only. Underlying NPZ maps stay absolute.
        shape=(*registered.shape,3)
        axes[2].imshow(np.ones(shape)*[1,.25,0],alpha=np.clip(assessed['body_protrusion_map']/.2,0,.9))
        axes[2].imshow(np.ones(shape)*[0,.6,1],alpha=np.clip(assessed['body_indentation_map']/.2,0,.9))
        axes[2].set_title('Body contour (registered)\nOrange: convex / blue: concave')
        axes[3].text(.02,.96,
            f"Body diagnostic: {row['body_diagnostic_score']:.1f}\nBody structure: {row['body_structural_score']:.1f}\n"
            f"Body contour cap: {row['body_contour_score_cap']:.1f}\n"
            f"Unassigned components: {row['unassigned_component_count']}\n"
            f"Unassigned ink: {100*row['unassigned_ink_fraction']:.2f}%\n\n"
            f"Strict full-crop prototype: {row['experimental_score']:.1f}\nRaw source: {row['raw_source_score']:.1f}\n\n"
            'Body score excludes magenta ink.\nReview its role before acceptance.\nPrototype; not human-calibrated.',va='top',fontsize=9)
        for ax in axes:ax.axis('off')
        fig.suptitle(sample.get('label',''))
        fig.savefig(destination/f'sample_{index:04d}.png',dpi=140)
        if show:plt.show()
        plt.close(fig)
    frame=pd.DataFrame(rows)
    frame.to_csv(destination/'quality_scores.csv',index=False,encoding='utf-8-sig')
    report=dict(version='source_quality_v5_component_diagnostics',trained=False,human_calibrated=False,
                body_score_scope='Excludes disconnected components outside expected support; role unverified',
                differentiable=False,config=asdict(cfg),score_config=asdict(score_cfg),
                source_config=asdict(source_cfg),results=rows,save_dir=str(destination))
    (destination/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if result.get('debug_zip'):
        with zipfile.ZipFile(result['debug_zip'],'a',zipfile.ZIP_DEFLATED) as archive:
            for file in destination.iterdir():
                archive.write(file,f'{destination.name}/{file.name}')
    return report,frame
