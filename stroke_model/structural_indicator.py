"""Deterministic structural indicator with explicit added-ink penalty.

No survey, fitted parameters, image embeddings, width or hard branch counts.
The 0..100 scale is a configurable engineering index, not calibrated quality.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import torch
from torch import nn
from .extra_ink import ExtraInkMetric


@dataclass(frozen=True)
class IndicatorConfig:
    missing_weight: float = 1.0
    split_weight: float = 1.0
    junction_weight: float = 1.0
    split_scale: float = 0.2
    junction_scale: float = 0.01
    worst_stroke_weight: float = 0.5
    extra_weight: float = 1.0
    extra_scale: float = 0.05
    extra_allowance: float = 0.02
    extra_radius: float = 8.0
    extra_transition: float = 0.5
    eps: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if min(self.missing_weight, self.split_weight, self.junction_weight, self.extra_weight) < 0:
            raise ValueError('Weights must be nonnegative')
        if self.split_scale <= 0 or self.junction_scale <= 0 or not 0 < self.eps < 1:
            raise ValueError('Scales must be positive; eps must be in (0,1)')
        if self.extra_scale <= 0 or not 0 <= self.extra_allowance < 1 or self.extra_radius < 0 or self.extra_transition <= 0:
            raise ValueError('Invalid extra-ink scale, allowance or geometry settings')
        for weight, scale in ((self.split_weight, self.split_scale), (self.junction_weight, self.junction_scale), (self.extra_weight, self.extra_scale)):
            if not math.isfinite(weight/scale) or weight/scale > torch.finfo(torch.float32).max:
                raise ValueError('weight/scale must fit a finite float32 coefficient')
        if not 0 <= self.worst_stroke_weight <= 1:
            raise ValueError('worst_stroke_weight must be in [0,1]')


class StructuralIndicator(nn.Module):
    """Map M,D,J and whole-glyph extra ink to scores without training.

    Inputs: K or B,K. Missing is in [0,1]; split/junction are nonnegative.
    q_k = 100*(1-M_k)^wM * exp(-wD*D_k/sD-wJ*J_k/sJ)
    Q0 = (1-alpha)*mean(q_k) + alpha*min(q_k), over active reference strokes.
    Q = Q0*exp(-wE*relu(E-extra_allowance)/extra_scale).
    min has subgradients; the full pipeline is not everywhere smooth.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or IndicatorConfig()

    def forward(self, missing, split, junction, active=None, *, extra=None):
        missing = torch.as_tensor(missing)
        if not missing.is_floating_point():
            missing = missing.float()
        split = torch.as_tensor(split, device=missing.device, dtype=missing.dtype)
        junction = torch.as_tensor(junction, device=missing.device, dtype=missing.dtype)
        if missing.ndim not in (1, 2) or missing.shape != split.shape or missing.shape != junction.shape:
            raise ValueError('missing, split, junction must have matching K or B,K shapes')
        if active is None:
            active = torch.ones_like(missing, dtype=torch.bool)
        if active.dtype != torch.bool or active.shape != missing.shape or active.device != missing.device:
            raise ValueError('active must be a bool tensor matching the input shape/device')
        if missing.ndim == 1:
            missing, split, junction, active = [x[None] for x in (missing, split, junction, active)]
        for name, value in [('missing', missing), ('split', split), ('junction', junction)]:
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f'{name} must be finite and nonnegative')
        if bool((missing > 1).any()) or not bool(active.any(1).all()):
            raise ValueError('Missing must be <=1 and each sample needs an active stroke')
        cfg = self.config
        # Smooth endpoint extension: exact 0/1 retention with finite gradients,
        # including fractional weights. Unlike a hard zero branch, full missing
        # still receives a restorative gradient at the default weight of one.
        if cfg.missing_weight == 0:
            retain_m = 1 + missing*0
        else:
            epsilon = missing.new_tensor(cfg.eps)
            lower = epsilon.pow(cfg.missing_weight)
            denominator = (1+epsilon).pow(cfg.missing_weight)-lower
            retain_m = ((1-missing+epsilon).pow(cfg.missing_weight)-lower)/denominator
        retain_d = torch.exp(-(cfg.split_weight/cfg.split_scale) * split)
        retain_j = torch.exp(-(cfg.junction_weight/cfg.junction_scale) * junction)
        stroke_scores = 100 * retain_m * retain_d * retain_j
        mean = (stroke_scores * active).sum(1) / active.sum(1)
        worst = stroke_scores.masked_fill(~active, torch.inf).amin(1)
        base_score = (1-cfg.worst_stroke_weight)*mean + cfg.worst_stroke_weight*worst
        if extra is None:
            if cfg.extra_weight > 0:
                raise ValueError('Added ink is required: provide extra or explicitly set extra_weight=0')
            extra = base_score*0
        extra = torch.as_tensor(extra, device=missing.device, dtype=missing.dtype)
        if extra.ndim == 0 and len(base_score) == 1:
            extra = extra[None]
        if extra.shape != base_score.shape or not bool(torch.isfinite(extra).all()) or bool(((extra < 0) | (extra > 1)).any()):
            raise ValueError('extra must contain one finite [0,1] fraction per glyph')
        extra_penalty = torch.relu(extra-cfg.extra_allowance)
        extra_retention = torch.exp(-(cfg.extra_weight/cfg.extra_scale)*extra_penalty)
        score = base_score * extra_retention
        return {'score': score, 'stroke_scores': stroke_scores, 'mean_score': mean,
                'worst_score': worst, 'penalty': 1-score/100, 'active': active,
                'base_score': base_score, 'extra': extra, 'extra_retention': extra_retention,
                'retention': {'missing': retain_m, 'split': retain_d, 'junction': retain_j}}


def _extra_for_sample(sample, source_path, config):
    """Old reports are remeasured from snapshots; never assume missing extra=0."""
    import numpy as np
    if 'extra' in sample:
        return float(sample['extra']), None, 'provided_value'
    if config.extra_weight == 0:
        return 0., None, 'disabled'
    for key in ('npz', 'input_npz'):
        value = sample.get(key)
        if not value:
            continue
        paths = [Path(value)]
        if source_path and not paths[0].is_absolute():
            paths.append(source_path.parent/paths[0])
        for path in paths:
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as arrays:
                target_key = next((key for key in ('reference_image', 'reference', 'reference_geometry', 'reference_strokes')
                                   if key in arrays), None)
                if 'image' not in arrays or target_key not in arrays:
                    continue
                image = torch.tensor(arrays['image'], dtype=torch.float32)
                target = torch.tensor(arrays[target_key], dtype=torch.float32)
            metric = ExtraInkMetric(target, radius=config.extra_radius, transition=config.extra_transition)
            with torch.no_grad():
                result = metric(image)
            return float(result['extra'][0]), {'image': image.squeeze().numpy(),
                'extra_map': result['extra_map'][0, 0].numpy(),
                'support': result['support'][0, 0].numpy()}, f'{path}::{target_key}'
    raise ValueError('Cannot assess added ink: keep the NPZ containing image and reference_geometry/reference_strokes. '
                     'Do not treat absent extra measurements as zero; extra_weight=0 explicitly opts out.')


def _image_for_sample(sample, source_path):
    """Prefer the scored input snapshot; never substitute a reference/montage."""
    import numpy as np
    from PIL import Image, ImageOps

    def candidates(value):
        if not value:
            return []
        path = Path(value)
        choices = [path]
        if not path.is_absolute() and source_path:
            choices.append(source_path.parent/path)
        return choices

    for key in ('npz', 'input_npz'):
        for path in candidates(sample.get(key)):
            if not path.is_file() or path.suffix.lower() != '.npz':
                continue
            with np.load(path, allow_pickle=False) as data:
                if 'image' not in data:
                    continue
                ink = np.asarray(data['image'], dtype=np.float32)
            if ink.ndim == 3 and ink.shape[0] == 1:
                ink = ink[0]
            if ink.ndim != 2 or not np.isfinite(ink).all() or (ink < 0).any() or (ink > 1).any():
                continue
            # Saved arrays use ink=1. Display black ink on a white background.
            content = np.rint(255*(1-ink)).astype(np.uint8)
            return Image.fromarray(content).convert('RGB'), str(path), 'scored_input_array'
    for path in candidates(sample.get('input')):
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff') and path.is_file():
            with Image.open(path) as image:
                return ImageOps.exif_transpose(image).convert('RGB'), str(path), 'original_file'
    return None, None, 'unavailable'


def _score_gallery(samples, scored, destination, source_path, columns, per_page, show):
    import base64
    import html
    import io
    import unicodedata
    import matplotlib.pyplot as plt

    cards, loaded, pngs = [], [], []
    for number, (sample, result) in enumerate(zip(samples, scored), 1):
        image, path, kind = _image_for_sample(sample, source_path)
        result.update(image_available=image is not None, image_source=path, image_kind=kind)
        name = str(sample.get('input') or sample.get('input_npz') or f'Image {number}').replace('\\', '/').rsplit('/', 1)[-1]
        loaded.append((image, name, result))
        if image is None:
            picture = '<div class="indicator-no-image">사진 경로를 찾지 못했습니다</div>'
        else:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            picture = f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(name, quote=True)}">'
        detail_rows = ''.join(f"<tr><td>{html.escape(str(s['stroke']))}</td><td>{s['stroke_score']:.1f}</td>"
                              f"<td>{s['missing']:.3g}</td><td>{s['split']:.3g}</td><td>{s['junction']:.3g}</td></tr>"
                              for s in result['strokes'])
        note = '평가에 사용한 입력 배열' if kind == 'scored_input_array' else '입력 사진'
        extra_detail = ''
        if result.get('extra_overlay'):
            encoded = base64.b64encode(Path(result['extra_overlay']).read_bytes()).decode('ascii')
            extra_detail = ('<details><summary>추가 잉크 위치 보기</summary>'
                f'<img src="data:image/png;base64,{encoded}" alt="추가 잉크 위치">'
                '<p>빨간색은 기준 허용 영역 밖의 잉크입니다.</p></details>')
        extra_note = (f"추가 잉크 {100*result['extra']:.1f}% · 추가 감점 전 {result['score_before_extra']:.1f}점"
                      if result['extra_assessed'] else '추가 잉크 검사 꺼짐')
        cards.append(f'<article class="indicator-card">{picture}'
                     f'<div class="indicator-name">{number}. {html.escape(name)}</div>'
                     f'<div class="indicator-score">{result["structural_score"]:.1f}<span> / 100</span></div>'
                     f'<div class="indicator-note">{note if image is not None else "이미지 없는 수치 입력"}</div>'
                     f'<div class="indicator-note">{extra_note}</div>{extra_detail}'
                     '<details><summary>획별 점수 보기</summary><table><thead><tr><th>획</th><th>점수</th>'
                     f'<th>누락</th><th>분리</th><th>정션</th></tr></thead><tbody>{detail_rows}</tbody></table></details></article>')
    style = f'''<style>
.indicator-gallery {{font-family:system-ui,sans-serif;color:#202124;}}
.indicator-grid {{display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:16px;}}
.indicator-card {{border:1px solid #ddd;border-radius:12px;padding:16px;background:white;min-width:0;}}
.indicator-card img,.indicator-no-image {{width:100%;height:240px;object-fit:contain;background:white;}}
.indicator-no-image {{display:flex;align-items:center;justify-content:center;color:#777;}}
.indicator-name {{font-size:14px;overflow-wrap:anywhere;margin:10px 0 4px;}}
.indicator-score {{font-size:36px;font-weight:700;}} .indicator-score span {{font-size:16px;font-weight:400;color:#666;}}
.indicator-note {{font-size:12px;color:#666;margin:4px 0 12px;}}
.indicator-gallery table {{font-size:12px;border-collapse:collapse;width:100%;margin-top:10px;}}
.indicator-gallery td,.indicator-gallery th {{padding:5px;text-align:right;border-bottom:1px solid #eee;}}
@media(max-width:650px){{.indicator-grid{{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
</style>'''
    fragment = style + '<section class="indicator-gallery"><p>사진별 구조 점수 · 누락·분리·정션·추가 잉크를 감점합니다.</p>'
    fragment += '<div class="indicator-grid">' + ''.join(cards) + '</div></section>'
    gallery_path = destination/'images_and_scores.html'
    gallery_path.write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'+fragment+'</html>', encoding='utf-8')
    for offset in range(0, len(loaded), per_page):
        page = loaded[offset:offset+per_page]
        cols = min(columns, len(page))
        rows = math.ceil(len(page)/cols)
        figure, axes = plt.subplots(rows, cols, squeeze=False, figsize=(3.5*cols, 4.2*rows), layout='constrained')
        for ax in axes.flat:
            ax.axis('off')
        for ax, (image, name, result) in zip(axes.flat, page):
            if image is not None:
                ax.imshow(image)
            else:
                ax.text(.5, .5, 'Image unavailable', transform=ax.transAxes, ha='center', color='#777')
            safe_name = unicodedata.normalize('NFC', name).encode('ascii', 'backslashreplace').decode('ascii')
            if len(safe_name) > 58:
                safe_name = safe_name[:55]+'...'
            ax.set_title(f'{result["sample"]}. {safe_name}\n{result["structural_score"]:.1f} / 100', fontsize=12)
        figure.suptitle('Image : structural score (0..100)', fontsize=14)
        path = destination/f'images_and_scores_{offset//per_page+1:02d}.png'
        figure.savefig(path, dpi=150)
        plt.close(figure)
        pngs.append(str(path))
    if show:
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if get_ipython() is not None:
                display(HTML(fragment))
        except ImportError:
            pass
    return str(gallery_path), pngs


def score_structure_report(report, config=None, save_dir=None, *, show_images=True,
                           images_per_row=3, images_per_page=12, show_stroke_table=False):
    """Score an existing structure_report.json/dict without rerunning extraction.

    Returns an auditable report and saves glyph_scores.csv, stroke_scores.csv,
    indicator_report.json. Only the three named error columns enter the formula.
    """
    import pandas as pd

    if any(not isinstance(v, int) or v < 1 for v in (images_per_row, images_per_page)):
        raise ValueError('Gallery row/page counts must be positive integers')
    cfg = config or IndicatorConfig()
    source_path = None
    if isinstance(report, (str, Path)):
        source_path = Path(report)
        report = json.loads(source_path.read_text(encoding='utf-8-sig'))
    if not isinstance(report, dict) or not report.get('results'):
        raise ValueError('Provide structure_results or a structure_report.json with nonempty results')
    if save_dir is None:
        base = source_path.parent if source_path else Path(report['results'][0].get('npz', '.')).parent
        save_dir = base / ('indicator_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=False)
    scorer = StructuralIndicator(cfg)
    glyph_rows, stroke_rows, results = [], [], []
    for number, sample in enumerate(report['results'], 1):
        strokes = sample.get('strokes', [])
        if not strokes:
            raise ValueError(f'Sample {number} has no per-stroke measurements')
        missing, split, junction = [torch.tensor([s[key] for s in strokes], dtype=torch.float64)
                                     for key in ('missing', 'split', 'junction')]
        extra, extra_maps, extra_source = _extra_for_sample(sample, source_path, cfg)
        with torch.no_grad():
            output = scorer(missing, split, junction, extra=[extra])
        identity = {'sample': number, 'csv_row': sample.get('csv_row'),
                    'input': sample.get('input') or sample.get('input_npz') or ''}
        glyph = {**identity, 'structural_score': float(output['score'][0]),
                 'score_before_extra': float(output['base_score'][0]), 'extra': extra,
                 'extra_retention': float(output['extra_retention'][0]),
                 'extra_assessed': extra_source != 'disabled',
                 'mean_stroke_score': float(output['mean_score'][0]),
                 'worst_stroke_score': float(output['worst_score'][0]),
                 'stroke_count': len(strokes)}
        glyph_rows.append(glyph)
        scored_strokes = []
        for k, stroke in enumerate(strokes):
            row = {**identity, 'stroke': stroke.get('stroke', k+1),
                   'missing': float(missing[k]), 'split': float(split[k]), 'junction': float(junction[k]),
                   'stroke_score': float(output['stroke_scores'][0, k]),
                   **{name+'_retention': float(value[0, k]) for name, value in output['retention'].items()}}
            stroke_rows.append(row)
            scored_strokes.append(row)
        extra_overlay = None
        if extra_maps is not None:
            import numpy as np
            from PIL import Image
            np.savez_compressed(destination/f'extra_{number:04d}.npz', **extra_maps)
            background = np.repeat((1-extra_maps['image'])[..., None], 3, axis=-1)
            alpha = np.clip(extra_maps['extra_map'], 0, 1)[..., None]
            colored = background*(1-alpha) + np.array([1., 0., 0.])*alpha
            extra_overlay = destination/f'extra_{number:04d}.png'
            Image.fromarray(np.rint(colored*255).astype(np.uint8)).save(extra_overlay)
        results.append({**glyph, 'extra_source': extra_source,
                        'extra_overlay': str(extra_overlay) if extra_overlay else None, 'strokes': scored_strokes})
    pd.DataFrame(glyph_rows).to_csv(destination / 'glyph_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stroke_rows).to_csv(destination / 'stroke_scores.csv', index=False, encoding='utf-8-sig')
    gallery, gallery_images = _score_gallery(report['results'], results, destination, source_path,
                                             images_per_row, images_per_page, show_images)
    output_report = {'indicator': 'structural_indicator_v2_added_ink', 'range': [0, 100],
                     'uses_only': ['missing', 'split', 'junction'] + (['extra'] if cfg.extra_weight > 0 else []), 'trained': False,
                     'calibrated_to_human_quality': False, 'config': asdict(cfg),
                     'measurement_config': report.get('config'), 'source_report': str(source_path) if source_path else None,
                     'save_dir': str(destination), 'gallery_html': gallery,
                     'gallery_images': gallery_images, 'results': results}
    (destination / 'indicator_report.json').write_text(json.dumps(output_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Structural indicator (0..100; higher is better; uncalibrated):')
    display = pd.DataFrame(glyph_rows)
    display['input'] = display['input'].map(lambda p: str(p).replace('\\', '/').rsplit('/', 1)[-1])
    print(display.round(2).to_string(index=False))
    if show_stroke_table:
        print('\nPer-stroke scores:')
        print(pd.DataFrame(stroke_rows)[['sample', 'stroke', 'missing', 'split', 'junction', 'stroke_score']]
              .to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    print('Image : score gallery:', gallery)
    print('Saved:', destination)
    return output_report
