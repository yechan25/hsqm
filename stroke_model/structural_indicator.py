"""Deterministic structural indicator using ONLY missing, split and junction.

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


@dataclass(frozen=True)
class IndicatorConfig:
    missing_weight: float = 1.0
    split_weight: float = 1.0
    junction_weight: float = 1.0
    split_scale: float = 0.2
    junction_scale: float = 0.01
    worst_stroke_weight: float = 0.5
    eps: float = 1e-6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if min(self.missing_weight, self.split_weight, self.junction_weight) < 0:
            raise ValueError('Weights must be nonnegative')
        if self.split_scale <= 0 or self.junction_scale <= 0 or not 0 < self.eps < 1:
            raise ValueError('Scales must be positive; eps must be in (0,1)')
        for weight, scale in ((self.split_weight, self.split_scale), (self.junction_weight, self.junction_scale)):
            if not math.isfinite(weight/scale) or weight/scale > torch.finfo(torch.float32).max:
                raise ValueError('weight/scale must fit a finite float32 coefficient')
        if not 0 <= self.worst_stroke_weight <= 1:
            raise ValueError('worst_stroke_weight must be in [0,1]')


class StructuralIndicator(nn.Module):
    """Map M,D,J to stroke and glyph scores without training.

    Inputs: K or B,K. Missing is in [0,1]; split/junction are nonnegative.
    q_k = 100*(1-M_k)^wM * exp(-wD*D_k/sD-wJ*J_k/sJ)
    Q = (1-alpha)*mean(q_k) + alpha*min(q_k), over active reference strokes.
    min has subgradients; the full pipeline is not everywhere smooth.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or IndicatorConfig()

    def forward(self, missing, split, junction, active=None):
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
        score = (1-cfg.worst_stroke_weight)*mean + cfg.worst_stroke_weight*worst
        return {'score': score, 'stroke_scores': stroke_scores, 'mean_score': mean,
                'worst_score': worst, 'penalty': 1-score/100, 'active': active,
                'retention': {'missing': retain_m, 'split': retain_d, 'junction': retain_j}}


def score_structure_report(report, config=None, save_dir=None):
    """Score an existing structure_report.json/dict without rerunning extraction.

    Returns an auditable report and saves glyph_scores.csv, stroke_scores.csv,
    indicator_report.json. Only the three named error columns enter the formula.
    """
    import pandas as pd

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
        with torch.no_grad():
            output = scorer(missing, split, junction)
        identity = {'sample': number, 'csv_row': sample.get('csv_row'),
                    'input': sample.get('input') or sample.get('input_npz') or ''}
        glyph = {**identity, 'structural_score': float(output['score'][0]),
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
        results.append({**glyph, 'strokes': scored_strokes})
    pd.DataFrame(glyph_rows).to_csv(destination / 'glyph_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stroke_rows).to_csv(destination / 'stroke_scores.csv', index=False, encoding='utf-8-sig')
    output_report = {'indicator': 'missing_split_junction_v1', 'range': [0, 100],
                     'uses_only': ['missing', 'split', 'junction'], 'trained': False,
                     'calibrated_to_human_quality': False, 'config': asdict(cfg),
                     'measurement_config': report.get('config'), 'source_report': str(source_path) if source_path else None,
                     'save_dir': str(destination), 'results': results}
    (destination / 'indicator_report.json').write_text(json.dumps(output_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Structural indicator (0..100; higher is better; uncalibrated):')
    display = pd.DataFrame(glyph_rows)
    display['input'] = display['input'].map(lambda p: str(p).replace('\\', '/').rsplit('/', 1)[-1])
    print(display.round(2).to_string(index=False))
    print('\nPer-stroke scores:')
    print(pd.DataFrame(stroke_rows)[['sample', 'stroke', 'missing', 'split', 'junction', 'stroke_score']]
          .to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    print('Saved:', destination)
    return output_report
