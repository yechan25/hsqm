"""Replay a user-provided ZIP locally; raw arrays stay in git-ignored outputs.

python tests/replay_junction_debug.py --archive C:/path/junction_debug.zip
These four cases are development regressions, NOT an independent validation set.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.ndimage import label
import torch

from stroke_model.stroke_structure import StructureConfig, StrokeStructureLoss
from stroke_model.score_stroke_structure import score_saved_strokes


def replay(archive, destination):
    torch.set_num_threads(1)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = destination / 'inputs'
    inputs.mkdir()
    paths, before = [], {}
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not re.fullmatch(r'row_\d{4}_structure\.npz', name):
                raise ValueError('Expected flat row_NNNN_structure.npz entries only')
            with np.load(zipped.open(name), allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            path = inputs / name
            np.savez_compressed(path, **arrays)
            paths.append(path)
            row = int(name[4:8])
            for k in range(len(arrays['soft_masks'])):
                length = max(float(arrays['centerline_maps'][k].sum()), 1.)
                old_j = float(arrays['junction_maps'][k].sum()) / length
                old_count = label(arrays['junction_hard_branches'][k], np.ones((3, 3)))[1]
                before[row, k+1] = {'old_J': old_j, 'old_branch_regions': old_count}
    report = score_saved_strokes(paths, save_dir=destination / 'scored', show=False)
    rows = []
    for result in report['results']:
        number = int(Path(result['input_npz']).name[4:8])
        for stroke in result['strokes']:
            key = number, stroke['stroke']
            rows.append({'row': number, 'stroke': stroke['stroke'], **before[key],
                         'new_J': stroke['junction'], 'new_branches': stroke['branch_regions_max'],
                         'unresolved': stroke['unresolved_regions_max'],
                         'new_split': stroke['split'], 'new_missing': stroke['missing']})
    table = pd.DataFrame(rows)
    table.to_csv(destination / 'before_after.csv', index=False)
    indexed = table.set_index(['row', 'stroke'])
    checks = {}
    if (38, 2) in indexed.index:
        checks['cut_bend_not_confirmed_junction'] = bool(indexed.loc[(38, 2), 'new_branches'] == 0)
    if (41, 4) in indexed.index:
        checks['missing_residue_soft_junction_removed'] = bool(indexed.loc[(41, 4), 'new_J'] < 1e-8)
    for row in (43, 44):
        if (row, 2) in indexed.index:
            checks[f'row_{row}_real_attachment_preserved'] = bool(indexed.loc[(row, 2), 'new_branches'] >= 1)
    if (43, 2) in indexed.index and (38, 2) in indexed.index:
        checks['attachment_soft_error_above_bend'] = bool(indexed.loc[(43, 2), 'new_J'] > indexed.loc[(38, 2), 'new_J'])
    (destination / 'regression_checks.json').write_text(json.dumps(
        {'development_cases_only': True, 'checks': checks, 'config': asdict(StructureConfig())},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nBEFORE / AFTER\n' + table.to_string(index=False, float_format=lambda x: f'{x:.6g}'))
    print('Regression checks:', checks)
    print('Saved:', destination)
    if not all(checks.values()):
        raise AssertionError('An actual-array regression check failed')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / 'outputs/junction_replay' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    replay(args.archive, output)
