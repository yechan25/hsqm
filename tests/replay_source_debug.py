"""Replay a diagnostic ZIP; supplied arrays remain in ignored outputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import torch
from stroke_model.structural_indicator import score_structure_report


def replay(archive,destination):
    torch.set_num_threads(1)
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        report=json.loads(zipped.read('structure_report.json'))
        old=json.loads(zipped.read('indicator_report.json'))
        for sample in report['results']:
            name=sample['npz']
            if not re.fullmatch(r'arrays/sample_\d{4}\.npz',name):
                raise ValueError('Expected evaluated sample snapshots only')
            target=destination/name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(zipped.read(name))
    (destination/'structure_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    result=score_structure_report(destination/'structure_report.json',measurement_mode='source',
                                  save_dir=destination/'scores',show_images=False)
    rows=[{'sample':new['sample'],'input':Path(new['input']).name,
           'before':previous['structural_score'],'after':new['structural_score'],
           'registration_dice':new['registration']['dice']}
          for previous,new in zip(old['results'],result['results'])]
    pd.DataFrame(rows).to_csv(destination/'before_after.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True)
    parser.add_argument('--save-dir',default=str(ROOT/'outputs/source_replay'/datetime.now().strftime('%Y%m%d_%H%M%S')))
    args=parser.parse_args()
    replay(args.archive,args.save_dir)
