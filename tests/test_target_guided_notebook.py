"""Verify standalone sources and case-loop orchestration without Colab/GPU."""
import ast
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
import zipfile
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]


class TargetNotebookTests(unittest.TestCase):
    def notebook(self):
        spec = importlib.util.spec_from_file_location('target_builder', ROOT/'notebooks/repair_audit/build_target_guided_repair.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return json.loads(module.build().read_text(encoding='utf-8'))

    def test_embedded_runtime_matches_sources_and_cells_compile(self):
        code = [''.join(c['source']) for c in self.notebook()['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),5)
        for cell in code: compile(cell,'<colab>','exec')
        assignments = {n.targets[0].id:n.value for n in ast.parse(code[0]).body
                       if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        helpers = ast.literal_eval(assignments['RUNTIME_HELPERS'])
        hashes = ast.literal_eval(assignments['SOURCE_HASHES'])
        for name, source in helpers.items():
            self.assertEqual(source,(ROOT/name).read_text(encoding='utf-8'))
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),hashes[name])
            compile(source,name,'exec')
        self.assertIn('stroke_model/stroke_structure.py',helpers)
        self.assertIn('diffusion_model/stroke_projected_diffusion.py',helpers)

    def test_case_loop_cache_fallback_backup_and_completed_skip(self):
        code = next(''.join(c['source']) for c in self.notebook()['cells']
                    if c['cell_type']=='code' and '# 4.' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'results'; output.mkdir()
            archive = Path(tmp)/'input.zip'
            with zipfile.ZipFile(archive,'w'): pass
            cases=[]
            for name in ['generated','normal']:
                root=output/name;root.mkdir();Image.new('RGB',(32,32),'white').save(root/'input.png')
                cases.append(dict(case_id=name,directory=str(root)))
            calls=[];fallback=[]
            class Extractor:
                def to(self,device): return self
            def repair(pipe,extractor,source,conditions,result,**kwargs):
                calls.append(result)
                self.assertIsNotNone(kwargs['cached_masks'])
                (result/'target_plan').mkdir(parents=True);(result/'reverse').mkdir()
                for name in ['target_plan/aligned_target.png','reverse/redenoise_only.png','reverse/stroke_projected.png']:
                    source.save(result/name)
                (result/'pipeline.json').write_text('{}')
                kwargs['progress']('stroke_projected',1,1,{})
                return SimpleNamespace(report=dict(strokes=[dict(edit_strength=1)])),dict(actual_steps=1)
            def extract(*args):
                fallback.append(True);return [np.ones((1,8,8),np.float32)]*4
            env=dict(Path=Path,datetime=datetime,shutil=shutil,zipfile=zipfile,json=json,
                gc=gc,torch=torch,time=time,traceback=traceback,Image=Image,ImageDraw=ImageDraw,
                display=lambda x:None,OUTPUT=output,DRIVE_RESULT_DIR=str(Path(tmp)/'drive'),
                CASES=cases,ARCHIVE=archive,extractor=Extractor(),pipe=object(),
                _cached_masks=lambda z,c,s,cond:[np.ones((1,8,8))]*4 if c['case_id']=='generated' else None,
                extract_sheet_masks=extract,load_conditions=lambda root:[],repair_generated_sheet=repair,
                SCORES={},PLAN_CONFIG={},TargetPlanConfig=lambda **k:None,
                REPAIR_STEPS=60,REPAIR_STRENGTH=.25,REPAIR_SEED=2026,
                json_write=lambda p,x:p.write_text(json.dumps(x)))
            exec(code,env)
            self.assertEqual(len(calls),2);self.assertEqual(len(fallback),1)
            self.assertTrue((Path(tmp)/'drive/results_checkpoint.zip').is_file())
            for case in cases:self.assertTrue((Path(case['directory'])/'diffusion_comparison.png').is_file())
            exec(code,env)
            self.assertEqual(len(calls),2)


if __name__ == '__main__': unittest.main()
