import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image, ImageDraw
import torch

from repair_audit.selectable_target_inputs import prepare_selected_sheet
from repair_audit.fresh_target_inputs import generate_fresh_cases
from repair_audit.stage1 import file_hash, json_write, load_conditions

ROOT=Path(__file__).resolve().parents[1]


class SelectableTests(unittest.TestCase):
    def notebook(self):
        folder=ROOT/'notebooks/repair_audit';sys.path.insert(0,str(folder))
        try:
            spec=importlib.util.spec_from_file_location('selectable_builder',folder/'build_selectable_target_repair.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return json.loads(module.build().read_text(encoding='utf-8'))
        finally:sys.path.remove(str(folder))

    def fixture(self,root):
        entries=[]
        for i in range(4):
            ref=np.zeros((1,8,8),np.float32);ref[0,1+i:3+i,1:7]=1
            path=root/f'row_{i}.npz';np.savez(path,reference=ref,strokes=ref)
            entries.append(dict(manifest_row=i+1,path=str(path),char_id=str(i)))
        return entries

    def test_six_cells_no_experiment_zip_and_exact_embedded_sources(self):
        codes=[''.join(c['source']) for c in self.notebook()['cells'] if c['cell_type']=='code']
        self.assertEqual(len(codes),6)
        for code in codes:
            compile(code,'<colab>','exec')
            tree=ast.parse(code)
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                                 and n.func.id=='read_archive' for n in ast.walk(tree)))
            self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                                 and n.func.attr=='upload' for n in ast.walk(tree)))
            self.assertFalse(any(isinstance(n,ast.Name) and n.id=='CASE_IDS' for n in ast.walk(tree)))
        assignments={n.targets[0].id:n.value for n in ast.parse(codes[0]).body
                     if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        helpers=ast.literal_eval(assignments['RUNTIME_HELPERS'])
        hashes=ast.literal_eval(assignments['SOURCE_HASHES'])
        for name,code in helpers.items():
            self.assertEqual(code,(ROOT/name).read_text(encoding='utf-8'))
            self.assertEqual(hashlib.sha256(code.encode()).hexdigest(),hashes[name])
        self.assertIn('stroke_model/reference_only_data.py',helpers)

    def test_freeze_order_duplicates_and_invalid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=self.fixture(root)
            cases=prepare_selected_sheet(catalog,[4,1,4,2],root/'chosen',size=32)
            self.assertEqual(cases[0]['original_rows'],[4,1,4,2])
            conditions=load_conditions(cases[0]['directory'])
            self.assertTrue(np.array_equal(conditions[0]['strokes'],conditions[2]['strokes']))
            self.assertFalse(np.array_equal(conditions[0]['strokes'],conditions[1]['strokes']))
            for rows in [[1,2],[1,2,3,99]]:
                with self.assertRaises(ValueError):prepare_selected_sheet(catalog,rows,root/'invalid')
                self.assertFalse((root/'invalid').exists())

    def test_new_selection_run_without_model_reload_preserves_previous_run(self):
        code=next(''.join(c['source']) for c in self.notebook()['cells']
                  if c['cell_type']=='code' and '# 5. 현재 선택' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalog=self.fixture(root);selected=[1,2,3,4];calls=[]
            checkpoint=root/'best.pt';checkpoint.write_bytes(b'fixture')
            manifest=root/'manifest.json';manifest.write_text('[]')
            class LocalPath:
                def __new__(cls,path):return root if str(path)=='/content' else Path(path)
            class Pipe:
                vae=SimpleNamespace(to=lambda **kwargs:None)
                def __call__(self,**kw):return SimpleNamespace(images=[kw['image'].copy()])
            class Extractor:
                def to(self,device):return self
            def repair(pipe,extractor,source,conditions,result,**kwargs):
                calls.append(result)
                (result/'target_plan').mkdir(parents=True);(result/'reverse').mkdir()
                source.save(result/'target_plan/aligned_target.png')
                source.save(result/'reverse/redenoise_only.png');source.save(result/'reverse/stroke_projected.png')
                (result/'pipeline.json').write_text('{}')
                return SimpleNamespace(report={'strokes':[{'edit_strength':1}]}),{'actual_steps':1}
            fake_torch=SimpleNamespace(float16=torch.float16,float32=torch.float32,
                cuda=SimpleNamespace(get_device_name=lambda i:'fixture',empty_cache=lambda:None))
            env=dict(Path=LocalPath,datetime=datetime,json=json,np=np,Image=Image,ImageDraw=ImageDraw,
                gc=gc,shutil=shutil,traceback=traceback,time=time,torch=fake_torch,display=lambda x:None,
                selected_rows=lambda:list(selected),CATALOG=catalog,CANVAS_SIZE=32,EXTRACTOR_SIZE=8,
                prepare_selected_sheet=prepare_selected_sheet,load_conditions=load_conditions,
                SCORES_JSON='',SOURCE_HASHES={},PREPARATION={},CHECKPOINT=checkpoint,TRAIN_MANIFEST=manifest,
                MODEL_ID='fixture',REVISION='fixture',GENERATION_STRENGTHS=[.65],GENERATION_SEEDS=[100],
                GENERATION_STEPS=60,REPAIR_STEPS=60,REPAIR_STRENGTH=.25,REPAIR_SEED=2026,PLAN_CONFIG={},
                file_hash=file_hash,json_write=json_write,SCHEDULER_CONFIG={},VERSIONS={},DRIVE_RESULT_DIR=str(root/'drive'),
                generate_fresh_cases=generate_fresh_cases,pipe=Pipe(),extractor=Extractor(),
                extract_sheet_masks=lambda s,c,e:[np.ones((1,8,8))]*4,
                repair_generated_sheet=repair,TargetPlanConfig=lambda **kw:None)
            exec(code,env);first=env['OUTPUT']
            selected[:]=[4,4,1,1]
            exec(code,env);second=env['OUTPUT']
            self.assertNotEqual(first,second);self.assertEqual(len(calls),4)
            first_rows=json.loads((first/'selected_reference/selection.json').read_text())['selected_rows']
            second_rows=json.loads((second/'selected_reference/selection.json').read_text())['selected_rows']
            self.assertEqual(first_rows,[1,2,3,4]);self.assertEqual(second_rows,[4,4,1,1])
            self.assertEqual(len(list((root/'drive').glob('*.zip'))),2)


if __name__=='__main__':unittest.main()
