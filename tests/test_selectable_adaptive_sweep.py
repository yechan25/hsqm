import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from diffusion_model.selectable_adaptive_sweep import prepare_groups, generate_cases, repair_cases, select_entries, ADAPTIVE_COMMIT


def catalog(root):
    records=[]
    for i in range(8):
        ink=np.zeros((1,32,32),np.float32);ink[0,4:28,4+i*2:7+i*2]=1
        np.savez(root/f'{i}.npz',reference=ink,strokes=ink)
        records.append(dict(path=f'{i}.npz',split='train',char_id=f'known-{i}'))
    manifest=root/'manifest.json';manifest.write_text(json.dumps(records));return manifest


class Pipeline:
    scheduler=SimpleNamespace(config={})
    def __init__(self):self.calls=[]
    def __call__(self,**kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[kwargs['image'].copy()],nsfw_content_detected=[len(self.calls)==2])


class Extractor(torch.nn.Module):
    def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.))
    def forward(self,ink,reference,strokes):
        p=torch.ones_like(strokes)/strokes.shape[1]
        return dict(probabilities=p,masks=p*ink,hints=strokes)


class SelectableTests(unittest.TestCase):
    def test_exactly_four_distinct_rows_and_order_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=catalog(root)
            self.assertEqual([e['manifest_row'] for e in select_entries(manifest,[8,2,5,1])],[8,2,5,1])
            for rows in ([1,1,2,3],[1,2,3],[1,2,3,99]):
                with self.assertRaises(ValueError):select_entries(manifest,rows)
            groups=prepare_groups(manifest,[[8,2,5,1],[3,4,6,7]],root/'runs')
            self.assertEqual(groups[0]['original_rows'],[8,2,5,1])
            local=json.loads(Path(groups[0]['manifest']).read_text())
            self.assertEqual([r['original_manifest_row'] for r in local],[8,2,5,1])

    def test_every_strength_seed_group_and_control_is_kept_and_generation_is_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=catalog(root)
            groups=prepare_groups(manifest,[[1,2,3,4],[5,6,7,8]],root/'runs')
            pipe=Pipeline()
            cases=generate_cases(groups,strengths=(.45,.65),seeds=(100,101),steps=10,canvas_size=64,pipeline=pipe)
            self.assertEqual(len(cases),10)
            self.assertEqual(len(pipe.calls),8)
            self.assertEqual(sum(c['control'] for c in cases),2)
            self.assertEqual(sum(c['safety_filtered'] is True for c in cases),1)
            self.assertEqual(len({c['case_id'] for c in cases}),10)
            self.assertTrue(all(c['image'] is pipe.calls[0]['image'] for c in pipe.calls[:4]))
            self.assertTrue(all(c['image'] is pipe.calls[4]['image'] for c in pipe.calls[4:]))
            self.assertIsNot(pipe.calls[0]['image'],pipe.calls[4]['image'])

    def test_partial_generation_failure_is_recorded_without_hiding_existing_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);groups=prepare_groups(catalog(root),[[1,2,3,4]],root/'runs')
            class Broken(Pipeline):
                def __call__(self,**kwargs):raise RuntimeError('generation failed')
            cases=generate_cases(groups,strengths=(.65,),seeds=(100,),steps=10,canvas_size=64,pipeline=Broken())
            self.assertEqual(len(cases),1)  # The normal input control survives.
            self.assertTrue((Path(groups[0]['root'])/'generation_failure.json').is_file())

    def test_each_case_is_fitted_and_repaired_without_modifying_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);groups=prepare_groups(catalog(root),[[1,2,3,4]],root/'runs')
            cases=generate_cases(groups,strengths=(.65,),seeds=(100,),steps=10,canvas_size=64,pipeline=Pipeline())
            before=[Path(c['input']).read_bytes() for c in cases]
            calls=[]
            def fake_run(source,candidate,metadata,output,**kwargs):
                calls.append((metadata,kwargs));Path(output).mkdir()
                source.save(Path(output)/'redenoise_only.png');source.save(Path(output)/'adaptive_guided.png')
                return dict(metrics={})
            with patch('diffusion_model.selectable_adaptive_sweep.run_pair',side_effect=fake_run):
                rows=repair_cases(cases,Extractor(),checkpoint_sha256='exact',revision='model-revision')
            self.assertEqual([r['status'] for r in rows],['completed','completed'])
            self.assertEqual(len(calls),2)
            for metadata,kwargs in calls:
                self.assertEqual(metadata['adaptive_runtime_commit'],ADAPTIVE_COMMIT)
                self.assertEqual(metadata['original_manifest_rows'],[1,2,3,4])
                self.assertEqual(kwargs['guidance_mode'],'adaptive_ink')
            self.assertEqual(before,[Path(c['input']).read_bytes() for c in cases])

    def test_repair_failure_remains_in_summary_and_other_cases_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);groups=prepare_groups(catalog(root),[[1,2,3,4]],root/'runs')
            cases=generate_cases(groups,strengths=(.65,),seeds=(100,),steps=10,canvas_size=64,pipeline=Pipeline())
            with patch('diffusion_model.selectable_adaptive_sweep.fit_case',side_effect=RuntimeError('fit failed')):
                rows=repair_cases(cases,Extractor(),checkpoint_sha256='exact',revision='revision')
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['status']=='failed' for r in rows))
            self.assertEqual(len(json.loads((root/'runs/all_results.json').read_text())),2)

    def test_notebook_embeds_exact_historical_core_and_exposes_selection_and_sweeps(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/selectable_adaptive_sweep.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),5)
        for source in code:ast.parse(source)
        assignments=[n for n in ast.parse(code[0]).body if isinstance(n,ast.Assign)]
        values={t.id:ast.literal_eval(n.value) for n in assignments for t in n.targets
            if isinstance(t,ast.Name) and t.id in ('PINNED_HASHES','RUNTIME_HELPERS')}
        for path,expected in values['PINNED_HASHES'].items():
            pinned=subprocess.check_output(['git','show',f'{ADAPTIVE_COMMIT}:{path}'],cwd=root)
            actual=values['RUNTIME_HELPERS'][path].encode()
            self.assertEqual(pinned,actual)
            self.assertEqual(hashlib.sha256(actual).hexdigest(),expected)
        self.assertNotIn('check_proposal',values['RUNTIME_HELPERS']['diffusion_model/fitted_candidate_diffusion.py'])
        self.assertIn('selectors=',code[2]);self.assertIn('SELECTED_GROUPS',code[2])
        self.assertIn('STRENGTHS=(.45,.65,.85)',code[3])
        self.assertIn('REPAIR_CASE_IDS=None',code[4])


if __name__=='__main__':unittest.main()
