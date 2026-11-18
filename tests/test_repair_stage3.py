import ast
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from repair_audit.stage3 import (choose_rows, validate_rows, materialize_inputs,
                                shape_comparison, write_stage3_tables)
from diffusion_model.score_pilot import training_catalog
from diffusion_model.selectable_adaptive_sweep import prepare_groups, generate_cases


class Stage3Tests(unittest.TestCase):
    def test_selection_excludes_previous_content_and_rejects_duplicates(self):
        catalog = [dict(manifest_row=i, reference_sha256=str(i)) for i in range(1, 22)]
        catalog[-1]['reference_sha256'] = '1'
        rows = choose_rows(catalog)
        self.assertEqual(len(rows), 12)
        self.assertFalse(set(rows) & {1, 2, 3, 5, 21})
        self.assertEqual(len(validate_rows(catalog, rows)), 3)
        for invalid in [[4, 4, 6, 7], [4, 6, 7, 21], [4, 6, 7, 999]]:
            with self.assertRaises(ValueError):
                validate_rows(catalog, invalid)

    def test_generation_controls_failure_retention_and_review_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = []
            for i in range(8):
                ink = np.zeros((1, 16, 16), np.float32)
                ink[:, 2:14, i+2:i+4] = 1
                np.savez_compressed(root/f'{i}.npz', reference=ink, strokes=ink)
                manifest.append(dict(path=f'{i}.npz', split='train', char_id=f'id{i}'))
            (root/'manifest.json').write_text(json.dumps(manifest))
            out = root/'output'; out.mkdir()
            groups = prepare_groups(root/'manifest.json', [[1,2,3,4], [5,6,7,8]], out/'sources')
            class FakeGenerator:
                scheduler = SimpleNamespace(config={})
                def __call__(self, **kw):
                    return SimpleNamespace(images=[kw['image'].copy()], nsfw_content_detected=[False])
            generate_cases(groups[:1], pipeline=FakeGenerator(), strengths=(.65,), seeds=(100,), steps=100)
            (Path(groups[1]['root'])/'generation_failure.json').write_text(json.dumps({'error':'test failure'}))
            cases, failures = materialize_inputs(groups, out, 'a'*64)
            self.assertEqual([c['kind'] for c in cases], ['normal_control', 'generated', 'normal_control'])
            self.assertEqual(len(failures), 1)
            self.assertEqual(cases[1]['generation_seed'], 100)
            case = cases[0]; audit = Path(case['directory'])/'audit'; audit.mkdir()
            for arm in ['input', 'vae_roundtrip', 'redenoise_only', 'cyclic_guided', 'boundary_guarded']:
                with Image.open(Path(case['directory'])/'input.png') as image:
                    image.save(audit/f'{arm}.png')
            shape_comparison(case)
            with Image.open(audit/'shape_comparison.png') as image:
                self.assertEqual(image.size, (960, 902))
            metrics = [dict(case_id=case['case_id'], arm='boundary_guarded', kind='normal_control', slot=1)]
            write_stage3_tables(out, cases, metrics, failures)
            path = out/'human_review.csv'
            with path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
            rows[0]['endpoint_deformed'] = 'yes'
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            write_stage3_tables(out, cases, metrics, failures)
            with path.open(encoding='utf-8-sig', newline='') as f:
                first = next(csv.DictReader(f))
            self.assertEqual(first['endpoint_deformed'], 'yes')
            self.assertEqual(first['original_manifest_row'], '1')
            self.assertEqual(first['error_reduced'], 'n/a')
            self.assertEqual(training_catalog(groups[0]['manifest'])[0]['char_id'], 'id0')

    def test_standalone_notebook_pins_stage2_and_contains_complete_new_runtime(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/repair_audit/stage3.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 7)
        for source in code:
            ast.parse(source)
        assigns = {t.id: ast.literal_eval(n.value) for n in ast.parse(code[0]).body
                   if isinstance(n, ast.Assign) for t in n.targets
                   if isinstance(t, ast.Name) and t.id in ('RUNTIME_HELPERS', 'SOURCE_HASHES')}
        helpers = assigns['RUNTIME_HELPERS']
        for name, source in helpers.items():
            ast.parse(source)
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), assigns['SOURCE_HASHES'][name])
        for name in ['repair_audit/stage1.py', 'repair_audit/stage3.py', 'stroke_model/boundary_guarded_repair.py']:
            self.assertEqual(helpers[name], (root/name).read_text(encoding='utf-8'))
        old = json.loads((root/'notebooks/repair_audit/stage2.ipynb').read_text(encoding='utf-8'))
        old_setup = next(''.join(c['source']) for c in old['cells'] if c['cell_type']=='code')
        old_helpers = next(ast.literal_eval(n.value) for n in ast.parse(old_setup).body
                           if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
        self.assertTrue(all(helpers[k] == v for k, v in old_helpers.items()))
        self.assertIn('boundary_guard=True', code[5])
        self.assertNotIn('ARCHIVE =', '\n'.join(code))


if __name__ == '__main__':
    unittest.main()
