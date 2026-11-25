import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
from stroke_model.attribute_guidance import straightness_plan

ROOT=Path(__file__).resolve().parents[1]


class AttributeTests(unittest.TestCase):
    def fixture(self):
        yy,xx=np.indices((32,32));center=10+2*np.sin((yy-4)/23*np.pi)
        stroke=(abs(xx-center)<1.7)&(yy>=4)&(yy<=27)
        other=(abs(xx-25)<1.7)&(yy>=5)&(yy<=26)
        masks=np.stack([stroke,other]).astype(np.float32)
        source=Image.fromarray(np.uint8(np.tile(1-masks.max(0),(2,2))*255)).convert('RGB')
        return source,[masks.copy() for _ in range(4)]

    def test_straightens_only_selected_region_and_keeps_endpoint_direction(self):
        source,masks=self.fixture();guide=straightness_plan(source,masks,0,0,amount=1)
        r=guide.report
        self.assertLess(r['tortuosity_after'],r['tortuosity_before'])
        self.assertAlmostEqual(r['tortuosity_after'],1.,places=5)
        self.assertTrue(np.allclose(r['endpoint_positions_before'],r['endpoint_positions_after']))
        original=np.asarray(source,np.float32)/255
        self.assertTrue(np.array_equal(guide.target[~guide.roi],original[~guide.roi]))
        self.assertTrue(np.array_equal(guide.target[32:],original[32:]))
        self.assertTrue(np.array_equal(guide.target[:32,23:28],original[:32,23:28]))
        unchanged=straightness_plan(source,masks,0,0,amount=0)
        self.assertFalse(unchanged.roi.any());self.assertTrue(np.array_equal(unchanged.target,original))

    def test_closed_and_empty_strokes_are_rejected(self):
        source,masks=self.fixture();yy,xx=np.indices((32,32))
        masks[0][0]=(abs(np.hypot(xx-15,yy-15)-8)<1.5)
        with self.assertRaises(ValueError):straightness_plan(source,masks,0,0)
        masks[0][0]=0
        with self.assertRaises(ValueError):straightness_plan(source,masks,0,0)

    def test_notebook_has_six_compilable_cells_and_shared_sampler(self):
        folder=ROOT/'notebooks/repair_audit';sys.path.insert(0,str(folder))
        try:
            spec=importlib.util.spec_from_file_location('attribute_builder',folder/'build_attribute_straightness.py')
            builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
            nb=json.loads(builder.build().read_text(encoding='utf-8'))
        finally:sys.path.remove(str(folder))
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertEqual(len(codes),6)
        for code in codes:compile(code,'<attribute_colab>','exec')
        self.assertIn('steps=120,strength=.25',codes[4])
        self.assertIn('save_final_projection=False',codes[4])
        self.assertNotIn('straightness_plan(',codes[2])  # generation/extraction is separate
        tree=ast.parse(codes[0])
        helpers=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='RUNTIME_HELPERS')
        for name,code in helpers.items():self.assertEqual(code,(ROOT/name).read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
