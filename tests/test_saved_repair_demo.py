import io
import json
import tempfile
import unittest
from pathlib import Path
import zipfile
import numpy as np
import torch
from PIL import Image
from diffusion_model.saved_repair_demo import prepare_saved_demo
from stroke_model.saved_extra_guide import extracted_extra_map


class SavedRepairTests(unittest.TestCase):
    def test_archive_keeps_actual_baseline_bytes_and_recovers_conditions(self):
        workspace=Path(__file__).resolve().parents[1]/'outputs';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            directory=Path(directory);archive=directory/'example.zip'
            b=io.BytesIO();Image.new('RGB',(64,64),'white').save(b,format='PNG');source=b.getvalue()
            report=dict(checkpoint='recorded/best.pt',checkpoint_sha256='abc',
                pages=[dict(method='baseline',input='/old/page_001_baseline.png')],results=[])
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('generation/page_001_baseline.png',source)
                for i in range(4):
                    report['results'].append(dict(method='reference',slot=i,npz=f'/old/sample_{i}.npz',
                        crop_bbox=[(i%2)*32,(i//2)*32,(i%2+1)*32,(i//2+1)*32]))
                    d=io.BytesIO();np.savez(d,reference=np.ones((1,16,16),np.float32),reference_strokes=np.ones((2,16,16),np.float32))
                    z.writestr(f'topo_evaluation/reference_slot_{i+1}.npz',d.getvalue())
                z.writestr('generation/generation_report.json',json.dumps(report))
            demo=prepare_saved_demo(archive,directory/'prepared')
            self.assertEqual(Path(demo['source']).read_bytes(),source)
            self.assertEqual(demo['checkpoint_sha256'],'abc')
            self.assertFalse(demo['provenance']['fresh_text_to_image'])
            self.assertEqual(len(json.loads(Path(demo['manifest']).read_text())),4)

    def test_assignment_errors_cannot_erase_supported_healthy_strokes(self):
        strokes=np.zeros((2,16,16),np.float32)
        strokes[0,2:5,2:14]=1;strokes[1,9:12,2:14]=1
        reference=strokes.max(0)[None]
        rgb=torch.from_numpy(1-reference).expand(1,3,16,16).clone()
        slot=dict(reference=reference,strokes=strokes,bbox=[0,0,16,16])
        class Extractor:
            def __init__(self,swap):self.swap=swap
            def __call__(self,ink,reference,strokes):
                return dict(masks=strokes.flip(1) if self.swap else strokes)
        good,geometry,_=extracted_extra_map(Extractor(False),rgb,[slot],tolerance=0)
        bad,geometry_bad,details=extracted_extra_map(Extractor(True),rgb,[slot],tolerance=0)
        self.assertEqual(good.sum(),0)
        self.assertGreater(details[0]['assignment_extra'].sum(),0)
        self.assertEqual(bad.sum(),0)
        torch.testing.assert_close(geometry,geometry_bad)
        self.assertEqual(geometry.sum(),0)

    def test_nonfocus_slots_receive_no_guide(self):
        strokes=np.zeros((1,16,16),np.float32);strokes[:,4:8,4:8]=1
        class Extractor:
            def __call__(self,ink,reference,strokes):return dict(masks=ink)
        slots=[dict(reference=strokes,strokes=strokes,bbox=[i*16,0,(i+1)*16,16]) for i in range(2)]
        rgb=torch.zeros(1,3,16,32)
        guide,_,_=extracted_extra_map(Extractor(),rgb,slots,focus_slots=(0,),tolerance=0)
        self.assertGreater(guide[...,:16].sum(),0)
        self.assertEqual(guide[...,16:].sum(),0)


if __name__=='__main__':unittest.main()
