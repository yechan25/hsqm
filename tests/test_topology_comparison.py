import hashlib
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import torch

from stroke_model.compare_topology import (assignment_changes, compare_checkpoints,
    find_before_checkpoint, inspect_training_supervision, load_comparison_model, compare_canvas)
from stroke_model.convert_reference_csv import load_ink
from stroke_model.reference_model import ReferenceStrokeModel


class PredictionFixture(torch.nn.Module):
    def __init__(self,flip=False):super().__init__();self.flip=flip
    def forward(self,image,reference,strokes):
        p=torch.empty_like(strokes);p[:,0]=.9;p[:,1]=.1
        if self.flip:p[:,0,16,16]=.4;p[:,1,16,16]=.6
        return dict(probabilities=p,masks=image*p,hints=strokes,flow=torch.zeros_like(strokes))


class ComparisonTests(unittest.TestCase):
    def test_one_canvas_submission_is_used_by_both_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'before.pt').write_bytes(b'original')
            torch.save({'image_size':32},root/'after.pt')
            original_before=(root/'before.pt').read_bytes();original_after=(root/'after.pt').read_bytes()
            gray=np.full((32,32),255,np.uint8);gray[8:24,14:18]=0
            Image.fromarray(gray).save(root/'reference.png')
            pd.DataFrame([dict(input_path='MISSING.png',target_path='reference.png',
                target_stroke_paths='reference.png;reference.png')]).to_csv(root/'path.csv',index=False)
            stream=io.BytesIO();Image.fromarray(gray).save(stream,format='PNG')
            answer=dict(row=0,png='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode())
            metadata=dict(image_size=32,epoch=3)
            with patch('stroke_model.canvas_topology.canvas_prompt',return_value=answer) as prompt, \
                 patch('stroke_model.compare_topology.load_comparison_model',side_effect=[
                     (PredictionFixture(),metadata),(PredictionFixture(True),metadata)]):
                report=compare_canvas(root/'before.pt',root/'after.pt',root/'path.csv',save_dir=root/'canvas',show=False)
            prompt.assert_called_once()
            self.assertEqual(report['reference_csv_row'],1)
            self.assertEqual(report['results'][0]['input'],report['canvas_input'])
            self.assertEqual((root/'before.pt').read_bytes(),original_before)
            self.assertEqual((root/'after.pt').read_bytes(),original_after)
            self.assertTrue((root/'canvas/input.png').is_file())

    def test_original_v2_checkpoint_loads_as_comparison_baseline(self):
        class Backbone(torch.nn.Module):
            def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(3,8,3,2,1)
            def forward(self,x):return [self.conv(x)]
        def tiny(**kwargs):return ReferenceStrokeModel(backbone=Backbone(),channels=[8],**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'v2.pt';original=tiny(image_size=32,width=8,pretrained=False)
            torch.save(dict(model=original.state_dict(),architecture='reference_v2',image_size=32,width=8,epoch=10),path)
            with patch('stroke_model.compare_topology.ReferenceStrokeModel',side_effect=tiny):
                loaded,metadata=load_comparison_model(path,'cpu')
            self.assertEqual(metadata['architecture'],'reference_v2')
            self.assertFalse(loaded.training)
            for key,value in original.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key],value)

    def test_transfer_matrix_reports_direction_without_quality_claim(self):
        b=np.full((2,4,4),.1,np.float32);b[0]=.9
        a=b.copy();a[:,1,1]=(.4,.6)
        before,after,transfer,rows,stats,margin=assignment_changes(b,a,np.ones((1,4,4)))
        self.assertEqual(transfer[0,1],1);self.assertEqual(transfer[1,0],0)
        self.assertEqual(stats['changed_pixels'],1)
        self.assertEqual(rows[0]['lost_pixels'],1);self.assertEqual(rows[1]['gained_pixels'],1)
        self.assertAlmostEqual(float(margin[1,1]),.2,places=6)

    def test_identical_saved_canvas_and_checkpoint_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('before.pt','after.pt'):(root/name).write_bytes(name.encode())
            before_bytes=(root/'before.pt').read_bytes();after_bytes=(root/'after.pt').read_bytes()
            gray=np.full((32,32),255,np.uint8);gray[8:24,14:18]=0
            Image.fromarray(gray).save(root/'canvas.png')
            pd.DataFrame([dict(input_path='MISSING_INPUT.png',target_path='canvas.png',
                target_stroke_paths='canvas.png;canvas.png',input_stroke_paths='MISSING_GT.png')]).to_csv(root/'path.csv',index=False)
            metadata=dict(image_size=32,architecture='reference_v2_topology_loss_v1',epoch=3)
            with patch('stroke_model.compare_topology.load_comparison_model',side_effect=[
                (PredictionFixture(),metadata),(PredictionFixture(True),metadata)]):
                report=compare_checkpoints(root/'before.pt',root/'after.pt',root/'path.csv',
                    input_image=root/'canvas.png',save_dir=root/'out',focus_strokes=[1,2],show=False)
            result=report['results'][0]
            self.assertEqual(result['diagnostics']['changed_pixels'],1)
            self.assertEqual(result['input_sha256'],hashlib.sha256(load_ink(root/'canvas.png',32).tobytes()).hexdigest())
            self.assertTrue(report['comparison_is_not_accuracy_score'])
            self.assertEqual((root/'before.pt').read_bytes(),before_bytes)
            self.assertEqual((root/'after.pt').read_bytes(),after_bytes)
            self.assertTrue((root/'out/row_00001_strokes.png').is_file())

    def test_original_checkpoint_source_is_explicit_not_previous_directory_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'before.pt').write_bytes(b'original')
            (root/'initial_checkpoint_source.txt').write_text(str(root/'before.pt'))
            self.assertEqual(find_before_checkpoint(root/'after.pt'),root/'before.pt')
            (root/'initial_checkpoint_source.txt').write_text('None')
            with self.assertRaises(FileNotFoundError):find_before_checkpoint(root/'after.pt')

    def test_saved_training_owners_are_exported_without_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            image=np.zeros((1,32,32),np.float32);image[:,8:24,14:18]=1
            target=np.repeat(image,2,axis=0);labels=np.full((32,32),-1)
            owners=np.full((32,32),-1,np.int16);owners[16,16]=1
            weight=np.zeros((32,32),np.uint8);weight[16,16]=200
            np.savez_compressed(root/'sample.npz',image=image,labels=labels,strokes=target,target_masks=target)
            np.savez_compressed(root/'owners.npz',owners=owners,reliability=weight)
            rows=[dict(path='sample.npz',split='train',group_id='g',ownership_path=str(root/'owners.npz'))]
            (root/'manifest.json').write_text(json.dumps(rows))
            result=inspect_training_supervision(root/'manifest.json',save_dir=root/'audit',show=False)
            self.assertEqual(result[0]['status'],'saved_training_map')
            with np.load(root/'audit/train_00001.npz') as arrays:
                np.testing.assert_array_equal(arrays['owners'],owners)
            del rows[0]['ownership_path'];(root/'manifest.json').write_text(json.dumps(rows))
            result=inspect_training_supervision(root/'manifest.json',show=False)
            self.assertEqual(result[0]['status'],'no_saved_training_ownership')


if __name__=='__main__':unittest.main()
