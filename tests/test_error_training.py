import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from scipy.ndimage import label
from stroke_model.error_synthesis import corrupt_extra, glyph_from_entry, build_error_sets
from stroke_model.train_error_localizer import fit_error_localizer, evaluate_errors


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.ink=np.zeros((1,32,32),np.float32)
        self.ink[:,5:27,16:19]=1;self.ink[:,15:18,19:26]=1

    def test_labels_are_only_visible_new_ink(self):
        for kind in (1,2):
            bad,extra,types=corrupt_extra(self.ink,kind,123)
            self.assertGreater(extra.sum(),0)
            self.assertTrue(np.array_equal(extra[0]>0,(bad[0]-self.ink[0])>.05))
            self.assertEqual(float((extra*self.ink).sum()),0)
            self.assertTrue(np.all(types[extra[0]>0]==kind))
            again=corrupt_extra(self.ink,kind,123)[0]
            self.assertTrue(np.array_equal(again,bad))

    def test_clean_negative_is_unchanged(self):
        bad,extra,kinds=corrupt_extra(self.ink,0,1)
        self.assertTrue(np.array_equal(bad,self.ink))
        self.assertEqual(extra.sum(),0);self.assertEqual(kinds.sum(),0)

    def test_corruption_kind_connectivity(self):
        original_count=label(self.ink[0]>.5)[1]
        attached=corrupt_extra(self.ink,1,10)[0]
        detached=corrupt_extra(self.ink,2,10)[0]
        self.assertEqual(label(attached[0]>.5)[1],original_count)
        self.assertEqual(label(detached[0]>.5)[1],original_count+1)

    def test_numeric_filenames_are_not_character_labels(self):
        self.assertIsNone(glyph_from_entry(dict(reference='123.png',char_id='123')))
        self.assertEqual(glyph_from_entry(dict(reference='NanumGothic_가.png')),'가')

    def test_splits_and_duplicate_font_content(self):
        class Font:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def getBestCmap(self):return {ord(c):'glyph' for c in '가나다라마바사아자차카타'}
        entries=[dict(char_id=c,manifest_row=i) for i,c in enumerate('가나다라마바사아자차카타')]
        fonts=dict(train=['font_A.ttf'],val=['font_B.ttf'],test=['font_C.ttf'])
        with patch('stroke_model.error_synthesis.TTFont',return_value=Font()), \
             patch.object(Path,'read_bytes',lambda p:str(p).encode()):
            sets,meta=build_error_sets(entries,fonts,max_characters=12)
        groups=[set(x) for x in meta['characters'].values()]
        self.assertFalse(groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
        self.assertEqual(sum(map(len,groups)),12)
        with patch.object(Path,'read_bytes',return_value=b'same font'):
            with self.assertRaises(ValueError):build_error_sets(entries,fonts)


class TinySet(Dataset):
    epoch=0
    def __len__(self):return 6
    def __getitem__(self,index):
        strokes=torch.zeros(2,24,24);strokes[0,4:20,12:15]=1;strokes[1,10:13,15:20]=1
        ref=strokes.max(0).values[None]
        bad,extra,kinds=corrupt_extra(ref.numpy(),index%3,index+self.epoch*100)
        return dict(rgb=torch.from_numpy(1-bad).repeat(3,1,1),reference=ref,strokes=strokes,
                    codepoint=torch.tensor(ord('가')),extra=torch.from_numpy(extra),
                    valid=torch.ones_like(ref),kind=torch.from_numpy(kinds),clean=(1-ref).repeat(3,1,1))


class TrainSmokeTests(unittest.TestCase):
    def test_complete_cpu_training_checkpoint_and_evaluation(self):
        torch.set_num_threads(1)
        class Extractor(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(4.))
            def forward(self,ink,ref,strokes,active=None):
                return dict(probabilities=(strokes*self.scale).softmax(1))
        extractor=Extractor();train=TinySet();val=TinySet()
        workspace=(Path(__file__).resolve().parents[1]/'outputs').resolve()
        workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            target=Path(directory)/'run'
            model,history=fit_error_localizer(extractor,train,val,target,device='cpu',
                epochs=2,batch_size=2,width=8,metadata=dict(test_only=True))
            self.assertEqual(len(history),2)
            checkpoint=torch.load(target/'best.pt',weights_only=True)
            self.assertFalse(checkpoint['threshold_calibrated'])
            self.assertEqual(checkpoint['architecture'],'stroke_error_localizer_v1')
            self.assertEqual(len(json.loads((target/'history.json').read_text())),2)
            metrics=evaluate_errors(model,val,'cpu',2)
            self.assertTrue(np.isfinite(metrics['loss']))
            self.assertEqual(metrics['samples'],6)
            self.assertIsNone(extractor.scale.grad)
            self.assertEqual(val.epoch,0)


if __name__=='__main__':unittest.main()
