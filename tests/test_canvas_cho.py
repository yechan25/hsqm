"""Canvas cell backend integration; browser pointer drawing needs Colab."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import stroke_model.structural_indicator as scoring


class CanvasCellTests(unittest.TestCase):
    def test_drawn_png_is_scored_against_existing_csv_target(self):
        cell=Path('notebooks/canvas_cho_cell.py').read_text(encoding='utf-8-sig')
        compile(cell,'canvas_cho_cell','exec')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stroke=np.zeros((128,128),np.uint8)
            stroke[20:105,62:67]=255
            picture=Image.fromarray(255-stroke)
            picture.save(root/'target.png')
            picture.save(root/'stroke.png')
            pd.DataFrame([dict(input_path='I_0_11.png',target_path='target.png',
                               target_stroke_paths='stroke.png')]).to_csv(root/'paths.csv',index=False)
            buffer=io.BytesIO()
            picture.resize((384,384)).convert('RGB').save(buffer,format='PNG')
            data='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
            output=types.ModuleType('google.colab.output')
            output.eval_js=lambda script:data
            namespace={'CSV_PATH':str(root/'paths.csv'),'CANVAS_SAVE_ROOT':str(root/'output')}
            with patch.dict(sys.modules,{'google.colab.output':output,'_structural_indicator_runtime':scoring}), \
                 patch('matplotlib.pyplot.show'):
                exec(cell,namespace)
            result=namespace['canvas_indicator_results']['results'][0]
            self.assertGreaterEqual(result['structural_score'],90)
            self.assertTrue(result['image_available'])
            self.assertEqual(result['measurement_mode'],'source')
            self.assertEqual(result['extractor_strokes'],[{'stroke':1}])
            raw=namespace['canvas_structure_results']
            self.assertEqual(raw['target'],str(root/'target.png'))
            with np.load(raw['results'][0]['npz']) as arrays:
                self.assertEqual(arrays['image'].shape,(1,128,128))
                self.assertEqual(arrays['reference_strokes'].shape,(1,128,128))


if __name__=='__main__':
    unittest.main()
