import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from diffusion_model.crossfont_input import compose_font_input

class CrossFontTests(unittest.TestCase):
    def test_independent_korean_font_sheet_has_four_populated_slots(self):
        font=Path('C:/Windows/Fonts/malgun.ttf')
        if not font.exists():self.skipTest('Local Korean font unavailable')
        slots=[dict(reference=np.zeros((1,32,32),np.float32),bbox=[x,y,x+64,y+64])
               for y in (0,64) for x in (0,64)]
        # Empty reference cannot supply a pixel target or bounding box to renderer.
        with patch('diffusion_model.crossfont_input.compose_sheet',return_value=(None,slots)):
            sheet,metadata=compose_font_input([{}]*4,['가','걔','곁','괴'],font,128)
        self.assertEqual(sheet.size,(128,128));self.assertEqual(metadata['placement'],'centered')
        a=np.asarray(sheet)
        for slot in slots:
            x0,y0,x1,y1=slot['bbox']
            self.assertLess(a[y0:y1,x0:x1].min(),100)
    def test_character_count_is_validated(self):
        with self.assertRaises(ValueError):compose_font_input([{}]*4,['가'],'missing.ttf')

if __name__=='__main__':unittest.main()
