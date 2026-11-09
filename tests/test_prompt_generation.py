import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from diffusion_model.prompt_generation import generate_prompt_image
from diffusion_model.two_stage_repair import generate_first_stage


class PromptOnlyTests(unittest.TestCase):
    def test_first_generation_receives_no_image_or_template(self):
        class Pipeline:
            def __init__(self):self.calls=[]
            # Explicit signature prevents any accidental image/latent injection.
            def __call__(self,*,prompt,negative_prompt,num_inference_steps,width,height,true_cfg_scale,generator):
                self.calls.append(dict(prompt=prompt,seed=generator.initial_seed(),steps=num_inference_steps))
                return SimpleNamespace(images=[Image.new('RGB',(width,height),'white')],nsfw_content_detected=[False])
        workspace=Path(__file__).resolve().parents[1]/'outputs';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            pipe=Pipeline();out=Path(directory)/'prompt'
            with patch('diffusion_model.multiglyph_pilot.compose_sheet',side_effect=AssertionError('No templates in first generation')):
                report=generate_prompt_image(out,prompt='A photo displaying 가나다라',size=64,steps=4,seed=123,pipeline=pipe)
            self.assertEqual(len(pipe.calls),1)
            self.assertEqual(pipe.calls[0]['seed'],123)
            self.assertTrue(Path(report['generated_image']).exists())
            self.assertFalse(report['input_image_used'] or report['template_used'] or report['font_rendering_used'])
            self.assertEqual(json.loads((out/'generation_report.json').read_text(encoding='utf-8'))['phase'],'prompt_only_generation')
        parameters=inspect.signature(generate_prompt_image).parameters
        self.assertNotIn('image',parameters);self.assertNotIn('reference',parameters);self.assertNotIn('font',parameters)

    def test_old_first_stage_cannot_silently_fall_back_to_img2img(self):
        with self.assertRaises(ValueError):
            generate_first_stage('unused',[],'unused','unused',init_mode='external',input_sheet='font.png')

    def test_main_notebook_generates_before_loading_stroke_models(self):
        notebook=Path(__file__).resolve().parents[1]/'notebooks/prompt_then_repair.ipynb'
        data=json.loads(notebook.read_text(encoding='utf-8'))
        generation=''.join(data['cells'][2]['source'])
        self.assertIn('generate_prompt_image(',generation)
        self.assertNotIn('compose_sheet',generation)
        self.assertNotIn('load_topology',generation)
        self.assertNotIn('load_error_head',generation)
        all_code='\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type']=='code')
        self.assertNotIn('compose_font_input',all_code)
        self.assertNotIn("glob('*/stage1/page_001_baseline.png')",all_code)
        self.assertIn("slot['bbox']=list(box)",all_code)
        self.assertIn("MODEL_ID='Qwen/Qwen-Image-2512'",generation)
        self.assertIn('repair_qwen_pair(',all_code)
        self.assertNotIn('generation=generate_guided_pair(',all_code)


if __name__=='__main__':unittest.main()
