"""First-stage text-to-image: prompt + seeded noise ONLY.

Deliberately has no image, template, font or stroke-checkpoint argument.
"""
import json
from pathlib import Path
import torch


def generate_prompt_image(output, *, prompt, negative_prompt='', model_id=
        'stable-diffusion-v1-5/stable-diffusion-v1-5', revision=None,
        seed=100, steps=50, size=576, pipeline=None):
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Explicit text prompt required')
    if type(seed) is not int or seed<0 or type(steps) is not int or steps<2:
        raise ValueError('Invalid seed or step count')
    if type(size) is not int or size<64 or size%8:raise ValueError('Size must be a multiple of eight, at least 64')
    destination=Path(output)
    if destination.exists():raise FileExistsError(destination)
    owns_pipeline=pipeline is None
    if owns_pipeline:
        if not torch.cuda.is_available():raise RuntimeError('Use a GPU runtime')
        from diffusers import StableDiffusionPipeline,DDIMScheduler
        kwargs=dict(torch_dtype=torch.float16,use_safetensors=True)
        if revision:kwargs['revision']=revision
        pipeline=StableDiffusionPipeline.from_pretrained(model_id,**kwargs)
        pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.to('cuda');pipeline.enable_attention_slicing()
    destination.mkdir(parents=True)
    report=dict(phase='prompt_only_generation',prompt=prompt,negative_prompt=negative_prompt,
        model_id=model_id,revision=revision,seed=seed,steps=steps,size=size,
        input_image_used=False,template_used=False,font_rendering_used=False,status='running')
    def save():
        (destination/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        result=pipeline(prompt=prompt,negative_prompt=negative_prompt,num_inference_steps=steps,
            width=size,height=size,guidance_scale=7.5,
            generator=torch.Generator('cpu').manual_seed(seed))
        path=destination/'generated.png';result.images[0].save(path)
        flags=getattr(result,'nsfw_content_detected',None)
        report.update(generated_image=str(path),status='complete',
            safety_filtered=bool(flags[0]) if flags is not None else None)
        save()
        return report
    except Exception as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        if owns_pipeline:
            del pipeline
            torch.cuda.empty_cache()
