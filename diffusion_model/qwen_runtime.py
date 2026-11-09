"""Qwen loading shared by pure text-to-image and native flow repair."""
import torch


def load_qwen_pipeline(model_id='Qwen/Qwen-Image-2512', *, revision=None):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Select a Colab A100 GPU; this Qwen configuration requires native BF16.')
    if torch.cuda.get_device_properties(0).total_memory < 35 * 1024**3:
        raise RuntimeError('Use an A100 with at least 40 GB VRAM. No silent SD1.5 fallback.')
    from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusionQuantization
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig as TextQuantization
    common=dict(revision=revision,torch_dtype=torch.bfloat16)
    quant=dict(load_in_4bit=True,bnb_4bit_quant_type='nf4',
               bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    transformer=QwenImageTransformer2DModel.from_pretrained(model_id,subfolder='transformer',
        quantization_config=DiffusionQuantization(**quant),**common)
    text_encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,subfolder='text_encoder',
        quantization_config=TextQuantization(**quant),**common)
    pipe=QwenImagePipeline.from_pretrained(model_id,transformer=transformer,
        text_encoder=text_encoder,**common).to('cuda')
    pipe.vae.enable_tiling()
    for component in (pipe.vae,pipe.transformer,pipe.text_encoder):
        component.eval().requires_grad_(False)
    return pipe
