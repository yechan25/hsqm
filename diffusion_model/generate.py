"""Generate from a reference glyph, using a trained small diffusion checkpoint."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from .core import sample
from .data import load_image, resize


def main():
    from diffusers import UNet2DModel, DDPMScheduler
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference', required=True, help='Dark glyph on white image')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    if not 1 <= args.steps <= 1000 or args.count < 1:
        p.error('steps must be 1..1000; count must be positive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state.get('architecture') != 'glyph_ddpm_v1':
        raise ValueError('Expected glyph_ddpm_v1 checkpoint')
    model = UNet2DModel.from_config(state['model_config']).to(device)
    model.load_state_dict(state['ema'], strict=True)
    model.eval()
    size = state.get('size', state.get('args', {}).get('size'))
    reference = resize(load_image(args.reference), size)[None].to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                              prediction_type='epsilon', clip_sample=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        result = sample(model, scheduler, reference, args.seed+i, args.steps)
        path = output/f'seed_{args.seed+i}.png'
        if path.exists():
            raise FileExistsError(path)
        Image.fromarray(((1-result[0, 0].cpu())*255).byte().numpy()).save(path)
        print(path)


if __name__ == '__main__':
    main()
