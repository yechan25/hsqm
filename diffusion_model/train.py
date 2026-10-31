"""A100 single-GPU supervised DDPM training; no reverse-time guidance yet."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .data import GlyphDataset, BatchStream, collate, read_manifest, fingerprint
from .core import build_model, clean_from_epsilon, EMA, OCRLoss, StrokeLoss, sample


def save_atomic(state, path):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(path)


def save_preview(path, references, generated, targets):
    size = references.shape[-1]
    canvas = Image.new('RGB', (size*3, (size+20)*len(references)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i in range(len(references)):
        draw.text((4, i*(size+20)+3), f'{i}: reference / generated / target', fill='black')
        for j, tensor in enumerate((references[i], generated[i], targets[i])):
            array = ((1-tensor.detach().float().cpu()[0].clamp(0, 1))*255).byte().numpy()
            canvas.paste(Image.fromarray(array).convert('RGB'), (j*size, i*(size+20)+20))
    canvas.save(path)


@torch.no_grad()
def validate(model, scheduler, dataset, device, count=16):
    generator = torch.Generator(device=device).manual_seed(8192)
    total = 0.
    model.eval()
    for i in range(min(count, len(dataset))):
        batch = collate([dataset[i]])
        clean = batch['image'].to(device)*2-1
        ref = batch['reference'].to(device)*2-1
        noise = torch.randn(clean.shape, device=device, generator=generator)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device, generator=generator)
        noisy = scheduler.add_noise(clean, noise, t)
        prediction = model(torch.cat((noisy, ref), 1), t).sample
        total += F.mse_loss(prediction.float(), noise.float()).item()
    model.train()
    return total / min(count, len(dataset))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=10000, help='Optimizer updates, not epochs')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--sample-every', type=int, default=500)
    p.add_argument('--sample-steps', type=int, default=50)
    p.add_argument('--ocr-weight', type=float, default=0.)
    p.add_argument('--stroke-weight', type=float, default=0.)
    p.add_argument('--stroke-checkpoint')
    p.add_argument('--aux-start', type=int, default=1000)
    p.add_argument('--aux-every', type=int, default=4)
    p.add_argument('--aux-max-t', type=int, default=200, help='Aux loss only at cleaner t < this value')
    p.add_argument('--resume', help='Exact resume from a trusted local checkpoint')
    p.add_argument('--init-from', help='New experiment from weights, optimizer/RNG reset')
    p.add_argument('--cpu', action='store_true', help='Tiny smoke tests only')
    p.add_argument('--small', action='store_true', help='Tiny smoke-test model')
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    args = p.parse_args()
    for name in ('steps', 'batch_size', 'accumulation', 'checkpoint_every', 'validate_every',
                 'sample_every', 'sample_steps', 'aux_every'):
        if getattr(args, name) <= 0:
            p.error(name + ' must be positive')
    if args.size < 16 or args.size % (2 if args.small else 8):
        p.error('size must be >=16 and divisible by the model downsampling factor')
    if not 0 < args.aux_max_t <= 1000 or min(args.ocr_weight, args.stroke_weight, args.lr) < 0 or args.lr == 0:
        p.error('Invalid weights, learning rate or aux-max-t')
    if args.resume and args.init_from:
        p.error('Use resume OR init-from')
    if args.stroke_weight and not args.stroke_checkpoint:
        p.error('stroke-weight requires stroke-checkpoint')
    if args.warmup < 0 or args.aux_start < 0:
        p.error('warmup/aux-start must be nonnegative')
    return args


def main():
    from diffusers import DDPMScheduler, UNet2DModel
    args = arguments()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime. --cpu is for tiny smoke tests only.')
    device = torch.device('cpu' if args.cpu else 'cuda')
    if args.cpu:
        torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        print('GPU:', torch.cuda.get_device_name(), 'VRAM GiB:',
              round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), flush=True)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(destination.iterdir()):
        raise FileExistsError('Output must be empty for a new run; use --resume or a new output')
    if args.resume and (destination/'last.pt').is_file() and Path(args.resume).resolve() != (destination/'last.pt').resolve():
        raise ValueError('Use last.pt to resume in-place; copy older checkpoints to a new output')
    rows = list(read_manifest(args.manifest))
    train = GlyphDataset([r for r in rows if r['split'] == 'train'], args.size)
    val = GlyphDataset([r for r in rows if r['split'] == 'val'], args.size)
    print(f'Train={len(train)}; val={len(val)}. Test rows never used for model selection.', flush=True)
    print('Checking dataset files and provenance...', flush=True)
    data_hash = fingerprint(args.manifest)
    if args.stroke_weight:
        for i in range(len(train)):
            if train[i]['strokes'] is None:
                raise ValueError('Every training sample needs reference stroke masks for stroke loss')
    for dataset in (train, val):
        for i in range(len(dataset)):
            dataset[i]  # fail before a long GPU run on malformed or empty samples
    state = torch.load(args.resume or args.init_from, map_location='cpu', weights_only=True) if args.resume or args.init_from else None
    if state and state.get('architecture') != 'glyph_ddpm_v1':
        raise ValueError('Not a glyph_ddpm_v1 diffusion checkpoint')
    if state and state['args']['size'] != args.size:
        raise ValueError('Checkpoint image size differs')
    if state and state['data_fingerprint'] != data_hash:
        raise ValueError('Dataset changed. Start an explicitly separate dataset experiment.')
    if args.resume:
        fixed = ('batch_size', 'accumulation', 'size', 'lr', 'warmup', 'seed', 'small',
                 'ocr_weight', 'stroke_weight', 'aux_start', 'aux_every', 'aux_max_t', 'stroke_checkpoint')
        for key in fixed:
            if state['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume config differs: {key}; use --init-from for a new experiment')
        if args.steps < state['step']:
            raise ValueError('--steps must not be less than the resumed step')
    model = (UNet2DModel.from_config(state['model_config']) if state else build_model(args.size, args.small)).to(device)
    if state:
        model.load_state_dict(state['model'], strict=True)
    if not args.no_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    ema = EMA(model)
    if state:
        ema.shadow = {k: v.to(device) for k, v in state['ema'].items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    amp = device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    stream = BatchStream(len(train), args.batch_size, args.seed)
    start, best = 0, float('inf')
    if args.resume:
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start, best = state['step'], state['best_val']
        stream = BatchStream(len(train), args.batch_size, args.seed, **state['stream'])
        metrics_path = destination/'metrics.jsonl'
        if metrics_path.is_file():
            complete = [line for line in metrics_path.read_text(encoding='utf-8').splitlines()
                        if json.loads(line)['step'] <= start]
            metrics_path.write_text(''.join(line+'\n' for line in complete), encoding='utf-8')
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2',
                              prediction_type='epsilon', clip_sample=True)
    ocr = OCRLoss([r['text'] for r in rows if r['split'] in ('train', 'val')], device) if args.ocr_weight else None
    stroke = StrokeLoss(args.stroke_checkpoint, args.size, device) if args.stroke_weight else None
    if args.resume:
        # Restore after model/OCR initialization, which may consume random numbers.
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            if state['cuda_rng'] is None:
                raise ValueError('Cannot exactly resume a CPU run on CUDA')
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    config = {**vars(args), 'architecture': 'glyph_ddpm_v1', 'data_fingerprint': data_hash,
              'parameters': sum(p.numel() for p in model.parameters()), 'torch_version': str(torch.__version__),
              'auxiliary_kind': 'normal-pair reconstruction; not a validated error-quality score',
              'source_sha256': {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}}
    (destination/'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Parameters:', config['parameters'], 'AMP:', str(dtype) if amp else 'off', flush=True)
    preview_batch = collate([val[i] for i in range(min(4, len(val)))])
    (destination/'preview_labels.json').write_text(json.dumps(preview_batch['text'], ensure_ascii=False), encoding='utf-8')

    def checkpoint(step):
        save_atomic({'architecture': 'glyph_ddpm_v1', 'args': vars(args), 'model_config': dict(model.config),
                     'model': model.state_dict(), 'ema': ema.shadow, 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'step': step, 'best_val': best, 'stream': stream.state_dict(),
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                     'data_fingerprint': data_hash}, destination/'last.pt')

    started = time.perf_counter()
    for step in tqdm(range(start+1, args.steps+1), initial=start, total=args.steps, desc='Optimizer updates', mininterval=5):
        optimizer.zero_grad(set_to_none=True)
        stats = {'denoise': 0., 'ocr': 0., 'stroke': 0., 'aux_samples': 0}
        optimizer.param_groups[0]['lr'] = args.lr * min(1., step/max(1, args.warmup))
        for _ in range(args.accumulation):
            batch = collate([train[i] for i in stream.next_indices()])
            clean, reference = [batch[key].to(device) for key in ('image', 'reference')]
            x0 = clean*2-1
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (len(clean),), device=device)
            noisy = scheduler.add_noise(x0, noise, t)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                prediction = model(torch.cat((noisy, reference*2-1), 1), t).sample
            base = F.mse_loss(prediction.float(), noise.float())
            loss = base
            stats['denoise'] += base.detach().item()/args.accumulation
            selected = t < args.aux_max_t
            if (ocr or stroke) and step >= args.aux_start and step % args.aux_every == 0 and selected.any():
                estimate = (clean_from_epsilon(noisy[selected], prediction[selected], t[selected], scheduler.alphas_cumprod)+1)/2
                # Low-noise region only; clamp bounds the OCR/segmenter inputs.
                # Monitor saturation: clipped pixels receive zero gradient outside [0,1].
                ink = estimate.clamp(0, 1)
                ids = selected.nonzero().flatten().tolist()
                stats['aux_samples'] += len(ids)
                if ocr:
                    ocr_value = ocr(ink, [batch['text'][i] for i in ids])
                    loss = loss + args.ocr_weight * ocr_value
                    stats['ocr'] += ocr_value.detach().item()/args.accumulation
                if stroke:
                    masks = batch['strokes'].to(device)[selected]
                    stroke_value = stroke(ink, clean[selected], reference[selected], masks)
                    loss = loss + args.stroke_weight * stroke_value
                    stats['stroke'] += stroke_value.detach().item()/args.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss; last completed checkpoint is preserved')
            scaler.scale(loss/args.accumulation).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        record = {'step': step, **stats, 'lr': optimizer.param_groups[0]['lr'],
                  'grad_norm': norm.item(), 'elapsed_seconds': time.perf_counter()-started}
        improved = False
        if step % args.validate_every == 0 or step == args.steps:
            with ema.apply(model):
                value = validate(model, scheduler, val, device)
            record['val_noise_mse'] = value
            if value < best:
                best, improved = value, True
                save_atomic({'architecture': 'glyph_ddpm_v1', 'model_config': dict(model.config),
                             'ema': ema.shadow, 'step': step, 'val_noise_mse': value,
                             'size': args.size}, destination/'best_weights.pt')
        if step % args.sample_every == 0 or step == args.steps:
            with ema.apply(model):
                refs = preview_batch['reference'].to(device)
                generated = sample(model, scheduler, refs, args.seed+10000, args.sample_steps)
            save_preview(destination/f'preview_{step:06d}.png', refs, generated, preview_batch['image'])
        with (destination/'metrics.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record)+'\n')
        if step % 25 == 0 or step == 1 or step == args.steps:
            print(json.dumps(record), flush=True)
        if step == 1 or step % args.checkpoint_every == 0 or step == args.steps or improved:
            checkpoint(step)
    print('Completed. best_weights.pt selects validation noise MSE, NOT human glyph correctness.', flush=True)


if __name__ == '__main__':
    main()
