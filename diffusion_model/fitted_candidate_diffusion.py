"""Paired DDIM re-denoising guided by an extractor-fitted candidate, not GT."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def load_case(archive, case_id):
    """Read named members only; recorded Colab absolute paths are not trusted."""
    with zipfile.ZipFile(archive) as z:
        cases = json.loads(z.read('case_manifest.json'))
        matches = [c for c in cases if c['case_id'] == case_id]
        if len(matches) != 1:
            raise ValueError('Select exactly one case from case_manifest.json')
        case = matches[0]
        page = f"generation/page_{case['page_id']:03d}.png"
        candidate = f"{case_id}/normal_candidate.png"
        def read_image(name):
            with Image.open(io.BytesIO(z.read(name))) as im:
                return im.convert('RGB').copy()
        source, target = read_image(page), read_image(candidate)
        if source.size != target.size or any(n % 8 for n in source.size):
            raise ValueError('Input and fitted candidate must have equal dimensions divisible by 8')
        report = json.loads(z.read('generation/generation_report.json'))
        metadata = dict(case=case, candidate_member=candidate, input_member=page,
                        generation_model=report['model_id'],
                        candidate_is_ground_truth=False,
                        prompt=report.get('prompts', {}).get('4', ''),
                        negative_prompt=report.get('negative_prompt', ''))
    return source, target, metadata


def image_tensor(image, device):
    a = np.asarray(image, dtype=np.float32) / 255
    return torch.from_numpy(a.copy()).permute(2, 0, 1)[None].to(device)


class CandidateObjective:
    """Fixed spatial weights; gradients stay continuous through the VAE."""
    def __init__(self, source, candidate, threshold=.10, radius=2, preserve_weight=.25):
        if source.shape != candidate.shape or not 0 < threshold < 1 or radius < 0:
            raise ValueError('Invalid images or edit-region settings')
        self.source, self.candidate = source.detach(), candidate.detach()
        difference = (source - candidate).abs().mean(1, keepdim=True)
        edit = (difference > threshold).float()
        self.edit = F.max_pool2d(edit, 2*radius+1, 1, radius).detach()
        self.preserve_weight = preserve_weight

    @staticmethod
    def mean_region(values, mask):
        return (values*mask).sum() / (mask.sum()*values.shape[1]).clamp_min(1)

    def __call__(self, image):
        match = self.mean_region((image-self.candidate).square(), self.edit)
        preserve = self.mean_region((image-self.source).square(), 1-self.edit)
        loss = match + self.preserve_weight*preserve
        return loss, dict(candidate_region_mse=float(match.detach()),
                          preserved_region_mse=float(preserve.detach()))


def optimize_clean(clean, decode, objective, *, iterations=4, step_ratio=.02,
                   budget_ratio=.08, backtracks=5):
    """Backtrack on the SAME clean estimate; bound total correction per step."""
    anchor = clean.detach().float()
    current = anchor.clone()
    scale = anchor.square().mean().sqrt().clamp_min(1e-3)
    with torch.no_grad():
        before, _ = objective(decode(current))
    accepted = 0
    for _ in range(iterations):
        with torch.enable_grad():
            leaf = current.detach().requires_grad_(True)
            loss, _ = objective(decode(leaf))
            gradient, = torch.autograd.grad(loss, leaf)
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite candidate guidance')
        norm = gradient.square().mean().sqrt()
        if norm < 1e-12:
            break
        direction = step_ratio*scale*gradient/norm
        found = False
        with torch.no_grad():
            for attempt in range(backtracks+1):
                proposal = current-direction*(.5**attempt)
                delta = proposal-anchor
                rms = delta.square().mean().sqrt().clamp_min(1e-12)
                proposal = anchor+delta*(budget_ratio*scale/rms).clamp(max=1)
                trial, _ = objective(decode(proposal))
                if torch.isfinite(trial) and trial < loss.detach():
                    current = proposal.detach()
                    accepted += 1
                    found = True
                    break
        if not found:
            break
    with torch.no_grad():
        after, details = objective(decode(current))
    return current, dict(loss_before=float(before), loss_after=float(after),
                         accepted_inner_steps=accepted, **details)


def run_pair(source, candidate, metadata, output, *, steps=60, strength=.35,
             seed=2026, cfg_scale=5., inner_steps=4, step_ratio=.02,
             model_id='stable-diffusion-v1-5/stable-diffusion-v1-5', revision=None,
             progress=None):
    """Identical source VAE encoding, noise, prompt, scheduler in both arms."""
    if not torch.cuda.is_available():
        raise RuntimeError('Select an L4 GPU runtime in Colab')
    if not 0 < strength <= 1 or steps < 4 or int(steps*strength) < 2:
        raise ValueError('Need at least two actual re-denoising steps')
    if inner_steps < 1 or step_ratio <= 0 or cfg_scale <= 1:
        raise ValueError('Invalid guidance configuration')
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    from copy import deepcopy
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source.save(out/'input.png')
    candidate.save(out/'fitted_candidate.png')
    kwargs = dict(torch_dtype=torch.float16, use_safetensors=True)
    if revision:
        kwargs['revision'] = revision
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,
                                              clip_sample=False, thresholding=False)
    pipe.to('cuda')
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.eval().requires_grad_(False)
    prompt = metadata.get('prompt') or ('four Korean Hangul glyphs, black ink on white, '
                                      'preserve the input typography and layout')
    with torch.no_grad():
        positive, negative = pipe.encode_prompt(prompt, 'cuda', 1, True,
            negative_prompt=metadata.get('negative_prompt', ''))
        embeddings = torch.cat([negative, positive])
        pipe.scheduler.set_timesteps(steps, device='cuda')
        timesteps, _ = pipe.get_timesteps(steps, strength, 'cuda')
        # VAE and noise initialization exactly once, then copy for both arms.
        initial = pipe.prepare_latents(pipe.image_processor.preprocess(source), timesteps[:1],
            1, 1, embeddings.dtype, 'cuda', torch.Generator('cpu').manual_seed(seed))
    pipe.vae.to(dtype=torch.float32)
    def decode(z):
        return pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor,
                               return_dict=False)[0].float()/2+.5
    def save_image(z, path):
        with torch.no_grad():
            array = decode(z)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        Image.fromarray(np.rint(array*255).astype('uint8')).save(path)
    objective = CandidateObjective(image_tensor(source, 'cuda'), image_tensor(candidate, 'cuda'))
    Image.fromarray((objective.edit[0, 0].cpu().numpy()*255).astype('uint8')).save(out/'edit_region.png')
    traces, metrics = {}, {}
    for method in ('redenoise_only', 'candidate_guided'):
        scheduler = deepcopy(pipe.scheduler)
        latent = initial.detach().float().clone()
        trace = []
        for i, t in enumerate(timesteps):
            with torch.no_grad():
                inputs = scheduler.scale_model_input(torch.cat([latent, latent]), t)
                u, c = pipe.unet(inputs.to(torch.float16), t,
                    encoder_hidden_states=embeddings, return_dict=False)[0].chunk(2)
                noise = (u+cfg_scale*(c-u)).float()
                base = scheduler.step(noise, t, latent, eta=0., return_dict=True)
            record = dict(step=i, timestep=int(t), active=False, accepted_inner_steps=0)
            latent = base.prev_sample.detach()
            clean = base.pred_original_sample.detach()
            if method == 'candidate_guided' and i/max(len(timesteps)-1, 1) >= .2:
                clean, result = optimize_clean(clean, decode, objective,
                    iterations=inner_steps if i < len(timesteps)-1 else inner_steps*2,
                    step_ratio=step_ratio)
                # DDIM eta=0: x_(t-1) = sqrt(a_prev)*x0 + sqrt(1-a_prev)*epsilon.
                # Hold the predicted epsilon fixed and change the clean component.
                prev_t = int(t)-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
                alpha = (scheduler.alphas_cumprod[prev_t] if prev_t >= 0
                         else scheduler.final_alpha_cumprod).to(latent)
                latent = latent+alpha.sqrt()*(clean-base.pred_original_sample)
                record.update(active=True, **result)
            latent = latent.detach()
            trace.append(record)
            if i % 4 == 0 or i == len(timesteps)-1:
                save_image(clean, out/f'{method}_step_{i:03d}.png')
            if progress:
                progress(method, i+1, len(timesteps), record)
        save_image(latent, out/f'{method}.png')
        with torch.no_grad():
            loss, values = objective(decode(latent))
        metrics[method] = dict(loss=float(loss), **values)
        traces[method] = trace
    with torch.no_grad():
        loss, values = objective(objective.source)
    metrics['input'] = dict(loss=float(loss), **values)
    report = dict(metadata=metadata, model_id=model_id, revision=revision,
        steps=steps, actual_steps=len(timesteps), strength=strength, seed=seed,
        cfg_scale=cfg_scale, inner_steps=inner_steps, step_ratio=step_ratio,
        initial_latent_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        paired_initial_latent=True, guidance_target='extractor_fitted_candidate',
        target_pixels_pasted_into_output=False, metrics=metrics, traces=traces,
        metric_note='Candidate agreement, not ground-truth accuracy or HSQM score')
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    del pipe
    torch.cuda.empty_cache()
    return report
