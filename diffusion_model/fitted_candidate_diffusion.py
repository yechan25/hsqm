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
        if 'case_manifest.json' not in z.namelist():
            report = json.loads(z.read('report.json'))
            metadata = dict(report['metadata'])
            if metadata['case']['case_id'] != case_id:
                raise ValueError('CASE_ID does not match the saved reverse experiment')
            with Image.open(io.BytesIO(z.read('input.png'))) as im:
                source = im.convert('RGB').copy()
            with Image.open(io.BytesIO(z.read('fitted_candidate.png'))) as im:
                target = im.convert('RGB').copy()
            if source.size != target.size or any(n % 8 for n in source.size):
                raise ValueError('Input/candidate dimensions mismatch')
            metadata['reused_archive_kind'] = 'reverse_result_original_input_and_candidate'
            return source, target, metadata
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


class InkConstraintObjective:
    """Candidate supplies fixed sparse constraints, never RGB target pixels."""
    def __init__(self, source, candidate, *, enable_restore=False, tolerance=5.,
                 halo=2, preserve_weight=.5, background_ink=.015, required_ink=.7):
        from scipy.ndimage import distance_transform_edt, binary_dilation
        from skimage.morphology import skeletonize
        if source.shape != candidate.shape or source.shape[:2] != (1, 3):
            raise ValueError('Expected matching single RGB sheets')
        if not np.isfinite(tolerance) or tolerance < 2 or type(halo) is not int or halo < 0:
            raise ValueError('Invalid constraint tolerance/halo')
        if not 0 <= background_ink < required_ink <= 1 or preserve_weight < 0:
            raise ValueError('Invalid ink constraints')
        self.source = source.detach()
        self.preserve_weight = preserve_weight
        self.background_ink, self.required_ink = background_ink, required_ink
        source_ink = self.ink(source).detach().cpu().numpy()[0, 0]
        target_ink = self.ink(candidate).detach().cpu().numpy()[0, 0]
        source_binary, target_binary = source_ink > .35, target_ink > .35
        if not source_binary.any() or not target_binary.any():
            raise ValueError('Empty source or candidate: cannot infer safe correspondence')
        distance = distance_transform_edt(~target_binary)
        # Ignore ordinary boundary/width differences within a tolerance corridor.
        seeds = source_binary & (distance > tolerance)
        nearby = binary_dilation(seeds, iterations=halo) if halo else seeds
        erase = nearby*np.clip((distance-1)/max(tolerance-1, 1), 0, 1)
        # Restore only an absent centerline, not the candidate's full silhouette.
        missing = skeletonize(target_binary) & (distance_transform_edt(~source_binary) > tolerance)
        restore = missing.astype(np.float32) if enable_restore else np.zeros_like(erase)
        erase[restore > 0] = 0  # Never issue conflicting instructions.
        def tensor(a):
            return torch.as_tensor(a.copy(), dtype=torch.float32, device=source.device)[None, None]
        self.erase, self.restore = tensor(erase), tensor(restore)
        self.edit = torch.maximum(self.erase, self.restore)
        occupied = (self.edit > 0).float()
        self.preserve = 1-F.max_pool2d(occupied, 5, 1, 2)
        self.metadata = dict(enable_restore=enable_restore, tolerance_pixels=tolerance,
            halo_pixels=halo, preserve_weight=preserve_weight,
            background_ink=background_ink, required_ink=required_ink,
            erase_pixels=int((self.erase > 0).sum()), restore_pixels=int((self.restore > 0).sum()),
            candidate_rgb_matching=False, masks_frozen_for_entire_pass=True)

    @staticmethod
    def ink(rgb):
        return 1-(rgb*rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)

    def project_gradient(self, gradient):
        # A small latent context halo permits VAE reconstruction at edit boundaries.
        support = F.adaptive_max_pool2d((self.edit > 0).float(), gradient.shape[-2:])
        return gradient*F.max_pool2d(support, 3, 1, 1)

    def __call__(self, image):
        ink = self.ink(image)
        remaining = F.relu(ink-self.background_ink)
        erase = CandidateObjective.mean_region(remaining, self.erase)
        # Penalize the darkest residual pixels as well as average remaining ink.
        active = remaining[self.erase > 0]
        tail = active.topk(max(1, int(active.numel()*.1))).values.mean() if active.numel() else erase*0
        restore = CandidateObjective.mean_region(F.relu(self.required_ink-ink), self.restore)
        preserve = CandidateObjective.mean_region((image-self.source).square(), self.preserve)
        healthy_ink = self.preserve*(self.ink(self.source) > .2).float()
        healthy_mse = CandidateObjective.mean_region((image-self.source).square(), healthy_ink)
        loss = erase+.25*tail+restore+self.preserve_weight*preserve
        return loss, dict(erase_ink=float(erase.detach()), erase_tail=float(tail.detach()),
            restore_deficit=float(restore.detach()), preserved_region_mse=float(preserve.detach()),
            healthy_ink_rgb_mse=float(healthy_mse.detach()))


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
            if hasattr(objective, 'project_gradient'):
                gradient = objective.project_gradient(gradient)
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
             progress=None, guidance_mode='candidate', constraint_config=None, objective_override=None):
    """Identical source VAE encoding, noise, prompt, scheduler in both arms."""
    if not torch.cuda.is_available():
        raise RuntimeError('Select an L4 GPU runtime in Colab')
    if not 0 < strength <= 1 or steps < 4 or int(steps*strength) < 2:
        raise ValueError('Need at least two actual re-denoising steps')
    if inner_steps < 1 or step_ratio <= 0 or cfg_scale <= 1:
        raise ValueError('Invalid guidance configuration')
    if guidance_mode not in ('candidate', 'ink_constraints', 'adaptive_ink'):
        raise ValueError('Unknown guidance mode')
    if (guidance_mode == 'adaptive_ink') != (objective_override is not None):
        raise ValueError('Adaptive mode requires an explicitly prepared stroke objective')
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
    source_rgb, candidate_rgb = image_tensor(source, 'cuda'), image_tensor(candidate, 'cuda')
    if objective_override is not None and (objective_override.source.shape != source_rgb.shape or
            not torch.allclose(objective_override.source, source_rgb, atol=1e-6)):
        raise ValueError('Adaptive objective must be prepared from this exact original input')
    objective = objective_override if objective_override is not None else (InkConstraintObjective(source_rgb, candidate_rgb, **(constraint_config or {}))
                 if guidance_mode == 'ink_constraints' else CandidateObjective(source_rgb, candidate_rgb))
    Image.fromarray((objective.edit[0, 0].cpu().numpy()*255).astype('uint8')).save(out/'edit_region.png')
    if guidance_mode in ('ink_constraints', 'adaptive_ink'):
        for name in ('erase', 'restore', 'preserve'):
            Image.fromarray(np.rint(getattr(objective, name)[0, 0].cpu().numpy()*255).astype('uint8')).save(out/f'{name}_region.png')
        (out/'constraints.json').write_text(json.dumps(objective.metadata, indent=2), encoding='utf-8')
    guided_name = {'candidate': 'candidate_guided', 'ink_constraints': 'ink_guided', 'adaptive_ink': 'adaptive_guided'}[guidance_mode]
    if hasattr(objective, 'export_state'):
        objective.export_state(out/'ownership_initial.npz')
    traces, metrics = {}, {}
    for method in ('redenoise_only', guided_name):
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
            if method == guided_name and i/max(len(timesteps)-1, 1) >= .2:
                if hasattr(objective, 'maybe_refresh'):
                    with torch.no_grad():
                        refresh = objective.maybe_refresh(decode(clean), i)
                    if refresh is not None:
                        record['constraint_refresh'] = refresh
                        for name in ('erase', 'restore'):
                            mask = getattr(objective, name)[0, 0].detach().cpu().numpy()
                            Image.fromarray(np.rint(mask*255).clip(0, 255).astype('uint8')).save(out/f'{name}_step_{i:03d}.png')
                        if hasattr(objective, 'export_state'):
                            objective.export_state(out/f'ownership_step_{i:03d}.npz')
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
            evaluate = getattr(objective, 'evaluate_fixed', objective)
            loss, values = evaluate(decode(latent))
        metrics[method] = dict(loss=float(loss), **values)
        traces[method] = trace
    with torch.no_grad():
        loss, values = getattr(objective, 'evaluate_fixed', objective)(objective.source)
    metrics['input'] = dict(loss=float(loss), **values)
    report = dict(metadata=metadata, model_id=model_id, revision=revision,
        steps=steps, actual_steps=len(timesteps), strength=strength, seed=seed,
        cfg_scale=cfg_scale, inner_steps=inner_steps, step_ratio=step_ratio,
        initial_latent_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        paired_initial_latent=True, guidance_target=guidance_mode,
        constraints=getattr(objective, 'metadata', None),
        constraint_history=getattr(objective, 'history', None),
        target_pixels_pasted_into_output=False, metrics=metrics, traces=traces,
        metric_note=('Fixed ink constraints and source preservation; not GT accuracy or HSQM score'
                     if guidance_mode in ('ink_constraints', 'adaptive_ink') else 'Candidate agreement, not ground-truth accuracy or HSQM score'))
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    del pipe
    torch.cuda.empty_cache()
    return report
