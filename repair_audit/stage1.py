"""First-stage diagnosis of the existing four-glyph cyclic repair, without training.

Reads data (never runtime code) from existing sweep/cyclic ZIPs. Automatic width
and reference-distance measurements are proxies, not character correctness.
"""
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _rgb(z, name):
    with Image.open(io.BytesIO(z.read(name))) as image:
        return image.convert('RGB').copy()


def _conditions(z, group):
    result = []
    for slot in range(1, 5):
        member = f'{group}/source_conditions/condition_{slot}.npz'
        with np.load(io.BytesIO(z.read(member)), allow_pickle=False) as arrays:
            reference = arrays['reference'].astype(np.float32)
            strokes = arrays['strokes'].astype(np.float32)
        if (reference.ndim != 3 or reference.shape[0] != 1 or
                reference.shape[1] != reference.shape[2] or strokes.ndim != 3 or
                strokes.shape[1:] != reference.shape[1:] or not len(strokes)):
            raise ValueError(f'Invalid reference/stroke dimensions: {member}')
        for value in (reference, strokes):
            if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError(f'Expected finite [0,1] ink arrays: {member}')
        if reference.max() <= .5 or (strokes.max((1, 2)) <= .5).any():
            raise ValueError(f'Empty glyph/stroke: {member}')
        result.append(dict(reference=reference, strokes=strokes))
    return result


def read_archive(path, output, selected_case_ids=None, include_controls=True):
    """Materialize only named PNG/NPZ data, ignoring saved absolute paths/code."""
    with zipfile.ZipFile(path) as check:
        if 'archive_metadata.json' in check.namelist():
            return read_stage1_archive(path, output, selected_case_ids, include_controls)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    selected = None if selected_case_ids is None else set(selected_case_ids)
    with zipfile.ZipFile(path) as z:
        experiment = json.loads(z.read('experiment.json'))
        expected = experiment.get('checkpoint_sha256', '')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
            raise ValueError('Archive experiment.json must record the extractor SHA256')
        cases = json.loads(z.read('cases.json'))
        available = {c['case_id'] for c in cases if not c.get('control')}
        if selected is not None and (not selected or selected - available):
            raise ValueError(f'Unknown/empty case selection; available: {sorted(available)}')
        chosen = [c for c in cases if not c.get('control') and
                  (selected is None or c['case_id'] in selected)]
        if not chosen:
            raise ValueError('No generated cases in this archive')
        prepared, controls = [], set()
        destination.mkdir(parents=True)
        for item in chosen:
            group, case_id = item['group_id'], item['case_id']
            if not re.fullmatch(r'group_[0-9]+', group) or not re.fullmatch(r'[A-Za-z0-9_-]+', case_id):
                raise ValueError('Invalid archive group/case identifier')
            conditions = _conditions(z, group)
            report = json.loads(z.read(f'{group}/generation/generation_report.json'))
            basename = PurePosixPath(str(item['input']).replace('\\', '/')).name
            matches = [p for p in report['pages'] if
                       PurePosixPath(p['input']).name == basename and not p.get('control')]
            if len(matches) != 1:
                raise ValueError(f'Generated page is not uniquely recorded: {case_id}')
            pages = [(case_id, matches[0], 'generated')]
            if include_controls and group not in controls:
                candidates = [p for p in report['pages'] if p.get('control') and p['glyph_count'] == 4]
                if len(candidates) != 1:
                    raise ValueError('Need one recorded four-glyph normal input control')
                pages.append((group + '_normal_control', candidates[0], 'normal_control'))
                controls.add(group)
            for identifier, page, kind in pages:
                if page['glyph_count'] != 4 or page.get('safety_filtered'):
                    raise ValueError('Requires a non-filtered four-glyph page')
                member = f"{group}/generation/page_{page['page_id']:03d}.png"
                source = _rgb(z, member)
                if source.width != source.height or source.width % 16:
                    raise ValueError('Expected square four-cell image with side divisible by 16')
                target = destination / identifier
                target.mkdir()
                source.save(target / 'input.png')
                for slot, condition in enumerate(conditions, 1):
                    np.savez_compressed(target / f'condition_{slot}.npz', **condition)
                metadata = dict(case_id=identifier, kind=kind, archive_member=member,
                    source_sha256=hashlib.sha256(z.read(member)).hexdigest(),
                    prompt=report.get('prompts', {}).get('4', item.get('prompt', '')),
                    negative_prompt=report.get('negative_prompt', ''),
                    generation_seed=page.get('seed'), generation_strength=page.get('strength'),
                    checkpoint_sha256=expected.lower())
                json_write(target / 'input_metadata.json', metadata)
                prepared.append(dict(directory=str(target), **metadata))
    json_write(destination / 'archive_metadata.json', dict(archive_sha256=file_hash(path),
        experiment=experiment, selected_cases=prepared))
    return prepared, experiment


def read_stage1_archive(path, output, selected_case_ids=None, include_controls=True):
    """Reuse original inputs from an audit ZIP, never the already guided outputs."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(path) as z:
        metadata = json.loads(z.read('archive_metadata.json'))
        experiment = metadata['experiment']
        expected = experiment.get('checkpoint_sha256', '')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
            raise ValueError('Missing original extractor SHA256')
        recorded = metadata['selected_cases']
        available = {c['case_id'] for c in recorded if c['kind'] == 'generated'}
        selected = available if selected_case_ids is None else set(selected_case_ids)
        if not selected or selected - available:
            raise ValueError(f'Unknown/empty audit selection; available: {sorted(available)}')
        chosen = [c for c in recorded if c['case_id'] in selected]
        groups = {re.fullmatch(r'(group_[0-9]+)_page_[0-9]+', c['case_id']).group(1)
                  for c in chosen if re.fullmatch(r'(group_[0-9]+)_page_[0-9]+', c['case_id'])}
        if len(groups) == 0:
            raise ValueError('Unsupported audit case identifiers')
        if include_controls:
            controls = {c['case_id']: c for c in recorded if c['kind'] == 'normal_control'}
            for group in sorted(groups):
                name = group + '_normal_control'
                if name not in controls:
                    raise ValueError('Audit ZIP lacks the matching normal control')
                chosen.append(controls[name])
        prepared = []
        destination.mkdir(parents=True)
        for entry in chosen:
            identifier = entry['case_id']
            if not re.fullmatch(r'group_[0-9]+_(page_[0-9]+|normal_control)', identifier):
                raise ValueError('Invalid audit identifier')
            target = destination / identifier
            target.mkdir()
            image = _rgb(z, identifier + '/input.png')
            if image.width != image.height or image.width % 16:
                raise ValueError('Invalid input sheet dimensions')
            image.save(target / 'input.png')
            # Same validation as the original four source conditions.
            for slot in range(1, 5):
                with np.load(io.BytesIO(z.read(f'{identifier}/condition_{slot}.npz')), allow_pickle=False) as a:
                    ref, strokes = a['reference'].astype(np.float32), a['strokes'].astype(np.float32)
                if (ref.ndim != 3 or ref.shape[0] != 1 or ref.shape[1] != ref.shape[2] or
                        strokes.ndim != 3 or not len(strokes) or strokes.shape[1:] != ref.shape[1:]):
                    raise ValueError('Invalid condition shape')
                if any(not np.isfinite(v).all() or v.min() < 0 or v.max() > 1 for v in (ref, strokes)):
                    raise ValueError('Invalid condition values')
                if ref.max() <= .5 or (strokes.max((1, 2)) <= .5).any():
                    raise ValueError('Empty condition')
                np.savez_compressed(target/f'condition_{slot}.npz', reference=ref, strokes=strokes)
            row = dict(entry, directory=str(target), checkpoint_sha256=expected.lower(),
                       reused_stage1_original=True)
            json_write(target/'input_metadata.json', row)
            prepared.append(row)
    json_write(destination/'archive_metadata.json', dict(archive_sha256=file_hash(path),
        experiment=experiment, selected_cases=prepared, original_input_reused=True))
    return prepared, experiment


def load_conditions(directory):
    result = []
    for slot in range(1, 5):
        with np.load(Path(directory) / f'condition_{slot}.npz', allow_pickle=False) as a:
            result.append(dict(reference=a['reference'].copy(), strokes=a['strokes'].copy()))
    return result


def fit_records(source, conditions, extractor, output, *, fit=True):
    """Same fitting parameters as the existing cyclic notebook."""
    import torch
    from torch.nn import functional as F
    from stroke_model.stroke_reference_fit import fit_reference_glyph, ReferenceFitConfig
    from diffusion_model.fitted_candidate_diffusion import image_tensor
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(extractor.parameters()).device
    rgb = image_tensor(source, device)
    ink = 1 - (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    side = source.width // 2
    records = []
    for i, condition in enumerate(conditions):
        ref, strokes = condition['reference'], condition['strokes']
        x, y = (i % 2) * side, (i // 2) * side
        crop = F.interpolate(ink[..., y:y+side, x:x+side], ref.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            prediction = extractor(crop, torch.as_tensor(ref, device=device)[None],
                                   torch.as_tensor(strokes, device=device)[None])
        masks = prediction['masks'][0].cpu().numpy()
        if not fit:
            np.savez_compressed(directory / f'slot_{i+1}.npz', **condition,
                extracted_masks=masks, probabilities=prediction['probabilities'][0].cpu().numpy())
            continue
        hints = prediction.get('hints', torch.as_tensor(strokes, device=device)[None])[0].cpu().numpy()
        fitted = fit_reference_glyph(strokes, masks, hints=hints, glyph_reference=ref,
            config=ReferenceFitConfig(max_displacement=16., smoothness=12., shape_prior=.15, max_junction_shift=2.))
        record = dict(reference=ref, strokes=strokes, fitted=fitted['strokes'], extracted_masks=masks,
                      fit_metadata=[r['metadata'] for r in fitted['fits']])
        records.append(record)
        np.savez_compressed(directory / f'slot_{i+1}.npz', **condition,
            extracted_masks=masks, fitted_strokes=fitted['strokes'],
            probabilities=prediction['probabilities'][0].cpu().numpy())
        json_write(directory / f'slot_{i+1}_fit.json', record['fit_metadata'])
    return records


def ink_array(image):
    a = np.asarray(image, np.float32) / 255
    return 1 - (a * np.array([.299, .587, .114], np.float32)).sum(-1)


def fixed_support(source, controller):
    """Freeze source-based diagnostic support before comparing ANY output arm."""
    initial_edit = np.maximum(controller.erase.cpu().numpy()[0, 0], controller.restore.cpu().numpy()[0, 0]) > .01
    preserve = ~binary_dilation(initial_edit, iterations=3)
    source_ink = ink_array(source)
    healthy = (source_ink > .5) & preserve
    center = skeletonize(source_ink > .5) & healthy
    return dict(healthy=healthy, center=center, preserve=preserve, initial_edit=initial_edit)


def measure(source, result, reference, support, threshold=.5):
    """Fixed-coordinate width proxy; movement/thresholds confound it. No OCR claim."""
    original, output, ref = map(ink_array, (source, result, reference))
    healthy, center = support['healthy'], support['center']
    source_radius = distance_transform_edt(original > threshold)[center]
    output_radius = distance_transform_edt(output > threshold)[center]
    ratio = float(output_radius.mean() / max(source_radius.mean(), 1e-8)) if len(source_radius) else None
    union = (original > .2) | (ref > .2)
    def average(values, mask):
        return float(values[mask].mean()) if mask.any() else None
    return dict(threshold=threshold, fixed_center_samples=int(center.sum()),
        healthy_ink_pixels=int(healthy.sum()),
        fixed_center_width_ratio_proxy=ratio,
        healthy_ink_drop=average(np.maximum(original-output, 0), healthy),
        healthy_ink_retention=average((output > threshold).astype(float), healthy),
        healthy_ink_mse=average((original-output)**2, healthy),
        reference_distance_on_fixed_union=average((output-ref)**2, union),
        source_reference_distance_on_fixed_union=average((original-ref)**2, union))


def reference_sheet(conditions, size):
    result = Image.new('RGB', (size, size), 'white')
    half = size // 2
    for i, c in enumerate(conditions):
        glyph = Image.fromarray(np.rint((1-c['reference'][0])*255).astype(np.uint8)).convert('RGB')
        result.paste(glyph.resize((half, half), Image.Resampling.BICUBIC), ((i % 2)*half, (i//2)*half))
    return result


def width_probe_conditions(conditions):
    """Optional +1 model-pixel dilation stress test, NOT a validated new font."""
    return [dict(reference=maximum_filter(c['reference'], size=(1, 3, 3)),
                 strokes=maximum_filter(c['strokes'], size=(1, 3, 3))) for c in conditions]


def run_case(pipe, extractor, case, *, steps=60, strength=.25, seed=2026,
             cfg_scale=5., width_probe=False, progress=None,
             boundary_guard=False, boundary_config=None, observed_width=False, width_config=None):
    import torch
    from diffusion_model.fitted_candidate_diffusion import image_tensor, optimize_clean
    from stroke_model.restore_settle_erase import RestoreSettleErase
    if steps < 4 or not 0 < strength < 1 or int(steps*strength) < 4:
        raise ValueError('Need at least four actual repair steps')
    directory = Path(case['directory'])
    if observed_width and not boundary_guard:
        raise ValueError('Observed-width comparison requires boundary_guard=True')
    out = directory / 'audit'
    out.mkdir(exist_ok=False)
    with Image.open(directory / 'input.png') as im:
        source = im.convert('RGB').copy()
    conditions = load_conditions(directory)
    records = fit_records(source, conditions, extractor, out / 'extraction_input')
    source_rgb = image_tensor(source, next(extractor.parameters()).device)
    controller = RestoreSettleErase(source_rgb, extractor, records)
    with torch.no_grad():
        controller.refresh_erase(source_rgb)  # Diagnostic only; first restore step clears erasure.
    support = fixed_support(source, controller)
    np.savez_compressed(out / 'fixed_diagnostic_regions.npz', **support)
    # Reset the controller: diagnostic setup must not change the repair experiment.
    controller = RestoreSettleErase(source_rgb, extractor, records)
    reference = reference_sheet(conditions, source.width)
    reference.save(out / 'reference_diagnostic_only.png')
    def decode(z):
        return pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor, return_dict=False)[0].float()/2+.5
    def encode(rgb):
        return pipe.vae.encode(rgb.float()*2-1).latent_dist.mode()*pipe.vae.config.scaling_factor
    def save_tensor(rgb, path):
        a = rgb[0].detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        Image.fromarray(np.rint(a*255).astype(np.uint8)).save(path)
    # Same frozen float32 VAE and posterior-mode encoding for every compared arm.
    # This is deliberately recorded: legacy cyclic used prepare_latents' sampled posterior.
    with torch.no_grad():
        clean_input = encode(source_rgb)
        save_tensor(decode(clean_input), out / 'vae_roundtrip.png')
        positive, negative = pipe.encode_prompt(case['prompt'], 'cuda', 1, True,
                                               negative_prompt=case['negative_prompt'])
        embeddings = torch.cat([negative, positive])
        pipe.scheduler.set_timesteps(steps, device='cuda')
        timesteps, _ = pipe.get_timesteps(steps, strength, 'cuda')
        noise = torch.randn(clean_input.shape, generator=torch.Generator('cpu').manual_seed(seed),
                            dtype=torch.float32).to(clean_input.device)
        initial = pipe.scheduler.add_noise(clean_input, noise, timesteps[:1])
    # Expose reconstruction before the actual rerun so failures still retain evidence.
    source.save(out / 'input.png')
    traces = {}
    arms = [('redenoise_only', None), ('cyclic_guided', controller)]
    if boundary_guard:
        from stroke_model.boundary_guarded_repair import BoundaryGuardConfig, BoundaryGuardedRepair
        guard_config = BoundaryGuardConfig(**(boundary_config or {}))
        guarded = BoundaryGuardedRepair(source_rgb, extractor, records, boundary_config=guard_config)
        with torch.no_grad():
            guarded.refresh_erase(source_rgb)
        np.savez_compressed(out/'boundary_initial_maps.npz',
            erase=guarded.erase.cpu().numpy(), restore=guarded.restore.cpu().numpy(),
            preserve=guarded.preserve.cpu().numpy())
        json_write(out/'boundary_initial_diagnostics.json', guarded.boundary_history)
        guarded = BoundaryGuardedRepair(source_rgb, extractor, records, boundary_config=guard_config)
        arms.append(('boundary_guarded', guarded))
    if observed_width:
        from stroke_model.observed_width_repair import ObservedWidthConfig, ObservedWidthRepair
        def calibrated_controller():
            return ObservedWidthRepair(source_rgb, extractor, records,
                boundary_config=guard_config, width_config=ObservedWidthConfig(**(width_config or {})))
        calibrated = calibrated_controller()
        with torch.no_grad():
            calibrated.refresh_erase(source_rgb)
        np.savez_compressed(out/'observed_width_initial_maps.npz',
            erase=calibrated.erase.cpu().numpy(), restore=calibrated.restore.cpu().numpy(),
            preserve=calibrated.preserve.cpu().numpy())
        json_write(out/'observed_width_calibration.json', calibrated.metadata)
        json_write(out/'observed_width_initial_diagnostics.json', calibrated.boundary_history)
        arms.append(('observed_width', calibrated_controller()))
    if width_probe:
        alternate = width_probe_conditions(conditions)
        alt_records = fit_records(source, alternate, extractor, out / 'width_probe_input')
        arms.append(('reference_width_probe', RestoreSettleErase(source_rgb, extractor, alt_records)))
    for arm, objective in arms:
        scheduler = deepcopy(pipe.scheduler)
        latent = initial.clone()
        trace = []
        frames = out / arm
        frames.mkdir()
        for index, t in enumerate(timesteps):
            with torch.no_grad():
                model_input = scheduler.scale_model_input(torch.cat([latent, latent]), t)
                u, c = pipe.unet(model_input.to(dtype=next(pipe.unet.parameters()).dtype), t,
                    encoder_hidden_states=embeddings, return_dict=False)[0].chunk(2)
                base = scheduler.step((u+cfg_scale*(c-u)).float(), t, latent, eta=0., return_dict=True)
            clean, latent = base.pred_original_sample.detach(), base.prev_sample.detach()
            row = dict(step=index, timestep=int(t), phase='baseline', accepted_inner_steps=0)
            if objective is not None:
                with torch.no_grad():
                    row.update(objective.begin_step(decode(clean), index))
                if row['phase'] == 'restore':
                    clean, projection = objective.project_clean(clean, decode, encode)
                    row.update(projection)
                clean, optimization = optimize_clean(clean, decode, objective,
                    iterations=2 if row['phase'] in ('restore', 'settle') else 4,
                    step_ratio=.008, budget_ratio=.06, backtracks=5)
                row.update(optimization)
                with torch.no_grad():
                    objective.finish_step(decode(clean), row)
                previous = int(t) - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
                alpha = (scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod).to(latent)
                latent = latent + alpha.sqrt() * (clean-base.pred_original_sample)
                for name in ('erase', 'restore', 'preserve'):
                    value = getattr(objective, name)[0, 0].detach().cpu().numpy()
                    Image.fromarray(np.rint(value.clip(0, 1)*255).astype(np.uint8)).save(frames / f'{name}_{index:03d}.png')
            with torch.no_grad():
                save_tensor(decode(clean), frames / f'clean_prediction_{index:03d}.png')
            latent = latent.detach()
            trace.append(row)
            if progress:
                progress(case['case_id'], arm, index+1, len(timesteps), row)
        with torch.no_grad():
            save_tensor(decode(latent), out / (arm+'.png'))
        traces[arm] = trace
        json_write(out / (arm+'_trace.json'), trace)
        if objective is not None and hasattr(objective, 'boundary_history'):
            json_write(out/(arm+'_map_history.json'), objective.boundary_history)
    results = {'input': source, 'vae_roundtrip': Image.open(out / 'vae_roundtrip.png').convert('RGB')}
    for arm, _ in arms:
        results[arm] = Image.open(out / (arm+'.png')).convert('RGB')
    metric_rows = []
    for arm, result in results.items():
        # Save final extraction for VISUAL review; masks are not quality probabilities.
        if arm != 'input':
            fit_records(result, conditions, extractor, out / ('extraction_'+arm), fit=False)
        for slot in range(4):
            half = source.width // 2
            x, y = slot % 2 * half, slot // 2 * half
            box = (x, y, x+half, y+half)
            local = {k: v[y:y+half, x:x+half] for k, v in support.items()}
            for threshold in (.35, .5, .65):
                values = measure(source.crop(box), result.crop(box), reference.crop(box), local, threshold)
                if boundary_guard:
                    # Include all source foreground, including tips excluded by
                    # the original automatic healthy mask. This still isn't GT.
                    ink = ink_array(source.crop(box))
                    all_source = dict(healthy=ink > .5, center=skeletonize(ink > .5))
                    full = measure(source.crop(box), result.crop(box), reference.crop(box), all_source, threshold)
                    values.update({'all_source_'+k:full[k] for k in
                        ('fixed_center_width_ratio_proxy','healthy_ink_drop','healthy_ink_retention','healthy_ink_mse')})
                metric_rows.append(dict(case_id=case['case_id'], kind=case['kind'], slot=slot+1, arm=arm, **values))
    json_write(out / 'report.json', dict(case=case, metrics=metric_rows, traces=traces,
        configuration=dict(steps=steps, actual_steps=len(timesteps), timesteps=[int(t) for t in timesteps],
            strength=strength, seed=seed, cfg_scale=cfg_scale, vae_dtype='float32',
            vae_posterior='mode_shared_by_all_arms', paired_initial_latent=True,
            initial_latent_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
            width_probe=width_probe, original_repair_parameters=True,
            boundary_guard=boundary_guard, boundary_config=boundary_config if boundary_guard else None,
            observed_width=observed_width, width_config=width_config if observed_width else None,
            legacy_difference='Posterior mode, not legacy stochastic VAE encoding'),
        limitations=['Widths are fixed-source-center distance proxies; movement is a confound.',
            'Healthy masks are heuristic and require human review.',
            'Reference similarity is not correctness and cannot prove template copying.',
            'No automatic pass/fail or independent text accuracy claim.',
            'Width probe is synthetic dilation and may be outside extractor training distribution.']))
    make_figures(out, source, reference, results, support)
    return metric_rows


def make_figures(out, source, reference, results, support):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = list(results)
    fig, axes = plt.subplots(1, len(labels)+1, figsize=(4*(len(labels)+1), 4), squeeze=False)
    for ax, (name, image) in zip(axes.flat, [('reference (diagnostic)', reference), *results.items()]):
        ax.imshow(image); ax.set_title(name); ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'comparison.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ('healthy', 'center', 'initial_edit')):
        ax.imshow(source); ax.imshow(np.ma.masked_where(~support[key], support[key]), cmap='autumn', alpha=.6)
        ax.set_title(key+' (heuristic)'); ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'diagnostic_regions.png', dpi=120); plt.close(fig)
    for slot in range(1, 5):
        with np.load(out / 'extraction_input' / f'slot_{slot}.npz') as a:
            count = len(a['extracted_masks'])
        fig, axes = plt.subplots(len(results), count+2,
            figsize=(2.2*(count+2), 2.3*len(results)), squeeze=False)
        half=source.width//2; x=(slot-1)%2*half; y=(slot-1)//2*half
        for row, (arm, image) in enumerate(results.items()):
            with np.load(out / ('extraction_'+arm) / f'slot_{slot}.npz') as a:
                images = [('reference', 1-a['reference'][0]),
                          (arm, np.asarray(image.crop((x,y,x+half,y+half))))] + [
                    (f'stroke {i+1}', 1-m) for i,m in enumerate(a['extracted_masks'])]
            for ax, (name, value) in zip(axes[row], images):
                ax.imshow(value, cmap='gray', vmin=0, vmax=1) if value.ndim==2 else ax.imshow(value)
                ax.set_title(name); ax.axis('off')
        fig.tight_layout(); fig.savefig(out / f'extraction_slot_{slot}.png', dpi=110); plt.close(fig)


def write_tables(root, cases, metrics, failures):
    root = Path(root)
    if metrics:
        with (root / 'automatic_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader(); writer.writerows(metrics)
    fields = ['case_id', 'kind', 'slot', 'arm', 'target_character', 'input_error_type',
              'extraction_ok', 'error_reduced', 'thinner', 'new_error', 'style_changed', 'notes']
    with (root / 'human_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            arms = sorted({r['arm'] for r in metrics if r['case_id'] == case['case_id']})
            for slot in range(1, 5):
                for arm in arms:
                    writer.writerow(dict(case_id=case['case_id'], kind=case['kind'], slot=slot, arm=arm))
    json_write(root / 'failures.json', failures)
