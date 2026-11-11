"""Paired DDIM with explicit missing-ink projection and restore/settle/erase phases."""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .fitted_candidate_diffusion import image_tensor, optimize_clean
from .selectable_adaptive_sweep import fit_case
from stroke_model.restore_settle_erase import RestoreSettleErase


def run_cycle_pair(source, candidate, controller, output, metadata, *, steps=60,
                   strength=.25, seed=2026, revision=None, progress=None):
    if not torch.cuda.is_available():raise RuntimeError('Select Colab L4 GPU')
    if not 0<strength<=1 or steps<4 or int(steps*strength)<4:raise ValueError('Need at least four actual steps')
    if source.size!=candidate.size or any(n%8 for n in source.size):raise ValueError('Matching image dimensions divisible by 8 required')
    source_rgb=image_tensor(source,controller.source.device)
    if source_rgb.shape!=controller.source.shape or not torch.allclose(source_rgb,controller.source,atol=1e-6):
        raise ValueError('Controller must use this exact original source image')
    from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source.save(out/'input.png');candidate.save(out/'candidate_diagnostic_only.png')
    kwargs=dict(torch_dtype=torch.float16,use_safetensors=True)
    if revision:kwargs['revision']=revision
    pipe=StableDiffusionImg2ImgPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',**kwargs)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False,thresholding=False)
    pipe.to('cuda');pipe.enable_attention_slicing();pipe.enable_vae_slicing()
    for model in (pipe.unet,pipe.vae,pipe.text_encoder):model.eval().requires_grad_(False)
    with torch.no_grad():
        positive,negative=pipe.encode_prompt(metadata['prompt'],'cuda',1,True,negative_prompt=metadata.get('negative_prompt',''))
        embeddings=torch.cat([negative,positive])
        pipe.scheduler.set_timesteps(steps,device='cuda')
        timesteps,_=pipe.get_timesteps(steps,strength,'cuda')
        initial=pipe.prepare_latents(pipe.image_processor.preprocess(source),timesteps[:1],1,1,
            embeddings.dtype,'cuda',torch.Generator('cpu').manual_seed(seed))
    pipe.vae.to(dtype=torch.float32)
    def decode(z):return pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor,return_dict=False)[0].float()/2+.5
    def encode(rgb):return pipe.vae.encode(rgb.float()*2-1).latent_dist.mode()*pipe.vae.config.scaling_factor
    def save_tensor(rgb,path):
        array=rgb[0].detach().permute(1,2,0).clamp(0,1).cpu().numpy()
        Image.fromarray(np.rint(array*255).astype('uint8')).save(path)
    def save_mask(mask,path):
        Image.fromarray(np.rint(mask[0,0].detach().cpu().numpy()*255).clip(0,255).astype('uint8')).save(path)
    save_mask(controller.restore,out/'initial_missing_band.png')
    traces={};metrics={}
    for arm in ('redenoise_only','cyclic_guided'):
        scheduler=deepcopy(pipe.scheduler);latent=initial.detach().float().clone();trace=[]
        for index,t in enumerate(timesteps):
            with torch.no_grad():
                inputs=scheduler.scale_model_input(torch.cat([latent,latent]),t)
                u,c=pipe.unet(inputs.half(),t,encoder_hidden_states=embeddings,return_dict=False)[0].chunk(2)
                base=scheduler.step((u+5.*(c-u)).float(),t,latent,eta=0.,return_dict=True)
            clean=base.pred_original_sample.detach();latent=base.prev_sample.detach()
            row=dict(step=index,timestep=int(t),phase='baseline',accepted_inner_steps=0)
            if arm=='cyclic_guided':
                with torch.no_grad():row.update(controller.begin_step(decode(clean),index))
                if row['phase']=='restore':
                    clean,projection=controller.project_clean(clean,decode,encode)
                    row.update(projection)
                    with torch.no_grad():save_tensor(decode(clean),out/f'inserted_step_{index:03d}.png')
                clean,optimization=optimize_clean(clean,decode,controller,
                    iterations=2 if row['phase'] in ('restore','settle') else 4,
                    step_ratio=.008,budget_ratio=.06,backtracks=5)
                row.update(optimization)
                with torch.no_grad():controller.finish_step(decode(clean),row)
                # Preserve the predicted noise residual when inserting corrected x0.
                previous=int(t)-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
                alpha=(scheduler.alphas_cumprod[previous] if previous>=0 else scheduler.final_alpha_cumprod).to(latent)
                latent=latent+alpha.sqrt()*(clean-base.pred_original_sample)
                save_mask(controller.erase,out/f'erase_step_{index:03d}.png')
                save_mask(controller.restore,out/f'restore_step_{index:03d}.png')
            latent=latent.detach();trace.append(row)
            with torch.no_grad():save_tensor(decode(clean),out/f'{arm}_step_{index:03d}_{row["phase"]}.png')
            if progress:progress(arm,index+1,len(timesteps),row)
        with torch.no_grad():
            final=decode(latent);save_tensor(final,out/f'{arm}.png');metrics[arm]=controller.diagnostics(final)
        traces[arm]=trace
    metrics['input']=controller.diagnostics(controller.source)
    final_ok=metrics['cyclic_guided']['all_paths_present']
    erasure_ran=any(r['phase']=='erase' for r in traces['cyclic_guided'])
    report=dict(metadata=metadata,configuration=controller.metadata,steps=steps,actual_steps=len(timesteps),
        strength=strength,seed=seed,revision=revision,initial_latent_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        paired_initial_latent=True,metrics=metrics,traces=traces,
        status=('restored_and_cleaned' if final_ok and erasure_ran else 'paths_present_without_erasure' if final_ok else 'restoration_incomplete'),
        erasure_ran=erasure_ran,missing_path_ink_inserted=True,candidate_rgb_copied=False,
        note='Actual decoded diffusion output. Coverage is reference-relative, not correctness or generalization proof.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    del pipe;gc.collect();torch.cuda.empty_cache()
    return report


def repair_cycle_cases(cases, extractor, *, checkpoint_sha256, revision, seed=2026,
                       steps=60, strength=.25, on_case=None, progress=None):
    summaries=[]
    for case in cases:
        row={k:case[k] for k in ('case_id','group_id','control','strength','seed','input','output')}
        try:
            source,candidate,records=fit_case(case,extractor)
            device=next(extractor.parameters(),torch.tensor(0.)).device
            controller=RestoreSettleErase(image_tensor(source,device),extractor,records)
            metadata=dict(prompt=case['prompt'],negative_prompt=case['negative_prompt'],
                checkpoint_sha256=checkpoint_sha256,original_manifest_rows=case['group']['original_rows'],
                case={k:v for k,v in case.items() if k!='group'})
            report=run_cycle_pair(source,candidate,controller,Path(case['output'])/'cycle',metadata,
                steps=steps,strength=strength,seed=seed,revision=revision,progress=progress)
            row.update(status=report['status'],guided=str(Path(case['output'])/'cycle/cyclic_guided.png'),
                baseline=str(Path(case['output'])/'cycle/redenoise_only.png'),
                candidate=str(Path(case['output'])/'fitted_candidate.png'),metrics=report['metrics'])
        except Exception as exc:
            row.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            Path(case['output']).mkdir(parents=True,exist_ok=True)
            (Path(case['output'])/'failure.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        summaries.append(row)
        (Path(case['group']['root']).parent/'all_results.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
        if on_case:on_case(row)
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return summaries
