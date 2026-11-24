"""Paired DDIM sampling with direct per-stroke x0 projection, no backward pass.

Supply a loaded SD1.5 img2img pipeline with DDIM scheduler and float32 VAE.
This is a sampling experiment, not a trained glyph-condition model.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def run_projected_pair(pipe, source, constraints, output, *, prompt='', negative_prompt='',
                       steps=60,strength=.25,seed=2026,cfg_scale=5.,start_from_noise=False,
                       save_final_projection=False,progress=None):
    from PIL import Image
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if not 0<strength<=1 or steps<1 or (not start_from_noise and int(steps*strength)<1) or any(v%8 for v in source.size):
        raise ValueError('Invalid schedule or image size')
    if type(pipe.scheduler).__name__ != 'DDIMScheduler':
        raise ValueError('This experimental sampler supports DDIM only')
    device=next(pipe.unet.parameters()).device
    rgb=torch.tensor(np.asarray(source.convert('RGB')).copy(),device=device).permute(2,0,1)[None].float()/255
    constraints.diagnostics(rgb)
    def encode(x):return pipe.vae.encode(x.float()*2-1).latent_dist.mode()*pipe.vae.config.scaling_factor
    def decode(z):return (pipe.vae.decode(z.float()/pipe.vae.config.scaling_factor,return_dict=False)[0].float()/2+.5).clamp(0,1)
    def save(x,path):
        Image.fromarray(np.rint(x[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype('uint8')).save(path)
    traces={}
    with torch.no_grad():
        positive,negative=pipe.encode_prompt(prompt,device,1,True,negative_prompt=negative_prompt)
        embeddings=torch.cat([negative,positive])
        pipe.scheduler.set_timesteps(steps,device=device)
        if start_from_noise:
            ts=pipe.scheduler.timesteps
        else:
            ts,_=pipe.get_timesteps(steps,strength,device)
        clean=encode(rgb)
        noise=torch.randn(clean.shape,generator=torch.Generator('cpu').manual_seed(seed)).to(device)
        initial=noise*pipe.scheduler.init_noise_sigma if start_from_noise else pipe.scheduler.add_noise(clean,noise,ts[:1])
        roi=torch.as_tensor(constraints.roi.copy(),device=device)[None,None].float()
        gate=F.adaptive_max_pool2d(roi,initial.shape[-2:])
        for arm in ('redenoise_only','stroke_projected'):
            scheduler=deepcopy(pipe.scheduler);latent=initial.clone();trace=[]
            for index,t in enumerate(ts):
                model_input=scheduler.scale_model_input(torch.cat([latent,latent]),t)
                u,c=pipe.unet(model_input.to(next(pipe.unet.parameters()).dtype),t,
                    encoder_hidden_states=embeddings,return_dict=False)[0].chunk(2)
                base=scheduler.step((u+cfg_scale*(c-u)).float(),t,latent,eta=0.,return_dict=True)
                latent=base.prev_sample.detach();row=dict(step=index,timestep=int(t))
                if arm=='stroke_projected':
                    prediction=decode(base.pred_original_sample)
                    target=constraints.project(prediction)
                    # Re-encoding is not an exact pixel projection. Record both.
                    corrected=base.pred_original_sample+gate*(encode(target)-base.pred_original_sample)
                    previous=int(t)-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
                    alpha=(scheduler.alphas_cumprod[previous] if previous>=0 else scheduler.final_alpha_cumprod).to(latent)
                    latent=latent+alpha.sqrt()*(corrected-base.pred_original_sample)
                    row.update(before=constraints.diagnostics(prediction),pixel_projected=constraints.diagnostics(target),
                               after_vae=constraints.diagnostics(decode(corrected)))
                trace.append(row)
                if progress:progress(arm,index+1,len(ts),row)
            final=decode(latent);save(final,out/(arm+'.png'))
            if arm=='stroke_projected' and save_final_projection:
                # Explicitly separate the raw diffusion output from the final
                # image-space enforcement. Never label this as raw generation.
                save(constraints.project(final),out/'stroke_projected_final_pixel_projection.png')
            traces[arm]=trace
    source.save(out/'input.png')
    report=dict(steps=steps,actual_steps=len(ts),strength=None if start_from_noise else strength,
        start_from_noise=start_from_noise,source_content_used=not start_from_noise,seed=seed,cfg_scale=cfg_scale,
        initial_latent_sha256=hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        paired_initial_latent=True,backward_passes=0,additional_training=False,traces=traces,
        guide=getattr(constraints,'metadata',dict(kind='per_stroke_ink_constraints')),
        final_pixel_projection_saved=save_final_projection,
        saved_png_ink_tolerance=1/255,
        limitations=['Target construction and extraction correspondence must be checked.',
                     'VAE re-encoding can change the target and leak changes beyond a pixel mask.',
                     'A separately requested final pixel projection is postprocessing, not raw generation.'])
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
