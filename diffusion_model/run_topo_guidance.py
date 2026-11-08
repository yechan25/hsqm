"""Paired four-glyph SD1.5 epsilon/DDIM experiment, fixed .65 / seed 100."""
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .guided_ddim import GuidanceConfig, decode_estimate, denoise
from .multiglyph_pilot import compose_sheet
from .score_pilot import scored_ink, training_catalog
from stroke_model.diffusion_guidance import TopoGuidance
from stroke_model.infer_topology import load_topology, export_hard_masks
from stroke_model.stroke_structure import StructureConfig


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def generate_guided_pair(manifest,manifest_rows,checkpoint,output,*,model_id=
        'stable-diffusion-v1-5/stable-diffusion-v1-5',revision=None,steps=30,canvas_size=576,
        guidance=None,structure_config=None,extra_weight=.05,stroke_focus=None,compare_uniform=True):
    if not torch.cuda.is_available():raise RuntimeError('Run this paired experiment on a Colab GPU')
    selected=list(manifest_rows)
    catalog={row['manifest_row']:row for row in training_catalog(manifest)}
    if len(selected)!=4 or len(set(selected))!=4 or any(row not in catalog for row in selected):
        raise ValueError('Select exactly four distinct training-reference rows')
    if type(steps) is not int or steps<4:raise ValueError('Need at least four nominal steps')
    config=(GuidanceConfig(**guidance) if isinstance(guidance,dict) else guidance or
            GuidanceConfig(clean_step_ratio=.01,end_fraction=1.,every=1))
    structure=StructureConfig(**structure_config) if isinstance(structure_config,dict) else structure_config or StructureConfig()
    focus=dict(focus_strength=2.,max_stroke_weight=3.)
    if stroke_focus is not None:focus.update(stroke_focus)
    initial_image,slots=compose_sheet([catalog[row] for row in selected],canvas_size)
    destination=Path(output)
    if destination.exists():raise FileExistsError(destination)
    model,size,epoch=load_topology(checkpoint,torch.device('cpu'))
    if any(slot['reference'].shape!=(1,size,size) for slot in slots):
        raise ValueError('Training reference and Topo checkpoint resolution mismatch')
    from diffusers import DDIMScheduler,StableDiffusionImg2ImgPipeline
    import diffusers
    kwargs=dict(torch_dtype=torch.float16,use_safetensors=True)
    if revision:kwargs['revision']=revision
    pipe=StableDiffusionImg2ImgPipeline.from_pretrained(model_id,**kwargs)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False,thresholding=False)
    pipe.to('cuda');pipe.enable_attention_slicing()
    for module in (pipe.unet,pipe.vae,pipe.text_encoder):module.eval().requires_grad_(False)
    device=pipe._execution_device
    prompt=('a typography sheet with 4 Korean Hangul glyphs arranged in 2 rows and 2 columns, '
            'black ink on plain white background, flat front view, no grid lines, '
            'preserve the glyphs and layout of the input image')
    negative='photograph, scenery, paper texture, shadows, colored background, grid lines, decorations'
    with torch.no_grad():
        positive,negative_embeds=pipe.encode_prompt(prompt,device,1,True,negative_prompt=negative)
        embeddings=torch.cat([negative_embeds,positive])
        pipe.scheduler.set_timesteps(steps,device=device)
        timesteps,effective_steps=pipe.get_timesteps(steps,.65,device)
        initial=pipe.prepare_latents(pipe.image_processor.preprocess(initial_image),timesteps[:1],
            1,1,embeddings.dtype,device,torch.Generator('cpu').manual_seed(100))
    # Encoding stays identical to ordinary fp16 img2img. Decode BOTH runs in fp32
    # for stable VAE/Topo input gradients; compare these paired runs, not old PNGs.
    pipe.vae.to(dtype=torch.float32)
    objective=TopoGuidance(model,slots,structure,extra_weight=extra_weight,**focus).to(device).eval()
    destination.mkdir(parents=True,exist_ok=False)
    checkpoint_hash=_file_hash(checkpoint)
    initial_hash=tensor_hash(initial)
    report=dict(experiment='paired_joint_topo_backward_guidance',status='running',model_id=model_id,
        requested_revision=revision,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=epoch,training_manifest=str(manifest),selected_manifest_rows=selected,
        seed=100,strength=.65,glyph_count=4,steps=steps,effective_steps=int(effective_steps),
        timesteps=[int(t) for t in timesteps],initial_latent_sha256=initial_hash,
        guidance_config=asdict(config),structure_config=asdict(structure),extra_weight=extra_weight,
        stroke_focus=focus,compare_uniform=bool(compare_uniform),
        primary_signal='frozen Topo soft-mask missing/split/junction',
        primary_score='StructuralIndicator on extracted masks with extra_weight=0',
        gradient_path='clean latent -> fp32 VAE -> four torch crops -> frozen Topo -> stroke loss',
        unet_backward=False,weights_updated=False,denoise_eta=0.,cfg_scale=7.5,
        vae_decode_dtype='float32',ddim_state_dtype='float32',
        unet_dtype=str(next(pipe.unet.parameters()).dtype).replace('torch.',''),
        diffusers_version=diffusers.__version__,torch_version=torch.__version__,
        prompt=prompt,negative_prompt=negative,scheduler=dict(pipe.scheduler.config),pages=[],results=[])

    def write_report():
        (destination/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    def save_page(image,method,filtered=None):
        page_id=len(report['pages']);path=destination/f'page_{page_id:03d}_{method}.png';image.save(path)
        label=f'{method} | 4 glyphs | strength=.65 seed=100'
        page=dict(page_id=page_id,method=method,label=label,input=str(path),glyph_count=4,
                  control=method=='reference',guidance_on=method in ('guided','uniform_guided'),initial_latent_sha256=initial_hash)
        report['pages'].append(page)
        for slot in slots:
            prefix=destination/f'sample_{len(report["results"]):04d}'
            crop=image.crop(tuple(slot['bbox']));crop.save(str(prefix)+'_raw.png')
            ink=scored_ink(crop,size)
            np.savez_compressed(str(prefix)+'.npz',image=ink,reference=slot['reference'],reference_strokes=slot['strokes'])
            report['results'].append(dict(**slot['entry'],input=str(prefix)+'_raw.png',npz=str(prefix)+'.npz',
                page_id=page_id,page_input=str(path),glyph_count=4,slot=slot['slot'],crop_bbox=slot['bbox'],
                method=method,guidance_on=method in ('guided','uniform_guided'),control=method=='reference',strength=.65,seed=100,
                safety_filtered=filtered,label=label+f' | slot={slot["slot"]+1}'))
        write_report()

    def callback(method):
        def step(record,before,after):
            print(f"{method} {record['step']+1}/{len(timesteps)} | guidance={record['accepted']} "
                  f"| loss={record.get('loss_before')} -> {record.get('loss_after')}",flush=True)
            with (destination/f'trace_{method}.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            if record['step']%config.snapshot_every==0 or record['step']==len(timesteps)-1:
                with torch.no_grad():
                    for tag,value in [('before',before),('after',after)] if record['accepted'] else [('estimate',after)]:
                        rgb=decode_estimate(pipe,value).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
                        Image.fromarray(np.rint(rgb*255).astype(np.uint8)).save(
                            destination/f'{method}_step_{record["step"]:02d}_{tag}.png')
        return step

    def finish(latents):
        with torch.no_grad():
            decoded=pipe.vae.decode(latents.float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
            checked,flags=pipe.run_safety_checker(decoded,device,embeddings.dtype)
            images=pipe.image_processor.postprocess(checked,output_type='pil',
                do_denormalize=[not flags[0]] if flags is not None else [True])
        return images[0],bool(flags[0]) if flags is not None else None

    try:
        save_page(initial_image,'reference')
        # Both receive copies of the same VAE-sampled, noised sheet; no re-seeding shortcut.
        methods=[('baseline',GuidanceConfig(noise_ratio=0))]
        if compare_uniform:methods.append(('uniform_guided',config))
        methods.append(('guided',config))
        for method,cfg in methods:
            objective.focus_strength=0. if method=='uniform_guided' else focus['focus_strength']
            latents,trace=denoise(pipe,initial,timesteps,embeddings,objective=objective,config=cfg,on_step=callback(method))
            report[method+'_final_latent_sha256']=tensor_hash(latents)
            if method=='baseline':baseline_final=latents.detach().float().cpu()
            else:
                delta=float((latents.detach().float().cpu()-baseline_final).square().mean().sqrt())
                report[method+'_final_latent_rms_difference']=delta
                if method=='guided':report['final_latent_rms_difference']=delta
            image,filtered=finish(latents);save_page(image,method,filtered)
            report[method+'_accepted_steps']=sum(row['accepted'] for row in trace)
        if tensor_hash(initial)!=initial_hash:raise RuntimeError('Initial latent changed')
        if _file_hash(checkpoint)!=checkpoint_hash:raise RuntimeError('Topo checkpoint changed')
        report['status']='complete';write_report()
    except Exception as exc:
        report.update(status='failed',error=repr(exc));write_report();raise
    finally:
        del objective,model,pipe
        gc.collect();torch.cuda.empty_cache()
    return report


def evaluate_topo_pair(report_path,checkpoint,*,show=True):
    """Primary numerical evaluation is based on predicted masks, not source ink."""
    import pandas as pd
    import matplotlib.pyplot as plt
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    if report.get('status')!='complete':raise ValueError('Paired generation is not complete')
    if _file_hash(checkpoint)!=report['checkpoint_sha256']:raise ValueError('Use the same frozen Topo checkpoint')
    slots=[]
    for sample in report['results'][:4]:
        with np.load(sample['npz']) as data:
            slots.append(dict(bbox=sample['crop_bbox'],reference=data['reference'].copy(),strokes=data['reference_strokes'].copy()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,size,epoch=load_topology(checkpoint,device)
    objective=TopoGuidance(model,slots,StructureConfig(**report['structure_config']),report['extra_weight'],
                           **report.get('stroke_focus',{})).to(device).eval()
    out=Path(report_path).parent/('topo_evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False);rows=[];stroke_rows=[];snapshots=[]
    for page in report['pages']:
        with Image.open(page['input']) as image:
            rgb=torch.from_numpy(np.asarray(image.convert('RGB'),np.float32).copy()/255).permute(2,0,1)[None].to(device)
        objective.focus_strength=(0. if page['method']=='uniform_guided' else report.get('stroke_focus',{}).get('focus_strength',0.))
        rgb.requires_grad_(True)
        with torch.enable_grad():loss,details=objective(rgb,return_masks=True)
        for values in details['per_glyph']:
            i=values['slot'];slot=slots[i]
            row={key:float(value.detach()) if torch.is_tensor(value) else value for key,value in values.items()
                 if key not in ('image','soft_masks','probabilities','per_stroke','defect_maps')}
            row.update(method=page['method'],sheet_loss=float(loss.detach()),primary_score='extracted stroke score, extra excluded')
            row['slot']=i+1  # Human-readable table; crop metadata remains zero-based.
            rows.append(row)
            arrays={key:values[value][0].detach().cpu().numpy() for key,value in
                    [('image','image'),('soft_masks','soft_masks'),('probabilities','probabilities')]}
            arrays.update(reference=slot['reference'],reference_strokes=slot['strokes'])
            gradients=[]
            for stroke in values['per_stroke']:
                stroke_row={key:float(value.detach()) if torch.is_tensor(value) else value for key,value in stroke.items()}
                stroke_row.update(method=page['method'],slot=i+1)
                stroke_rows.append(stroke_row)
                # Input-RGB derivative of THIS weighted stroke contribution; not a latent gradient.
                with torch.enable_grad():gradient,=torch.autograd.grad(stroke['contribution'],rgb,retain_graph=True)
                x0,y0,x1,y1=slot['bbox']
                gradient=torch.nn.functional.interpolate(gradient[...,y0:y1,x0:x1],
                    slot['reference'].shape[-2:],mode='bilinear',align_corners=False)
                gradients.append(gradient[0].mean(0).detach().cpu().numpy())
            arrays['stroke_rgb_gradients']=np.stack(gradients)
            for name in ('missing','split','junction'):
                arrays[name+'_maps']=np.stack([m[name][0,0].cpu().numpy() for m in values['defect_maps']])
            prefix=out/f'{page["method"]}_slot_{i+1}'
            np.savez_compressed(str(prefix)+'.npz',**arrays)
            sample=next(s for s in report['results'] if s['page_id']==page['page_id'] and s['slot']==i)
            snapshots.append({**sample,'npz':str(prefix)+'.npz'})
            masks=arrays['soft_masks'];hard=export_hard_masks(arrays['image'],masks,
                output_mode=getattr(model,'output_mode','partition'))
            fig,axes=plt.subplots(6,len(masks)+1,figsize=(2*(len(masks)+1),13),squeeze=False)
            for ax in axes.flat:ax.axis('off')
            axes[0,0].imshow(arrays['image'][0],cmap='gray_r',vmin=0,vmax=1)
            axes[1,0].imshow(slot['reference'][0],cmap='gray_r',vmin=0,vmax=1)
            axes[0,0].set_title('Generated crop');axes[1,0].set_title('Reference')
            for ax,label in zip(axes[2:,0],('Missing map','Split map','Junction map','RGB loss gradient\nred: darken / blue: brighten')):
                ax.text(.5,.5,label,ha='center',va='center',transform=ax.transAxes)
            for k in range(len(masks)):
                s=values['per_stroke'][k]
                axes[0,k+1].imshow(masks[k],cmap='gray_r',vmin=0,vmax=1)
                axes[0,k+1].set_title(f"Stroke {k+1} | {float(s['score'].detach()):.1f}\nw={float(s['weight']):.2f}, share={float(s['weight_share']):.0%}")
                axes[1,k+1].imshow(hard[k],cmap='gray_r',vmin=0,vmax=1);axes[1,k+1].set_title(f'Export {k+1}')
                for axis,name in zip(axes[2:5,k+1],('missing','split','junction')):
                    axis.imshow(arrays[name+'_maps'][k],cmap='magma',vmin=0)
                g=arrays['stroke_rgb_gradients'][k];limit=max(float(abs(g).max()),1e-12)
                axes[5,k+1].imshow(g,cmap='bwr',vmin=-limit,vmax=limit)
            fig.suptitle(f"{page['method']} | slot {i+1} | TOPO extracted-stroke score={row['extracted_stroke_score_no_extra']:.1f}")
            fig.tight_layout();fig.savefig(str(prefix)+'.png',dpi=140)
            if show:plt.show()
            plt.close(fig)
        del loss,details,rgb,values,stroke
    frame=pd.DataFrame(rows);frame.to_csv(out/'topo_scores.csv',index=False,encoding='utf-8-sig')
    stroke_frame=pd.DataFrame(stroke_rows)
    stroke_frame.to_csv(out/'stroke_scores.csv',index=False,encoding='utf-8-sig')
    paired=stroke_frame.pivot(index=['slot','stroke'],columns='method',values='score')
    paired['guided_minus_baseline']=paired['guided']-paired['baseline']
    if 'uniform_guided' in paired:paired['guided_minus_uniform']=paired['guided']-paired['uniform_guided']
    paired.to_csv(out/'stroke_comparison.csv',encoding='utf-8-sig')
    if show:
        from IPython.display import display
        display(stroke_frame[['method','slot','stroke','score','weight','weight_share','missing','split','junction']])
        display(paired)
    extraction=dict(checkpoint=str(checkpoint),checkpoint_epoch=epoch,output_mode=getattr(model,'output_mode','partition'),results=snapshots)
    (out/'topo_report.json').write_text(json.dumps(dict(rows=rows,stroke_rows=stroke_rows,extraction=extraction),ensure_ascii=False,indent=2),encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(out/'guidance_debug.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in out.iterdir():
            if file.suffix!='.zip':archive.write(file,'topo_evaluation/'+file.name)
        for file in Path(report_path).parent.iterdir():
            if file.is_file() and file.suffix in ('.png','.json','.jsonl','.csv'):archive.write(file,'generation/'+file.name)
    return frame,extraction,str(out/'guidance_debug.zip')
