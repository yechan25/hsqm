"""Detector loss guidance on Qwen's flow clean prediction, without RGB painting.

Flow: clean = sample - sigma * velocity. Optimize that clean latent through
the frozen VAE, then carry its residual into the next flow state while holding
the inferred noise fixed. The transformer is never differentiated.
"""
import json
import hashlib
from pathlib import Path
from contextlib import contextmanager
import torch


def optimize_clean(clean, decode, objective, *, step_size=.02, backtracks=5):
    with torch.enable_grad():
        value=clean.detach().float().requires_grad_(True)
        loss,_=objective(decode(value))
        gradient,=torch.autograd.grad(loss,value)
    before=loss.detach().item()
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise RuntimeError('Non-finite Qwen guidance loss/gradient')
    rms=gradient.square().mean().sqrt()
    if rms.item()==0:return clean,dict(before=before,after=before,accepted=False)
    direction=gradient/rms.clamp_min(1e-12)
    for attempt in range(backtracks):
        candidate=(value.detach()-step_size*(.5**attempt)*direction).to(clean.dtype)
        with torch.no_grad():after,_=objective(decode(candidate))
        if torch.isfinite(after) and after<loss.detach():
            return candidate,dict(before=before,after=after.item(),accepted=True)
    return clean,dict(before=before,after=before,accepted=False)


@contextmanager
def guided_flow_steps(scheduler, decode, objective, history, *, step_size=.02, backtracks=5, snapshot=None):
    """Wrap the native scheduler only during the guided call; always restore it."""
    original=scheduler.step
    def step(velocity,timestep,sample,*args,**kwargs):
        # Native pipeline uses return_dict=False; reject unexpected contracts.
        if kwargs.get('return_dict',True):raise ValueError('Tuple scheduler output required')
        if scheduler.step_index is None:scheduler._init_step_index(timestep)
        index=scheduler.step_index
        sigma=scheduler.sigmas[index].to(sample)
        following=scheduler.sigmas[index+1].to(sample)
        clean=sample-sigma*velocity
        edited,record=optimize_clean(clean,decode,objective,step_size=step_size,backtracks=backtracks)
        result=original(velocity,timestep,sample,*args,**kwargs)
        residual=(1-following)*(edited-clean)
        history.append(dict(step=len(history),sigma=float(sigma),**record))
        if snapshot is not None and (len(history)%3==0 or following.item()==0):
            with torch.no_grad():snapshot(len(history),decode(edited))
        return (result[0]+residual.to(result[0]),*result[1:])
    scheduler.step=step
    try:yield
    finally:scheduler.step=original


def repair_qwen_pair(pipeline, source_image, output, objective, *, prompt,
                     negative_prompt=' ',seed=101,steps=50,strength=.3,
                     step_size=.02,backtracks=5):
    from diffusers import QwenImageImg2ImgPipeline
    from PIL import Image
    if not 0<strength<=1 or int(steps*strength)<1:raise ValueError('Invalid repair schedule')
    destination=Path(output)
    destination.mkdir(parents=True,exist_ok=False)
    with Image.open(source_image) as image:source=image.convert('RGB')
    width,height=source.size
    if width%16 or height%16:raise ValueError('Qwen repair dimensions must be divisible by 16')
    pipe=QwenImageImg2ImgPipeline.from_pipe(pipeline)
    # Separate scheduler avoids changing first-generation state.
    pipe.scheduler=type(pipeline.scheduler).from_config(pipeline.scheduler.config)
    def decode(packed):
        latent=pipe._unpack_latents(packed,height,width,pipe.vae_scale_factor)
        mean=latent.new_tensor(pipe.vae.config.latents_mean).view(1,-1,1,1,1)
        std=latent.new_tensor(pipe.vae.config.latents_std).view(1,-1,1,1,1)
        raw=pipe.vae.decode((latent*std+mean).to(pipe.vae.dtype),return_dict=False)[0]
        return (raw[:,:,0].float()/2+.5).clamp(0,1)
    starts=[]
    def run():
        original=pipe.scheduler.step
        recorded=[]
        def capture(velocity,timestep,sample,*args,**kwargs):
            if not recorded:
                digest=hashlib.sha256(sample.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
                recorded.append(digest)
                if starts and digest!=starts[0]:raise RuntimeError('Paired initial latents differ')
                starts.append(digest)
            return original(velocity,timestep,sample,*args,**kwargs)
        pipe.scheduler.step=capture
        try:
            return pipe(image=source,prompt=prompt,negative_prompt=negative_prompt,
                width=width,height=height,strength=strength,num_inference_steps=steps,
                true_cfg_scale=4.,generator=torch.Generator('cpu').manual_seed(seed)).images[0]
        finally:pipe.scheduler.step=original
    baseline=run();baseline.save(destination/'baseline.png')
    history=[]
    def snapshot(index,rgb):
        pixels=(rgb.detach()[0].permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype('uint8')
        Image.fromarray(pixels).save(destination/f'guided_step_{index:04d}_after.png')
    with guided_flow_steps(pipe.scheduler,decode,objective,history,step_size=step_size,backtracks=backtracks,snapshot=snapshot):
        guided=run()
    guided.save(destination/'guided.png')
    report=dict(backend='qwen_flow',seed=seed,strength=strength,steps=steps,
        prompt=prompt,negative_prompt=negative_prompt,step_size=step_size,backtracks=backtracks,
        input_image=str(source_image),clean_reference_used=False,rgb_painting=False,
        paired_initial_latent_sha256=starts,source_sha256=hashlib.sha256(Path(source_image).read_bytes()).hexdigest(),
        guided_accepted_steps=sum(r['accepted'] for r in history),history=history,
        pages=[dict(method=name,input=str(destination/(name+'.png'))) for name in ('baseline','guided')])
    (destination/'generation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
