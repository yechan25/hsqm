"""Backward clean-latent guidance for a frozen epsilon DDIM model.

Gradient is through VAE -> frozen Topo -> soft stroke loss, NOT through U-Net.
Each clean-latent correction is converted back to epsilon and backtracked until
the SAME timestep's objective decreases. This does not guarantee final quality.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class GuidanceConfig:
    noise_ratio: float=.05
    clean_step_ratio: float | None=None
    start_fraction: float=.30
    end_fraction: float=.90
    every: int=2
    backtracks: int=2
    snapshot_every: int=4

    def __post_init__(self):
        if (not math.isfinite(self.noise_ratio) or self.noise_ratio<0
                or (self.clean_step_ratio is not None and
                    (not math.isfinite(self.clean_step_ratio) or self.clean_step_ratio<0))
                or not 0<=self.start_fraction<self.end_fraction<=1
                or type(self.every) is not int or self.every<1
                or type(self.backtracks) is not int or self.backtracks<0
                or type(self.snapshot_every) is not int or self.snapshot_every<1):
            raise ValueError('Invalid guidance schedule')


def decode_estimate(pipe,z0):
    dtype=next(pipe.vae.parameters()).dtype
    decoded=pipe.vae.decode(z0.to(dtype)/pipe.vae.config.scaling_factor,return_dict=False)[0]
    return decoded.float()/2+.5


def _scalar_rows(details):
    return [{key:float(value.detach()) if torch.is_tensor(value) else value
             for key,value in row.items()} for row in details['per_glyph']]


def _mean_score(details):
    key='extracted_stroke_score_no_extra'
    if not all(key in row for row in details['per_glyph']):return None
    return sum(float(row[key].detach()) for row in details['per_glyph'])/len(details['per_glyph'])


def denoise(pipe,initial,timesteps,embeddings,*,objective=None,config=None,cfg_scale=7.5,on_step=None):
    config=config or GuidanceConfig(noise_ratio=0)
    enabled=(config.clean_step_ratio>0 if config.clean_step_ratio is not None else config.noise_ratio>0)
    if enabled and objective is None:
        raise ValueError('An actual differentiable Topo objective is required for guidance')
    scheduler=deepcopy(pipe.scheduler)
    if scheduler.config.prediction_type!='epsilon':raise ValueError('Epsilon prediction required')
    if scheduler.config.clip_sample or getattr(scheduler.config,'thresholding',False):
        raise ValueError('This clean-latent correction requires clip_sample=False and thresholding=False')
    # Keep small DDIM state updates in fp32. Only U-Net inputs use model dtype.
    latent=initial.detach().float().clone()
    unet_dtype=next(pipe.unet.parameters()).dtype
    trace=[]
    for index,t in enumerate(timesteps):
        with torch.no_grad():
            inputs=torch.cat([latent,latent]) if cfg_scale>1 else latent
            inputs=scheduler.scale_model_input(inputs,t)
            predicted=pipe.unet(inputs.to(unet_dtype),t,encoder_hidden_states=embeddings,return_dict=False)[0]
            if cfg_scale>1:
                uncond,text=predicted.chunk(2)
                predicted=uncond+cfg_scale*(text-uncond)
            predicted=predicted.float()
            original=scheduler.step(predicted,t,latent,eta=0.,return_dict=True)
        progress=index/max(len(timesteps)-1,1)
        active=(objective is not None and enabled and index%config.every==0
                and config.start_fraction<progress<=config.end_fraction)
        record=dict(step=index,timestep=int(t),active=active,accepted=False,
                    base_transition_rms=float((original.prev_sample-latent).square().mean().sqrt()))
        chosen=original
        if active:
            # Single-step leaf: never retain the entire denoising trajectory.
            with torch.enable_grad():
                clean=original.pred_original_sample.detach().float().requires_grad_(True)
                loss,details=objective(decode_estimate(pipe,clean))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Topo guidance loss')
                gradient,=torch.autograd.grad(loss,clean)
            if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite clean-latent gradient')
            norm=gradient.square().mean().sqrt()
            record.update(loss_before=float(loss.detach()),gradient_rms=float(norm),
                topo_score_before=_mean_score(details),per_glyph_before=_scalar_rows(details),attempts=[])
            del loss,details,clean
            if norm>1e-12:
                if config.clean_step_ratio is not None:
                    # Specify a clean-latent step, then convert it EXACTLY to epsilon.
                    # Ramp in, but keep final-step correction instead of tapering to zero.
                    schedule=min(1.,(progress-config.start_fraction)/max(
                        .25*(config.end_fraction-config.start_fraction),1e-6))
                    alpha=scheduler.alphas_cumprod[int(t)].to(latent.device).float()
                    clean_rms=original.pred_original_sample.float().square().mean().sqrt().clamp_min(1e-3)
                    desired=config.clean_step_ratio*schedule*clean_rms*gradient/norm
                    correction=(alpha/(1-alpha).clamp_min(1e-12)).sqrt()*desired
                    record['correction_budget']='clean_latent_rms'
                else:
                    schedule=math.sin(math.pi*(progress-config.start_fraction)/(config.end_fraction-config.start_fraction))
                    rms=predicted.float().square().mean().sqrt().clamp_min(1e-3)
                    correction=config.noise_ratio*schedule*rms*gradient/norm
                    record['correction_budget']='epsilon_rms'
                with torch.no_grad():
                    for attempt in range(config.backtracks+1):
                        candidate_noise=(predicted.float()+correction*(.5**attempt)).to(predicted.dtype)
                        actual_delta=candidate_noise.float()-predicted.float()
                        candidate=scheduler.step(candidate_noise,t,latent,eta=0.,return_dict=True)
                        after,after_details=objective(decode_estimate(pipe,candidate.pred_original_sample.float()))
                        if not torch.isfinite(after):raise FloatingPointError('Nonfinite candidate loss')
                        after_score=_mean_score(after_details)
                        score_ok=(record['topo_score_before'] is None or after_score>=record['topo_score_before']-1e-5)
                        state_delta=float((candidate.prev_sample-original.prev_sample).square().mean().sqrt())
                        clean_delta=float((candidate.pred_original_sample-original.pred_original_sample).square().mean().sqrt())
                        reason=('loss_not_lower' if float(after)>=record['loss_before'] else
                                'primary_score_lower' if not score_ok else
                                'no_state_change' if state_delta==0 else 'accepted')
                        record['attempts'].append(dict(attempt=attempt,loss_after=float(after),
                            topo_score_after=after_score,state_delta_rms=state_delta,
                            clean_delta_rms=clean_delta,reason=reason))
                        if reason=='accepted':
                            chosen=candidate
                            record.update(accepted=True,backtracks=attempt,loss_after=float(after),
                                correction_rms=float(actual_delta.square().mean().sqrt()),
                                topo_score_after=after_score,per_glyph_after=_scalar_rows(after_details),
                                state_delta_rms=state_delta,clean_delta_rms=clean_delta,
                                state_change_to_base_step_ratio=state_delta/max(record['base_transition_rms'],1e-12))
                            break
            if not record['accepted']:record['loss_after']=record['loss_before']
        latent=chosen.prev_sample.detach()
        trace.append(record)
        if on_step:
            on_step(record,original.pred_original_sample.detach(),chosen.pred_original_sample.detach())
        del original,chosen,predicted
    return latent,trace
