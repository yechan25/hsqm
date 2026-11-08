"""Backward clean-latent guidance for a frozen epsilon DDIM model.

Gradient is through VAE -> frozen Topo -> soft stroke loss, NOT through U-Net.
Either convert a clean-latent correction back to epsilon (legacy), or apply an
explicit guided mean update with the U-Net residual held fixed. Backtrack until
the SAME timestep's objective decreases. This does not guarantee final quality.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from stroke_model.source_ink_guard import local_ink_change


@dataclass(frozen=True)
class GuidanceConfig:
    noise_ratio: float=.05
    clean_step_ratio: float | None=None
    start_fraction: float=.30
    end_fraction: float=.90
    every: int=2
    backtracks: int=2
    snapshot_every: int=4
    healthy_stroke_score: float=90.
    max_healthy_stroke_drop: float=1.
    state_update: str='epsilon'
    max_new_local_ink: float | None=None
    local_edit: bool=False
    local_iterations: int=24
    local_halo_radius: int=0
    local_final_iterations: int=0
    local_erase_l1_weight: float=0.
    local_erase_tail_weight: float=0.
    max_state_change_to_base_step_ratio: float | None=None

    def __post_init__(self):
        if (self.max_state_change_to_base_step_ratio is not None and
                (not math.isfinite(self.max_state_change_to_base_step_ratio) or
                 self.max_state_change_to_base_step_ratio<=0 or self.state_update!='clean_residual' or self.local_edit)):
            raise ValueError('State change cap requires nonlocal clean_residual guidance and a positive ratio')
        if (not math.isfinite(self.local_erase_tail_weight) or self.local_erase_tail_weight<0
                or not math.isfinite(self.local_erase_l1_weight) or self.local_erase_l1_weight<0
                or type(self.local_halo_radius) is not int or self.local_halo_radius<0
                or type(self.local_final_iterations) is not int or self.local_final_iterations<0
                or type(self.local_iterations) is not int or self.local_iterations<1
                or (self.local_edit and self.state_update!='clean_residual')
                or not math.isfinite(self.noise_ratio) or self.noise_ratio<0
                or (self.clean_step_ratio is not None and
                    (not math.isfinite(self.clean_step_ratio) or self.clean_step_ratio<0))
                or not 0<=self.start_fraction<self.end_fraction<=1
                or type(self.every) is not int or self.every<1
                or type(self.backtracks) is not int or self.backtracks<0
                or type(self.snapshot_every) is not int or self.snapshot_every<1
                or not math.isfinite(self.healthy_stroke_score) or not 0<=self.healthy_stroke_score<=100
                or not math.isfinite(self.max_healthy_stroke_drop) or self.max_healthy_stroke_drop<0
                or self.state_update not in ('epsilon','clean_residual')
                or (self.state_update=='clean_residual' and self.clean_step_ratio is None)
                or (self.max_new_local_ink is not None and
                    (not math.isfinite(self.max_new_local_ink) or self.max_new_local_ink<0))):
            raise ValueError('Invalid guidance schedule')


def decode_estimate(pipe,z0):
    dtype=next(pipe.vae.parameters()).dtype
    decoded=pipe.vae.decode(z0.to(dtype)/pipe.vae.config.scaling_factor,return_dict=False)[0]
    return decoded.float()/2+.5


def _plain(value):
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(item) for item in value]
    return value


def _scalar_rows(details):
    return _plain(details['per_glyph'])


def _healthy_guard(before,after,config):
    """Protect EACH previously healthy stroke, independent of sheet averages."""
    drops=[]
    for a,b in zip(before,after):
        sa,sb=a.get('per_stroke',[]),b.get('per_stroke',[])
        if len(sa)!=len(sb):raise ValueError('Candidate stroke identities changed')
        for first,second in zip(sa,sb):
            if first['stroke']!=second['stroke']:raise ValueError('Candidate stroke order changed')
            if first['score']>=config.healthy_stroke_score:
                drops.append(float(first['score']-second['score']))
    drop=max([0.]+drops)
    return drop<=config.max_healthy_stroke_drop+1e-5,drop


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
    trace=[];ink_anchor=None;editor=None
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
        record=dict(step=index,timestep=int(t),active=active,accepted=False,state_update=config.state_update,
                    base_transition_rms=float((original.prev_sample-latent).square().mean().sqrt()))
        chosen=original
        if active and config.local_edit:
            from stroke_model.local_reference_edit import LocalReferenceEdit,optimize_local
            if objective.source_guard is None:raise ValueError('Local edit requires reference sheet')
            if editor is None:
                with torch.no_grad():
                    anchor=decode_estimate(pipe,original.pred_original_sample)
                    reference=(1-objective.source_guard.reference_ink).expand_as(anchor)
                    editor=LocalReferenceEdit(reference,anchor,halo_radius=config.local_halo_radius,
                        erase_l1_weight=config.local_erase_l1_weight,
                        erase_tail_weight=config.local_erase_tail_weight)
                    objective.local_editor=editor
            with torch.no_grad():editor.rebase(decode_estimate(pipe,original.pred_original_sample))
            edited,summary=optimize_local(original.pred_original_sample,
                lambda z:decode_estimate(pipe,z),editor,iterations=(max(config.local_iterations,config.local_final_iterations)
                    if index==len(timesteps)-1 else config.local_iterations),
                step_ratio=config.clean_step_ratio)
            record.update(summary)
            if summary['accepted']:
                delta=edited-original.pred_original_sample
                prev_t=int(t)-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
                a=(scheduler.alphas_cumprod[prev_t] if prev_t>=0 else scheduler.final_alpha_cumprod).to(latent)
                chosen=type(original)(prev_sample=original.prev_sample+a.sqrt()*delta,
                                      pred_original_sample=edited)
                record['state_delta_rms']=float((chosen.prev_sample-original.prev_sample).square().mean().sqrt())
        elif active:
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
            priorities=objective.freeze_priority(details) if hasattr(objective,'freeze_priority') else None
            source_before=details.get('source_spill_map')
            if config.max_new_local_ink is not None and source_before is None:
                raise ValueError('Local ink guard requires the fixed native reference sheet')
            if details.get('source_spill_loss') is not None:
                record['source_spill_loss_before']=float(details['source_spill_loss'].detach())
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
                        if config.state_update=='clean_residual':
                            # Keep the U-Net residual direction FIXED. Modifying both
                            # x0 and epsilon in ordinary DDIM largely cancels the step.
                            # This is an explicit guided-mean update, not an ordinary
                            # DDIM step from a corrected epsilon at fixed z_t.
                            delta=desired*(.5**attempt)
                            prev_t=int(t)-scheduler.config.num_train_timesteps//scheduler.num_inference_steps
                            prev_alpha=(scheduler.alphas_cumprod[prev_t] if prev_t>=0
                                        else scheduler.final_alpha_cumprod).to(latent.device).float()
                            if config.max_state_change_to_base_step_ratio is not None:
                                cap=config.max_state_change_to_base_step_ratio*record['base_transition_rms']
                                rms=prev_alpha.sqrt()*delta.square().mean().sqrt()
                                delta=delta*(cap/rms.clamp_min(1e-12)).clamp(max=1.)
                            candidate=type(original)(
                                prev_sample=original.prev_sample-prev_alpha.sqrt()*delta,
                                pred_original_sample=original.pred_original_sample-delta)
                        else:
                            candidate=scheduler.step(candidate_noise,t,latent,eta=0.,return_dict=True)
                        image=decode_estimate(pipe,candidate.pred_original_sample.float())
                        after,after_details=(objective(image,stroke_weights=priorities) if priorities is not None
                                             else objective(image))
                        if not torch.isfinite(after):raise FloatingPointError('Nonfinite candidate loss')
                        after_score=_mean_score(after_details)
                        score_ok=(record['topo_score_before'] is None or after_score>=record['topo_score_before']-1e-5)
                        healthy_ok,healthy_drop=_healthy_guard(record['per_glyph_before'],
                            _scalar_rows(after_details),config)
                        ink_ok=True;ink_increase=ink_violation=0.
                        if config.max_new_local_ink is not None:
                            if after_details.get('source_spill_map') is None:
                                raise ValueError('Candidate lost its source ink guard')
                            ink_ok,ink_increase,ink_violation,ink_anchor=local_ink_change(
                                source_before,after_details['source_spill_map'],ink_anchor,config.max_new_local_ink)
                        state_delta=float((candidate.prev_sample-original.prev_sample).square().mean().sqrt())
                        clean_delta=float((candidate.pred_original_sample-original.pred_original_sample).square().mean().sqrt())
                        reason=('loss_not_lower' if float(after)>=record['loss_before'] else
                                'primary_score_lower' if not score_ok else
                                'healthy_stroke_worse' if not healthy_ok else
                                'new_local_ink' if not ink_ok else
                                'no_state_change' if state_delta==0 else 'accepted')
                        record['attempts'].append(dict(attempt=attempt,loss_after=float(after),
                            topo_score_after=after_score,state_delta_rms=state_delta,
                            clean_delta_rms=clean_delta,max_healthy_stroke_drop=healthy_drop,reason=reason,
                            new_local_ink_increase=ink_increase,new_local_ink_violation=ink_violation,
                            state_to_clean_transfer_ratio=state_delta/max(clean_delta,1e-12)))
                        if reason=='accepted':
                            chosen=candidate
                            record.update(accepted=True,backtracks=attempt,loss_after=float(after),
                                correction_rms=float(actual_delta.square().mean().sqrt()),
                                topo_score_after=after_score,per_glyph_after=_scalar_rows(after_details),
                                state_delta_rms=state_delta,clean_delta_rms=clean_delta,
                                state_to_clean_transfer_ratio=state_delta/max(clean_delta,1e-12),
                                state_change_to_base_step_ratio=state_delta/max(record['base_transition_rms'],1e-12))
                            if after_details.get('source_spill_loss') is not None:
                                record['source_spill_loss_after']=float(after_details['source_spill_loss'])
                            if ink_anchor is not None:
                                ink_anchor=torch.minimum(ink_anchor,after_details['source_spill_map'])
                            break
            if not record['accepted']:record['loss_after']=record['loss_before']
        latent=chosen.prev_sample.detach()
        trace.append(record)
        if on_step:
            on_step(record,original.pred_original_sample.detach(),chosen.pred_original_sample.detach())
        del original,chosen,predicted
    return latent,trace
