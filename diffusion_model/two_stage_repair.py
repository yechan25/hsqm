"""Generate once; explicitly re-noise the saved result for paired repair."""
import json
from pathlib import Path
from .run_topo_guidance import generate_guided_pair,_file_hash


def _save(report,output,**metadata):
    report.update(metadata)
    (Path(output)/'generation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def generate_first_stage(manifest,manifest_rows,checkpoint,output,**kwargs):
    kwargs.setdefault("init_mode","noise")
    if kwargs["init_mode"]!="noise":raise ValueError("First generation must start from prompt/noise; use a separate explicit benchmark for img2img")
    report=generate_guided_pair(manifest,manifest_rows,checkpoint,output,
        guidance=dict(noise_ratio=0,local_edit=False,max_new_local_ink=None),
        guidance_signal='template_structure',baseline_only=True,
        compare_structure=False,compare_uniform=False,appearance_weight=0,
        extra_weight=0,source_spill_weight=0,direct_reference=False,**kwargs)
    return _save(report,output,phase='initial_generation',
        generated_image=next(p['input'] for p in report['pages'] if p['method']=='baseline'))


def repair_generated_image(manifest,manifest_rows,checkpoint,generated_image,output,*,
        steps=100,strength=.2,seed=101,preservation_weight=1.,guidance=None,**kwargs):
    source=Path(generated_image)
    digest=_file_hash(source)
    config=guidance or dict(local_edit=False,clean_step_ratio=.003,start_fraction=.25,
        end_fraction=1.,every=1,backtracks=6,state_update='clean_residual',
        max_new_local_ink=None,max_state_change_to_base_step_ratio=.25)
    report=generate_guided_pair(manifest,manifest_rows,checkpoint,output,
        steps=steps,strength=strength,seed=seed,init_mode='external',input_sheet=source,
        guidance=config,guidance_signal='template_structure',compare_structure=False,
        compare_uniform=False,appearance_weight=0,extra_weight=0,source_spill_weight=0,
        direct_reference=False,preservation_weight=preservation_weight,**kwargs)
    if _file_hash(source)!=digest:raise RuntimeError('The saved first generation was modified')
    return _save(report,output,phase='second_pass_repair',first_generated_image=str(source),
        first_generated_sha256=digest,clean_target_pixels_used=False,
        method_meanings=dict(baseline='re-denoise only',guided='same re-noised latent plus stroke guidance'))
