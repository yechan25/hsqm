"""Supervised extractor fine-tuning objective. Never a source-quality reward.

The caller supplies a live StrokeStructureLoss result using fixed, aligned
reference geometry and a frozen teacher prediction for the same batch.
"""
from dataclasses import dataclass
import math

import torch
from .reference_model import partition_loss


@dataclass(frozen=True)
class ExtractionTrainingConfig:
    structure_weight: float = 0.05
    mass_guard_weight: float = 0.2
    consistency_weight: float = 0.1
    mass_tolerance: float = 0.15
    eps: float = 1e-6

    def __post_init__(self):
        for name in ('structure_weight', 'mass_guard_weight', 'consistency_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 <= self.mass_tolerance < 1 or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('Invalid mass_tolerance or eps')


def extraction_training_loss(output, labels, structure_result=None,
                             teacher_output=None, config=None, *, topology_valid=None):
    """Base CE/Dice + extractor topology + teacher safeguards.

    Missing/width reference deficits are deliberately excluded. A real missing
    input stroke must not be fabricated by this extractor (masks=image*p).
    topology_valid is explicit B,K metadata: only known structurally complete
    source strokes may receive the no-junction/connectedness regularizer. Do not
    derive this metadata from the student's own predictions.
    Teacher safeguards reduce deletion/channel-transfer shortcuts but do not
    prove their absence. Validate supervised Dice AND structural diagnostics.
    """
    cfg = config or ExtractionTrainingConfig()
    base = partition_loss(output, labels)
    zero = output['masks'].sum() * 0
    if cfg.structure_weight == 0:
        return {**base, 'supervised': base['loss'], 'structure': zero,
                'mass_guard': zero, 'consistency': zero}
    if structure_result is None or teacher_output is None:
        raise ValueError('Structural fine-tuning requires live structure_result and frozen teacher_output')
    if cfg.mass_guard_weight <= 0:
        raise ValueError('Enable mass_guard_weight to protect against deleting strokes')
    if not all(c['pairs'] and c['attachment'] for c in structure_result['convergence']):
        raise ValueError('Structure propagation did not converge; increase iteration caps')
    masks = output['masks'].float()
    teacher = teacher_output['masks'].detach().to(device=masks.device, dtype=torch.float32)
    active = output['active']
    if teacher.shape != masks.shape or active.shape != masks.shape[:2]:
        raise ValueError('Student/teacher mask shapes and active channels must match')
    if topology_valid is None or topology_valid.shape != active.shape or topology_valid.dtype != torch.bool:
        raise ValueError('Provide topology_valid: B,K bool metadata for known-complete input strokes')
    valid = active & topology_valid.to(active.device)
    if not bool(torch.isfinite(teacher).all()) or bool((teacher < 0).any()) or bool((teacher > 1).any()):
        raise ValueError('Teacher masks must be finite absolute ink in [0,1]')
    errors = structure_result['per_stroke']['extraction_error']
    if errors.shape != active.shape:
        raise ValueError('Structure results must be B,K for the same live output')
    if masks.requires_grad and not errors.requires_grad:
        raise ValueError('Do not detach structural losses in extractor training')
    structure = errors[valid].mean() if bool(valid.any()) else zero
    target_mass = teacher.sum((-2, -1))
    mass = masks.sum((-2, -1))
    guarded = valid & (target_mass > cfg.eps)
    deficit = torch.relu(((1 - cfg.mass_tolerance) * target_mass - mass)
                         / target_mass.clamp_min(cfg.eps)).square()
    mass_guard = deficit[guarded].mean() if bool(guarded.any()) else zero
    # Only reference channel activity is discrete. Teacher tensors are detached.
    consistency = (masks - teacher).square().sum((-2, -1))
    consistency = (consistency / target_mass.clamp_min(1))[valid].mean() if bool(valid.any()) else zero
    loss = (base['loss'] + cfg.structure_weight * structure
            + cfg.mass_guard_weight * mass_guard + cfg.consistency_weight * consistency)
    return {**base, 'loss': loss, 'supervised': base['loss'], 'structure': structure,
            'mass_guard': mass_guard, 'consistency': consistency}
