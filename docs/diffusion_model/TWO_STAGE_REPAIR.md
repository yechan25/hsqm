> 2026-11-09 수정: 아래의 외부 폰트 초기화 설명은 과거 대조 실험 기록이다. 현재 기본 실행은 [프롬프트 전용 첫 생성](PROMPT_THEN_REPAIR.md)이며 첫 생성에 이미지를 주지 않는다.

# Two-stage generation and re-denoising

Entry: notebooks/two_stage_repair.ipynb, built by notebooks/build_two_stage_repair.py.

1. Prepare templates/checkpoint, choose four characters and an independent input font. Stage 1 runs ordinary denoising only and saves the generated PNG. Default conditions remain external font input, strength=0.65, 100 nominal steps, seed=100. Optional noise initialization is still experimental; FIRST_IMAGE can instead specify any already generated sheet of the correct size and character order.
2. Inspect the first image. The notebook does not claim a calibrated automatic error detector; the user decides whether to run the repair cell.
3. Encode the SAVED generated PNG once and add fresh noise. Repair defaults: strength=0.2, 100 nominal steps (20 actual steps), seed=101. Two denoising branches copy that identical noised latent: baseline is re-denoising alone; guided adds the frozen extractor/template descriptor objective.
4. Compare first generation, re-denoising alone and stroke-guided re-denoising. Preserve all results; do not count improvements in the plain re-denoising branch as the extractor's contribution.

## Objective and preservation

L = existing normalized template-descriptor smooth L1 (plus its 0.01 RGB range term) + preservation_weight * MSE(decoded RGB, FIRST generated RGB). Default preservation_weight=1. The anchor is the imperfect generated input, not a clean target, and may resist correction as well as prevent drift. No oracle erase/fill map is used. Template pixels still condition the frozen extractor and define normalized geometry targets; this is template-conditioned, not reference-free recognition. The old descriptor's spur/serif ambiguities remain unvalidated.

Repair guidance uses clean_step_ratio=0.003, active after fraction 0.25, six backtracks, and clean_residual state transfer. State displacement at each guided candidate is capped at 0.25 times that same step's unguided DDIM displacement. This cap has a default of None in the sampler for backward compatibility and is supported only by non-local clean_residual guidance. These are experimental starting values, not optimized hyperparameters.

## Reproducibility

Stage 1 has only reference (conditioning display) and baseline pages; its baseline trace has no active guidance. Stage 2 reports first-generated path/hash and verifies the file is unchanged, preserves external_input.png, records both seeds/schedules, initial latent hashes, descriptor/preservation losses and all attempted updates. The ZIP includes both stages, the exact first input and comparison image. The same selected templates, model revision and checkpoint must be used across stages.

## Validation limits

Tests cover the displacement cap, preservation gradients and an immutable generated-image anchor, plus mocked complete stage1->saved PNG->stage2 initialization and shared-noise comparisons. Actual SD/real-checkpoint GPU quality is not tested locally. The method reuses the current descriptor; a second pass is not evidence that its error detection is correct. Manually inspect character correctness, attached spurs, disconnected blobs and damage to normal style.
