import ast
import json
from pathlib import Path
import unittest

import torch

from diffusion_model.fitted_candidate_diffusion import InkConstraintObjective, optimize_clean
from stroke_model.guarded_ink_constraints import GuardedInkConstraints
from tests.test_adaptive_ink_constraints import example, Extractor


def make():
    source, records = example()
    candidate = torch.ones_like(source)
    for y, x in [(0, 0), (0, 64), (64, 0), (64, 64)]:
        ink = torch.from_numpy(records[0]['fitted'].max(0))[None, None]
        ink = torch.nn.functional.interpolate(ink, (64, 64), mode='nearest')
        candidate[..., y:y+64, x:x+64] = 1-ink
    return source, candidate, records


class GuardedTests(unittest.TestCase):
    def test_static_foundation_replaces_aggressive_initial_adaptive_masks(self):
        source, candidate, records = make()
        objective = GuardedInkConstraints(source, candidate, Extractor(), records)
        base = InkConstraintObjective(source, candidate)
        self.assertTrue(torch.equal(objective.erase, base.erase))
        self.assertTrue(objective.metadata['fixed_base'])

    def test_consensus_requires_repeated_evidence_and_can_revoke_new_regions(self):
        source, candidate, records = make()
        objective = GuardedInkConstraints(source, candidate, Extractor(), records)
        changed = source.clone();changed[..., 2:8, 54:60] = 0
        objective.refresh(changed, step=3)
        self.assertEqual(float(objective.dynamic[..., 2:8, 54:60].sum()), 0)
        objective.refresh(changed, step=6)
        self.assertGreater(float(objective.dynamic[..., 2:8, 54:60].sum()), 0)
        for step in range(9, 42, 3):objective.refresh(source, step=step)
        self.assertEqual(float(objective.dynamic[..., 2:8, 54:60].sum()), 0)

    def test_guard_rejects_normal_stroke_damage_despite_error_removal(self):
        source, candidate, records = make()
        objective = GuardedInkConstraints(source, candidate, Extractor(), records)
        objective.record_baseline(0, source);objective.begin_step(0)
        damaged = source.clone();damaged[..., 14:22, 17:21] = 1
        self.assertFalse(objective.check_proposal(source, damaged))
        self.assertTrue(objective.check_proposal(source, source))

    def test_backtracking_records_guard_rejections_and_accepts_only_small_damage(self):
        source, candidate, records = make()
        objective = GuardedInkConstraints(source, candidate, Extractor(), records)
        objective.record_baseline(0, source);objective.begin_step(0)
        influence = (objective.erase > .01).float().expand_as(source).clone()
        influence[..., 14:22, 17:21] = 1  # Simulated VAE leakage into a normal stroke.
        decode = lambda z: source+(z-1)*influence
        start = torch.ones(1, 1, 1, 1)
        result, row = optimize_clean(start, decode, objective, iterations=1,
            step_ratio=.3, budget_ratio=.4, backtracks=8)
        self.assertGreater(row['rejected_guard_attempts'], 0)
        self.assertTrue(objective.check_proposal(source, decode(result)))

    def test_final_policy_selects_improved_whole_arm_or_original_never_composites(self):
        source, candidate, records = make()
        objective = GuardedInkConstraints(source, candidate, Extractor(), records)
        good = source+(1-source)*(objective.base_erase > .01).float()
        objective.finish_arm('redenoise_only', source)
        objective.finish_arm('adaptive_guided', good)
        choice = objective.select_result('adaptive_guided')
        self.assertEqual(choice['file'], 'adaptive_guided.png')
        damaged = good.clone();damaged[..., 14:22, 17:21] = 1
        objective.finish_arm('adaptive_guided', damaged)
        self.assertEqual(objective.select_result('adaptive_guided')['file'], 'input.png')

    def test_clean_input_does_not_count_as_success_without_error_reduction(self):
        _, candidate, records = make()
        objective = GuardedInkConstraints(candidate, candidate, Extractor(), records)
        objective.finish_arm('redenoise_only', candidate)
        objective.finish_arm('adaptive_guided', candidate)
        self.assertEqual(objective.select_result('adaptive_guided')['status'], 'original_retained')

    def test_straight_branch_proposals_leave_caps_out(self):
        source, candidate, records = make()
        objective = GuardedInkConstraints(source, candidate, Extractor(), records)
        self.assertGreater(float(objective.uncertain[..., 26:32, 2:12].sum()), 0)
        self.assertEqual(float(objective.uncertain[..., 8:11, 17:21].sum()), 0)

    def test_restore_only_improvement_is_not_rejected_for_having_no_extra_ink(self):
        _, candidate, records = make()
        source = candidate.clone();source[..., 36:50, 36:58] = 1
        objective = GuardedInkConstraints(source, candidate, Extractor(), records, enable_restore=True)
        self.assertGreater(float(objective.base_restore.sum()), 0)
        good = source*(1-objective.base_restore)+.1*objective.base_restore
        objective.finish_arm('redenoise_only', source)
        objective.finish_arm('adaptive_guided', good)
        self.assertEqual(objective.select_result('adaptive_guided')['file'], 'adaptive_guided.png')

    def test_release_notebook_embeds_sources_and_displays_selected_output(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/guarded_ink_release.ipynb').read_text(encoding='utf-8'))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for source in code:ast.parse(source)
        assignments=[n for n in ast.parse(code[0]).body if isinstance(n,ast.Assign)]
        helpers=next(ast.literal_eval(n.value) for n in assignments
            if any(isinstance(t,ast.Name) and t.id=='RUNTIME_HELPERS' for t in n.targets))
        for path, source in helpers.items():self.assertEqual(source,(root/path).read_text(encoding='utf-8'))
        self.assertIn('GuardedInkConstraints',code[2])
        self.assertIn("OUTPUT/'selected.png'",code[3])
        self.assertNotIn('MARGIN_RATIO',code[3])


if __name__=='__main__':unittest.main()
