import ast
import json
from pathlib import Path
import unittest

import torch

from diffusion_model.fitted_candidate_diffusion import InkConstraintObjective, optimize_clean


def example():
    source = torch.ones(1, 3, 64, 64)
    source[:, :, 8:54, 29:34] = 0
    source[:, :, 26:31, 10:30] = 0  # Extra branch, attached to a healthy straight stroke.
    target = torch.ones_like(source)
    target[:, :, 8:54, 29:34] = .08  # Different shade is not a target RGB value.
    target[:, :, 16:48, 50:55] = 0  # Entirely missing stroke.
    return source, target


class InkConstraintTests(unittest.TestCase):
    def test_sparse_erase_retains_normal_straight_stroke_and_default_disables_restore(self):
        source, target = example()
        objective = InkConstraintObjective(source, target)
        self.assertGreater(float(objective.erase[0, 0, 28, 15]), 0)
        self.assertEqual(float(objective.erase[0, 0, 20, 31]), 0)
        self.assertEqual(float(objective.restore.sum()), 0)
        result, trace = optimize_clean(source, lambda z: z, objective,
                                      iterations=8, step_ratio=.01, budget_ratio=.2)
        self.assertLess(trace['loss_after'], trace['loss_before'])
        self.assertGreater(float(result[0, :, 28, 15].mean()), 0)
        self.assertTrue(torch.equal(result[:, :, 10:20, 29:34], source[:, :, 10:20, 29:34]))

    def test_restore_only_centerline_and_stops_when_ink_sufficient(self):
        source, target = example()
        objective = InkConstraintObjective(source, target, enable_restore=True)
        self.assertGreater(float(objective.restore.sum()), 0)
        self.assertFalse(bool(((objective.erase > 0) & (objective.restore > 0)).any()))
        candidate_band = (target < .35).float().mean(1, keepdim=True)
        self.assertLess(float(objective.restore.sum()), float(candidate_band.sum()))
        leaf = source.clone().requires_grad_(True)
        loss, _ = objective(leaf)
        gradient, = torch.autograd.grad(loss, leaf)
        self.assertGreater(float(gradient.expand_as(leaf)[objective.restore.expand_as(leaf) > 0].mean()), 0)
        enough = source*(1-objective.restore)+.2*objective.restore
        _, details = objective(enough)
        self.assertAlmostEqual(details['restore_deficit'], 0)

    def test_shade_and_minor_boundary_difference_do_not_become_candidate_matching(self):
        source, target = example()
        a = InkConstraintObjective(source, target)
        alternate = target.clone()
        alternate[target < .35] = .15
        b = InkConstraintObjective(source, alternate)
        self.assertTrue(torch.equal(a.erase, b.erase))
        self.assertAlmostEqual(float(a(source)[0]), float(b(source)[0]))
        self.assertFalse(a.metadata['candidate_rgb_matching'])
        source2 = torch.ones_like(source)
        source2[:, :, 8:54, 29:34] = 0
        target2 = torch.ones_like(source)
        target2[:, :, 9:53, 28:35] = 0
        self.assertEqual(float(InkConstraintObjective(source2, target2).erase.sum()), 0)

    def test_empty_masks_have_no_spurious_gradient_and_gradient_is_local(self):
        source, _ = example()
        objective = InkConstraintObjective(source, source)
        self.assertEqual(float(objective.edit.sum()), 0)
        self.assertEqual(float(objective.project_gradient(torch.ones(1, 4, 8, 8)).sum()), 0)
        self.assertEqual(float(objective(source)[0]), 0)

    def test_notebook_embeds_current_runtime_and_explicit_delete_only_controls(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/'notebooks/ink_constraint_diffusion.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        for source in code:
            ast.parse(source)
        nodes = [n for n in ast.parse(code[0]).body if isinstance(n, ast.Assign)]
        helper = next(ast.literal_eval(n.value) for n in nodes
                      if any(isinstance(t, ast.Name) and t.id == 'HELPER_SOURCE' for t in n.targets))
        self.assertEqual(helper, (root/'diffusion_model/fitted_candidate_diffusion.py').read_text(encoding='utf-8'))
        self.assertIn('ENABLE_RESTORE=False', code[1])
        self.assertIn("guidance_mode='ink_constraints'", code[2])
        self.assertIn('preview.erase', code[1])


if __name__ == '__main__':
    unittest.main()
