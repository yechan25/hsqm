import io
import unittest
import torch
from torch import nn
from stroke_model.error_localizer import (
    ErrorLocalizerConfig, StrokeErrorLocalizer, FrozenExtractorErrorLocalizer,
    error_localizer_loss, hangul_tokens,
)
from stroke_model.local_repair_proposal import LocalRepairProposal, proposal_loss, FixedRepairPlan


class ErrorArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.rgb = torch.rand(2, 3, 20, 24)
        self.p = torch.randn(2, 3, 20, 24).softmax(1)
        self.chars = torch.tensor([ord('가'), ord('곁')])
        self.model = StrokeErrorLocalizer(ErrorLocalizerConfig(width=8))

    def test_shapes_and_odd_resolution(self):
        out = self.model(self.rgb[..., :19, :21], self.p[..., :19, :21], self.chars)
        self.assertEqual(out['extra_logits'].shape, (2, 1, 19, 21))
        self.assertEqual(out['kind_logits'].shape, (2, 2, 19, 21))
        self.assertEqual(out['has_error_logit'].shape, (2, 1))
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))

    def test_stroke_permutation_does_not_change_detection(self):
        active = torch.tensor([[True, True, False], [True, True, True]])
        kinds = torch.tensor([[1, 2, 0], [2, 4, 5]])
        edges = torch.randint(0, 4, (2, 3, 3))
        order = torch.tensor([2, 0, 1])
        a = self.model(self.rgb, self.p, self.chars, active, kinds, edges)
        b = self.model(self.rgb, self.p[:, order], self.chars, active[:, order],
                       kinds[:, order], edges[:, order][:, :, order])
        torch.testing.assert_close(a['extra_logits'], b['extra_logits'], atol=2e-6, rtol=2e-5)

    def test_padding_cannot_change_output(self):
        a = self.model(self.rgb, self.p, self.chars)
        padded = torch.cat((self.p, torch.rand(2, 2, 20, 24)), 1)
        active = torch.tensor([[True, True, True, False, False]] * 2)
        b = self.model(self.rgb, padded, self.chars, active)
        torch.testing.assert_close(a['extra_logits'], b['extra_logits'], atol=2e-6, rtol=2e-5)

    def test_character_condition_is_connected(self):
        a = self.model(self.rgb, self.p, self.chars)['extra_logits']
        b = self.model(self.rgb, self.p, self.chars.flip(0))['extra_logits']
        self.assertGreater(float((a-b).abs().sum().detach()), 0)
        l, v, t = hangul_tokens(torch.tensor([ord('가'), ord('힣')]))
        self.assertEqual(l.tolist(), [0, 18]); self.assertEqual(v.tolist(), [0, 20])
        self.assertEqual(t.tolist(), [0, 27])
        with self.assertRaises(ValueError):hangul_tokens(torch.tensor([ord('A')]))

    def test_training_updates_head_not_extractor(self):
        class Extractor(nn.Module):
            def __init__(self):
                super().__init__(); self.gain = nn.Parameter(torch.tensor(2.))
            def forward(self, ink, reference, strokes, active=None):
                assert not self.training
                return dict(probabilities=(strokes*self.gain).softmax(1))
        extractor = Extractor()
        wrapper = FrozenExtractorErrorLocalizer(extractor, self.model).train()
        out = wrapper(self.rgb, torch.zeros(2, 1, 20, 24), self.p, self.chars)
        target = torch.zeros(2, 1, 20, 24); target[..., 8:12, 7:13] = 1
        valid = torch.ones_like(target)
        kind = torch.zeros(2, 20, 24, dtype=torch.long); kind[:, 8:12, 7:13] = 1
        loss = error_localizer_loss(out, target, valid, kind)['loss']
        loss.backward()
        self.assertIsNone(extractor.gain.grad)
        self.assertFalse(extractor.gain.requires_grad)
        self.assertGreater(float(self.model.extra.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.medial.weight.grad.abs().sum()), 0)

    def test_unknown_annotations_have_no_supervision(self):
        logits = torch.randn(2, 1, 20, 24, requires_grad=True)
        presence = torch.randn(2, 1, requires_grad=True)
        out = dict(extra_logits=logits, has_error_logit=presence)
        loss = error_localizer_loss(out, torch.ones_like(logits), torch.zeros_like(logits))['loss']
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0)
        self.assertEqual(float(presence.grad.abs().sum()), 0)

    def test_partial_clean_annotation_does_not_label_whole_image_clean(self):
        logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
        presence = torch.zeros(1, 1, requires_grad=True)
        valid = torch.zeros_like(logits); valid[..., :8, :] = 1
        out = dict(extra_logits=logits, has_error_logit=presence)
        loss = error_localizer_loss(out, torch.zeros_like(logits), valid)
        self.assertEqual(float(loss['presence'].detach()), 0)
        loss['loss'].backward()
        self.assertEqual(float(presence.grad.abs().sum()), 0)
        self.assertGreater(float(logits.grad[..., :8, :].abs().sum()), 0)
        self.assertEqual(float(logits.grad[..., 8:, :].abs().sum()), 0)

    def test_clean_empty_mask_and_blank_input_are_finite(self):
        out = self.model(torch.ones_like(self.rgb), self.p, self.chars)
        target = torch.zeros(2, 1, 20, 24)
        loss = error_localizer_loss(out, target, torch.ones_like(target))['loss']
        loss.backward(); self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(self.model.extra.weight.grad).all())

    def test_architecture_roundtrip(self):
        self.model.eval()
        stream = io.BytesIO(); torch.save(self.model.state_dict(), stream); stream.seek(0)
        other = StrokeErrorLocalizer(self.model.config).eval()
        other.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        torch.testing.assert_close(self.model(self.rgb, self.p, self.chars)['extra_logits'],
                                   other(self.rgb, self.p, self.chars)['extra_logits'], rtol=0, atol=0)


class ProposalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_proposal_starts_as_identity_and_does_not_retrain_detector(self):
        rgb = torch.rand(1, 3, 20, 24)
        features = torch.randn(1, 8, 20, 24, requires_grad=True)
        probability = torch.rand(1, 1, 20, 24, requires_grad=True)
        model = LocalRepairProposal(feature_width=8)
        proposal = model(rgb, features, probability)
        self.assertTrue(torch.equal(proposal, rgb))
        target = rgb.clone(); target[..., 7:11, 6:10] = 1
        mask = torch.zeros_like(probability); mask[..., 7:11, 6:10] = 1
        loss = proposal_loss(proposal, rgb, target, mask, torch.ones_like(mask))['loss']
        loss.backward()
        self.assertIsNone(features.grad); self.assertIsNone(probability.grad)
        self.assertGreater(float(model.residual.weight.grad.abs().sum()), 0)

    def test_fixed_plan_edits_error_and_preserves_source_outside(self):
        source = torch.full((1, 3, 20, 24), .5)
        target = torch.ones_like(source, requires_grad=True)
        mask = torch.zeros(1, 1, 20, 24); mask[..., 7:11, 6:10] = 1
        plan = FixedRepairPlan(source, target, mask)
        mask.fill_(1); source.fill_(0)  # buffers must remain frozen snapshots
        candidate = torch.full_like(source, .5, requires_grad=True)
        values = plan(candidate); values['loss'].backward()
        self.assertIsNone(target.grad)
        self.assertLess(float(candidate.grad[..., 7:11, 6:10].sum()), 0)
        self.assertEqual(float(candidate.grad[..., :5, :].abs().sum()), 0)
        edited = plan.composite(torch.ones_like(source))
        self.assertTrue(torch.all(edited[..., :5, :] == .5))
        self.assertTrue(torch.all(edited[..., 7:11, 6:10] == 1))

    def test_empty_plan_is_no_edit(self):
        source = torch.rand(1, 3, 16, 16)
        plan = FixedRepairPlan(source, torch.ones_like(source), torch.zeros(1, 1, 16, 16))
        self.assertEqual(float(plan(source)['loss']), 0)
        self.assertTrue(torch.equal(plan.composite(torch.zeros_like(source)), source))


if __name__ == '__main__':unittest.main()
