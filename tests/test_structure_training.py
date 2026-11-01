import unittest
import torch

from stroke_model.extraction_training import ExtractionTrainingConfig, extraction_training_loss
from stroke_model.reference_model import partition_loss
from stroke_model.stroke_structure import StructureConfig, StrokeStructureLoss


class ExtractionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reference = torch.zeros(2, 32, 32)
        cls.reference[0, 16, 4:28] = 1
        cls.reference[1, 4:28, 10] = 1
        masks = torch.nn.functional.max_pool2d(cls.reference[None], 3, 1, 1)
        cls.image = masks.amax(1, keepdim=True)
        cls.labels = torch.where(masks.sum(1) == 1, masks.argmax(1), -1)
        cls.base_logits = masks*5
        cls.topology_valid = torch.ones(1, 2, dtype=torch.bool)
        cls.criterion = StrokeStructureLoss(cls.reference,
            StructureConfig(anchor_spacing=6, max_pairs=12))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def output(self, logits):
        p = logits.softmax(1)
        return {'logits': logits, 'probabilities': p, 'masks': self.image*p,
                'active': torch.ones(1, 2, dtype=torch.bool),
                'flow': torch.zeros(1, 2, 32, 32)}

    def test_supervised_only_matches_original_loss(self):
        output = self.output(self.base_logits.clone().requires_grad_())
        expected = partition_loss(output, self.labels)
        result = extraction_training_loss(output, self.labels,
                                          config=ExtractionTrainingConfig(structure_weight=0))
        torch.testing.assert_close(result['loss'], expected['loss'])

    def test_live_student_gradient_and_detached_teacher(self):
        logits = self.base_logits.clone().requires_grad_()
        teacher_logits = self.base_logits.clone().requires_grad_()
        output, teacher = self.output(logits), self.output(teacher_logits)
        structure = self.criterion(output['masks'])
        result = extraction_training_loss(output, self.labels, structure, teacher,
                                          topology_valid=self.topology_valid)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher_logits.grad)
        self.assertNotIn('reference_deficit', result)

    def test_deleting_channel_triggers_mass_guard(self):
        teacher = self.output(self.base_logits)
        normal = self.output(self.base_logits.clone().requires_grad_())
        old = extraction_training_loss(normal, self.labels, self.criterion(normal['masks']), teacher,
                                       topology_valid=self.topology_valid)
        collapsed = torch.zeros_like(self.base_logits)
        collapsed[:, 0] = -15
        collapsed[:, 1] = 15
        bad = self.output(collapsed.requires_grad_())
        new = extraction_training_loss(bad, self.labels, self.criterion(bad['masks']), teacher,
                                       topology_valid=self.topology_valid)
        self.assertAlmostEqual(old['mass_guard'].item(), 0)
        self.assertGreater(new['mass_guard'].item(), .3)
        self.assertGreater(new['supervised'].item(), old['supervised'].item())

    def test_requires_safeguards_and_live_converged_structure(self):
        output = self.output(self.base_logits.clone().requires_grad_())
        teacher = self.output(self.base_logits)
        with self.assertRaisesRegex(ValueError, 'requires'):
            extraction_training_loss(output, self.labels)
        structure = self.criterion(output['masks'])
        detached = {**structure, 'per_stroke': {key: value.detach() for key, value in structure['per_stroke'].items()}}
        with self.assertRaisesRegex(ValueError, 'detach'):
            extraction_training_loss(output, self.labels, detached, teacher,
                                     topology_valid=self.topology_valid)
        with self.assertRaisesRegex(ValueError, 'mass_guard'):
            extraction_training_loss(output, self.labels, structure, teacher,
                                     ExtractionTrainingConfig(mass_guard_weight=0), topology_valid=self.topology_valid)
        truncated = {**structure, 'convergence': [{'pairs': False, 'attachment': True}]}
        with self.assertRaisesRegex(ValueError, 'converge'):
            extraction_training_loss(output, self.labels, truncated, teacher,
                                     topology_valid=self.topology_valid)

    def test_unknown_or_damaged_source_not_forced_to_be_complete(self):
        output = self.output(self.base_logits.clone().requires_grad_())
        teacher = self.output(self.base_logits)
        structure = self.criterion(output['masks'])
        with self.assertRaisesRegex(ValueError, 'topology_valid'):
            extraction_training_loss(output, self.labels, structure, teacher)
        result = extraction_training_loss(output, self.labels, structure, teacher,
                                          topology_valid=torch.zeros_like(self.topology_valid))
        self.assertEqual(result['structure'].item(), 0)
        torch.testing.assert_close(result['loss'], result['supervised'])


if __name__ == '__main__':
    unittest.main()
