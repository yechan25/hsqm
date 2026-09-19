"""Offline real-Swin smoke test; random weights, not an accuracy evaluation."""
import json
import torch
from stroke_model.reference_model import ReferenceStrokeModel, partition_loss, stroke_moments

torch.set_num_threads(2)
torch.manual_seed(9)
model = ReferenceStrokeModel(pretrained=False)
image = torch.rand(1, 1, 128, 128)
strokes = torch.zeros(1, 3, 128, 128)
strokes[:, 0, 20:30, 10:100] = 1
strokes[:, 1, 20:110, 60:70] = 1
reference = strokes.amax(1, keepdim=True)
labels = torch.full((1, 128, 128), -1, dtype=torch.long)
labels[:, 20:30, 10:100] = 0
labels[:, 20:110, 60:70] = 1
out = model(image, reference, strokes)
loss = partition_loss(out, labels)['loss']
loss.backward()
assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert all(p.grad is None for p in model.backbone.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
model.zero_grad(set_to_none=True)
model.eval().requires_grad_(False)
image = image.requires_grad_()
out = model(image, reference, strokes)
torch.testing.assert_close(out['masks'].sum(1, keepdim=True), image)
objective = out['probabilities'][:, 0, :64].sum() + stroke_moments(out['masks'])[:, :2].square().mean()
gradient, = torch.autograd.grad(objective, image)
assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
print(json.dumps({'shape': list(out['masks'].shape), 'trainable': trainable,
                  'total': sum(p.numel() for p in model.parameters()),
                  'image_gradient_l1': gradient.abs().sum().item(),
                  'random_weight_training_loss': loss.item(), 'pretrained_weights_used': False}))
