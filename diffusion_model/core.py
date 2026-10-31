"""Epsilon-prediction DDPM, EMA and differentiable auxiliary objectives."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


def build_model(size=128, small=False):
    from diffusers import UNet2DModel
    widths = (16, 32) if small else (64, 128, 128, 256)
    return UNet2DModel(sample_size=size, in_channels=2, out_channels=1,
                       layers_per_block=1 if small else 2, block_out_channels=widths,
                       down_block_types=('DownBlock2D',) * (len(widths)-1) + ('AttnDownBlock2D',),
                       up_block_types=('AttnUpBlock2D',) + ('UpBlock2D',) * (len(widths)-1),
                       norm_num_groups=8, attention_head_dim=8)


def clean_from_epsilon(noisy, epsilon, timesteps, alphas):
    alpha = alphas.to(noisy.device)[timesteps].float().view(-1, 1, 1, 1)
    return (noisy.float() - (1-alpha).sqrt() * epsilon.float()) / alpha.sqrt().clamp_min(1e-6)


class EMA:
    def __init__(self, model, decay=.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1-self.decay)
            else:
                self.shadow[key].copy_(value)

    @contextmanager
    def apply(self, model):
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(original)


class OCRLoss:
    """Frozen EasyOCR recognizer, tensor-only preprocessing and target CTC."""
    def __init__(self, texts, device):
        import easyocr
        reader = easyocr.Reader(['ko', 'en'], gpu=device.type == 'cuda', detector=False,
                                quantize=False, verbose=False)
        self.model = reader.recognizer.to(device).eval().requires_grad_(False)
        self.table = {char: i+1 for i, char in enumerate(reader.character)}
        for text in texts:
            if not text or len(text) != 1 or any(char not in self.table for char in text):
                raise ValueError(f'OCR expects one supported NFC character per sample: {text!r}')

    def __call__(self, ink, texts):
        image = F.interpolate(1-ink.float(), (64, 128), mode='bilinear', align_corners=False) * 2-1
        # cuDNN RNN eval forward cannot backprop to its input. Use native RNN.
        with torch.backends.cudnn.flags(enabled=False):
            logits = self.model(image, None).float()
        log_probs = logits.log_softmax(-1).transpose(0, 1).contiguous()
        targets = torch.tensor([self.table[c] for text in texts for c in text], device=ink.device)
        target_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
        input_lengths = torch.full_like(target_lengths, log_probs.shape[0])
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                          blank=0, reduction='none', zero_infinity=False)
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Non-finite CTC loss; check vocabulary and sequence lengths')
        return (loss / target_lengths.to(ink.device)).mean()


class StrokeLoss:
    """Normal-pair reconstruction supervision, NOT a validated stroke-error score."""
    def __init__(self, checkpoint, size, device):
        from stroke_model.reference_model import ReferenceStrokeModel
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('architecture') != 'reference_v2' or state.get('image_size') != size:
            raise ValueError('Need a trained reference_v2 checkpoint with matching image_size')
        self.model = ReferenceStrokeModel(image_size=size, width=state.get('width', 64),
                                          pretrained=False).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(state['model'], strict=True)

    def __call__(self, predicted, target, reference, strokes):
        if strokes is None:
            raise ValueError('Stroke auxiliary loss needs per-sample reference strokes in NPZ')
        with torch.no_grad():
            truth = self.model(target, reference, strokes)['masks']
        masks = self.model(predicted, reference, strokes)['masks']
        active = strokes.flatten(2).sum(-1) > 0
        # Include empty target strokes: extra predicted mass must still be penalized.
        mass_error = (masks.mean((-2, -1)) - truth.mean((-2, -1))).abs()
        dice = (2 * (masks*truth).sum((-2, -1)) + 1e-5) / (
            masks.square().sum((-2, -1)) + truth.square().sum((-2, -1)) + 1e-5)
        return ((1-dice + mass_error)[active]).mean()


@torch.no_grad()
def sample(model, scheduler, references, seed=42, steps=50):
    from diffusers import DDIMScheduler
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(steps, device=references.device)
    generator = torch.Generator(device=references.device).manual_seed(seed)
    noisy = torch.randn(references.shape, device=references.device, generator=generator)
    was_training = model.training
    model.eval()
    try:
        for timestep in sampler.timesteps:
            epsilon = model(torch.cat((noisy, references*2-1), 1), timestep).sample
            noisy = sampler.step(epsilon, timestep, noisy, eta=0).prev_sample
    finally:
        model.train(was_training)
    return ((noisy+1)/2).clamp(0, 1)
