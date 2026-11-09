"""Trainable, character-conditioned extra-ink detector (untrained architecture).

No clean/template RGB enters this head. Existing extractor probabilities are
features, not labels. Checkpoint compatibility with V2 is NOT implied.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'stroke_error_localizer_v1'
STROKE_TYPES = ('unknown', 'horizontal', 'vertical', 'diagonal', 'curve', 'bent', 'loop')
RELATION_TYPES = ('unknown', 'contact_expected', 'separate_expected', 'continuation_expected')


@dataclass(frozen=True)
class ErrorLocalizerConfig:
    width: int = 32
    def __post_init__(self):
        if type(self.width) is not int or self.width < 8 or self.width % 4:
            raise ValueError('width must be a multiple of 4, at least 8')


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1),
                         nn.GroupNorm(4, cout), nn.GELU())


def _unit(tensor, name):
    if not torch.isfinite(tensor).all() or (tensor < 0).any() or (tensor > 1).any():
        raise ValueError(f'{name} must be finite in [0,1]')


def hangul_tokens(codepoints):
    """Unicode syllable -> initial/medial/final IDs. Final ID 0 means absent."""
    if codepoints.ndim != 1 or codepoints.dtype != torch.long:
        raise ValueError('One int64 Unicode codepoint per image required')
    syllable = codepoints - 0xAC00
    if ((syllable < 0) | (syllable >= 11172)).any():
        raise ValueError('V1 supports precomposed Hangul syllables only')
    return syllable // 588, syllable % 588 // 28, syllable % 28


class StrokeErrorLocalizer(nn.Module):
    architecture = ARCHITECTURE

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ErrorLocalizerConfig()
        w = self.config.width
        self.initial = nn.Embedding(19, w)
        self.medial = nn.Embedding(21, w)
        self.final = nn.Embedding(28, w)
        self.stroke_type = nn.Embedding(len(STROKE_TYPES), w, padding_idx=0)
        self.relation_type = nn.Embedding(len(RELATION_TYPES), w, padding_idx=0)
        self.relation_message = nn.Sequential(nn.Linear(2 * w, w), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(6, w), nn.GELU(), nn.Linear(w, w))
        self.stroke_pixels = nn.Sequential(block(6, w), block(w, w))
        self.stroke_context = nn.Linear(w, w)
        self.context = nn.Linear(w * 2, w)
        self.enc0 = block(5 + 2 * w, w)
        self.enc1 = block(w, 2 * w, 2)
        self.enc2 = block(2 * w, 4 * w, 2)
        self.film = nn.Linear(2 * w, 8 * w)
        self.dec1 = block(6 * w, 2 * w)
        self.dec0 = block(3 * w, w)
        self.extra = nn.Conv2d(w, 1, 1)
        self.kind = nn.Conv2d(w, 2, 1)  # attached branch / disconnected extra
        self.has_error = nn.Linear(w, 1)  # supervised presence, NOT uncertainty

    def forward(self, rgb, probabilities, codepoints, active=None,
                stroke_types=None, relations=None):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 16:
            raise ValueError('RGB must be B,3,H,W with H,W >=16')
        b, _, h, w = rgb.shape
        if probabilities.ndim != 4 or probabilities.shape[0] != b or probabilities.shape[-2:] != (h, w):
            raise ValueError('Probabilities must be B,K,H,W')
        k = probabilities.shape[1]
        if k < 1 or codepoints.shape != (b,):
            raise ValueError('At least one stroke and one character per sample required')
        _unit(rgb, 'RGB'); _unit(probabilities, 'Probabilities')
        if active is None:
            active = torch.ones(b, k, device=rgb.device, dtype=torch.bool)
        if active.shape != (b, k) or active.dtype != torch.bool or not active.any(1).all():
            raise ValueError('Each sample needs at least one active stroke')
        # Prevent the new head from optimizing the frozen extractor's ownership.
        p = probabilities.detach() * active[:, :, None, None]
        total = p.sum(1, keepdim=True)
        if (total <= 0).any():
            raise ValueError('Active probabilities must cover every pixel')
        p = p / total
        ink = 1 - (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        entropy = -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype),
                                torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype), indexing='ij')
        xy = torch.stack((xx, yy))
        masks = ink * p
        mass = masks.sum((-2, -1)).clamp_min(1e-6)
        cx = (masks * xx).sum((-2, -1)) / mass
        cy = (masks * yy).sum((-2, -1)) / mass
        dx, dy = xx - cx[..., None, None], yy - cy[..., None, None]
        geom = torch.stack((mass * active / (mass * active).sum(1, keepdim=True), cx, cy,
                            (masks * dx.square()).sum((-2, -1)) / mass,
                            (masks * dy.square()).sum((-2, -1)) / mass,
                            (masks * dx * dy).sum((-2, -1)) / mass), -1)
        if stroke_types is None:
            stroke_types = torch.zeros(b, k, device=rgb.device, dtype=torch.long)
        if relations is None:
            relations = torch.zeros(b, k, k, device=rgb.device, dtype=torch.long)
        if stroke_types.shape != (b, k) or relations.shape != (b, k, k):
            raise ValueError('Invalid stroke/relationship metadata shape')
        if stroke_types.dtype != torch.long or relations.dtype != torch.long:
            raise ValueError('Stroke/relationship metadata must be int64')
        if ((stroke_types < 0) | (stroke_types >= len(STROKE_TYPES))).any() or ((relations < 0) | (relations >= len(RELATION_TYPES))).any():
            raise ValueError('Unknown stroke/relationship metadata ID')
        neighbors = active[:, None, :].expand(b, k, k) & ~torch.eye(k, device=rgb.device, dtype=torch.bool)[None]
        base_tokens = self.geometry(geom) + self.stroke_type(stroke_types)
        known_edges = neighbors & (relations != 0)
        messages = self.relation_message(torch.cat((self.relation_type(relations),
            base_tokens[:, None].expand(b, k, k, -1)), -1))
        edge = (messages * known_edges[..., None]).sum(2) / known_edges.sum(2, keepdim=True).clamp_min(1)
        tokens = base_tokens + edge
        valid = active[..., None]
        pooled = torch.cat(((tokens * valid).sum(1) / valid.sum(1),
                            tokens.masked_fill(~valid, -torch.inf).amax(1)), 1)
        context = self.context(pooled)
        def repeat(t):
            return t[:, None].expand(b, k, *t.shape[1:])
        pixels = torch.cat((repeat(ink), p[:, :, None], masks[:, :, None],
                            xy[None, None].expand(b, k, 2, h, w), repeat(entropy)), 2)
        pixels = self.stroke_pixels(pixels.reshape(b * k, 6, h, w)).reshape(b, k, -1, h, w)
        pixels = pixels + self.stroke_context(tokens)[..., None, None]
        pixel_mean = (pixels * valid[..., None, None]).sum(1) / valid.sum(1)[..., None, None]
        pixel_max = pixels.masked_fill(~valid[..., None, None], -torch.inf).amax(1)
        c0 = self.enc0(torch.cat((rgb, ink, entropy, pixel_mean, pixel_max), 1))
        c1 = self.enc1(c0); c2 = self.enc2(c1)
        l, v, t = hangul_tokens(codepoints)
        char = self.initial(l) + self.medial(v) + self.final(t)
        gain, bias = self.film(torch.cat((char, context), 1)).chunk(2, 1)
        c2 = c2 * (1 + .1 * gain[..., None, None].tanh()) + bias[..., None, None]
        d1 = self.dec1(torch.cat((F.interpolate(c2, c1.shape[-2:], mode='bilinear', align_corners=False), c1), 1))
        features = self.dec0(torch.cat((F.interpolate(d1, c0.shape[-2:], mode='bilinear', align_corners=False), c0), 1))
        logits = self.extra(features)
        return dict(extra_logits=logits, extra_probability=logits.sigmoid(),
                    kind_logits=self.kind(features),
                    has_error_logit=self.has_error(features.mean((-2, -1))), features=features)


class FrozenExtractorErrorLocalizer(nn.Module):
    """V2 still uses a standard template to identify stroke IDs, not a clean target."""
    def __init__(self, extractor, localizer):
        super().__init__()
        self.extractor = extractor.eval().requires_grad_(False)
        self.localizer = localizer

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, rgb, reference_ink, reference_strokes, codepoints, active=None, **metadata):
        ink = 1 - (rgb * rgb.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        if active is None:
            active = reference_strokes.sum((-2, -1)) > 0
        with torch.no_grad():
            result = self.extractor(ink, reference_ink, reference_strokes, active=active)
        return self.localizer(rgb, result['probabilities'], codepoints, active, **metadata)


def error_localizer_loss(output, extra_target, valid, kind_target=None):
    """Dense SYNTHETIC/verified annotations only; unknown pixels are excluded."""
    logits = output['extra_logits']
    if extra_target.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError('Target and valid must match B,1,H,W logits')
    _unit(extra_target, 'Extra labels'); _unit(valid, 'Validity')
    y, v = extra_target.to(logits), valid.to(logits)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    pt = probability * y + (1 - probability) * (1 - y)
    focal = (bce * (1 - pt).square() * v).sum() / v.sum().clamp_min(1)
    intersection = (probability * y * v).sum((1, 2, 3))
    denom = ((probability + y) * v).sum((1, 2, 3))
    present = v.sum((1, 2, 3)) > 0
    dice_rows = 1 - (2 * intersection + 1e-6) / (denom + 1e-6)
    dice = (dice_rows * present).sum() / present.sum().clamp_min(1)
    kinds = logits.sum() * 0
    if kind_target is not None:
        if kind_target.shape != logits[:, 0].shape or kind_target.dtype != torch.long:
            raise ValueError('Kind labels must be B,H,W int64: -1 unknown, 0 normal, 1 attached, 2 detached')
        if ((kind_target < -1) | (kind_target > 2)).any():
            raise ValueError('Invalid kind label')
        known = (kind_target > 0) & (v[:, 0] > 0) & (y[:, 0] > .5)
        target = (kind_target - 1).clamp_min(0)
        ce = F.cross_entropy(output['kind_logits'], target, reduction='none')
        kinds = (ce * known).sum() / known.sum().clamp_min(1)
    has_error = (y * v).flatten(1).amax(1)[:, None]
    presence_rows = F.binary_cross_entropy_with_logits(output['has_error_logit'], has_error, reduction='none')[:, 0]
    # A partially annotated image with no known error is not a clean negative.
    presence_valid = present & ((has_error[:, 0] > .5) | (v == 1).flatten(1).all(1))
    presence = (presence_rows * presence_valid).sum() / presence_valid.sum().clamp_min(1)
    total = focal + dice + .2 * kinds + .1 * presence
    return dict(loss=total, focal=focal, dice=dice, kind=kinds, presence=presence)
